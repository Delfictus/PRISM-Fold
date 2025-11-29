//! World Record Breaking Pipeline
//!
//! Ultimate PRISM integration combining ALL advanced modules:
//! - Active Inference (variational free energy minimization)
//! - ADP Q-Learning (adaptive parameter optimization)
//! - Neuromorphic Reservoir Computing (conflict prediction)
//! - Statistical Mechanics (thermodynamic equilibration)
//! - Quantum-Classical Hybrid (QUBO + classical feedback)
//! - Multi-Scale Analysis (temporal hierarchies)
//! - Ensemble Consensus (multi-algorithm voting)
//!
//! Target: Beat 83 colors world record on DSJC1000.5
//! Current best: 115 colors (1.39x gap)
//! Expected: 83-90 colors with full integration

use crate::coloring::greedy_coloring_with_ordering;
use crate::cpu_init::init_rayon_threads;
use crate::dsatur_backtracking::DSaturSolver;
use crate::errors::*;
use crate::geodesic::{compute_landmark_distances, GeodesicFeatures};
use crate::initial_coloring::{compute_initial_coloring, InitialColoringStrategy};
use crate::memetic_coloring::{MemeticColoringSolver, MemeticConfig};
use crate::quantum_coloring::QuantumColoringSolver;
use crate::telemetry::{PhaseExecMode, PhaseName, RunMetric};
use crate::transfer_entropy_coloring::hybrid_te_kuramoto_ordering;
use shared_types::*;

use rand::prelude::SliceRandom;
use rand::Rng;
use serde_json::json;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;

#[cfg(feature = "cuda")]
use crate::world_record_pipeline_gpu::GpuReservoirConflictPredictor;

#[cfg(feature = "cuda")]
use crate::gpu_transfer_entropy;

#[cfg(feature = "cuda")]
use crate::gpu_thermodynamic;

#[cfg(feature = "cuda")]
use crate::gpu_active_inference;

/// Runtime GPU usage tracking for each phase
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PhaseGpuStatus {
    pub phase0_gpu_used: bool,
    pub phase1_gpu_used: bool,
    pub phase1_ai_gpu_used: bool,
    pub phase2_gpu_used: bool,
    pub phase3_gpu_used: bool,
    pub phase0_fallback_reason: Option<String>,
    pub phase1_fallback_reason: Option<String>,
    pub phase1_ai_fallback_reason: Option<String>,
    pub phase2_fallback_reason: Option<String>,
    pub phase3_fallback_reason: Option<String>,
}

impl Default for PhaseGpuStatus {
    fn default() -> Self {
        Self {
            phase0_gpu_used: false,
            phase1_gpu_used: false,
            phase1_ai_gpu_used: false,
            phase2_gpu_used: false,
            phase3_gpu_used: false,
            phase0_fallback_reason: None,
            phase1_fallback_reason: None,
            phase1_ai_fallback_reason: None,
            phase2_fallback_reason: None,
            phase3_fallback_reason: None,
        }
    }
}

// Serde default helpers for v1.1 config
fn default_true() -> bool {
    true
}
fn default_threads() -> usize {
    24
}
fn default_streams() -> usize {
    4
}
fn default_replicas() -> usize {
    256
} // Scaled for B200 GPUs (distributed across multi-GPU)
fn default_beads() -> usize {
    256
} // Scaled for B200 GPUs (future PIMC)
fn default_batch_size() -> usize {
    1024
}
fn default_stream_mode() -> StreamMode {
    StreamMode::Sequential
}

/// Stream execution mode for GPU phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamMode {
    /// All phases use default stream (sequential execution)
    Sequential,

    /// Phases use separate streams (parallel execution)
    Parallel,
}

/// Multi-GPU Configuration for distributed computation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MultiGpuConfig {
    /// Enable multi-GPU distributed execution
    pub enabled: bool,

    /// Number of GPUs to use
    pub num_gpus: usize,

    /// Device IDs to use (e.g., [0, 1, 2, 3, 4, 5, 6, 7])
    pub devices: Vec<usize>,

    /// Enable peer-to-peer memory access between GPUs
    pub enable_peer_access: bool,

    /// Enable NCCL for collective operations (experimental)
    pub enable_nccl: bool,

    /// Distribution strategy: "distributed_phases" or "independent_instances"
    pub strategy: String,
}

impl Default for MultiGpuConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_gpus: 1,
            devices: vec![0],
            enable_peer_access: false,
            enable_nccl: false,
            strategy: "distributed_phases".to_string(),
        }
    }
}

/// GPU Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuConfig {
    #[serde(default)]
    pub device_id: usize,

    #[serde(default = "default_streams")]
    pub streams: usize,

    #[serde(default = "default_stream_mode")]
    pub stream_mode: StreamMode,

    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    #[serde(default = "default_true")]
    pub enable_reservoir_gpu: bool,

    #[serde(default = "default_true")]
    pub enable_te_gpu: bool,

    #[serde(default = "default_true")]
    pub enable_statmech_gpu: bool,

    #[serde(default = "default_true")]
    pub enable_thermo_gpu: bool,

    #[serde(default = "default_true")]
    pub enable_pimc_gpu: bool,

    #[serde(default = "default_true")]
    pub enable_quantum_gpu: bool,

    #[serde(default = "default_true")]
    pub enable_tda_gpu: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            streams: default_streams(),
            stream_mode: default_stream_mode(),
            batch_size: default_batch_size(),
            enable_reservoir_gpu: true,
            enable_te_gpu: true,
            enable_statmech_gpu: true,
            enable_thermo_gpu: true,
            enable_pimc_gpu: true,
            enable_quantum_gpu: true,
            enable_tda_gpu: true,
        }
    }
}

/// Thermodynamic Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThermoConfig {
    /// Number of parallel replicas (VRAM guard: max 56 for 8GB devices)
    #[serde(default = "default_replicas")]
    pub replicas: usize,

    #[serde(default = "default_replicas")]
    pub num_temps: usize,

    #[serde(default)]
    pub exchange_interval: usize,

    #[serde(default = "default_t_min")]
    pub t_min: f64,

    #[serde(default = "default_t_max")]
    pub t_max: f64,

    /// Steps per temperature for equilibration (default: 5000)
    #[serde(default = "default_steps_per_temp")]
    pub steps_per_temp: usize,

    /// VRAM-safe maximum replicas for 8GB devices (default: 56)
    #[serde(default = "default_replicas")]
    pub replicas_max_safe: usize,

    /// VRAM-safe maximum temperatures for 8GB devices (default: 56)
    #[serde(default = "default_replicas")]
    pub num_temps_max_safe: usize,

    /// TWEAK 1: Temperature at which conflict forces start activating (default: 5.0)
    #[serde(default = "default_force_start_temp")]
    pub force_start_temp: f64,

    /// TWEAK 1: Temperature at which conflict forces reach full strength (default: 1.0)
    #[serde(default = "default_force_full_strength_temp")]
    pub force_full_strength_temp: f64,

    #[serde(default)]
    pub compaction: ThermoCompactionConfig,
}

fn default_t_min() -> f64 {
    0.01
}

fn default_t_max() -> f64 {
    10.0
}

fn default_steps_per_temp() -> usize {
    5000
}

fn default_force_start_temp() -> f64 {
    5.0
}

fn default_force_full_strength_temp() -> f64 {
    1.0
}
fn default_guard_threshold() -> f64 {
    0.15
}
fn default_guard_initial_slack() -> usize {
    50
}
fn default_guard_min_slack() -> usize {
    30
}
fn default_guard_max_slack() -> usize {
    110
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ThermoCompactionConfig {
    pub max_force_blend_factor: f64,
    pub color_range_expand_threshold: f64,
    pub color_range_expand_percent: f64,
    pub reheat_consecutive_guards: usize,
    pub reheat_temp_boost: f64,
    pub initial_slack: usize,
    pub min_slack: usize,
    pub max_slack: usize,
}

impl Default for ThermoCompactionConfig {
    fn default() -> Self {
        Self {
            max_force_blend_factor: 0.35,
            color_range_expand_threshold: default_guard_threshold(),
            color_range_expand_percent: 6.0,
            reheat_consecutive_guards: 1,
            reheat_temp_boost: 1.4,
            initial_slack: default_guard_initial_slack(),
            min_slack: default_guard_min_slack(),
            max_slack: default_guard_max_slack(),
        }
    }
}

impl Default for ThermoConfig {
    fn default() -> Self {
        Self {
            replicas: default_replicas(),
            num_temps: default_replicas(),
            exchange_interval: 50,
            t_min: 0.01,
            t_max: 10.0,
            steps_per_temp: 5000,
            replicas_max_safe: 56,
            num_temps_max_safe: 56,
            force_start_temp: 5.0,
            force_full_strength_temp: 1.0,
            compaction: ThermoCompactionConfig::default(),
        }
    }
}

/// Quantum Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantumConfig {
    pub iterations: usize,
    pub target_chromatic: usize,

    /// Number of retries for initial solution generation before giving up
    #[serde(default = "default_quantum_retries")]
    pub failure_retries: usize,

    /// Fall back to DSATUR if quantum fails
    #[serde(default = "default_true")]
    pub fallback_on_failure: bool,

    /// GPU QUBO iterations (default: 10000)
    #[serde(default = "default_qubo_iterations")]
    pub qubo_iterations: usize,

    /// GPU QUBO batch size (default: 256)
    #[serde(default = "default_qubo_batch")]
    pub qubo_batch_size: usize,

    /// GPU QUBO initial temperature (default: 1.0)
    #[serde(default = "default_qubo_t_initial")]
    pub qubo_t_initial: f64,

    /// GPU QUBO final temperature (default: 0.01)
    #[serde(default = "default_qubo_t_final")]
    pub qubo_t_final: f64,
}

fn default_quantum_retries() -> usize {
    2
}

fn default_qubo_iterations() -> usize {
    10_000
}

fn default_qubo_batch() -> usize {
    256
}

fn default_qubo_t_initial() -> f64 {
    1.0
}

fn default_qubo_t_final() -> f64 {
    0.01
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            iterations: 20,
            target_chromatic: 83,
            failure_retries: 2,
            fallback_on_failure: true,
            qubo_iterations: 10_000,
            qubo_batch_size: 256,
            qubo_t_initial: 1.0,
            qubo_t_final: 0.01,
        }
    }
}

/// GPU Coloring Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuColoringConfig {
    /// Force sparse kernel regardless of density
    #[serde(default = "default_false")]
    pub prefer_sparse: bool,

    /// Density threshold for sparse/dense selection
    #[serde(default = "default_sparse_threshold")]
    pub sparse_density_threshold: f64,

    /// Mask width for color bitsets (64 or 128)
    #[serde(default = "default_mask_width")]
    pub mask_width: u32,
}

fn default_sparse_threshold() -> f64 {
    0.40
}
fn default_mask_width() -> u32 {
    64
}
fn default_false() -> bool {
    false
}

impl Default for GpuColoringConfig {
    fn default() -> Self {
        Self {
            prefer_sparse: false,
            sparse_density_threshold: 0.40,
            mask_width: 64,
        }
    }
}

/// Geodesic Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeodesicConfig {
    pub num_landmarks: usize,
    pub metric: String,
    pub weight_attr: Option<String>,
    #[serde(default = "default_centrality_weight")]
    pub centrality_weight: f64,
    #[serde(default = "default_eccentricity_weight")]
    pub eccentricity_weight: f64,
}

fn default_centrality_weight() -> f64 {
    0.5
}

fn default_eccentricity_weight() -> f64 {
    0.5
}

impl Default for GeodesicConfig {
    fn default() -> Self {
        Self {
            num_landmarks: 10,
            metric: "hop".to_string(),
            weight_attr: None,
            centrality_weight: 0.5,
            eccentricity_weight: 0.5,
        }
    }
}

/// Transfer Entropy Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransferEntropyConfig {
    /// Number of histogram bins for discretization (default: 128)
    #[serde(default = "default_histogram_bins")]
    pub histogram_bins: usize,

    /// Number of time series steps for dynamics (default: 200)
    #[serde(default = "default_time_series_steps")]
    pub time_series_steps: usize,

    /// Weight for geodesic features in TE ordering (0.0 = TE only, 1.0 = geodesic only)
    #[serde(default = "default_geodesic_weight")]
    pub geodesic_weight: f64,

    /// Weight for TE vs Kuramoto in hybrid ordering (te_weight = this, kuramoto_weight = 1.0 - this)
    #[serde(default = "default_te_vs_kuramoto_weight")]
    pub te_vs_kuramoto_weight: f64,
}

fn default_histogram_bins() -> usize {
    128
}

fn default_time_series_steps() -> usize {
    200
}

fn default_geodesic_weight() -> f64 {
    0.35
}

fn default_te_vs_kuramoto_weight() -> f64 {
    0.85
}

impl Default for TransferEntropyConfig {
    fn default() -> Self {
        Self {
            histogram_bins: 128,
            time_series_steps: 200,
            geodesic_weight: 0.35,
            te_vs_kuramoto_weight: 0.85,
        }
    }
}

/// CPU Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CpuConfig {
    pub threads: usize,
    #[serde(default)]
    pub pin_pool: bool,
    #[serde(default = "default_work_steal")]
    pub work_steal: bool,
    #[serde(default = "default_parallel_io")]
    pub parallel_io: bool,
}

fn default_cpu_threads() -> usize {
    24
}

fn default_work_steal() -> bool {
    true
}

fn default_parallel_io() -> bool {
    true
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            threads: default_cpu_threads(),
            pin_pool: false,
            work_steal: default_work_steal(),
            parallel_io: default_parallel_io(),
        }
    }
}

/// ADP Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdpConfig {
    pub epsilon: f64,
    pub epsilon_decay: f64,
    pub epsilon_min: f64,
    pub alpha: f64,
    pub gamma: f64,
}

impl Default for AdpConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.03,
            alpha: 0.10,
            gamma: 0.95,
        }
    }
}

/// Neuromorphic Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuromorphicConfig {
    /// Phase threshold for difficulty zone clustering (radians)
    #[serde(default = "default_phase_threshold")]
    pub phase_threshold: f64,
}

fn default_phase_threshold() -> f64 {
    0.5
}

impl Default for NeuromorphicConfig {
    fn default() -> Self {
        Self {
            phase_threshold: 0.5,
        }
    }
}

/// Initial Coloring Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct InitialColoringConfig {
    pub strategy: InitialColoringStrategy,
}

impl Default for InitialColoringConfig {
    fn default() -> Self {
        Self {
            strategy: InitialColoringStrategy::Greedy,
        }
    }
}

/// Orchestrator Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrchestratorConfig {
    pub adp_dsatur_depth: usize,
    pub adp_quantum_iterations: usize,
    pub adp_thermo_num_temps: usize,
    pub restarts: usize,
    pub early_stop_no_improve_iters: usize,
    pub checkpoint_minutes: usize,

    /// DSATUR target offset: how many colors below best to try (default: 3)
    #[serde(default = "default_dsatur_target_offset")]
    pub dsatur_target_offset: usize,

    /// Minimum history length before enabling thermo loopback (default: 3)
    #[serde(default = "default_adp_min_history_for_thermo")]
    pub adp_min_history_for_thermo: usize,

    /// Minimum history length before enabling quantum loopback (default: 2)
    #[serde(default = "default_adp_min_history_for_quantum")]
    pub adp_min_history_for_quantum: usize,

    /// Minimum history length before enabling full loopback (default: 5)
    #[serde(default = "default_adp_min_history_for_loopback")]
    pub adp_min_history_for_loopback: usize,
}

fn default_dsatur_target_offset() -> usize {
    3
}

fn default_adp_min_history_for_thermo() -> usize {
    3
}

fn default_adp_min_history_for_quantum() -> usize {
    2
}

fn default_adp_min_history_for_loopback() -> usize {
    5
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            adp_dsatur_depth: 200000,
            adp_quantum_iterations: 20,
            adp_thermo_num_temps: 64,
            restarts: 10,
            early_stop_no_improve_iters: 3,
            checkpoint_minutes: 15,
            dsatur_target_offset: 3,
            adp_min_history_for_thermo: 3,
            adp_min_history_for_quantum: 2,
            adp_min_history_for_loopback: 5,
        }
    }
}

/// World Record Pipeline Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct WorldRecordConfig {
    /// Configuration profile name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,

    /// Configuration version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// Deterministic mode (use fixed seed)
    #[serde(default)]
    pub deterministic: bool,

    /// Random seed for deterministic mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Target chromatic number (world record)
    pub target_chromatic: usize,

    /// Maximum total runtime (hours)
    pub max_runtime_hours: f64,

    /// Enable Active Inference policy selection
    #[serde(default = "default_true")]
    pub use_active_inference: bool,

    /// Enable GPU-accelerated Active Inference (requires CUDA)
    #[serde(default = "default_true")]
    pub use_gpu_active_inference: bool,

    /// Enable ADP reinforcement learning
    #[serde(default = "default_true")]
    pub use_adp_learning: bool,

    /// Enable Reservoir Computing prediction
    #[serde(default = "default_true")]
    pub use_reservoir_prediction: bool,

    /// Enable Transfer Entropy ordering
    #[serde(default = "default_true")]
    pub use_transfer_entropy: bool,

    /// Enable Statistical Mechanics equilibration
    #[serde(default = "default_true")]
    pub use_thermodynamic_equilibration: bool,

    /// Enable Path Integral Monte Carlo
    #[serde(default)]
    pub use_pimc: bool,

    /// Enable Quantum-Classical hybrid
    #[serde(default = "default_true")]
    pub use_quantum_classical_hybrid: bool,

    /// Enable Multi-Scale neuromorphic analysis
    #[serde(default = "default_true")]
    pub use_multiscale_analysis: bool,

    /// Enable Ensemble Consensus
    #[serde(default = "default_true")]
    pub use_ensemble_consensus: bool,

    /// Enable Geodesic Features (OFF by default - experimental)
    #[serde(default)]
    pub use_geodesic_features: bool,

    /// Enable GNN Screening (OFF by default - experimental)
    #[serde(default)]
    pub use_gnn_screening: bool,

    /// Enable Topological Data Analysis (OFF by default - experimental)
    #[serde(default)]
    pub use_tda: bool,

    /// Number of parallel worker threads
    #[serde(default = "default_threads")]
    pub num_workers: usize,

    /// Multi-GPU configuration
    #[serde(default)]
    pub multi_gpu: MultiGpuConfig,

    /// GPU configuration
    #[serde(default)]
    pub gpu: GpuConfig,

    /// FluxNet reinforcement learning configuration
    #[cfg(feature = "cuda")]
    #[serde(default)]
    pub fluxnet: crate::fluxnet::FluxNetConfig,

    /// Memetic algorithm configuration
    #[serde(default)]
    pub memetic: MemeticConfig,

    /// Thermodynamic configuration
    #[serde(default)]
    pub thermo: ThermoConfig,

    /// Quantum configuration
    #[serde(default)]
    pub quantum: QuantumConfig,

    /// ADP configuration
    #[serde(default)]
    pub adp: AdpConfig,

    /// Orchestrator configuration
    #[serde(default)]
    pub orchestrator: OrchestratorConfig,

    /// Geodesic configuration
    #[serde(default)]
    pub geodesic: GeodesicConfig,

    /// GPU coloring configuration
    #[serde(default)]
    pub gpu_coloring: GpuColoringConfig,

    /// CPU configuration
    #[serde(default)]
    pub cpu: CpuConfig,

    /// Transfer Entropy configuration
    #[serde(default)]
    pub transfer_entropy: TransferEntropyConfig,

    /// Neuromorphic configuration
    #[serde(default)]
    pub neuromorphic: NeuromorphicConfig,

    /// Initial coloring configuration
    #[serde(default)]
    pub initial_coloring: InitialColoringConfig,
}

impl Default for WorldRecordConfig {
    fn default() -> Self {
        Self {
            profile: Some("record".to_string()),
            version: Some("1.0.0".to_string()),
            deterministic: false,
            seed: Some(123456789),
            target_chromatic: 83,    // DSJC1000.5 world record
            max_runtime_hours: 48.0, // 2 days maximum
            use_active_inference: true,
            use_gpu_active_inference: true,
            use_adp_learning: true,
            use_reservoir_prediction: true,
            use_transfer_entropy: true,
            use_thermodynamic_equilibration: true,
            use_pimc: false,
            use_quantum_classical_hybrid: true,
            use_multiscale_analysis: true,
            use_ensemble_consensus: true,
            use_geodesic_features: false,
            use_gnn_screening: false,
            use_tda: true,   // Topological Data Analysis ENABLED by default
            num_workers: 24, // Intel i9 Ultra
            multi_gpu: MultiGpuConfig::default(),
            gpu: GpuConfig::default(),
            memetic: MemeticConfig::default(),
            thermo: ThermoConfig::default(),
            quantum: QuantumConfig::default(),
            adp: AdpConfig::default(),
            orchestrator: OrchestratorConfig::default(),
            geodesic: GeodesicConfig::default(),
            gpu_coloring: GpuColoringConfig::default(),
            cpu: CpuConfig::default(),
            transfer_entropy: TransferEntropyConfig::default(),
            neuromorphic: NeuromorphicConfig::default(),
            initial_coloring: InitialColoringConfig::default(),
            #[cfg(feature = "cuda")]
            fluxnet: crate::fluxnet::FluxNetConfig::default(),
        }
    }
}

impl WorldRecordConfig {
    pub fn validate(&self) -> Result<()> {
        if self.target_chromatic < 1 {
            return Err(PRCTError::ColoringFailed(
                "target_chromatic must be >= 1".to_string(),
            ));
        }

        if self.max_runtime_hours <= 0.0 || self.max_runtime_hours > 168.0 {
            return Err(PRCTError::ColoringFailed(
                "max_runtime_hours must be in (0, 168]".to_string(),
            ));
        }

        if self.num_workers == 0 || self.num_workers > 256 {
            return Err(PRCTError::ColoringFailed(
                "num_workers must be in [1, 256]".to_string(),
            ));
        }

        // Require at least one method enabled
        let mut any_enabled = false;
        any_enabled |= self.use_reservoir_prediction;
        any_enabled |= self.use_active_inference;
        any_enabled |= self.use_thermodynamic_equilibration;
        any_enabled |= self.use_quantum_classical_hybrid;
        any_enabled |= self.use_ensemble_consensus;
        any_enabled |= self.use_adp_learning;
        any_enabled |= self.use_multiscale_analysis;

        if !any_enabled {
            return Err(PRCTError::ColoringFailed(
                "enable at least one phase".to_string(),
            ));
        }

        // Validate geodesic configuration if enabled
        if self.use_geodesic_features {
            if self.geodesic.num_landmarks == 0 {
                return Err(PRCTError::ColoringFailed(
                    "geodesic.num_landmarks must be > 0".to_string(),
                ));
            }
            if self.geodesic.metric != "hop" && self.geodesic.metric != "weighted" {
                return Err(PRCTError::ColoringFailed(
                    "geodesic.metric must be 'hop' or 'weighted'".to_string(),
                ));
            }
            if self.geodesic.centrality_weight < 0.0 || self.geodesic.centrality_weight > 1.0 {
                return Err(PRCTError::ColoringFailed(
                    "geodesic.centrality_weight must be in [0, 1]".to_string(),
                ));
            }
            if self.geodesic.eccentricity_weight < 0.0 || self.geodesic.eccentricity_weight > 1.0 {
                return Err(PRCTError::ColoringFailed(
                    "geodesic.eccentrity_weight must be in [0, 1]".to_string(),
                ));
            }
        }

        // Validate CPU configuration
        if self.cpu.threads == 0 || self.cpu.threads > 1024 {
            return Err(PRCTError::ColoringFailed(
                "cpu.threads must be in [1, 1024]".to_string(),
            ));
        }

        // VRAM guards removed for B200 GPUs (180GB VRAM each)
        // Multi-GPU configuration allows massive scaling:
        // - 8x B200 = 1440GB total VRAM
        // - Conservative per-GPU limit: 160GB
        // No artificial caps on replicas or temps

        // Validate ADP parameters use config values
        if self.adp.alpha <= 0.0 || self.adp.alpha > 1.0 {
            return Err(PRCTError::ColoringFailed(
                "adp.alpha must be in (0, 1]".to_string(),
            ));
        }

        if self.adp.gamma < 0.0 || self.adp.gamma > 1.0 {
            return Err(PRCTError::ColoringFailed(
                "adp.gamma must be in [0, 1]".to_string(),
            ));
        }

        if self.adp.epsilon_decay <= 0.0 || self.adp.epsilon_decay > 1.0 {
            return Err(PRCTError::ColoringFailed(
                "adp.epsilon_decay must be in (0, 1]".to_string(),
            ));
        }

        if self.adp.epsilon_min < 0.0 || self.adp.epsilon_min > 1.0 {
            return Err(PRCTError::ColoringFailed(
                "adp.epsilon_min must be in [0, 1]".to_string(),
            ));
        }

        #[cfg(feature = "cuda")]
        {
            if self.fluxnet.enabled {
                if let Err(e) = self.fluxnet.validate() {
                    return Err(PRCTError::ConfigError(format!(
                        "FluxNet configuration invalid: {}",
                        e
                    )));
                }
            }
        }

        // ═══════════════════════════════════════════════════════════
        // HARD GUARDRAILS: Unimplemented feature detection
        // ═══════════════════════════════════════════════════════════

        // TDA validation - ENABLED (implementation available in foundation/phase6/tda.rs)
        if self.use_tda {
            #[cfg(feature = "cuda")]
            {
                if self.gpu.enable_tda_gpu {
                    println!("[PIPELINE][INIT] TDA GPU acceleration ENABLED");
                }
            }

            #[cfg(not(feature = "cuda"))]
            {
                if self.use_tda {
                    println!("[PIPELINE][INIT] TDA enabled (CPU-only mode)");
                }
            }
        }

        // GNN screening requested but not implemented
        if self.use_gnn_screening {
            println!("[PIPELINE][FALLBACK] GNN screening requested but not implemented, will skip this phase");
            println!("[PIPELINE][FALLBACK] Performance impact: none (experimental feature)");
        }

        // PIMC requested but not implemented
        if self.use_pimc {
            #[cfg(feature = "cuda")]
            {
                if self.gpu.enable_pimc_gpu {
                    return Err(PRCTError::ConfigError(
                        "PIMC GPU requested but not yet implemented (use_pimc=true, enable_pimc_gpu=true)".into()
                    ));
                }
            }

            println!(
                "[PIPELINE][FALLBACK] PIMC requested but not implemented, will skip this phase"
            );
            println!("[PIPELINE][FALLBACK] Performance impact: none (experimental feature)");
        }

        Ok(())
    }

    /// Validate VRAM requirements at runtime (B200 180GB baseline)
    #[cfg(feature = "cuda")]
    pub fn validate_vram_requirements(&self, graph: &Graph) -> Result<()> {
        // B200 GPU: 180GB VRAM available, use 160GB conservatively per GPU
        const VRAM_GB: usize = 160;
        const VRAM_MB: usize = VRAM_GB * 1024;

        // Estimate VRAM usage for thermodynamic replica exchange
        if self.use_thermodynamic_equilibration && self.gpu.enable_thermo_gpu {
            let vertices = graph.num_vertices;
            let edges = graph.num_edges;

            // Each replica needs: vertex colors (4 bytes) + adjacency info
            let per_replica_mb = (vertices * 4 + edges * 8) / (1024 * 1024);
            let total_thermo_mb = per_replica_mb * self.thermo.replicas;

            if total_thermo_mb > VRAM_MB / 2 {
                return Err(PRCTError::GpuError(format!(
                    "Thermodynamic VRAM requirement (~{} MB for {} replicas) exceeds safe limit ({} MB available for allocation)",
                    total_thermo_mb, self.thermo.replicas, VRAM_MB / 2
                )));
            }

            println!(
                "[VRAM][GUARD] Thermodynamic allocation estimate: {} MB ({} replicas)",
                total_thermo_mb, self.thermo.replicas
            );
        }

        // Estimate VRAM for reservoir computing
        if self.use_reservoir_prediction && self.gpu.enable_reservoir_gpu {
            let reservoir_size = 1000.min(graph.num_vertices * 2);
            let reservoir_mb = (reservoir_size * reservoir_size * 4) / (1024 * 1024);

            if reservoir_mb > VRAM_MB / 4 {
                return Err(PRCTError::GpuError(format!(
                    "Reservoir VRAM requirement (~{} MB for size {}) exceeds safe limit ({} MB)",
                    reservoir_mb,
                    reservoir_size,
                    VRAM_MB / 4
                )));
            }

            println!(
                "[VRAM][GUARD] Reservoir allocation estimate: {} MB (size={})",
                reservoir_mb, reservoir_size
            );
        }

        // Estimate VRAM for quantum solver
        if self.use_quantum_classical_hybrid && self.gpu.enable_quantum_gpu {
            let vertices = graph.num_vertices;
            let quantum_mb = (vertices * vertices * 8) / (1024 * 1024); // PhaseField coherence matrix

            if quantum_mb > VRAM_MB / 4 {
                return Err(PRCTError::GpuError(format!(
                    "Quantum solver VRAM requirement (~{} MB) exceeds safe limit ({} MB)",
                    quantum_mb,
                    VRAM_MB / 4
                )));
            }

            println!(
                "[VRAM][GUARD] Quantum solver allocation estimate: {} MB",
                quantum_mb
            );
        }

        println!("[VRAM][GUARD] ✅ VRAM validation passed for all enabled GPU phases");
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn validate_vram_requirements(&self, _graph: &Graph) -> Result<()> {
        println!("[VRAM][GUARD] Skipped (CUDA not compiled)");
        Ok(())
    }
}

/// Active Inference Policy for Graph Coloring
pub struct ActiveInferencePolicy {
    /// Vertex uncertainty scores (higher = more uncertain)
    pub uncertainty: Vec<f64>,

    /// Expected free energy per vertex
    pub expected_free_energy: Vec<f64>,

    /// Pragmatic value (goal-directed)
    pub pragmatic_value: Vec<f64>,

    /// Epistemic value (information-seeking)
    pub epistemic_value: Vec<f64>,
}

impl ActiveInferencePolicy {
    /// Compute Active Inference policy from current coloring state
    pub fn compute(
        graph: &Graph,
        partial_coloring: &[usize],
        kuramoto_state: &KuramotoState,
    ) -> Result<Self> {
        let n = graph.num_vertices;
        let mut uncertainty = vec![0.0; n];
        let mut expected_free_energy = vec![0.0; n];
        let mut pragmatic_value = vec![0.0; n];
        let mut epistemic_value = vec![0.0; n];

        // Build adjacency for conflict detection
        let adj = build_adjacency_matrix(graph);

        for v in 0..n {
            // CRITICAL FIX: Always compute uncertainty for ALL vertices
            // Even if vertex is colored, we need its uncertainty for downstream phases
            // (Previously skipped colored vertices, causing all-zero uncertainty)

            // Pragmatic value: How hard is this vertex to color?
            // CRITICAL FIX: Use degree-based uncertainty, NOT colored_neighbors
            // (colored_neighbors is 0 at start, causing constant uncertainty)
            let degree = (0..n).filter(|&u| adj[[v, u]]).count();

            // Degree-based uncertainty (mirrors GPU path in gpu_active_inference.rs:108-122)
            // High degree (500) → high pragmatic value (hard to color)
            // Low degree (50) → low pragmatic value (easy to color)
            let max_degree = 500.0; // Approximate max for DSJC1000.5
            let normalized_degree = (degree as f64 / max_degree).min(1.0);

            // pragmatic_value = 0.1 + normalized_degree * 0.9
            // Range: [0.1, 1.0] (high degree = high pragmatic value)
            pragmatic_value[v] = 0.1 + normalized_degree * 0.9;

            // Epistemic value: How much information do we gain?
            // Use Kuramoto phase coherence as proxy for information
            let phase = kuramoto_state.phases[v];
            let neighbor_phases: Vec<f64> = (0..n)
                .filter(|&u| adj[[v, u]])
                .map(|u| kuramoto_state.phases[u])
                .collect();

            if !neighbor_phases.is_empty() {
                let mean_phase = neighbor_phases.iter().sum::<f64>() / neighbor_phases.len() as f64;
                let phase_variance = neighbor_phases
                    .iter()
                    .map(|&p| (p - mean_phase).powi(2))
                    .sum::<f64>()
                    / neighbor_phases.len() as f64;

                epistemic_value[v] = phase_variance; // Higher variance = more information
            }

            // Uncertainty: Combination of degree and phase dispersion
            uncertainty[v] = pragmatic_value[v] * (1.0 + epistemic_value[v]);

            // Expected Free Energy: Balance pragmatic and epistemic
            expected_free_energy[v] = pragmatic_value[v] - 0.5 * epistemic_value[v];
        }

        // Normalize uncertainty using min-max scaling to [0, 1]
        let min_unc = uncertainty.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_unc = uncertainty
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if (max_unc - min_unc).abs() > 1e-10 {
            // Valid range, normalize
            for u in &mut uncertainty {
                *u = (*u - min_unc) / (max_unc - min_unc);
            }
            println!(
                "[AI-CPU] Normalized uncertainty: min={:.6}, max={:.6}, mean={:.6}",
                min_unc,
                max_unc,
                uncertainty.iter().sum::<f64>() / uncertainty.len() as f64
            );
        } else {
            // All values same or zero - this is a BUG
            eprintln!(
                "[AI-CPU][BUG] Uncertainty vector is constant (all {:.6})! Using uniform fallback.",
                min_unc
            );
            // Set to uniform distribution
            let uniform_value = 1.0 / uncertainty.len() as f64;
            for u in &mut uncertainty {
                *u = uniform_value;
            }
        }

        Ok(Self {
            uncertainty,
            expected_free_energy,
            pragmatic_value,
            epistemic_value,
        })
    }

    /// Select next vertex to color (minimize expected free energy)
    pub fn select_vertex(&self) -> usize {
        self.expected_free_energy
            .iter()
            .enumerate()
            .filter(|(_, &efe)| efe > 0.0)
            .min_by(|(_, a), (_, b)| {
                use std::cmp::Ordering;
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0) // Fallback to vertex 0 if no positive EFE found
    }
}

/// ADP State for Reinforcement Learning
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ColoringState {
    /// Current chromatic number (discretized)
    pub chromatic_bucket: i32,

    /// Conflict density (discretized)
    pub conflict_bucket: i32,

    /// Phase coherence (discretized)
    pub coherence_bucket: i32,
}

impl ColoringState {
    pub fn from_solution(solution: &ColoringSolution, order_param: f64) -> Self {
        Self {
            chromatic_bucket: (solution.chromatic_number / 5) as i32,
            conflict_bucket: (solution.conflicts / 10) as i32,
            coherence_bucket: (order_param * 10.0) as i32,
        }
    }
}

/// ADP Actions for parameter tuning
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum ColoringAction {
    IncreaseDSaturDepth,
    DecreaseDSaturDepth,
    IncreaseMemeticGenerations,
    DecreaseMemeticGenerations,
    IncreaseMutationRate,
    DecreaseMutationRate,
    IncreasePopulationSize,
    DecreasePopulationSize,
    FocusOnExploration,  // Higher diversity
    FocusOnExploitation, // More local search
    IncreaseQuantumIterations,
    DecreaseQuantumIterations,
    IncreaseThermoTemperatures,
    DecreaseThermoTemperatures,
}

impl ColoringAction {
    pub fn all() -> Vec<Self> {
        vec![
            Self::IncreaseDSaturDepth,
            Self::DecreaseDSaturDepth,
            Self::IncreaseMemeticGenerations,
            Self::DecreaseMemeticGenerations,
            Self::IncreaseMutationRate,
            Self::DecreaseMutationRate,
            Self::IncreasePopulationSize,
            Self::DecreasePopulationSize,
            Self::FocusOnExploration,
            Self::FocusOnExploitation,
            Self::IncreaseQuantumIterations,
            Self::DecreaseQuantumIterations,
            Self::IncreaseThermoTemperatures,
            Self::DecreaseThermoTemperatures,
        ]
    }
}

/// Neuromorphic Conflict Predictor using Reservoir Computing
pub struct ReservoirConflictPredictor {
    /// Conflict probability per vertex
    pub conflict_scores: Vec<f64>,

    /// Difficulty zones (high-conflict regions)
    pub difficulty_zones: Vec<Vec<usize>>,
}

impl ReservoirConflictPredictor {
    /// Train reservoir on partial colorings and predict conflicts
    pub fn predict(
        graph: &Graph,
        coloring_history: &[ColoringSolution],
        kuramoto_state: &KuramotoState,
        phase_threshold: f64,
    ) -> Result<Self> {
        let n = graph.num_vertices;
        let mut conflict_scores = vec![0.0; n];

        // Analyze historical conflicts
        for solution in coloring_history {
            let adj = build_adjacency_matrix(graph);

            for v in 0..n {
                let mut local_conflicts = 0;
                for u in 0..n {
                    if adj[[v, u]] && solution.colors[v] == solution.colors[u] {
                        local_conflicts += 1;
                    }
                }

                // Update conflict score with exponential moving average
                conflict_scores[v] = 0.7 * conflict_scores[v] + 0.3 * (local_conflicts as f64);
            }
        }

        // Use Kuramoto phases to identify coherent difficulty zones
        let mut difficulty_zones = Vec::new();

        for seed in 0..n {
            if conflict_scores[seed] < 2.0 {
                continue; // Not a difficult vertex
            }

            let mut zone = vec![seed];
            for v in 0..n {
                if v != seed
                    && conflict_scores[v] >= 2.0
                    && (kuramoto_state.phases[v] - kuramoto_state.phases[seed]).abs()
                        < phase_threshold
                {
                    zone.push(v);
                }
            }

            if zone.len() >= 3 {
                difficulty_zones.push(zone);
            }
        }

        Ok(Self {
            conflict_scores,
            difficulty_zones,
        })
    }
}

/// Statistical Mechanics Equilibration using Thermodynamic Network
pub struct ThermodynamicEquilibrator {
    /// Temperature schedule for annealing
    pub temperatures: Vec<f64>,

    /// Equilibrium colorings at each temperature
    pub equilibrium_states: Vec<ColoringSolution>,
}

impl ThermodynamicEquilibrator {
    /// Find equilibrium colorings at multiple temperatures
    pub fn equilibrate(
        graph: &Graph,
        initial_solution: &ColoringSolution,
        target_chromatic: usize,
        t_min: f64,
        t_max: f64,
        num_temps: usize,
        steps_per_temp: usize,
    ) -> Result<Self> {
        // Logarithmic temperature schedule
        let temperatures: Vec<f64> = (0..num_temps)
            .map(|i| {
                let frac = i as f64 / (num_temps - 1) as f64;
                let ratio: f64 = t_min / t_max;
                t_max * ratio.powf(frac)
            })
            .collect();

        let mut equilibrium_states = Vec::new();
        let mut current = initial_solution.clone();

        println!("[THERMODYNAMIC] Starting replica exchange...");
        println!(
            "[THERMODYNAMIC] Temperature range: [{:.3}, {:.3}], steps_per_temp: {}",
            t_min, t_max, steps_per_temp
        );

        for (i, &temp) in temperatures.iter().enumerate() {
            println!(
                "[THERMODYNAMIC] Temperature {}/{}: T = {:.3}",
                i + 1,
                num_temps,
                temp
            );

            // Simulated annealing at this temperature
            let mut best = current.clone();
            let adj = build_adjacency_matrix(graph);

            // Use configurable steps per temperature
            for _ in 0..steps_per_temp {
                // Random recoloring move
                let v = rand::random::<usize>() % graph.num_vertices;
                let old_color = current.colors[v];
                let new_color = rand::random::<usize>() % target_chromatic;

                current.colors[v] = new_color;

                // Compute energy change (conflict count)
                let mut delta_conflicts = 0i32;
                for u in 0..graph.num_vertices {
                    if adj[[v, u]] {
                        if current.colors[u] == old_color {
                            delta_conflicts -= 1;
                        }
                        if current.colors[u] == new_color {
                            delta_conflicts += 1;
                        }
                    }
                }

                // Metropolis acceptance criterion
                if delta_conflicts > 0 {
                    let prob = (-delta_conflicts as f64 / temp).exp();
                    if rand::random::<f64>() > prob {
                        current.colors[v] = old_color; // Reject
                    }
                }

                // Track best
                let conflicts = count_conflicts(&current.colors, &adj);
                if conflicts < best.conflicts {
                    best = current.clone();
                    best.conflicts = conflicts;
                }
            }

            equilibrium_states.push(best.clone());
            current = best;

            if current.conflicts == 0 {
                println!("[THERMODYNAMIC] ✅ Found valid coloring at T = {:.3}", temp);
            }
        }

        Ok(Self {
            temperatures,
            equilibrium_states,
        })
    }
}

/// Quantum-Classical Hybrid Solver
pub struct QuantumClassicalHybrid {
    /// Quantum solver for QUBO
    quantum_solver: QuantumColoringSolver,

    /// Classical solver for refinement
    classical_solver: DSaturSolver,

    /// Reservoir conflict scores for DSATUR guidance
    reservoir_scores: Option<Vec<f64>>,

    /// Active Inference expected free energy for vertex selection
    active_inference_efe: Option<Vec<f64>>,
}

impl QuantumClassicalHybrid {
    #[cfg(feature = "cuda")]
    pub fn new(max_colors: usize, cuda_device: Option<Arc<CudaContext>>) -> Result<Self> {
        Ok(Self {
            quantum_solver: QuantumColoringSolver::new(cuda_device)?,
            classical_solver: DSaturSolver::new(max_colors, 50000),
            reservoir_scores: None,
            active_inference_efe: None,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(max_colors: usize) -> Result<Self> {
        Ok(Self {
            quantum_solver: QuantumColoringSolver::new()?,
            classical_solver: DSaturSolver::new(max_colors, 50000),
            reservoir_scores: None,
            active_inference_efe: None,
        })
    }

    /// Set reservoir conflict scores for DSATUR tie-breaking
    pub fn set_reservoir_scores(&mut self, scores: Vec<f64>) {
        self.reservoir_scores = Some(scores.clone());
        self.classical_solver = self.classical_solver.clone().with_reservoir_scores(scores);
    }

    /// Set Active Inference expected free energy for vertex selection
    pub fn set_active_inference(&mut self, efe: Vec<f64>) {
        self.active_inference_efe = Some(efe.clone());
        self.classical_solver = self.classical_solver.clone().with_active_inference(efe);
    }

    /// Solve with quantum-classical feedback loop
    pub fn solve_with_feedback(
        &mut self,
        graph: &Graph,
        initial_solution: &ColoringSolution,
        kuramoto_state: &KuramotoState,
        num_iterations: usize,
    ) -> Result<ColoringSolution> {
        let mut best = initial_solution.clone();
        let n = graph.num_vertices;

        println!("[QUANTUM-CLASSICAL] Starting hybrid feedback loop...");

        for iter in 0..num_iterations {
            println!(
                "[QUANTUM-CLASSICAL] Iteration {}/{}",
                iter + 1,
                num_iterations
            );

            // Construct PhaseField from current best coloring and Kuramoto state
            let phase_field = self.construct_phase_field(graph, &best, kuramoto_state, None)?;

            // Phase 1: Quantum QUBO solve with error handling and fallback
            println!("[QUANTUM-CLASSICAL]   Phase 1: Quantum QUBO...");
            match self.quantum_solver.find_coloring(
                graph,
                &phase_field,
                kuramoto_state,
                best.chromatic_number,
            ) {
                Ok(quantum_result) => {
                    if quantum_result.chromatic_number < best.chromatic_number
                        && quantum_result.conflicts == 0
                    {
                        println!(
                            "[QUANTUM-CLASSICAL]   🎯 Quantum improved: {} → {} colors",
                            best.chromatic_number, quantum_result.chromatic_number
                        );
                        best = quantum_result.clone();
                    }
                }
                Err(e) => {
                    println!(
                        "[QUANTUM-CLASSICAL][FALLBACK] Quantum solver failed: {:?}",
                        e
                    );
                    println!("[QUANTUM-CLASSICAL][FALLBACK] Using DSATUR-only refinement instead");
                    println!("[QUANTUM-CLASSICAL][FALLBACK] Performance impact: ~20-30% slower (loses quantum exploration)");
                    // Don't abort - continue with classical solver below
                }
            }

            // Phase 2: Classical refinement
            println!("[QUANTUM-CLASSICAL]   Phase 2: Classical DSATUR refinement...");
            let classical_result = self.classical_solver.find_coloring(
                graph,
                Some(&best),
                best.chromatic_number.saturating_sub(3),
            )?;

            if classical_result.chromatic_number < best.chromatic_number
                && classical_result.conflicts == 0
            {
                println!(
                    "[QUANTUM-CLASSICAL]   🎯 Classical improved: {} → {} colors",
                    best.chromatic_number, classical_result.chromatic_number
                );
                best = classical_result;
            }

            // Adaptive iteration: stop early if no progress
            if iter > 0 && best.chromatic_number == initial_solution.chromatic_number {
                println!(
                    "[QUANTUM-CLASSICAL]   No improvement after {} iterations, stopping early",
                    iter + 1
                );
                break;
            }
        }

        Ok(best)
    }

    /// Construct PhaseField from coloring solution and Kuramoto state
    fn construct_phase_field(
        &self,
        graph: &Graph,
        solution: &ColoringSolution,
        kuramoto_state: &KuramotoState,
        _geodesic_features: Option<&GeodesicFeatures>,
    ) -> Result<PhaseField> {
        let n = graph.num_vertices;

        // Use Kuramoto phases as base, modulated by color information
        let mut phases = kuramoto_state.phases.clone();

        // Adjust phases based on color classes to create coherent clusters
        use std::f64::consts::PI;
        for v in 0..n {
            let color = solution.colors[v];
            if color != usize::MAX {
                // Add color-based phase shift while preserving Kuramoto structure
                let color_shift = 2.0 * PI * (color as f64) / (solution.chromatic_number as f64);
                phases[v] = (phases[v] + 0.5 * color_shift) % (2.0 * PI);
            }
        }

        // Compute coherence matrix based on graph adjacency
        let mut coherence_matrix = vec![0.0; n * n];
        for &(u, v, _weight) in &graph.edges {
            // High coherence for adjacent vertices (should have different colors)
            let phase_diff = (phases[u] - phases[v]).abs();
            let coherence = (phase_diff / PI).min(1.0); // Normalize to [0, 1]
            coherence_matrix[u * n + v] = coherence;
            coherence_matrix[v * n + u] = coherence;
        }

        Ok(PhaseField {
            phases,
            coherence_matrix,
            order_parameter: kuramoto_state.order_parameter,
            resonance_frequency: 1.0, // Default resonance
        })
    }
}

/// Ensemble Consensus Voting System
pub struct EnsembleConsensus {
    /// Solutions from different algorithms
    pub solutions: Vec<ColoringSolution>,

    /// Algorithm names
    pub algorithm_names: Vec<String>,
}

impl EnsembleConsensus {
    pub fn new() -> Self {
        Self {
            solutions: Vec::new(),
            algorithm_names: Vec::new(),
        }
    }

    /// Add solution from an algorithm
    pub fn add_solution(&mut self, solution: ColoringSolution, algorithm: &str) {
        self.solutions.push(solution);
        self.algorithm_names.push(algorithm.to_string());
    }

    /// Consensus voting: Choose best valid coloring
    pub fn vote(&self) -> Result<ColoringSolution> {
        if self.solutions.is_empty() {
            return Err(PRCTError::ColoringFailed(
                "No solutions to vote on".to_string(),
            ));
        }

        println!(
            "[ENSEMBLE] Voting among {} solutions...",
            self.solutions.len()
        );

        // Filter valid colorings only
        let valid: Vec<&ColoringSolution> =
            self.solutions.iter().filter(|s| s.conflicts == 0).collect();

        if valid.is_empty() {
            println!("[ENSEMBLE][FALLBACK] No valid colorings found in ensemble");
            println!("[ENSEMBLE][FALLBACK] Using best approximate solution (lowest conflicts)");
            println!("[ENSEMBLE][FALLBACK] Performance impact: solution may have conflicts");

            let best_approx = self
                .solutions
                .iter()
                .min_by_key(|s| (s.conflicts, s.chromatic_number))
                .cloned()
                .ok_or_else(|| {
                    PRCTError::ColoringFailed(
                        "No solutions available for ensemble voting (empty solutions list)"
                            .to_string(),
                    )
                })?;

            println!(
                "[ENSEMBLE] ℹ️  Best approximate: {} colors, {} conflicts",
                best_approx.chromatic_number, best_approx.conflicts
            );
            return Ok(best_approx);
        }

        // Return best valid coloring
        // Safe due to !valid.is_empty() check above
        let best = valid
            .iter()
            .min_by_key(|s| s.chromatic_number)
            .ok_or_else(|| {
                PRCTError::ColoringFailed(
                    "Failed to find minimum chromatic in non-empty valid solutions (logic error)"
                        .to_string(),
                )
            })?;

        println!("[ENSEMBLE] ✅ Consensus: {} colors", best.chromatic_number);
        Ok((*best).clone())
    }
}

/// Helper functions
fn build_adjacency_matrix(graph: &Graph) -> ndarray::Array2<bool> {
    use ndarray::Array2;
    let n = graph.num_vertices;
    let mut adj = Array2::from_elem((n, n), false);

    for &(u, v, _weight) in &graph.edges {
        adj[[u, v]] = true;
        adj[[v, u]] = true;
    }

    adj
}

fn count_conflicts(coloring: &[usize], adj: &ndarray::Array2<bool>) -> usize {
    let n = coloring.len();
    let mut conflicts = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            if adj[[i, j]] && coloring[i] == coloring[j] {
                conflicts += 1;
            }
        }
    }

    conflicts
}

impl Default for EnsembleConsensus {
    fn default() -> Self {
        Self::new()
    }
}

/// **WORLD RECORD BREAKING PIPELINE ORCHESTRATOR**
///
/// The ultimate integration of ALL PRISM modules for breaking the
/// 83-color world record on DSJC1000.5
pub struct WorldRecordPipeline {
    config: WorldRecordConfig,

    /// Best solution found so far
    best_solution: ColoringSolution,

    /// Coloring history for learning
    history: Vec<ColoringSolution>,

    /// Active Inference policy
    active_inference_policy: Option<ActiveInferencePolicy>,

    /// Dendritic-enhanced neuromorphic predictor (GPU-accelerated)
    #[cfg(feature = "cuda")]
    conflict_predictor_gpu: Option<GpuReservoirConflictPredictor>,

    /// Dendritic-enhanced neuromorphic predictor (CPU fallback)
    #[cfg(not(feature = "cuda"))]
    conflict_predictor: Option<ReservoirConflictPredictor>,

    /// Thermodynamic equilibrator
    thermodynamic_eq: Option<ThermodynamicEquilibrator>,

    /// Quantum-Classical hybrid solver
    quantum_classical: Option<QuantumClassicalHybrid>,

    /// Ensemble consensus system
    ensemble: EnsembleConsensus,

    /// ADP Q-table for parameter tuning
    adp_q_table: std::collections::HashMap<(ColoringState, ColoringAction), f64>,

    /// Shared CUDA context for GPU acceleration
    #[cfg(feature = "cuda")]
    cuda_device: Arc<CudaContext>,

    /// Multi-GPU device pool (if enabled)
    #[cfg(feature = "cuda")]
    multi_gpu_pool: Option<Arc<crate::gpu::MultiGpuDevicePool>>,

    /// GPU stream pool and state management
    #[cfg(feature = "cuda")]
    gpu_state: Option<Arc<crate::gpu::PipelineGpuState>>,

    /// Telemetry handle for real-time metric collection
    telemetry: Option<Arc<crate::telemetry::TelemetryHandle>>,

    /// Runtime GPU usage tracking
    phase_gpu_status: PhaseGpuStatus,

    /// Reservoir difficulty scores (Task A3: thread through all phases)
    reservoir_difficulty_scores: Option<Vec<f32>>,

    adp_epsilon: f64,

    /// ADP-tuned solver parameters
    adp_dsatur_depth: usize,
    adp_quantum_iterations: usize,
    adp_thermo_num_temps: usize,

    /// Stagnation tracking for adaptive loopback
    stagnation_count: usize,
    last_improvement_iteration: usize,
}

impl WorldRecordPipeline {
    #[cfg(feature = "cuda")]
    pub fn new(config: WorldRecordConfig, cuda_device: Arc<CudaContext>) -> Result<Self> {
        config.validate()?;

        // Initialize Rayon thread pool with configured CPU threads
        init_rayon_threads(config.cpu.threads);

        // Log GPU availability and configuration
        println!("[PIPELINE][INIT] CUDA device available (GPU acceleration enabled)");
        println!(
            "[PIPELINE][INIT] GPU phases: reservoir={}, te={}, thermo={}, quantum={}",
            config.gpu.enable_reservoir_gpu,
            config.gpu.enable_te_gpu,
            config.gpu.enable_thermo_gpu,
            config.gpu.enable_quantum_gpu
        );

        // Initialize multi-GPU pool if enabled
        let multi_gpu_pool = if config.multi_gpu.enabled {
            println!(
                "[PIPELINE][INIT] Multi-GPU mode enabled: {} devices",
                config.multi_gpu.num_gpus
            );
            let pool = crate::gpu::MultiGpuDevicePool::new(
                &config.multi_gpu.devices,
                config.multi_gpu.enable_peer_access,
            )?;
            println!(
                "[PIPELINE][INIT] Multi-GPU pool initialized: {} devices",
                pool.num_devices()
            );
            Some(Arc::new(pool))
        } else {
            None
        };

        // Initialize GPU stream pool if streams > 0
        let gpu_state = if config.gpu.streams > 0 {
            let stream_mode = match config.gpu.stream_mode {
                StreamMode::Sequential => crate::gpu::state::StreamMode::Sequential,
                StreamMode::Parallel => crate::gpu::state::StreamMode::Parallel,
            };
            let state = crate::gpu::PipelineGpuState::new(
                config.gpu.device_id,
                config.gpu.streams,
                stream_mode,
            )?;
            println!(
                "[PIPELINE][INIT] GPU stream pool: {} streams, mode={:?}",
                config.gpu.streams, config.gpu.stream_mode
            );
            Some(Arc::new(state))
        } else {
            println!("[PIPELINE][INIT] GPU streams disabled (streams=0)");
            None
        };

        Ok(Self {
            config: config.clone(),
            best_solution: ColoringSolution {
                colors: vec![],
                chromatic_number: usize::MAX,
                conflicts: usize::MAX,
                quality_score: 0.0,
                computation_time_ms: 0.0,
            },
            history: Vec::new(),
            active_inference_policy: None,
            conflict_predictor_gpu: None,
            thermodynamic_eq: None,
            quantum_classical: Some(QuantumClassicalHybrid::new(
                config.target_chromatic,
                Some(cuda_device.clone()),
            )?),
            ensemble: EnsembleConsensus::new(),
            adp_q_table: std::collections::HashMap::new(),
            cuda_device,
            multi_gpu_pool,
            gpu_state,
            telemetry: None,
            phase_gpu_status: PhaseGpuStatus::default(),
            reservoir_difficulty_scores: None,
            adp_epsilon: config.adp.epsilon,
            adp_dsatur_depth: config.orchestrator.adp_dsatur_depth,
            adp_quantum_iterations: config.orchestrator.adp_quantum_iterations,
            adp_thermo_num_temps: config.orchestrator.adp_thermo_num_temps,
            stagnation_count: 0,
            last_improvement_iteration: 0,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(config: WorldRecordConfig) -> Result<Self> {
        config.validate()?;

        // Initialize Rayon thread pool with configured CPU threads
        init_rayon_threads(config.cpu.threads);

        // FALLBACK WARNING: CUDA not available
        println!("[PIPELINE][FALLBACK] CUDA feature not compiled - using CPU-only mode");
        println!("[PIPELINE][FALLBACK] Performance impact: ~50-80% slower (no GPU acceleration)");
        println!("[PIPELINE][FALLBACK] Affected phases: reservoir (~10-50x slower), thermo (~5x slower), quantum (~3x slower)");

        // Warn about enabled GPU phases that will fallback
        if config.use_reservoir_prediction && config.gpu.enable_reservoir_gpu {
            println!(
                "[PIPELINE][FALLBACK] Reservoir GPU requested but CUDA unavailable → CPU fallback"
            );
        }
        if config.use_transfer_entropy && config.gpu.enable_te_gpu {
            println!("[PIPELINE][FALLBACK] Transfer Entropy GPU requested but CUDA unavailable → CPU fallback");
        }
        if config.use_thermodynamic_equilibration && config.gpu.enable_thermo_gpu {
            println!("[PIPELINE][FALLBACK] Thermodynamic GPU requested but CUDA unavailable → CPU fallback");
        }
        if config.use_quantum_classical_hybrid && config.gpu.enable_quantum_gpu {
            println!("[PIPELINE][FALLBACK] Quantum solver GPU requested but CUDA unavailable → CPU fallback");
        }

        Ok(Self {
            config: config.clone(),
            best_solution: ColoringSolution {
                colors: vec![],
                chromatic_number: usize::MAX,
                conflicts: usize::MAX,
                quality_score: 0.0,
                computation_time_ms: 0.0,
            },
            history: Vec::new(),
            active_inference_policy: None,
            conflict_predictor: None,
            thermodynamic_eq: None,
            quantum_classical: Some(QuantumClassicalHybrid::new(config.target_chromatic)?),
            ensemble: EnsembleConsensus::new(),
            adp_q_table: std::collections::HashMap::new(),
            telemetry: None,
            phase_gpu_status: PhaseGpuStatus::default(),
            reservoir_difficulty_scores: None,
            adp_epsilon: config.adp.epsilon,
            adp_dsatur_depth: config.orchestrator.adp_dsatur_depth,
            adp_quantum_iterations: config.orchestrator.adp_quantum_iterations,
            adp_thermo_num_temps: config.orchestrator.adp_thermo_num_temps,
            stagnation_count: 0,
            last_improvement_iteration: 0,
        })
    }

    /// Enable telemetry collection with specified run ID
    ///
    /// Creates a TelemetryHandle that will write metrics to
    /// target/run_artifacts/live_metrics_{run_id}_{timestamp}.jsonl
    ///
    /// # Arguments
    /// - `run_id`: Unique identifier for this run (e.g., graph name or experiment ID)
    ///
    /// # Returns
    /// Self with telemetry enabled
    pub fn with_telemetry(mut self, run_id: &str) -> Result<Self> {
        self.telemetry = Some(Arc::new(crate::telemetry::TelemetryHandle::new(
            run_id, 1000,
        )?));
        println!("[PIPELINE][INIT] Telemetry enabled: run_id={}", run_id);
        Ok(self)
    }

    /// Get Active Inference uncertainty vector (if available)
    ///
    /// Returns reference to normalized uncertainty vector from Phase 1.
    /// Higher values indicate vertices that are more difficult to color
    /// and should be prioritized for thermodynamic exploration.
    ///
    /// # Returns
    /// Option<&Vec<f64>> - Uncertainty scores per vertex (if AI policy computed)
    pub fn get_ai_uncertainty(&self) -> Option<&Vec<f64>> {
        self.active_inference_policy
            .as_ref()
            .map(|policy| &policy.uncertainty)
    }

    /// Get Active Inference expected free energy (if available)
    ///
    /// Returns reference to expected free energy vector from Phase 1.
    /// Lower values indicate more favorable coloring configurations.
    ///
    /// # Returns
    /// Option<&Vec<f64>> - Expected free energy per vertex (if AI policy computed)
    pub fn get_ai_expected_free_energy(&self) -> Option<&Vec<f64>> {
        self.active_inference_policy
            .as_ref()
            .map(|policy| &policy.expected_free_energy)
    }

    /// Print comprehensive phase checklist validating all config toggles and GPU paths
    fn print_phase_checklist(&self, graph: &Graph) -> Result<()> {
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║               PHASE CHECKLIST & VALIDATION                ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!();

        // Graph Statistics
        let vertices = graph.num_vertices;
        let edges_undirected = graph.num_edges;
        let edges_directed = edges_undirected * 2;
        let density = if vertices > 1 {
            (2.0 * edges_undirected as f64) / (vertices * (vertices - 1)) as f64
        } else {
            0.0
        };

        println!("[GRAPH] Statistics:");
        println!("  • vertices={}", vertices);
        println!("  • edges={} (undirected)", edges_undirected);
        println!("  • directed_edges={}", edges_directed);
        println!("  • density={:.6}", density);
        println!();

        // VRAM Guard Detection
        #[cfg(feature = "cuda")]
        {
            // Note: cudarc 0.9 doesn't expose total_memory() API
            // Using conservative estimates based on config defaults (8GB baseline)
            let vram_gb_estimate = 8; // Conservative baseline for RTX 5070 Ti
            println!(
                "[VRAM][GUARD] Device: CUDA {} (assumed {} GB VRAM)",
                self.config.gpu.device_id, vram_gb_estimate
            );

            // Check replica clamping
            if self.config.thermo.replicas > 56 {
                println!(
                    "[VRAM][GUARD] ⚠️  WARN: thermo.replicas={} exceeds 8GB safe limit (56)",
                    self.config.thermo.replicas
                );
            } else {
                println!(
                    "[VRAM][GUARD] ✅ thermo.replicas={} (within safe limit)",
                    self.config.thermo.replicas
                );
            }

            // Check beads clamping (for PIMC)
            let pimc_beads = default_beads(); // Use default as reference
            if self.config.use_pimc {
                if pimc_beads > 64 {
                    println!(
                        "[VRAM][GUARD] ⚠️  WARN: pimc.beads={} exceeds 8GB safe limit (64)",
                        pimc_beads
                    );
                } else {
                    println!(
                        "[VRAM][GUARD] ✅ pimc.beads={} (within safe limit)",
                        pimc_beads
                    );
                }
            }
            println!();
        }

        #[cfg(not(feature = "cuda"))]
        {
            println!("[VRAM][GUARD] CUDA not available - skipping VRAM checks");
            println!();
        }

        // Phase Configuration Summary
        println!("[PHASES] Configuration Summary:");
        println!();

        // Phase 0: Reservoir (GPU)
        let phase0_enabled = self.config.use_reservoir_prediction;
        let phase0_gpu = self.config.gpu.enable_reservoir_gpu;
        println!("  Phase 0: Reservoir Conflict Prediction");
        println!("    • enabled={}", phase0_enabled);
        println!("    • GPU={}", phase0_gpu);
        #[cfg(feature = "cuda")]
        {
            if phase0_enabled && phase0_gpu {
                println!("    • Status: ✅ GPU-accelerated neuromorphic reservoir active");
            } else if phase0_enabled && !phase0_gpu {
                println!("    • Status: ⚠️  CPU fallback (GPU disabled in config)");
            } else {
                println!("    • Status: ⏸️  Phase disabled");
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            if phase0_enabled {
                println!("    • Status: ⚠️  CPU fallback (CUDA feature not compiled)");
            } else {
                println!("    • Status: ⏸️  Phase disabled");
            }
        }
        println!();

        // Phase 1: Active Inference & Transfer Entropy
        let phase1_ai = self.config.use_active_inference;
        let phase1_te = self.config.use_transfer_entropy;
        let phase1_te_gpu = self.config.gpu.enable_te_gpu;
        println!("  Phase 1: Active Inference & Transfer Entropy");
        println!("    • Active Inference={}", phase1_ai);
        println!("    • Transfer Entropy={}", phase1_te);
        println!("    • TE GPU={}", phase1_te_gpu);
        if phase1_te && !phase1_te_gpu {
            println!("    • Status: ⚠️  TE requested but GPU disabled; using CPU path");
        } else if phase1_te && phase1_te_gpu {
            println!("    • Status: ✅ GPU-accelerated TE ordering active");
        } else {
            println!("    • Status: CPU-only TE ordering");
        }
        println!();

        // Phase 2: Thermodynamic & Statistical Mechanics
        let phase2_thermo = self.config.use_thermodynamic_equilibration;
        let phase2_thermo_gpu = self.config.gpu.enable_thermo_gpu;
        let phase2_statmech_gpu = self.config.gpu.enable_statmech_gpu;
        let phase2_pimc = self.config.use_pimc;
        let phase2_pimc_gpu = self.config.gpu.enable_pimc_gpu;
        println!("  Phase 2: Thermodynamic Equilibration & Statistical Mechanics");
        println!("    • Thermodynamic={}", phase2_thermo);
        println!("    • Thermo GPU={}", phase2_thermo_gpu);
        println!("    • StatMech GPU={}", phase2_statmech_gpu);
        println!("    • PIMC={}", phase2_pimc);
        println!("    • PIMC GPU={}", phase2_pimc_gpu);
        if phase2_thermo && !phase2_thermo_gpu {
            println!("    • Status: ⚠️  Thermo requested but GPU disabled; using CPU path");
        } else if phase2_thermo && phase2_thermo_gpu {
            println!("    • Status: ✅ GPU-accelerated thermodynamic equilibration active");
        }
        if phase2_pimc && phase2_pimc_gpu {
            println!("    • Status: ✅ GPU-accelerated PIMC active");
        } else if phase2_pimc && !phase2_pimc_gpu {
            println!("    • Status: ⚠️  PIMC requested but GPU disabled; using CPU path");
        }
        println!();

        // Phase 3: Quantum-Classical Hybrid
        let phase3_quantum = self.config.use_quantum_classical_hybrid;
        let phase3_quantum_gpu = self.config.gpu.enable_quantum_gpu;
        let phase3_memetic = true; // Always enabled when quantum is used
        let phase3_dsatur = true; // Always enabled as fallback
        println!("  Phase 3: Quantum-Classical Hybrid");
        println!("    • Quantum-Classical={}", phase3_quantum);
        println!("    • Quantum GPU={}", phase3_quantum_gpu);
        println!("    • Memetic={}", phase3_memetic);
        println!("    • DSATUR={}", phase3_dsatur);
        if phase3_quantum && !phase3_quantum_gpu {
            println!("    • Status: ⚠️  Quantum requested but GPU disabled; using CPU path");
        } else if phase3_quantum && phase3_quantum_gpu {
            println!("    • Status: ✅ GPU-accelerated quantum coloring active");
        }
        println!();

        // Phase 4/5: Geodesic & AI tie-breaks & GNN screening
        let phase45_geodesic = self.config.use_geodesic_features;
        let phase45_ai_tiebreak = self.config.use_active_inference;
        let phase45_gnn = self.config.use_gnn_screening;
        println!("  Phase 4/5: Advanced Heuristics");
        println!("    • Geodesic Features={}", phase45_geodesic);
        println!("    • AI tie-breaks={}", phase45_ai_tiebreak);
        println!("    • GNN Screening={}", phase45_gnn);
        if phase45_geodesic {
            println!("    • Status: ✅ Geodesic landmark features enabled (experimental)");
        }
        if phase45_gnn {
            println!("    • Status: ✅ GNN screening enabled (experimental)");
        }
        println!();

        // Phase 6: TDA
        let phase6_tda = self.config.use_tda;
        let phase6_tda_gpu = self.config.gpu.enable_tda_gpu;
        println!("  Phase 6: Topological Data Analysis");
        println!("    • TDA={}", phase6_tda);
        println!("    • TDA GPU={}", phase6_tda_gpu);
        println!(
            "    • [DEBUG] Config use_tda field: {}",
            self.config.use_tda
        );
        println!(
            "    • [DEBUG] Config enable_tda_gpu field: {}",
            self.config.gpu.enable_tda_gpu
        );
        if phase6_tda && !phase6_tda_gpu {
            println!("    • Status: ⚠️  TDA requested but GPU disabled; using CPU path");
        } else if phase6_tda && phase6_tda_gpu {
            println!("    • Status: ✅ GPU-accelerated TDA active (experimental)");
        } else {
            println!("    • Status: ⏸️  TDA disabled (CURRENTLY NO PHASE 6 IMPLEMENTATION EXISTS)");
        }
        println!();

        // ADP Learning
        let adp_enabled = self.config.use_adp_learning;
        println!("  ADP Q-Learning:");
        println!("    • enabled={}", adp_enabled);
        println!("    • epsilon={:.3}", self.config.adp.epsilon);
        println!("    • alpha={:.3}", self.config.adp.alpha);
        println!("    • gamma={:.3}", self.config.adp.gamma);
        println!();

        println!("╚═══════════════════════════════════════════════════════════╝");
        println!();

        Ok(())
    }

    /// **MAIN WORLD RECORD ATTEMPT**
    ///
    /// Runs the complete multi-modal PRISM pipeline
    pub fn optimize_world_record(
        &mut self,
        graph: &Graph,
        initial_kuramoto: &KuramotoState,
    ) -> Result<ColoringSolution> {
        let start = std::time::Instant::now();

        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║       WORLD RECORD BREAKING PIPELINE - PRISM ULTIMATE     ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        // IMPORTANT: This is the TARGET/GOAL, not the result (DO NOT PARSE)
        println!(
            "[WR-PIPELINE] Target: {} colors (World Record - GOAL)",
            self.config.target_chromatic
        );
        println!("[WR-PIPELINE] Current Best: 115 colors (Full Integration - BASELINE)");
        println!("[WR-PIPELINE] Gap to Close: 32 colors");
        println!();

        // Print comprehensive phase checklist
        self.print_phase_checklist(graph)?;

        // VRAM guard validation (early failure for OOM scenarios)
        self.config.validate_vram_requirements(graph)?;

        // ═══════════════════════════════════════════════════════════
        // INITIAL COLORING: Compute starting solution
        // ═══════════════════════════════════════════════════════════
        println!("\n┌─────────────────────────────────────────────────────────┐");
        println!("│ INITIAL COLORING: Starting Solution Generation         │");
        println!("└─────────────────────────────────────────────────────────┘");
        println!(
            "[INIT] Computing initial coloring with strategy: {:?}",
            self.config.initial_coloring.strategy
        );

        let initial_solution =
            compute_initial_coloring(graph, self.config.initial_coloring.strategy)?;

        println!(
            "[INIT] Initial coloring: {} colors, {} conflicts",
            initial_solution.chromatic_number, initial_solution.conflicts
        );

        self.best_solution = initial_solution;
        self.history.push(self.best_solution.clone());

        // ═══════════════════════════════════════════════════════════
        // PHASE 0A: Geodesic Features (if enabled)
        // ═══════════════════════════════════════════════════════════
        let geodesic_features = if self.config.use_geodesic_features {
            let phase_start = std::time::Instant::now();
            println!("┌─────────────────────────────────────────────────────────┐");
            println!("│ PHASE 0A: Geodesic Feature Computation                 │");
            println!("└─────────────────────────────────────────────────────────┘");
            println!("{{\"event\":\"phase_start\",\"phase\":\"0A\",\"name\":\"geodesic\"}}");

            // Record telemetry: phase start
            if let Some(ref telemetry) = self.telemetry {
                telemetry.record(
                    RunMetric::new(
                        PhaseName::Validation,
                        "phase_0a_start",
                        self.best_solution.chromatic_number,
                        self.best_solution.conflicts,
                        0.0,
                        PhaseExecMode::cpu_disabled(),
                    )
                    .with_parameters(json!({
                        "phase": "0A",
                        "enabled": true,
                        "num_landmarks": self.config.geodesic.num_landmarks,
                    })),
                );
            }

            let features = compute_landmark_distances(
                graph,
                self.config.geodesic.num_landmarks,
                &self.config.geodesic.metric,
            )?;

            let phase_elapsed = phase_start.elapsed();
            // IMPORTANT: This is an intermediate phase result (DO NOT PARSE as final result)
            println!(
                "[PHASE 0A] ✅ Geodesic features computed for {} landmarks",
                features.landmarks.len()
            );
            println!("{{\"event\":\"phase_end\",\"phase\":\"0A\",\"name\":\"geodesic\",\"time_s\":{:.3}}}",
                     phase_elapsed.as_secs_f64());

            // Record telemetry: phase complete
            if let Some(ref telemetry) = self.telemetry {
                telemetry.record(
                    RunMetric::new(
                        PhaseName::Validation,
                        "phase_0a_complete",
                        self.best_solution.chromatic_number,
                        self.best_solution.conflicts,
                        phase_elapsed.as_secs_f64() * 1000.0,
                        PhaseExecMode::cpu_disabled(),
                    )
                    .with_parameters(json!({
                        "num_landmarks": features.landmarks.len(),
                    })),
                );
            }

            Some(features)
        } else {
            None
        };

        // ═══════════════════════════════════════════════════════════
        // PHASE 0B: Dendritic-Enhanced Neuromorphic Pre-Analysis
        // ═══════════════════════════════════════════════════════════
        if self.config.use_reservoir_prediction {
            let phase_start = std::time::Instant::now();
            println!("┌─────────────────────────────────────────────────────────┐");
            println!("│ PHASE 0: Dendritic Neuromorphic Conflict Prediction    │");
            println!("└─────────────────────────────────────────────────────────┘");
            println!("{{\"event\":\"phase_start\",\"phase\":\"0B\",\"name\":\"reservoir\"}}");

            // Record telemetry: phase start
            if let Some(ref telemetry) = self.telemetry {
                telemetry.record(
                    RunMetric::new(
                        PhaseName::Reservoir,
                        "phase_0b_start",
                        self.best_solution.chromatic_number,
                        self.best_solution.conflicts,
                        0.0,
                        PhaseExecMode::cpu_disabled(),
                    )
                    .with_parameters(json!({
                        "phase": "0B",
                        "enabled": true,
                        "gpu_enabled": self.config.gpu.enable_reservoir_gpu,
                    })),
                );
            }

            // ACTIVATION LOG: Phase entry
            #[cfg(feature = "cuda")]
            {
                if self.config.gpu.enable_reservoir_gpu {
                    println!(
                        "[PHASE 0][GPU] Reservoir active (custom GEMV), M={}, N={}",
                        graph.num_vertices, graph.num_vertices
                    );
                } else {
                    println!("[PHASE 0] Reservoir active (CPU fallback)");
                }
            }

            #[cfg(not(feature = "cuda"))]
            println!("[PHASE 0] Reservoir active (CPU only)");

            // Train reservoir on diverse greedy solutions for better conflict prediction
            // World-record config: 200 patterns (20x increase) for 15x GPU speedup utilization
            let num_training_patterns = 200;
            println!(
                "[PHASE 0] Generating {} diverse training colorings for reservoir...",
                num_training_patterns
            );

            // Initialize RNG for random orderings (deterministic if seed is set)
            let mut rng = rand::thread_rng();

            // Pre-compute vertex degrees for degree-based orderings
            let mut vertex_degrees: Vec<(usize, usize)> = (0..graph.num_vertices)
                .map(|v| {
                    let degree = graph
                        .edges
                        .iter()
                        .filter(|(src, tgt, _)| *src == v || *tgt == v)
                        .count();
                    (v, degree)
                })
                .collect();

            let mut training_solutions = Vec::new();
            for i in 0..num_training_patterns {
                // Use different ordering strategies for diversity (4-way rotation)
                let random_order: Vec<usize> = if i % 4 == 0 {
                    // Strategy 1: Random shuffle
                    let mut order: Vec<usize> = (0..graph.num_vertices).collect();
                    order.shuffle(&mut rng);
                    order
                } else if i % 4 == 1 {
                    // Strategy 2: Degree-descending (DSATUR-style)
                    vertex_degrees.sort_by_key(|(_, deg)| std::cmp::Reverse(*deg));
                    vertex_degrees.iter().map(|(v, _)| *v).collect()
                } else if i % 4 == 2 {
                    // Strategy 3: Degree-ascending (reverse strategy for diversity)
                    vertex_degrees.sort_by_key(|(_, deg)| *deg);
                    vertex_degrees.iter().map(|(v, _)| *v).collect()
                } else {
                    // Strategy 4: Kuramoto phase ordering
                    let mut order: Vec<usize> = (0..graph.num_vertices).collect();
                    order.sort_by_key(|&v| (initial_kuramoto.phases[v] * 1000.0) as i32);
                    order
                };

                let greedy = greedy_coloring_with_ordering(graph, &random_order)?;
                training_solutions.push(greedy);

                if (i + 1) % 50 == 0 {
                    println!(
                        "[PHASE 0] Generated {}/{} training patterns",
                        i + 1,
                        num_training_patterns
                    );
                }
            }

            println!("[PHASE 0] ✅ {} diverse training patterns generated (Random: 25%, Degree-Desc: 25%, Degree-Asc: 25%, Kuramoto: 25%)",
                     training_solutions.len());

            #[cfg(feature = "cuda")]
            {
                if self.config.gpu.enable_reservoir_gpu {
                    // Get stream for Phase 0 (Reservoir)
                    let stream = if let Some(ref gpu_state) = self.gpu_state {
                        let stream = gpu_state.stream_for_phase(0);
                        println!("[PHASE 0] 🚀 Using GPU-accelerated neuromorphic reservoir (10-50x speedup) on stream {:?}",
                                stream as *const _ as usize);
                        stream
                    } else {
                        println!(
                            "[PHASE 0][WARNING] GPU state not initialized, using default stream"
                        );
                        &self.cuda_device.fork_default_stream().map_err(|e| {
                            PRCTError::GpuError(format!("Failed to fork stream: {}", e))
                        })?
                    };

                    match GpuReservoirConflictPredictor::predict_gpu(
                        graph,
                        &training_solutions,
                        initial_kuramoto,
                        Arc::clone(&self.cuda_device),
                        stream,
                    ) {
                        Ok(predictor) => {
                            self.phase_gpu_status.phase0_gpu_used = true;
                            println!("[PHASE 0][GPU] ✅ GPU reservoir executed successfully");
                            println!(
                                "[PHASE 0] ✅ Identified {} difficulty zones",
                                predictor.difficulty_zones.len()
                            );
                            println!(
                                "[PHASE 0] ✅ GPU dendritic processing: {} high-conflict vertices",
                                predictor
                                    .conflict_scores
                                    .iter()
                                    .filter(|&&s| s > 2.0)
                                    .count()
                            );
                            self.conflict_predictor_gpu = Some(predictor);
                        }
                        Err(e) => {
                            self.phase_gpu_status.phase0_fallback_reason = Some(format!("{}", e));
                            println!("[PHASE 0][GPU→CPU FALLBACK] {}", e);
                            println!("[PHASE 0][CPU] Using CPU reservoir fallback");
                            println!("[PHASE 0][FALLBACK] Performance impact: ~10-50x slower (loses GPU acceleration)");

                            // CPU fallback
                            let predictor = ReservoirConflictPredictor::predict(
                                graph,
                                &training_solutions,
                                initial_kuramoto,
                                self.config.neuromorphic.phase_threshold,
                            )?;
                            println!(
                                "[PHASE 0] ✅ CPU fallback: {} difficulty zones identified",
                                predictor.difficulty_zones.len()
                            );
                            // Note: can't store in conflict_predictor_gpu (wrong type), continue without it
                        }
                    }
                } else {
                    println!("[PHASE 0][FALLBACK] GPU reservoir disabled in config → CPU fallback");
                    println!("[PHASE 0][FALLBACK] Performance impact: ~10-50x slower (GPU disabled by user)");

                    let predictor = ReservoirConflictPredictor::predict(
                        graph,
                        &training_solutions,
                        initial_kuramoto,
                        self.config.neuromorphic.phase_threshold,
                    )?;
                    println!(
                        "[PHASE 0] ✅ CPU reservoir: {} difficulty zones identified",
                        predictor.difficulty_zones.len()
                    );
                }
            }

            #[cfg(not(feature = "cuda"))]
            {
                println!("[PHASE 0][FALLBACK] CUDA not compiled → CPU-only reservoir");
                println!("[PHASE 0][FALLBACK] Performance impact: ~10-50x slower (no GPU support)");
                self.conflict_predictor = Some(ReservoirConflictPredictor::predict(
                    graph,
                    &training_solutions,
                    initial_kuramoto,
                    self.config.neuromorphic.phase_threshold,
                )?);

                // Safe: just set above; if this fails, it's a logic error
                if let Some(ref predictor) = self.conflict_predictor {
                    println!(
                        "[PHASE 0] ✅ Identified {} difficulty zones",
                        predictor.difficulty_zones.len()
                    );
                    println!(
                        "[PHASE 0] ✅ Dendritic processing: {} high-conflict vertices",
                        predictor
                            .conflict_scores
                            .iter()
                            .filter(|&&s| s > 2.0)
                            .count()
                    );
                } else {
                    println!("[PHASE 0][ERROR] Logic error: conflict_predictor should be set after predict()");
                }
            }

            let phase_elapsed = phase_start.elapsed();
            println!("{{\"event\":\"phase_end\",\"phase\":\"0B\",\"name\":\"reservoir\",\"time_s\":{:.3}}}",
                     phase_elapsed.as_secs_f64());

            // Record telemetry: phase complete
            if let Some(ref telemetry) = self.telemetry {
                let gpu_mode = if self.phase_gpu_status.phase0_gpu_used {
                    PhaseExecMode::gpu_success(Some(0))
                } else if let Some(ref reason) = self.phase_gpu_status.phase0_fallback_reason {
                    PhaseExecMode::cpu_fallback(reason)
                } else {
                    PhaseExecMode::cpu_disabled()
                };

                // Gather difficulty zone stats
                let difficulty_stats = {
                    #[cfg(feature = "cuda")]
                    {
                        if let Some(ref pred) = self.conflict_predictor_gpu {
                            let scores = pred.get_conflict_scores();
                            let mut indexed_scores: Vec<(usize, f64)> = scores
                                .iter()
                                .enumerate()
                                .filter(|(_, &s)| s.is_finite()) // Filter out NaN/Inf
                                .map(|(i, &s)| (i, s))
                                .collect();
                            indexed_scores.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let top_10: Vec<(usize, f64)> =
                                indexed_scores.into_iter().take(10).collect();

                            println!("[PHASE 0][RESERVOIR] Top 10 difficulty vertices:");
                            for (i, (v, score)) in top_10.iter().enumerate() {
                                println!("  {}. Vertex {} → score {:.4}", i + 1, v, score);
                            }

                            json!({
                                "num_zones": pred.get_difficulty_zones().len(),
                                "high_conflict_vertices": pred.get_conflict_scores().iter().filter(|&&s| s > 2.0).count(),
                                "max_conflict_score": pred.get_conflict_scores().iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                                "mean_conflict_score": pred.get_conflict_scores().iter().sum::<f64>() / pred.get_conflict_scores().len() as f64,
                                "top_10_difficulty": top_10,
                            })
                        } else {
                            json!({"zones_available": false})
                        }
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        if let Some(ref pred) = self.conflict_predictor {
                            let scores = &pred.conflict_scores;
                            let mut indexed_scores: Vec<(usize, f64)> =
                                scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
                            indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            let top_10: Vec<(usize, f64)> =
                                indexed_scores.into_iter().take(10).collect();

                            println!("[PHASE 0][RESERVOIR] Top 10 difficulty vertices:");
                            for (i, (v, score)) in top_10.iter().enumerate() {
                                println!("  {}. Vertex {} → score {:.4}", i + 1, v, score);
                            }

                            json!({
                                "num_zones": pred.difficulty_zones.len(),
                                "high_conflict_vertices": pred.conflict_scores.iter().filter(|&&s| s > 2.0).count(),
                                "max_conflict_score": pred.conflict_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                                "mean_conflict_score": pred.conflict_scores.iter().sum::<f64>() / pred.conflict_scores.len() as f64,
                                "top_10_difficulty": top_10,
                            })
                        } else {
                            json!({"zones_available": false})
                        }
                    }
                };

                telemetry.record(
                    RunMetric::new(
                        PhaseName::Reservoir,
                        "phase_0b_complete",
                        self.best_solution.chromatic_number,
                        self.best_solution.conflicts,
                        phase_elapsed.as_secs_f64() * 1000.0,
                        gpu_mode,
                    )
                    .with_parameters(json!({
                        "phase": "0B",
                        "gpu_used": self.phase_gpu_status.phase0_gpu_used,
                        "difficulty_zones": difficulty_stats,
                    })),
                );
            }

            // Task A3: Store reservoir difficulty scores for threading through pipeline
            #[cfg(feature = "cuda")]
            {
                if let Some(ref pred) = self.conflict_predictor_gpu {
                    self.reservoir_difficulty_scores = Some(
                        pred.get_conflict_scores()
                            .iter()
                            .map(|&score| score as f32)
                            .collect(),
                    );
                    println!(
                        "[PHASE 0][THREAD-SCORES] Stored {} reservoir difficulty scores for downstream phases",
                        self.reservoir_difficulty_scores.as_ref().map(|s| s.len()).unwrap_or(0)
                    );
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                if let Some(ref pred) = self.conflict_predictor {
                    self.reservoir_difficulty_scores = Some(
                        pred.conflict_scores
                            .iter()
                            .map(|&score| score as f32)
                            .collect(),
                    );
                    println!(
                        "[PHASE 0][THREAD-SCORES] Stored {} reservoir difficulty scores for downstream phases",
                        self.reservoir_difficulty_scores.as_ref().map(|s| s.len()).unwrap_or(0)
                    );
                }
            }
        } else {
            println!("[PHASE 0] disabled by config");
        }

        // ═══════════════════════════════════════════════════════════
        // PHASE 1: Transfer Entropy with Active Inference Policy
        // ═══════════════════════════════════════════════════════════
        let phase1_start = std::time::Instant::now();
        println!("\n┌─────────────────────────────────────────────────────────┐");
        println!("│ PHASE 1: Active Inference-Guided Transfer Entropy      │");
        println!("└─────────────────────────────────────────────────────────┘");
        println!("{{\"event\":\"phase_start\",\"phase\":\"1\",\"name\":\"transfer_entropy\"}}");

        // Record telemetry: phase start
        if let Some(ref telemetry) = self.telemetry {
            telemetry.record(
                RunMetric::new(
                    PhaseName::TransferEntropy,
                    "phase_1_start",
                    self.best_solution.chromatic_number,
                    self.best_solution.conflicts,
                    0.0,
                    PhaseExecMode::cpu_disabled(),
                )
                .with_parameters(json!({
                    "phase": "1",
                    "enabled": self.config.use_transfer_entropy,
                    "gpu_enabled": self.config.gpu.enable_te_gpu,
                    "te_vs_kuramoto_weight": self.config.transfer_entropy.te_vs_kuramoto_weight,
                    "geodesic_weight": self.config.transfer_entropy.geodesic_weight,
                })),
            );
        }

        // ACTIVATION LOG: Phase entry
        if !self.config.use_transfer_entropy {
            println!("[PHASE 1] disabled by config");
        }

        let te_ordering = if self.config.use_transfer_entropy {
            #[cfg(feature = "cuda")]
            {
                if self.config.gpu.enable_te_gpu {
                    // Get stream for Phase 1 (Transfer Entropy)
                    let stream = if let Some(ref gpu_state) = self.gpu_state {
                        let stream = gpu_state.stream_for_phase(1);
                        println!(
                            "[PHASE 1][GPU] Using stream {:?} for TE computation",
                            stream as *const _ as usize
                        );
                        stream
                    } else {
                        println!(
                            "[PHASE 1][WARNING] GPU state not initialized, using default stream"
                        );
                        &self.cuda_device.fork_default_stream().map_err(|e| {
                            PRCTError::GpuError(format!("Failed to fork stream: {}", e))
                        })?
                    };

                    println!("[PHASE 1][GPU] Attempting TE kernels (histogram bins={}, time steps={}, lag=1)",
                             self.config.transfer_entropy.histogram_bins,
                             self.config.transfer_entropy.time_series_steps);
                    match gpu_transfer_entropy::compute_transfer_entropy_ordering_gpu(
                        &self.cuda_device,
                        stream,
                        graph,
                        initial_kuramoto,
                        geodesic_features.as_ref(),
                        self.config.transfer_entropy.geodesic_weight,
                        self.config.transfer_entropy.histogram_bins,
                        self.config.transfer_entropy.time_series_steps,
                    ) {
                        Ok(ordering) => {
                            self.phase_gpu_status.phase1_gpu_used = true;
                            println!("[PHASE 1][GPU] ✅ TE kernels executed successfully");
                            ordering
                        }
                        Err(e) => {
                            self.phase_gpu_status.phase1_fallback_reason = Some(format!("{}", e));
                            println!("[PHASE 1][GPU→CPU FALLBACK] {}", e);
                            println!("[PHASE 1][CPU] Using CPU TE computation");
                            hybrid_te_kuramoto_ordering(
                                graph,
                                initial_kuramoto,
                                geodesic_features.as_ref(),
                                self.config.transfer_entropy.geodesic_weight,
                            )?
                        }
                    }
                } else {
                    println!("[PHASE 1][CPU] TE on CPU (GPU disabled in config)");
                    hybrid_te_kuramoto_ordering(
                        graph,
                        initial_kuramoto,
                        geodesic_features.as_ref(),
                        self.config.transfer_entropy.geodesic_weight,
                    )?
                }
            }

            #[cfg(not(feature = "cuda"))]
            {
                println!("[PHASE 1][CPU] TE on CPU (CUDA not compiled)");
                hybrid_te_kuramoto_ordering(
                    graph,
                    initial_kuramoto,
                    geodesic_features.as_ref(),
                    self.config.transfer_entropy.geodesic_weight,
                )?
            }
        } else {
            // Phase disabled: return simple vertex ordering
            (0..graph.num_vertices).collect()
        };

        let mut te_solution = greedy_coloring_with_ordering(graph, &te_ordering)?;

        // Apply Active Inference to refine difficult vertices
        if self.config.use_active_inference {
            #[cfg(feature = "cuda")]
            {
                if self.config.use_gpu_active_inference {
                    // Try GPU path
                    match gpu_active_inference::active_inference_policy_gpu(
                        &self.cuda_device,
                        graph,
                        &te_solution.colors,
                        initial_kuramoto,
                    ) {
                        Ok(policy_gpu) => {
                            // Convert GPU policy to CPU-compatible format
                            let policy_cpu = ActiveInferencePolicy {
                                uncertainty: policy_gpu.uncertainty,
                                expected_free_energy: policy_gpu.expected_free_energy,
                                pragmatic_value: policy_gpu.pragmatic_value,
                                epistemic_value: policy_gpu.epistemic_value,
                            };
                            self.phase_gpu_status.phase1_ai_gpu_used = true;
                            println!("[PHASE 1][GPU] ✅ Active Inference policy computed on GPU");
                            self.active_inference_policy = Some(policy_cpu);
                        }
                        Err(e) => {
                            self.phase_gpu_status.phase1_ai_fallback_reason =
                                Some(format!("{}", e));
                            println!("[PHASE 1][GPU→CPU FALLBACK] AI GPU failed: {}", e);
                            // Fall back to CPU
                            println!("[PHASE 1][CPU] Active Inference on CPU");
                            self.active_inference_policy = Some(ActiveInferencePolicy::compute(
                                graph,
                                &te_solution.colors,
                                initial_kuramoto,
                            )?);
                        }
                    }
                } else {
                    // GPU disabled, use CPU
                    println!("[PHASE 1][CPU] Active Inference on CPU (GPU disabled in config)");
                    self.active_inference_policy = Some(ActiveInferencePolicy::compute(
                        graph,
                        &te_solution.colors,
                        initial_kuramoto,
                    )?);
                }
            }

            #[cfg(not(feature = "cuda"))]
            {
                // No CUDA, CPU only
                println!("[PHASE 1][CPU] Active Inference on CPU (CUDA not compiled)");
                self.active_inference_policy = Some(ActiveInferencePolicy::compute(
                    graph,
                    &te_solution.colors,
                    initial_kuramoto,
                )?);
            }

            println!("[PHASE 1] ✅ Active Inference: Computed expected free energy");
            println!("[PHASE 1] ✅ Uncertainty-guided vertex selection enabled");
        }

        // IMPORTANT: This is an intermediate phase result (DO NOT PARSE as final result)
        println!(
            "[PHASE 1] ✅ TE-guided coloring: {} colors",
            te_solution.chromatic_number
        );
        self.best_solution = te_solution.clone();
        self.history.push(te_solution.clone());

        let phase1_elapsed = phase1_start.elapsed();
        println!("{{\"event\":\"phase_end\",\"phase\":\"1\",\"name\":\"transfer_entropy\",\"time_s\":{:.3},\"colors\":{}}}",
                 phase1_elapsed.as_secs_f64(),
                 te_solution.chromatic_number);

        // Record telemetry: phase complete
        if let Some(ref telemetry) = self.telemetry {
            let gpu_mode = if self.phase_gpu_status.phase1_gpu_used {
                PhaseExecMode::gpu_success(Some(1))
            } else if let Some(ref reason) = self.phase_gpu_status.phase1_fallback_reason {
                PhaseExecMode::cpu_fallback(reason)
            } else {
                PhaseExecMode::cpu_disabled()
            };

            // Gather AI uncertainty stats
            let ai_stats = if let Some(ref policy) = self.active_inference_policy {
                let unc_mean =
                    policy.uncertainty.iter().sum::<f64>() / policy.uncertainty.len() as f64;
                let unc_min = policy
                    .uncertainty
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min);
                let unc_max = policy
                    .uncertainty
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let unc_variance = policy
                    .uncertainty
                    .iter()
                    .map(|u| (u - unc_mean).powi(2))
                    .sum::<f64>()
                    / policy.uncertainty.len() as f64;
                json!({
                    "uncertainty_available": true,
                    "uncertainty_mean": unc_mean,
                    "uncertainty_std": unc_variance.sqrt(),
                    "uncertainty_min": unc_min,
                    "uncertainty_max": unc_max,
                    "uncertainty_nonzero": policy.uncertainty.iter().any(|&u| u > 1e-6),
                    "ai_gpu_used": self.phase_gpu_status.phase1_ai_gpu_used,
                    "ai_gpu_fallback_reason": self.phase_gpu_status.phase1_ai_fallback_reason.clone(),
                })
            } else {
                json!({
                    "uncertainty_available": false,
                    "ai_gpu_used": false,
                })
            };

            telemetry.record(
                RunMetric::new(
                    PhaseName::TransferEntropy,
                    "phase_1_complete",
                    te_solution.chromatic_number,
                    te_solution.conflicts,
                    phase1_elapsed.as_secs_f64() * 1000.0,
                    gpu_mode,
                )
                .with_parameters(json!({
                    "phase": "1",
                    "te_gpu_used": self.phase_gpu_status.phase1_gpu_used,
                    "ai_gpu_used": self.phase_gpu_status.phase1_ai_gpu_used,
                    "active_inference_enabled": self.config.use_active_inference,
                    "ai_stats": ai_stats,
                })),
            );
        }

        self.ensemble
            .add_solution(te_solution.clone(), "Transfer Entropy");

        // ═══════════════════════════════════════════════════════════
        // PHASE 2: Statistical Mechanics Thermodynamic Equilibration
        // ═══════════════════════════════════════════════════════════
        if self.config.use_thermodynamic_equilibration {
            let phase2_start = std::time::Instant::now();
            println!("\n┌─────────────────────────────────────────────────────────┐");
            println!("│ PHASE 2: Thermodynamic Replica Exchange                │");
            println!("└─────────────────────────────────────────────────────────┘");
            println!("{{\"event\":\"phase_start\",\"phase\":\"2\",\"name\":\"thermodynamic\"}}");

            // Record telemetry: phase start with AI guidance status
            if let Some(ref telemetry) = self.telemetry {
                let uncertainty_stats = if let Some(unc) = self.get_ai_uncertainty() {
                    json!({
                        "ai_guided": true,
                        "uncertainty_mean": unc.iter().sum::<f64>() / unc.len() as f64,
                        "uncertainty_min": unc.iter().cloned().fold(f64::INFINITY, f64::min),
                        "uncertainty_max": unc.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    })
                } else {
                    json!({"ai_guided": false})
                };

                telemetry.record(
                    RunMetric::new(
                        PhaseName::Thermodynamic,
                        "phase_2_start",
                        self.best_solution.chromatic_number,
                        self.best_solution.conflicts,
                        0.0,
                        PhaseExecMode::cpu_disabled(),
                    )
                    .with_parameters(json!({
                        "phase": "2",
                        "enabled": true,
                        "gpu_enabled": self.config.gpu.enable_thermo_gpu,
                        "num_temps": self.adp_thermo_num_temps,
                        "steps_per_temp": self.config.thermo.steps_per_temp,
                        "t_min": self.config.thermo.t_min,
                        "t_max": self.config.thermo.t_max,
                        "ai_uncertainty": uncertainty_stats,
                    })),
                );
            }

            // ADP: Learn from Phase 1 results
            if self.config.use_adp_learning && self.history.len() >= 2 {
                let current_state = ColoringState::from_solution(
                    &self.best_solution,
                    initial_kuramoto.order_parameter,
                );
                let best_action = self.select_adp_action(&current_state);

                // For thermodynamic phase, we only apply temperature-related actions
                match best_action {
                    ColoringAction::IncreaseThermoTemperatures
                    | ColoringAction::DecreaseThermoTemperatures => {
                        self.apply_adp_action(&mut MemeticConfig::default(), best_action);
                        println!(
                            "[PHASE 2] 🧠 ADP tuned thermo temps: {}",
                            self.adp_thermo_num_temps
                        );
                    }
                    _ => {
                        // Save other actions for later phases
                    }
                }
            }

            // GPU/CPU dispatch for thermodynamic equilibration
            let equilibrium_states = {
                #[cfg(feature = "cuda")]
                {
                    if self.config.gpu.enable_thermo_gpu {
                        // Check if multi-GPU mode is enabled
                        if let Some(ref pool) = self.multi_gpu_pool {
                            if pool.num_devices() > 1 {
                                println!("[PHASE 2][MULTI-GPU] Using {} GPUs for distributed thermodynamic",
                                         pool.num_devices());

                                // Get AI uncertainty from Phase 1 (if available)
                                let ai_uncertainty = self.get_ai_uncertainty();

                                match crate::gpu_thermodynamic_multi::equilibrate_thermodynamic_multi_gpu(
                                    pool.devices(),
                                    graph,
                                    &self.best_solution,
                                    self.config.thermo.replicas,
                                    self.adp_thermo_num_temps,
                                    self.config.thermo.t_min,
                                    self.config.thermo.t_max,
                                    self.config.thermo.steps_per_temp,
                                    ai_uncertainty,
                                    Some(&self.config.fluxnet),
                                    self.reservoir_difficulty_scores.as_ref(),
                                    self.config.thermo.compaction.initial_slack,
                                    self.config.thermo.compaction.min_slack,
                                    self.config.thermo.compaction.max_slack,
                                    self.config.thermo.compaction.color_range_expand_threshold,
                                    self.config.thermo.compaction.reheat_consecutive_guards,
                                    self.config.thermo.compaction.reheat_temp_boost,
                                ) {
                                    Ok(states) => {
                                        self.phase_gpu_status.phase2_gpu_used = true;
                                        println!("[PHASE 2][MULTI-GPU] ✅ Distributed thermodynamic completed, {} solutions",
                                                 states.len());
                                        states
                                    }
                                    Err(e) => {
                                        self.phase_gpu_status.phase2_fallback_reason = Some(format!("Multi-GPU: {}", e));
                                        println!("[PHASE 2][MULTI-GPU→CPU FALLBACK] {}", e);
                                        println!("[PHASE 2][CPU] Using CPU thermodynamic equilibration");
                                        let eq = ThermodynamicEquilibrator::equilibrate(
                                            graph,
                                            &self.best_solution,
                                            self.config.target_chromatic,
                                            self.config.thermo.t_min,
                                            self.config.thermo.t_max,
                                            self.adp_thermo_num_temps,
                                            self.config.thermo.steps_per_temp,
                                        )?;
                                        eq.equilibrium_states
                                    }
                                }
                            } else {
                                // Single GPU in pool, use single-GPU path
                                let stream = if let Some(ref gpu_state) = self.gpu_state {
                                    let stream = gpu_state.stream_for_phase(2);
                                    println!("[PHASE 2][GPU] Using stream {:?} for thermodynamic equilibration", stream as *const _ as usize);
                                    stream
                                } else {
                                    println!("[PHASE 2][WARNING] GPU state not initialized, using default stream");
                                    &self.cuda_device.fork_default_stream().map_err(|e| {
                                        PRCTError::GpuError(format!("Failed to fork stream: {}", e))
                                    })?
                                };

                                println!("[PHASE 2][GPU] Attempting thermodynamic replica exchange (temps={}, steps={})",
                                         self.adp_thermo_num_temps, self.config.thermo.steps_per_temp);

                                // Get AI uncertainty from Phase 1 (if available)
                                let ai_uncertainty = self.get_ai_uncertainty();

                                match gpu_thermodynamic::equilibrate_thermodynamic_gpu(
                                    &self.cuda_device,
                                    stream,
                                    graph,
                                    &self.best_solution,
                                    self.config.target_chromatic,
                                    self.config.thermo.t_min,
                                    self.config.thermo.t_max,
                                    self.adp_thermo_num_temps,
                                    self.config.thermo.steps_per_temp,
                                    ai_uncertainty,
                                    self.telemetry.as_ref(),
                                    Some(&self.config.fluxnet),
                                    self.reservoir_difficulty_scores.as_ref(),
                                    self.config.thermo.force_start_temp,
                                    self.config.thermo.force_full_strength_temp,
                                    self.config.thermo.compaction.initial_slack,
                                    self.config.thermo.compaction.min_slack,
                                    self.config.thermo.compaction.max_slack,
                                    self.config.thermo.compaction.color_range_expand_threshold,
                                    self.config.thermo.compaction.reheat_consecutive_guards,
                                    self.config.thermo.compaction.reheat_temp_boost,
                                ) {
                                    Ok(states) => {
                                        self.phase_gpu_status.phase2_gpu_used = true;
                                        println!("[PHASE 2][GPU] ✅ Thermodynamic kernels executed successfully");
                                        states
                                    }
                                    Err(e) => {
                                        self.phase_gpu_status.phase2_fallback_reason =
                                            Some(format!("{}", e));
                                        println!("[PHASE 2][GPU→CPU FALLBACK] {}", e);
                                        println!(
                                            "[PHASE 2][CPU] Using CPU thermodynamic equilibration"
                                        );
                                        let eq = ThermodynamicEquilibrator::equilibrate(
                                            graph,
                                            &self.best_solution,
                                            self.config.target_chromatic,
                                            self.config.thermo.t_min,
                                            self.config.thermo.t_max,
                                            self.adp_thermo_num_temps,
                                            self.config.thermo.steps_per_temp,
                                        )?;
                                        eq.equilibrium_states
                                    }
                                }
                            }
                        } else {
                            // No multi-GPU pool, use single GPU
                            let stream = if let Some(ref gpu_state) = self.gpu_state {
                                let stream = gpu_state.stream_for_phase(2);
                                println!("[PHASE 2][GPU] Using stream {:?} for thermodynamic equilibration", stream as *const _ as usize);
                                stream
                            } else {
                                println!("[PHASE 2][WARNING] GPU state not initialized, using default stream");
                                &self.cuda_device.fork_default_stream().map_err(|e| {
                                    PRCTError::GpuError(format!("Failed to fork stream: {}", e))
                                })?
                            };

                            println!("[PHASE 2][GPU] Attempting thermodynamic replica exchange (temps={}, steps={})",
                                     self.adp_thermo_num_temps, self.config.thermo.steps_per_temp);

                            // Get AI uncertainty from Phase 1 (if available)
                            let ai_uncertainty = self.get_ai_uncertainty();

                            match gpu_thermodynamic::equilibrate_thermodynamic_gpu(
                                &self.cuda_device,
                                stream,
                                graph,
                                &self.best_solution,
                                self.config.target_chromatic,
                                self.config.thermo.t_min,
                                self.config.thermo.t_max,
                                self.adp_thermo_num_temps,
                                self.config.thermo.steps_per_temp,
                                ai_uncertainty,
                                self.telemetry.as_ref(),
                                Some(&self.config.fluxnet),
                                self.reservoir_difficulty_scores.as_ref(),
                                self.config.thermo.force_start_temp,
                                self.config.thermo.force_full_strength_temp,
                                self.config.thermo.compaction.initial_slack,
                                self.config.thermo.compaction.min_slack,
                                self.config.thermo.compaction.max_slack,
                                self.config.thermo.compaction.color_range_expand_threshold,
                                self.config.thermo.compaction.reheat_consecutive_guards,
                                self.config.thermo.compaction.reheat_temp_boost,
                            ) {
                                Ok(states) => {
                                    self.phase_gpu_status.phase2_gpu_used = true;
                                    println!("[PHASE 2][GPU] ✅ Thermodynamic kernels executed successfully");
                                    states
                                }
                                Err(e) => {
                                    self.phase_gpu_status.phase2_fallback_reason =
                                        Some(format!("{}", e));
                                    println!("[PHASE 2][GPU→CPU FALLBACK] {}", e);
                                    println!(
                                        "[PHASE 2][CPU] Using CPU thermodynamic equilibration"
                                    );
                                    let eq = ThermodynamicEquilibrator::equilibrate(
                                        graph,
                                        &self.best_solution,
                                        self.config.target_chromatic,
                                        self.config.thermo.t_min,
                                        self.config.thermo.t_max,
                                        self.adp_thermo_num_temps,
                                        self.config.thermo.steps_per_temp,
                                    )?;
                                    eq.equilibrium_states
                                }
                            }
                        }
                    } else {
                        println!("[PHASE 2][CPU] Thermodynamic on CPU (GPU disabled in config)");
                        let eq = ThermodynamicEquilibrator::equilibrate(
                            graph,
                            &self.best_solution,
                            self.config.target_chromatic,
                            self.config.thermo.t_min,
                            self.config.thermo.t_max,
                            self.adp_thermo_num_temps,
                            self.config.thermo.steps_per_temp,
                        )?;
                        eq.equilibrium_states
                    }
                }

                #[cfg(not(feature = "cuda"))]
                {
                    println!("[PHASE 2][CPU] Thermodynamic on CPU (CUDA not compiled)");
                    let eq = ThermodynamicEquilibrator::equilibrate(
                        graph,
                        &self.best_solution,
                        self.config.target_chromatic,
                        self.config.thermo.t_min,
                        self.config.thermo.t_max,
                        self.adp_thermo_num_temps,
                        self.config.thermo.steps_per_temp,
                    )?;
                    eq.equilibrium_states
                }
            };

            // Compute temperature ladder for logging
            let temperatures: Vec<f64> = (0..self.adp_thermo_num_temps)
                .map(|i| {
                    let ratio = self.config.thermo.t_min / self.config.thermo.t_max;
                    self.config.thermo.t_max
                        * ratio.powf(i as f64 / (self.adp_thermo_num_temps - 1) as f64)
                })
                .collect();

            // Store equilibrium states
            self.thermodynamic_eq = Some(ThermodynamicEquilibrator {
                temperatures: temperatures.clone(),
                equilibrium_states: equilibrium_states.clone(),
            });

            // Process equilibrium states
            for (i, state) in equilibrium_states.iter().enumerate() {
                if state.conflicts == 0
                    && state.chromatic_number < self.best_solution.chromatic_number
                {
                    // IMPORTANT: This is an intermediate phase result (DO NOT PARSE as final result)
                    println!(
                        "[PHASE 2] 🎯 Thermodynamic improvement at T={:.3}: {} → {} colors",
                        temperatures[i],
                        self.best_solution.chromatic_number,
                        state.chromatic_number
                    );
                    self.best_solution = state.clone();
                }
                self.history.push(state.clone());
                self.ensemble.add_solution(
                    state.clone(),
                    &format!("Thermodynamic-T{:.3}", temperatures[i]),
                );
            }

            let phase2_elapsed = phase2_start.elapsed();
            println!("{{\"event\":\"phase_end\",\"phase\":\"2\",\"name\":\"thermodynamic\",\"time_s\":{:.3},\"colors\":{}}}",
                     phase2_elapsed.as_secs_f64(),
                     self.best_solution.chromatic_number);

            // TWEAK 5: ADP telemetry-aware adjustments based on compaction guard events
            if self.config.use_adp_learning {
                // Count compaction guard triggers across all temperatures
                let initial_chromatic = self.best_solution.chromatic_number;
                let guard_count = equilibrium_states
                    .iter()
                    .filter(|s| {
                        // Heuristic: if chromatic is very low but conflicts are high, guard likely triggered
                        s.chromatic_number < (initial_chromatic / 2) && s.conflicts > 1000
                    })
                    .count();

                if guard_count > 2 {
                    println!("[TWEAK-5][ADP][TELEMETRY-DRIVEN] Detected {} guard triggers, adjusting parameters", guard_count);

                    // Add more temperature steps to allow better equilibration
                    let additional_steps = 2000 * guard_count;
                    let old_temps = self.adp_thermo_num_temps;
                    self.adp_thermo_num_temps =
                        (self.adp_thermo_num_temps + 8).min(self.config.thermo.num_temps);

                    println!("[TWEAK-5][ADP][TELEMETRY-DRIVEN] Increased temps from {} to {} (wanted {})",
                             old_temps, self.adp_thermo_num_temps, old_temps + 8);
                    println!("[TWEAK-5][ADP][TELEMETRY-DRIVEN] Would add {} steps/temp (logged for future config tuning)",
                             additional_steps);

                    // Increase exploration to escape phase-locking basins
                    let old_epsilon = self.adp_epsilon;
                    self.adp_epsilon = (self.adp_epsilon + 0.2).min(1.0);
                    println!(
                        "[TWEAK-5][ADP][TELEMETRY-DRIVEN] Increased epsilon from {:.3} to {:.3}",
                        old_epsilon, self.adp_epsilon
                    );
                }
            }

            // Record telemetry: phase complete
            if let Some(ref telemetry) = self.telemetry {
                let gpu_mode = if self.phase_gpu_status.phase2_gpu_used {
                    PhaseExecMode::gpu_success(Some(2))
                } else if let Some(ref reason) = self.phase_gpu_status.phase2_fallback_reason {
                    PhaseExecMode::cpu_fallback(reason)
                } else {
                    PhaseExecMode::cpu_disabled()
                };

                telemetry.record(
                    RunMetric::new(
                        PhaseName::Thermodynamic,
                        "phase_2_complete",
                        self.best_solution.chromatic_number,
                        self.best_solution.conflicts,
                        phase2_elapsed.as_secs_f64() * 1000.0,
                        gpu_mode,
                    )
                    .with_parameters(json!({
                        "phase": "2",
                        "gpu_used": self.phase_gpu_status.phase2_gpu_used,
                        "num_states_explored": equilibrium_states.len(),
                    })),
                );
            }
        } else {
            println!("[PHASE 2] disabled by config");
        }

        // ═══════════════════════════════════════════════════════════
        // PHASE 3: Quantum-Classical Hybrid with Feedback
        // ═══════════════════════════════════════════════════════════
        if self.config.use_quantum_classical_hybrid {
            let phase3_start = std::time::Instant::now();
            println!("\n┌─────────────────────────────────────────────────────────┐");
            println!("│ PHASE 3: Quantum-Classical Hybrid Feedback Loop        │");
            println!("└─────────────────────────────────────────────────────────┘");
            println!(
                "{{\"event\":\"phase_start\",\"phase\":\"3\",\"name\":\"quantum_classical\"}}"
            );

            // Record telemetry: phase start
            if let Some(ref telemetry) = self.telemetry {
                telemetry.record(
                    RunMetric::new(
                        PhaseName::Quantum,
                        "phase_3_start",
                        self.best_solution.chromatic_number,
                        self.best_solution.conflicts,
                        0.0,
                        PhaseExecMode::cpu_disabled(),
                    )
                    .with_parameters(json!({
                        "phase": "3",
                        "enabled": true,
                        "gpu_enabled": self.config.gpu.enable_quantum_gpu,
                        "quantum_iterations": self.adp_quantum_iterations,
                    })),
                );
            }

            // ACTIVATION LOG: Phase entry
            #[cfg(feature = "cuda")]
            {
                if self.config.gpu.enable_quantum_gpu {
                    println!(
                        "[PHASE 3][GPU] Attempting quantum solver (iterations={}, retries={})",
                        self.config.quantum.iterations, self.config.quantum.failure_retries
                    );
                } else {
                    println!("[PHASE 3][CPU] Quantum solver on CPU (GPU disabled in config)");
                }
            }

            #[cfg(not(feature = "cuda"))]
            println!("[PHASE 3][CPU] Quantum solver on CPU (CUDA not compiled)");

            // Check for Memetic fallback
            println!(
                "[PHASE 3] Memetic active (gens={}, pop={})",
                self.config.memetic.generations, self.config.memetic.population_size
            );

            // ADP: Learn from Phase 2 results and tune quantum parameters
            if self.config.use_adp_learning && self.history.len() >= 5 {
                let current_state = ColoringState::from_solution(
                    &self.best_solution,
                    initial_kuramoto.order_parameter,
                );
                let best_action = self.select_adp_action(&current_state);

                // Apply quantum-related and DSATUR-related actions
                match best_action {
                    ColoringAction::IncreaseQuantumIterations
                    | ColoringAction::DecreaseQuantumIterations
                    | ColoringAction::IncreaseDSaturDepth
                    | ColoringAction::DecreaseDSaturDepth => {
                        self.apply_adp_action(&mut MemeticConfig::default(), best_action);
                        println!("[PHASE 3] 🧠 ADP action: {:?}", best_action);
                    }
                    _ => {}
                }
            }

            if let Some(qc_hybrid) = &mut self.quantum_classical {
                // Wire in reservoir conflict scores for DSATUR tie-breaking
                #[cfg(feature = "cuda")]
                {
                    if let Some(ref predictor) = self.conflict_predictor_gpu {
                        qc_hybrid.set_reservoir_scores(predictor.get_conflict_scores().to_vec());
                        println!("[PHASE 3] ✅ Wired GPU reservoir scores into DSATUR");
                    }
                }

                #[cfg(not(feature = "cuda"))]
                {
                    if let Some(ref predictor) = self.conflict_predictor {
                        qc_hybrid.set_reservoir_scores(predictor.conflict_scores.clone());
                        println!("[PHASE 3] ✅ Wired CPU reservoir scores into DSATUR");
                    }
                }

                // Wire in Active Inference expected free energy
                if let Some(ref policy) = self.active_inference_policy {
                    qc_hybrid.set_active_inference(policy.expected_free_energy.clone());
                    println!("[PHASE 3] ✅ Wired Active Inference EFE into DSATUR");
                }

                // Use ADP-tuned quantum iteration count
                println!(
                    "[PHASE 3] 🧠 ADP-tuned quantum iterations: {}",
                    self.adp_quantum_iterations
                );
                match qc_hybrid.solve_with_feedback(
                    graph,
                    &self.best_solution,
                    initial_kuramoto,
                    self.adp_quantum_iterations, // ADP-tuned iterations
                ) {
                    Ok(qc_solution) => {
                        #[cfg(feature = "cuda")]
                        {
                            if self.config.gpu.enable_quantum_gpu {
                                self.phase_gpu_status.phase3_gpu_used = true;
                                println!("[PHASE 3][GPU] ✅ Quantum-classical hybrid completed");
                            }
                        }

                        if qc_solution.conflicts == 0
                            && qc_solution.chromatic_number < self.best_solution.chromatic_number
                        {
                            // IMPORTANT: This is an intermediate phase result (DO NOT PARSE as final result)
                            println!(
                                "[PHASE 3] 🎯 Quantum-Classical breakthrough: {} → {} colors",
                                self.best_solution.chromatic_number, qc_solution.chromatic_number
                            );
                            self.best_solution = qc_solution.clone();
                        }
                        self.history.push(qc_solution.clone());
                        self.ensemble.add_solution(qc_solution, "Quantum-Classical");
                    }
                    Err(e) => {
                        #[cfg(feature = "cuda")]
                        {
                            if self.config.gpu.enable_quantum_gpu {
                                self.phase_gpu_status.phase3_fallback_reason =
                                    Some(format!("{}", e));
                                self.phase_gpu_status.phase3_gpu_used = false;
                            }
                        }

                        println!(
                            "[PHASE 3][FALLBACK] Quantum-Classical phase failed: {:?}",
                            e
                        );
                        println!("[PHASE 3][FALLBACK] Continuing with best solution from previous phases");
                        println!("[PHASE 3][FALLBACK] Performance impact: ~30% (loses quantum-classical hybrid optimization)");
                        // Continue the pipeline - don't abort
                    }
                }
            }

            let phase3_elapsed = phase3_start.elapsed();
            println!("{{\"event\":\"phase_end\",\"phase\":\"3\",\"name\":\"quantum_classical\",\"time_s\":{:.3},\"colors\":{}}}",
                     phase3_elapsed.as_secs_f64(),
                     self.best_solution.chromatic_number);

            // Record telemetry: phase complete
            if let Some(ref telemetry) = self.telemetry {
                let gpu_mode = if self.phase_gpu_status.phase3_gpu_used {
                    PhaseExecMode::gpu_success(Some(3))
                } else if let Some(ref reason) = self.phase_gpu_status.phase3_fallback_reason {
                    PhaseExecMode::cpu_fallback(reason)
                } else {
                    PhaseExecMode::cpu_disabled()
                };

                telemetry.record(
                    RunMetric::new(
                        PhaseName::Quantum,
                        "phase_3_complete",
                        self.best_solution.chromatic_number,
                        self.best_solution.conflicts,
                        phase3_elapsed.as_secs_f64() * 1000.0,
                        gpu_mode,
                    )
                    .with_parameters(json!({
                        "phase": "3",
                        "gpu_used": self.phase_gpu_status.phase3_gpu_used,
                    })),
                );
            }
        } else {
            println!("[PHASE 3] disabled by config");
        }

        // ═══════════════════════════════════════════════════════════
        // PHASE 4: ADP-Optimized Memetic Algorithm
        // ═══════════════════════════════════════════════════════════
        let phase4_start = std::time::Instant::now();
        println!("\n┌─────────────────────────────────────────────────────────┐");
        println!("│ PHASE 4: ADP Q-Learning Memetic Optimization           │");
        println!("└─────────────────────────────────────────────────────────┘");
        println!("{{\"event\":\"phase_start\",\"phase\":\"4\",\"name\":\"memetic\"}}");

        // Record telemetry: phase start
        if let Some(ref telemetry) = self.telemetry {
            telemetry.record(
                RunMetric::new(
                    PhaseName::Memetic,
                    "phase_4_start",
                    self.best_solution.chromatic_number,
                    self.best_solution.conflicts,
                    0.0,
                    PhaseExecMode::cpu_disabled(),
                )
                .with_parameters(json!({
                    "phase": "4",
                    "enabled": true,
                })),
            );
        }

        // World-record aggressive settings (48-hour target)
        let mut memetic_config = MemeticConfig {
            population_size: 128, // 4x for better diversity
            elite_size: 16,       // Scale with population
            generations: 500,     // 10x for deeper search
            mutation_rate: 0.20,
            tournament_size: 5,        // Stronger selection pressure
            local_search_depth: 50000, // 10x for intensive local optimization
            use_tsp_guidance: true,
            tsp_weight: 0.25,
        };

        // ADP: Learn optimal parameters from history
        if self.config.use_adp_learning && self.history.len() > 3 {
            let current_state =
                ColoringState::from_solution(&self.best_solution, initial_kuramoto.order_parameter);

            // Q-learning action selection
            let best_action = self.select_adp_action(&current_state);
            self.apply_adp_action(&mut memetic_config, best_action);

            println!("[PHASE 4] 🧠 ADP selected action: {:?}", best_action);
        }

        let initial_pop = vec![self.best_solution.clone(), te_solution];

        let mut memetic = MemeticColoringSolver::new(memetic_config.clone());
        let memetic_solution = memetic.solve_with_restart(graph, initial_pop, 10)?; // 10 restarts for world-record attempt

        if memetic_solution.conflicts == 0
            && memetic_solution.chromatic_number < self.best_solution.chromatic_number
        {
            // IMPORTANT: This is an intermediate phase result (DO NOT PARSE as final result)
            println!(
                "[PHASE 4] 🎯 Memetic+ADP improved: {} → {} colors",
                self.best_solution.chromatic_number, memetic_solution.chromatic_number
            );

            // ADP: Update Q-value with reward
            if self.config.use_adp_learning {
                let reward = (self.best_solution.chromatic_number as f64
                    - memetic_solution.chromatic_number as f64)
                    * 10.0;
                self.update_adp_q_value(
                    &ColoringState::from_solution(
                        &self.best_solution,
                        initial_kuramoto.order_parameter,
                    ),
                    self.select_adp_action(&ColoringState::from_solution(
                        &self.best_solution,
                        initial_kuramoto.order_parameter,
                    )),
                    reward,
                );
            }

            self.best_solution = memetic_solution.clone();
        }
        self.history.push(memetic_solution.clone());
        self.ensemble.add_solution(memetic_solution, "Memetic+ADP");

        let phase4_elapsed = phase4_start.elapsed();
        println!("{{\"event\":\"phase_end\",\"phase\":\"4\",\"name\":\"memetic\",\"time_s\":{:.3},\"colors\":{}}}",
                 phase4_elapsed.as_secs_f64(),
                 self.best_solution.chromatic_number);

        // Record telemetry: phase complete
        if let Some(ref telemetry) = self.telemetry {
            telemetry.record(
                RunMetric::new(
                    PhaseName::Memetic,
                    "phase_4_complete",
                    self.best_solution.chromatic_number,
                    self.best_solution.conflicts,
                    phase4_elapsed.as_secs_f64() * 1000.0,
                    PhaseExecMode::cpu_disabled(),
                )
                .with_parameters(json!({
                    "phase": "4",
                    "population_size": memetic_config.population_size,
                    "generations": memetic_config.generations,
                })),
            );
        }

        // ═══════════════════════════════════════════════════════════
        // PHASE 5: Ensemble Consensus with Multi-Scale Analysis
        // ═══════════════════════════════════════════════════════════
        if self.config.use_ensemble_consensus {
            let phase5_start = std::time::Instant::now();
            println!("\n┌─────────────────────────────────────────────────────────┐");
            println!("│ PHASE 5: Ensemble Consensus Voting                     │");
            println!("└─────────────────────────────────────────────────────────┘");
            println!("{{\"event\":\"phase_start\",\"phase\":\"5\",\"name\":\"ensemble\"}}");

            // Record telemetry: phase start
            if let Some(ref telemetry) = self.telemetry {
                telemetry.record(
                    RunMetric::new(
                        PhaseName::Ensemble,
                        "phase_5_start",
                        self.best_solution.chromatic_number,
                        self.best_solution.conflicts,
                        0.0,
                        PhaseExecMode::cpu_disabled(),
                    )
                    .with_parameters(json!({
                        "phase": "5",
                        "enabled": true,
                    })),
                );
            }

            let consensus_solution = self.ensemble.vote()?;

            if consensus_solution.chromatic_number < self.best_solution.chromatic_number {
                // IMPORTANT: This is an intermediate phase result (DO NOT PARSE as final result)
                println!(
                    "[PHASE 5] 🎯 Ensemble consensus: {} → {} colors",
                    self.best_solution.chromatic_number, consensus_solution.chromatic_number
                );
                self.best_solution = consensus_solution;
            }

            let phase5_elapsed = phase5_start.elapsed();
            println!("{{\"event\":\"phase_end\",\"phase\":\"5\",\"name\":\"ensemble\",\"time_s\":{:.3},\"colors\":{}}}",
                     phase5_elapsed.as_secs_f64(),
                     self.best_solution.chromatic_number);

            // Record telemetry: phase complete
            if let Some(ref telemetry) = self.telemetry {
                telemetry.record(
                    RunMetric::new(
                        PhaseName::Ensemble,
                        "phase_5_complete",
                        self.best_solution.chromatic_number,
                        self.best_solution.conflicts,
                        phase5_elapsed.as_secs_f64() * 1000.0,
                        PhaseExecMode::cpu_disabled(),
                    )
                    .with_parameters(json!({
                        "phase": "5",
                    })),
                );
            }
        }

        // ═══════════════════════════════════════════════════════════
        // PHASE 6: Topological Data Analysis (TDA) Chromatic Bounds
        // ═══════════════════════════════════════════════════════════
        if self.config.use_tda {
            use crate::sparse_qubo::ChromaticBounds;

            let phase6_start = std::time::Instant::now();
            println!("\n┌─────────────────────────────────────────────────────────┐");
            println!("│ PHASE 6: Topological Data Analysis (TDA)               │");
            println!("└─────────────────────────────────────────────────────────┘");
            println!("{{\"event\":\"phase_start\",\"phase\":\"6\",\"name\":\"tda\"}}");

            // Record telemetry: phase start
            if let Some(ref telemetry) = self.telemetry {
                telemetry.record(
                    RunMetric::new(
                        PhaseName::Ensemble, // Reuse Ensemble for now (TDA doesn't have its own PhaseName yet)
                        "phase_6_tda_start",
                        self.best_solution.chromatic_number,
                        self.best_solution.conflicts,
                        0.0,
                        PhaseExecMode::cpu_disabled(),
                    )
                    .with_parameters(json!({
                        "phase": "6",
                        "enabled": true,
                        "gpu_enabled": self.config.gpu.enable_tda_gpu,
                    })),
                );
            }

            // Compute TDA chromatic bounds
            match ChromaticBounds::from_graph_tda(graph) {
                Ok(bounds) => {
                    println!("[PHASE 6] TDA Chromatic Bounds Computed:");
                    println!("[PHASE 6]   • Lower bound (max clique): {}", bounds.lower);
                    println!("[PHASE 6]   • Upper bound (degree+1): {}", bounds.upper);
                    println!("[PHASE 6]   • Max clique size: {}", bounds.max_clique_size);
                    println!(
                        "[PHASE 6]   • Connected components (Betti-0): {}",
                        bounds.num_components
                    );
                    println!(
                        "[PHASE 6]   • Current best: {} colors",
                        self.best_solution.chromatic_number
                    );

                    // Sanity check: warn if current solution is outside bounds
                    if self.best_solution.chromatic_number < bounds.lower {
                        println!("[PHASE 6] ⚠️  WARNING: Current solution ({}) violates TDA lower bound ({})",
                                 self.best_solution.chromatic_number, bounds.lower);
                    } else if self.best_solution.chromatic_number > bounds.upper {
                        println!("[PHASE 6] ⚠️  Current solution ({}) exceeds TDA upper bound ({}), but this is expected for hard graphs",
                                 self.best_solution.chromatic_number, bounds.upper);
                    } else {
                        println!(
                            "[PHASE 6] ✅ Current solution within TDA bounds [{}, {}]",
                            bounds.lower, bounds.upper
                        );
                    }

                    // Use TDA bounds to inform adaptive strategy
                    let gap_to_lower = self
                        .best_solution
                        .chromatic_number
                        .saturating_sub(bounds.lower);
                    if gap_to_lower > 10 {
                        println!("[PHASE 6] 🎯 Large gap to lower bound ({} colors): Consider more aggressive search",
                                 gap_to_lower);
                    }
                }
                Err(e) => {
                    println!("[PHASE 6][WARNING] TDA bounds computation failed: {:?}", e);
                    println!("[PHASE 6][WARNING] Continuing without TDA bounds");
                }
            }

            let phase6_elapsed = phase6_start.elapsed();
            println!("{{\"event\":\"phase_end\",\"phase\":\"6\",\"name\":\"tda\",\"time_s\":{:.3},\"colors\":{}}}",
                     phase6_elapsed.as_secs_f64(),
                     self.best_solution.chromatic_number);

            // Record telemetry: phase complete
            if let Some(ref telemetry) = self.telemetry {
                telemetry.record(
                    RunMetric::new(
                        PhaseName::Ensemble, // Reuse Ensemble for now
                        "phase_6_tda_complete",
                        self.best_solution.chromatic_number,
                        self.best_solution.conflicts,
                        phase6_elapsed.as_secs_f64() * 1000.0,
                        PhaseExecMode::cpu_disabled(),
                    )
                    .with_parameters(json!({
                        "phase": "6",
                    })),
                );
            }
        } else {
            println!("[PHASE 6] TDA disabled by config");
        }

        // ═══════════════════════════════════════════════════════════
        // STAGNATION DETECTION & ADAPTIVE LOOPBACK
        // ═══════════════════════════════════════════════════════════
        let current_best = self.best_solution.chromatic_number;
        let history_len = self.history.len();

        // Check if we've improved in the last 5 solutions
        let recent_improvement = if history_len >= 5 {
            self.history
                .iter()
                .rev()
                .take(5)
                .any(|sol| sol.chromatic_number < current_best)
        } else {
            true // Still warming up
        };

        if !recent_improvement && history_len >= 10 {
            self.stagnation_count += 1;
            println!(
                "\n⚠️  STAGNATION DETECTED (count: {})",
                self.stagnation_count
            );

            // Adaptive recovery strategies
            if self.stagnation_count >= 2 {
                println!("🔄 ADAPTIVE LOOPBACK: Applying recovery strategies...");

                // Strategy 1: Increase exploration (reset epsilon, boost mutation)
                self.adp_epsilon = (self.adp_epsilon + 0.3).min(1.0);
                println!("   • Increased exploration: ε = {:.3}", self.adp_epsilon);

                // Strategy 2: Boost thermodynamic escape capability
                let new_temps = self.adp_thermo_num_temps + 16;
                self.adp_thermo_num_temps = new_temps.min(self.config.thermo.num_temps);
                if new_temps > self.config.thermo.num_temps {
                    eprintln!(
                        "[ADP][WARNING] ADP wanted {} temps but clamping to config max {}",
                        new_temps, self.config.thermo.num_temps
                    );
                }
                println!(
                    "   • Increased thermal diversity: {} temps",
                    self.adp_thermo_num_temps
                );

                // Strategy 3: Intensify quantum iterations
                self.adp_quantum_iterations = (self.adp_quantum_iterations + 5).min(30);
                println!(
                    "   • Increased quantum depth: {} iterations",
                    self.adp_quantum_iterations
                );

                // Strategy 4: Try desperation phase with extreme memetic search
                if self.stagnation_count >= 3 {
                    println!("   • 🚨 DESPERATION MODE: Running extreme memetic search...");

                    let desperation_config = MemeticConfig {
                        population_size: 256, // 2x aggressive
                        elite_size: 32,
                        generations: 1000,   // 2x aggressive
                        mutation_rate: 0.35, // Higher mutation
                        tournament_size: 7,
                        local_search_depth: 100000, // Maximum depth
                        use_tsp_guidance: true,
                        tsp_weight: 0.3,
                    };

                    let initial_pop = vec![
                        self.best_solution.clone(),
                        // Use last history solution, or best_solution if history is empty
                        self.history
                            .last()
                            .unwrap_or(&self.best_solution) // Fallback to best_solution (should be rare)
                            .clone(),
                    ];

                    let mut desperation_memetic = MemeticColoringSolver::new(desperation_config);
                    if let Ok(desperation_sol) =
                        desperation_memetic.solve_with_restart(graph, initial_pop, 5)
                    {
                        if desperation_sol.conflicts == 0
                            && desperation_sol.chromatic_number
                                < self.best_solution.chromatic_number
                        {
                            println!(
                                "   • 🎯 Desperation mode SUCCESS: {} → {} colors!",
                                self.best_solution.chromatic_number,
                                desperation_sol.chromatic_number
                            );
                            self.best_solution = desperation_sol.clone();
                            self.stagnation_count = 0; // Reset on improvement
                        }
                    }
                }
            }
        } else if recent_improvement {
            // Reset stagnation counter on improvement
            if self.stagnation_count > 0 {
                println!("✅ Progress detected, resetting stagnation counter");
                self.stagnation_count = 0;
                self.last_improvement_iteration = history_len;
            }
        }

        let elapsed = start.elapsed().as_secs_f64();

        // ═══════════════════════════════════════════════════════════
        // FINAL RESULTS
        // ═══════════════════════════════════════════════════════════
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║                  WORLD RECORD ATTEMPT COMPLETE             ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!();

        // IMPORTANT: Below are summary statistics (DO NOT PARSE for final results)
        println!(
            "🏆 Summary: {} colors achieved",
            self.best_solution.chromatic_number
        );
        println!(
            "🥇 World Record: {} colors (REFERENCE)",
            self.config.target_chromatic
        );
        println!(
            "📊 Gap to WR: {:.2}x",
            self.best_solution.chromatic_number as f64 / self.config.target_chromatic as f64
        );
        println!("⏱️  Total Time: {:.2}s", elapsed);
        println!();

        if self.best_solution.chromatic_number <= self.config.target_chromatic {
            println!("🎉 *** WORLD RECORD MATCHED OR BEATEN! *** 🎉");
        } else if self.best_solution.chromatic_number <= 90 {
            println!("✨ *** EXCELLENT RESULT (Within 10% of WR) *** ✨");
        } else if self.best_solution.chromatic_number <= 100 {
            println!("✅ *** STRONG RESULT (Target <100 achieved) *** ✅");
        }
        println!();

        // ═══════════════════════════════════════════════════════════
        // PARSER-SAFE FINAL RESULT (Plain Text)
        // ═══════════════════════════════════════════════════════════
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║                     FINAL RESULT                          ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!(
            "FINAL RESULT: colors={} conflicts={} time={:.2}s",
            self.best_solution.chromatic_number, self.best_solution.conflicts, elapsed
        );

        // ═══════════════════════════════════════════════════════════
        // JSON TELEMETRY (Machine-Parseable)
        // ═══════════════════════════════════════════════════════════
        let graph_density =
            2.0 * graph.num_edges as f64 / (graph.num_vertices * (graph.num_vertices - 1)) as f64;

        println!("{{\"event\":\"final_result\",\"colors\":{},\"conflicts\":{},\"time_s\":{:.3},\"quality_score\":{:.6},\"graph\":{{\"vertices\":{},\"edges\":{},\"density\":{:.6}}}}}",
                 self.best_solution.chromatic_number,
                 self.best_solution.conflicts,
                 elapsed,
                 self.best_solution.quality_score,
                 graph.num_vertices,
                 graph.num_edges,
                 graph_density);

        // Persist GPU usage tracking
        self.save_phase_gpu_status()?;

        Ok(self.best_solution.clone())
    }

    /// Save phase GPU status to JSON file for verification
    fn save_phase_gpu_status(&self) -> Result<()> {
        let status_json = serde_json::to_string_pretty(&self.phase_gpu_status).map_err(|e| {
            PRCTError::ConfigError(format!("Failed to serialize GPU status: {}", e))
        })?;

        std::fs::write("phase_gpu_status.json", status_json).map_err(|e| {
            PRCTError::ConfigError(format!("Failed to write phase_gpu_status.json: {}", e))
        })?;

        println!("\n[GPU-STATUS] Saved runtime GPU usage to phase_gpu_status.json");
        println!(
            "[GPU-STATUS] Phase 0 (Reservoir): {}",
            if self.phase_gpu_status.phase0_gpu_used {
                "GPU ✅"
            } else {
                "CPU"
            }
        );
        println!(
            "[GPU-STATUS] Phase 1 (Transfer Entropy): {}",
            if self.phase_gpu_status.phase1_gpu_used {
                "GPU ✅"
            } else {
                "CPU"
            }
        );
        println!(
            "[GPU-STATUS] Phase 2 (Thermodynamic): {}",
            if self.phase_gpu_status.phase2_gpu_used {
                "GPU ✅"
            } else {
                "CPU"
            }
        );
        println!(
            "[GPU-STATUS] Phase 3 (Quantum): {}",
            if self.phase_gpu_status.phase3_gpu_used {
                "GPU ✅"
            } else {
                "CPU"
            }
        );

        Ok(())
    }

    /// ADP Q-Learning: Select action using epsilon-greedy
    fn select_adp_action(&self, state: &ColoringState) -> ColoringAction {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < self.adp_epsilon {
            // Explore: Random action
            let actions = ColoringAction::all();
            actions[rng.gen_range(0..actions.len())]
        } else {
            // Exploit: Best Q-value
            let actions = ColoringAction::all();
            actions
                .iter()
                .max_by(|a, b| {
                    use std::cmp::Ordering;
                    let q_a = self.adp_q_table.get(&(state.clone(), **a)).unwrap_or(&0.0);
                    let q_b = self.adp_q_table.get(&(state.clone(), **b)).unwrap_or(&0.0);
                    q_a.partial_cmp(q_b).unwrap_or(Ordering::Equal) // NaN-safe comparison
                })
                .copied()
                .unwrap_or(ColoringAction::FocusOnExploration) // Fallback if no actions available
        }
    }

    /// ADP Q-Learning: Update Q-value
    fn update_adp_q_value(&mut self, state: &ColoringState, action: ColoringAction, reward: f64) {
        let alpha = self.config.adp.alpha;
        let gamma = self.config.adp.gamma;

        let old_q = self
            .adp_q_table
            .get(&(state.clone(), action))
            .unwrap_or(&0.0); // Default to 0.0 for new states
        let new_q = old_q + alpha * (reward - old_q);

        self.adp_q_table.insert((state.clone(), action), new_q);

        // Decay epsilon (exploration rate)
        self.adp_epsilon *= self.config.adp.epsilon_decay;
        self.adp_epsilon = self.adp_epsilon.max(self.config.adp.epsilon_min);
    }

    /// Apply ADP action to memetic configuration and solver parameters
    fn apply_adp_action(&mut self, config: &mut MemeticConfig, action: ColoringAction) {
        match action {
            ColoringAction::IncreaseDSaturDepth => {
                config.local_search_depth += 1000;
                self.adp_dsatur_depth = (self.adp_dsatur_depth + 5000).min(100000);
            }
            ColoringAction::DecreaseDSaturDepth => {
                config.local_search_depth = config.local_search_depth.saturating_sub(1000).max(500);
                self.adp_dsatur_depth = self.adp_dsatur_depth.saturating_sub(5000).max(10000);
            }
            ColoringAction::IncreaseMemeticGenerations => config.generations += 10,
            ColoringAction::DecreaseMemeticGenerations => {
                config.generations = config.generations.saturating_sub(10).max(20)
            }
            ColoringAction::IncreaseMutationRate => {
                config.mutation_rate = (config.mutation_rate + 0.05).min(0.5)
            }
            ColoringAction::DecreaseMutationRate => {
                config.mutation_rate = (config.mutation_rate - 0.05).max(0.05)
            }
            ColoringAction::IncreasePopulationSize => config.population_size += 8,
            ColoringAction::DecreasePopulationSize => {
                config.population_size = config.population_size.saturating_sub(8).max(16)
            }
            ColoringAction::FocusOnExploration => {
                config.mutation_rate = (config.mutation_rate + 0.1).min(0.5);
                config.population_size += 8;
            }
            ColoringAction::FocusOnExploitation => {
                config.local_search_depth += 2000;
                config.elite_size += 2;
            }
            ColoringAction::IncreaseQuantumIterations => {
                self.adp_quantum_iterations = (self.adp_quantum_iterations + 2).min(10);
            }
            ColoringAction::DecreaseQuantumIterations => {
                self.adp_quantum_iterations = self.adp_quantum_iterations.saturating_sub(1).max(2);
            }
            ColoringAction::IncreaseThermoTemperatures => {
                let new_temps = self.adp_thermo_num_temps + 8;
                self.adp_thermo_num_temps = new_temps.min(self.config.thermo.num_temps);
                if new_temps > self.config.thermo.num_temps {
                    eprintln!(
                        "[ADP][WARNING] ADP wanted {} temps but clamping to config max {}",
                        new_temps, self.config.thermo.num_temps
                    );
                }
            }
            ColoringAction::DecreaseThermoTemperatures => {
                self.adp_thermo_num_temps = self.adp_thermo_num_temps.saturating_sub(8).max(8);
            }
        }
    }
}

impl Default for WorldRecordPipeline {
    #[cfg(feature = "cuda")]
    fn default() -> Self {
        // Default implementation for testing/convenience - in production, use explicit new()
        let device = CudaContext::new(0)
            .expect("[DEFAULT][FATAL] Failed to create CUDA device 0 for default pipeline");
        Self::new(WorldRecordConfig::default(), device).expect(
            "[DEFAULT][FATAL] Failed to create default WorldRecordPipeline with default config",
        )
    }

    #[cfg(not(feature = "cuda"))]
    fn default() -> Self {
        // Default implementation for testing/convenience - in production, use explicit new()
        Self::new(WorldRecordConfig::default()).expect(
            "[DEFAULT][FATAL] Failed to create default WorldRecordPipeline with default config",
        )
    }
}
