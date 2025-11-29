//! PRCT Core Domain
//!
//! Pure domain logic for Phase Resonance Chromatic-TSP algorithm.
//! This crate contains ONLY business logic - no infrastructure dependencies.
//!
//! Architecture: Hexagonal (Ports & Adapters)
//! - Domain logic depends on port abstractions (traits)
//! - Infrastructure adapters implement ports
//! - Dependency arrows point INWARD to domain

pub mod adapters;
pub mod algorithm;
pub mod coloring;
pub mod coupling;
pub mod cpu_init;
pub mod dimacs_parser;
pub mod drpp_algorithm;
pub mod errors;
pub mod ports;
pub mod simulated_annealing;
pub mod tsp; // ADDED: Adapter implementations

// Re-export main types
pub use algorithm::*;
pub use coloring::{greedy_coloring_with_ordering, phase_guided_coloring};
pub use coupling::*;
pub use cpu_init::init_rayon_threads;
pub use dimacs_parser::{parse_dimacs_file, parse_graph_file, parse_mtx_file};
pub use drpp_algorithm::*;
pub use errors::*;
pub use ports::*;
pub use simulated_annealing::*;

// Re-export adapters
#[cfg(feature = "cuda")]
pub use adapters::NeuromorphicAdapter;
pub use adapters::{CouplingAdapter, QuantumAdapter};

// Re-export shared types for convenience
pub use shared_types::*;
pub mod gpu_prct;
pub use gpu_prct::GpuPRCT;

#[cfg(feature = "cuda")]
pub mod gpu_kuramoto;
#[cfg(feature = "cuda")]
pub use gpu_kuramoto::GpuKuramotoSolver;

#[cfg(feature = "cuda")]
pub mod gpu_quantum;
#[cfg(feature = "cuda")]
pub use gpu_quantum::GpuQuantumSolver;

pub mod quantum_coloring;
pub use quantum_coloring::QuantumColoringSolver;

pub mod sparse_qubo;
pub use sparse_qubo::{ChromaticBounds, SparseQUBO};

pub mod dsatur_backtracking;
pub use dsatur_backtracking::DSaturSolver;

pub mod transfer_entropy_coloring;
pub use transfer_entropy_coloring::{
    compute_transfer_entropy_ordering, hybrid_te_kuramoto_ordering,
};

pub mod memetic_coloring;
pub use memetic_coloring::{MemeticColoringSolver, MemeticConfig};

pub mod geodesic;

pub mod cascading_pipeline;
pub use cascading_pipeline::CascadingPipeline;

pub mod world_record_pipeline;
pub use world_record_pipeline::{
    ActiveInferencePolicy, AdpConfig, EnsembleConsensus, GpuConfig, OrchestratorConfig,
    QuantumClassicalHybrid, QuantumConfig, ReservoirConflictPredictor, ThermoConfig,
    ThermodynamicEquilibrator, WorldRecordConfig, WorldRecordPipeline,
};

pub mod config_io;

#[cfg(feature = "cuda")]
pub mod world_record_pipeline_gpu;
#[cfg(feature = "cuda")]
pub use world_record_pipeline_gpu::GpuReservoirConflictPredictor;

#[cfg(feature = "cuda")]
pub mod gpu_transfer_entropy;
#[cfg(feature = "cuda")]
pub use gpu_transfer_entropy::compute_transfer_entropy_ordering_gpu;

#[cfg(feature = "cuda")]
pub mod gpu_thermodynamic;
#[cfg(feature = "cuda")]
pub use gpu_thermodynamic::equilibrate_thermodynamic_gpu;

#[cfg(feature = "cuda")]
pub mod gpu_thermodynamic_multi;
#[cfg(feature = "cuda")]
pub use gpu_thermodynamic_multi::equilibrate_thermodynamic_multi_gpu;

#[cfg(feature = "cuda")]
pub mod gpu_thermodynamic_streams;
#[cfg(feature = "cuda")]
pub use gpu_thermodynamic_streams::{ThermodynamicContext, ReplicaState, GraphGpuData};

#[cfg(feature = "cuda")]
pub mod gpu_active_inference;
#[cfg(feature = "cuda")]
pub use gpu_active_inference::{active_inference_policy_gpu, ActiveInferencePolicyGpu};

#[cfg(feature = "cuda")]
pub mod gpu_quantum_annealing;
#[cfg(feature = "cuda")]
pub use gpu_quantum_annealing::{
    gpu_qubo_simulated_annealing, qubo_solution_to_coloring, GpuQuboSolver,
};

#[cfg(feature = "cuda")]
pub mod gpu_quantum_multi;
#[cfg(feature = "cuda")]
pub use gpu_quantum_multi::{extract_coloring_from_qubo, quantum_annealing_multi_gpu};

// GPU stream management and telemetry
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub use gpu::{CudaStreamPool, EventRegistry, MultiGpuDevicePool, PipelineGpuState};

pub mod telemetry;
pub use telemetry::{PhaseExecMode, PhaseName, RunMetric, TelemetryHandle};

pub mod initial_coloring;
pub use initial_coloring::{compute_initial_coloring, InitialColoringStrategy};

pub mod iterative_controller;
pub use iterative_controller::{run_iterative_pipeline, IterativeConfig};

pub mod hypertune;
pub use hypertune::{AdpControl, HypertuneController};

pub mod reservoir_sampling;
pub use reservoir_sampling::{select_diverse_training_set, select_training_set};

// FluxNet RL force profile system (GPU-accelerated)
#[cfg(feature = "cuda")]
pub mod fluxnet;
#[cfg(feature = "cuda")]
pub use fluxnet::{
    CommandResult, Experience, FluxNetConfig, ForceBand, ForceBandStats, ForceCommand,
    ForceProfile, ForceProfileConfig, MemoryTier, PersistenceConfig, QTable, RLConfig,
    RLController, RLState, ReplayBuffer,
};
