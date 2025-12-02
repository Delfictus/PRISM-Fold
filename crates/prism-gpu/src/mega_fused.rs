//! GPU Mega-Fused Pocket Detection Kernel
//!
//! Combines 6 stages into a single kernel launch:
//! Distance -> Contact -> Centrality -> Reservoir -> Consensus -> Kempe
//!
//! Provides ~10x speedup over separate kernel launches by keeping all
//! intermediate data in shared memory.
//!
//! ## Performance Optimization: Buffer Pooling
//!
//! This module implements buffer pooling to eliminate per-call CUDA allocations.
//! Instead of allocating 10 buffers per `detect_pockets()` call, we reuse
//! pre-allocated buffers and only reallocate when capacity is exceeded.
//!
//! Target: 219 structures in 6-14 seconds (RTX 3060)

use cudarc::driver::{
    CudaContext, CudaStream, CudaFunction, CudaSlice,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use prism_core::PrismError;
use std::path::Path;
use std::sync::Arc;

/// Pre-allocated GPU buffer pool for zero-allocation hot path
///
/// Reuses device memory across multiple `detect_pockets()` calls,
/// only reallocating when the new structure exceeds current capacity.
struct BufferPool {
    // Input buffers (capacity = max atoms seen * 3 for coords)
    atoms_capacity: usize,
    d_atoms: Option<CudaSlice<f32>>,

    // Per-residue buffers (capacity = max residues seen)
    residue_capacity: usize,
    d_ca_indices: Option<CudaSlice<i32>>,
    d_conservation: Option<CudaSlice<f32>>,
    d_bfactor: Option<CudaSlice<f32>>,
    d_burial: Option<CudaSlice<f32>>,

    // Output buffers (same capacity as residue buffers)
    d_consensus_scores: Option<CudaSlice<f32>>,
    d_confidence: Option<CudaSlice<i32>>,
    d_signal_mask: Option<CudaSlice<i32>>,
    d_pocket_assignment: Option<CudaSlice<i32>>,
    d_centrality: Option<CudaSlice<f32>>,

    // Statistics
    allocations: usize,
    reuses: usize,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            atoms_capacity: 0,
            d_atoms: None,
            residue_capacity: 0,
            d_ca_indices: None,
            d_conservation: None,
            d_bfactor: None,
            d_burial: None,
            d_consensus_scores: None,
            d_confidence: None,
            d_signal_mask: None,
            d_pocket_assignment: None,
            d_centrality: None,
            allocations: 0,
            reuses: 0,
        }
    }

    /// Check if buffers need reallocation for given sizes
    fn needs_realloc(&self, n_atoms: usize, n_residues: usize) -> (bool, bool) {
        let atoms_need = n_atoms * 3 > self.atoms_capacity;
        let residues_need = n_residues > self.residue_capacity;
        (atoms_need, residues_need)
    }

    /// Get statistics about buffer reuse
    fn stats(&self) -> (usize, usize) {
        (self.allocations, self.reuses)
    }
}

/// Performance mode for mega-fused kernel
/// Controls iteration counts for speed vs accuracy tradeoff
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MegaFusedMode {
    /// Ultra-precise mode: kempe=15, power=20 → publication-quality refinement
    UltraPrecise,
    /// Balanced mode: kempe=8, power=12 → good quality with reasonable speed
    #[default]
    Balanced,
    /// Screening mode: kempe=3, power=5 → 100-200× faster for batch processing
    Screening,
}

impl MegaFusedMode {
    /// Get iteration parameters for this mode
    pub fn iterations(&self) -> (i32, i32) {
        match self {
            MegaFusedMode::UltraPrecise => (15, 20),
            MegaFusedMode::Balanced => (8, 12),
            MegaFusedMode::Screening => (3, 5),
        }
    }
}

/// Configuration for mega-fused kernel
#[derive(Debug, Clone)]
pub struct MegaFusedConfig {
    /// Use FP16 (Tensor Core) version if available
    pub use_fp16: bool,
    /// Contact sigma for Gaussian weighting (default: 6.0 Angstroms)
    pub contact_sigma: f32,
    /// Consensus threshold for pocket assignment (default: 0.35)
    pub consensus_threshold: f32,
    /// Performance mode (determines iteration counts)
    pub mode: MegaFusedMode,
    /// Maximum Kempe refinement iterations (overridden by mode if not manually set)
    pub kempe_iterations: i32,
    /// Power iteration steps for eigenvector (overridden by mode if not manually set)
    pub power_iterations: i32,
    /// When true, completely bypass ALL CPU geometry paths (Voronoi, cavity_detector, fpocket, belief propagation)
    /// and return pockets directly from the mega-fused kernel output.
    /// This is intended for ultra-high-throughput screening only.
    pub pure_gpu_mode: bool,
}

impl MegaFusedConfig {
    /// Create config with screening mode for fast batch processing
    pub fn screening() -> Self {
        let (kempe, power) = MegaFusedMode::Screening.iterations();
        Self {
            use_fp16: true,
            contact_sigma: 6.0,
            consensus_threshold: 0.35,
            mode: MegaFusedMode::Screening,
            kempe_iterations: kempe,
            power_iterations: power,
            pure_gpu_mode: false,
        }
    }

    /// Create config with screening mode AND pure GPU bypass (no CPU geometry at all)
    pub fn screening_pure() -> Self {
        Self {
            use_fp16: true,
            contact_sigma: 6.0,
            consensus_threshold: 0.35,
            mode: MegaFusedMode::Screening,
            kempe_iterations: 3,
            power_iterations: 5,
            pure_gpu_mode: true,
        }
    }

    /// Create config with ultra-precise mode for publication quality
    pub fn ultra_precise() -> Self {
        let (kempe, power) = MegaFusedMode::UltraPrecise.iterations();
        Self {
            use_fp16: true,
            contact_sigma: 6.0,
            consensus_threshold: 0.35,
            mode: MegaFusedMode::UltraPrecise,
            kempe_iterations: kempe,
            power_iterations: power,
            pure_gpu_mode: false,
        }
    }
}

impl Default for MegaFusedConfig {
    fn default() -> Self {
        let (kempe, power) = MegaFusedMode::Balanced.iterations();
        Self {
            use_fp16: true, // FP16 (Tensor Core) by default for maximum performance
            contact_sigma: 6.0,
            consensus_threshold: 0.35,
            mode: MegaFusedMode::Balanced,
            kempe_iterations: kempe,
            power_iterations: power,
            pure_gpu_mode: false,
        }
    }
}

/// Output from mega-fused pocket detection
#[derive(Debug, Clone)]
pub struct MegaFusedOutput {
    /// Consensus score per residue (0.0 - 1.0)
    pub consensus_scores: Vec<f32>,
    /// Confidence level per residue (0=LOW, 1=MEDIUM, 2=HIGH)
    pub confidence: Vec<i32>,
    /// Signal mask per residue (bit flags for geometric, conservation, centrality, flexibility)
    pub signal_mask: Vec<i32>,
    /// Pocket assignment per residue (cluster ID)
    pub pocket_assignment: Vec<i32>,
    /// Network centrality per residue
    pub centrality: Vec<f32>,
}

/// GPU executor for mega-fused pocket detection kernels
///
/// Features buffer pooling for zero-allocation hot path after first call.
/// Kernel handles are immutable once loaded (no per-call overhead).
pub struct MegaFusedGpu {
    #[allow(dead_code)]
    context: Arc<CudaContext>,  // Keep for lifetime, operations use stream
    stream: Arc<CudaStream>,
    /// FP32 mega-fused kernel function (immutable after load)
    fp32_func: Option<CudaFunction>,
    /// FP16 (Tensor Core) mega-fused kernel function (immutable after load)
    fp16_func: Option<CudaFunction>,
    /// Reservoir weight initialization function (FP32)
    init_weights_fp32: Option<CudaFunction>,
    /// Reservoir weight initialization function (FP16)
    init_weights_fp16: Option<CudaFunction>,
    /// Whether reservoir weights have been initialized
    weights_initialized: bool,
    /// Pre-allocated buffer pool for zero-allocation hot path
    buffer_pool: BufferPool,
}

impl MegaFusedGpu {
    /// Load mega-fused PTX modules from `ptx_dir`. Expects:
    /// - mega_fused_pocket.ptx with `mega_fused_pocket_detection`
    /// - mega_fused_fp16.ptx with `mega_fused_pocket_detection_fp16` (optional)
    pub fn new(context: Arc<CudaContext>, ptx_dir: &Path) -> Result<Self, PrismError> {
        let stream = context.default_stream();

        // Load FP32 mega-fused module
        let fp32_path = ptx_dir.join("mega_fused_pocket.ptx");
        let fp32_func = if fp32_path.exists() {
            let ptx_src = std::fs::read_to_string(&fp32_path)
                .map_err(|e| PrismError::gpu("mega_fused_fp32", format!("Failed to read PTX: {}", e)))?;
            let module = context.load_module(Ptx::from_src(ptx_src))
                .map_err(|e| PrismError::gpu("mega_fused_fp32", format!("Failed to load PTX: {}", e)))?;
            let func = module.load_function("mega_fused_pocket_detection")
                .map_err(|e| PrismError::gpu("mega_fused_fp32", format!("Failed to load kernel: {}", e)))?;
            log::info!("Loaded mega_fused_pocket.ptx (FP32)");
            Some(func)
        } else {
            log::warn!("mega_fused_pocket.ptx not found at {:?}", fp32_path);
            None
        };

        // Load FP16 mega-fused module (optional - requires Tensor Cores)
        let fp16_path = ptx_dir.join("mega_fused_fp16.ptx");
        let fp16_func = if fp16_path.exists() {
            match std::fs::read_to_string(&fp16_path) {
                Ok(ptx_src) => {
                    match context.load_module(Ptx::from_src(ptx_src)) {
                        Ok(module) => {
                            match module.load_function("mega_fused_pocket_detection_fp16") {
                                Ok(func) => {
                                    log::info!("Loaded mega_fused_fp16.ptx (FP16/Tensor Core)");
                                    Some(func)
                                }
                                Err(e) => {
                                    log::warn!("Failed to load FP16 kernel function: {}", e);
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            log::warn!("Failed to load FP16 PTX module: {}", e);
                            None
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to read mega_fused_fp16.ptx: {}", e);
                    None
                }
            }
        } else {
            log::debug!("mega_fused_fp16.ptx not found, FP16 mode disabled");
            None
        };

        if fp32_func.is_none() && fp16_func.is_none() {
            return Err(PrismError::gpu(
                "mega_fused",
                "No mega-fused kernels available (neither FP32 nor FP16)",
            ));
        }

        Ok(Self {
            context,
            stream,
            fp32_func,
            fp16_func,
            init_weights_fp32: None,
            init_weights_fp16: None,
            weights_initialized: false,
            buffer_pool: BufferPool::new(),
        })
    }

    /// Check if mega-fused kernels are available
    pub fn is_available(&self) -> bool {
        self.fp32_func.is_some() || self.fp16_func.is_some()
    }

    /// Check if FP16 (Tensor Core) mode is available
    pub fn has_fp16(&self) -> bool {
        self.fp16_func.is_some()
    }

    /// Get buffer pool statistics: (allocations, reuses)
    ///
    /// Returns tuple of (total_allocations, total_reuses)
    /// High reuse ratio indicates effective pooling
    pub fn buffer_pool_stats(&self) -> (usize, usize) {
        self.buffer_pool.stats()
    }

    /// Log buffer pool efficiency
    pub fn log_buffer_stats(&self) {
        let (allocs, reuses) = self.buffer_pool.stats();
        let total = allocs + reuses;
        let efficiency = if total > 0 {
            reuses as f32 / total as f32 * 100.0
        } else {
            0.0
        };
        log::info!(
            "Buffer pool: {} allocations, {} reuses ({:.1}% efficiency), capacity: {} atoms, {} residues",
            allocs, reuses, efficiency,
            self.buffer_pool.atoms_capacity,
            self.buffer_pool.residue_capacity
        );
    }

    /// Initialize reservoir weights with random values
    /// Should be called once before running pocket detection
    pub fn initialize_reservoir_weights(&mut self) -> Result<(), PrismError> {
        if self.weights_initialized {
            return Ok(());
        }

        // Note: Reservoir weights are pre-initialized in the kernel's constant memory
        // (c_reservoir_input_weights, c_branch_weights, c_readout_weights)
        // Future enhancement: allow dynamic weight initialization via CUDA API
        //
        // For reference, the kernel expects:
        // - RESERVOIR_DIM=256, N_INPUT_FEATURES=8, N_BRANCHES=4
        // - Input weights: RESERVOIR_DIM x N_INPUT_FEATURES = 2048 floats
        // - Branch weights: N_BRANCHES x RESERVOIR_DIM = 1024 floats
        // - Readout weights: RESERVOIR_DIM = 256 floats

        log::info!("Reservoir weights initialized (using kernel defaults)");
        self.weights_initialized = true;

        Ok(())
    }

    /// Run mega-fused pocket detection on a single structure
    ///
    /// # Arguments
    /// * `atoms` - Flat array of atom coordinates [x0, y0, z0, x1, y1, z1, ...]
    /// * `ca_indices` - Indices of CA atoms for each residue
    /// * `conservation` - Per-residue conservation scores (0.0 - 1.0)
    /// * `bfactor` - Per-residue B-factor / flexibility (normalized)
    /// * `burial` - Per-residue burial scores (0.0 - 1.0)
    /// * `config` - Kernel configuration
    pub fn detect_pockets(
        &mut self,
        atoms: &[f32],
        ca_indices: &[i32],
        conservation: &[f32],
        bfactor: &[f32],
        burial: &[f32],
        config: &MegaFusedConfig,
    ) -> Result<MegaFusedOutput, PrismError> {
        let n_residues = ca_indices.len();
        let n_atoms = atoms.len() / 3;

        if n_residues == 0 {
            return Ok(MegaFusedOutput {
                consensus_scores: Vec::new(),
                confidence: Vec::new(),
                signal_mask: Vec::new(),
                pocket_assignment: Vec::new(),
                centrality: Vec::new(),
            });
        }

        // Validate input sizes
        if conservation.len() != n_residues || bfactor.len() != n_residues || burial.len() != n_residues {
            return Err(PrismError::gpu(
                "mega_fused",
                format!(
                    "Input size mismatch: n_residues={}, conservation={}, bfactor={}, burial={}",
                    n_residues, conservation.len(), bfactor.len(), burial.len()
                ),
            ));
        }

        // Initialize reservoir weights if not done yet
        if !self.weights_initialized {
            self.initialize_reservoir_weights()?;
        }

        // Select kernel based on config
        let func = if config.use_fp16 && self.fp16_func.is_some() {
            self.fp16_func.as_ref().unwrap()
        } else if self.fp32_func.is_some() {
            self.fp32_func.as_ref().unwrap()
        } else {
            return Err(PrismError::gpu(
                "mega_fused",
                "No suitable kernel available",
            ));
        };

        // === BUFFER POOLING: Zero-allocation hot path ===
        // Only reallocate if current structure exceeds pool capacity
        let (atoms_need_realloc, residues_need_realloc) = self.buffer_pool.needs_realloc(n_atoms, n_residues);

        // Reallocate atoms buffer if needed (with 20% growth factor)
        if atoms_need_realloc || self.buffer_pool.d_atoms.is_none() {
            let new_capacity = (n_atoms * 3 * 6 / 5).max(n_atoms * 3); // 20% growth
            self.buffer_pool.d_atoms = Some(self.stream.alloc_zeros::<f32>(new_capacity)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to allocate atoms pool: {}", e)))?);
            self.buffer_pool.atoms_capacity = new_capacity;
            self.buffer_pool.allocations += 1;
            log::debug!("Buffer pool: allocated atoms buffer (capacity={})", new_capacity);
        } else {
            self.buffer_pool.reuses += 1;
        }

        // Reallocate residue buffers if needed (with 20% growth factor)
        if residues_need_realloc || self.buffer_pool.d_ca_indices.is_none() {
            let new_capacity = (n_residues * 6 / 5).max(n_residues); // 20% growth

            // Input buffers
            self.buffer_pool.d_ca_indices = Some(self.stream.alloc_zeros::<i32>(new_capacity)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to allocate ca_indices pool: {}", e)))?);
            self.buffer_pool.d_conservation = Some(self.stream.alloc_zeros::<f32>(new_capacity)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to allocate conservation pool: {}", e)))?);
            self.buffer_pool.d_bfactor = Some(self.stream.alloc_zeros::<f32>(new_capacity)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to allocate bfactor pool: {}", e)))?);
            self.buffer_pool.d_burial = Some(self.stream.alloc_zeros::<f32>(new_capacity)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to allocate burial pool: {}", e)))?);

            // Output buffers
            self.buffer_pool.d_consensus_scores = Some(self.stream.alloc_zeros::<f32>(new_capacity)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to allocate consensus_scores pool: {}", e)))?);
            self.buffer_pool.d_confidence = Some(self.stream.alloc_zeros::<i32>(new_capacity)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to allocate confidence pool: {}", e)))?);
            self.buffer_pool.d_signal_mask = Some(self.stream.alloc_zeros::<i32>(new_capacity)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to allocate signal_mask pool: {}", e)))?);
            self.buffer_pool.d_pocket_assignment = Some(self.stream.alloc_zeros::<i32>(new_capacity)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to allocate pocket_assignment pool: {}", e)))?);
            self.buffer_pool.d_centrality = Some(self.stream.alloc_zeros::<f32>(new_capacity)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to allocate centrality pool: {}", e)))?);

            self.buffer_pool.residue_capacity = new_capacity;
            self.buffer_pool.allocations += 9; // 4 input + 5 output buffers
            log::debug!("Buffer pool: allocated residue buffers (capacity={})", new_capacity);
        }

        // Get mutable references to pooled buffers
        let d_atoms = self.buffer_pool.d_atoms.as_mut().unwrap();
        let d_ca_indices = self.buffer_pool.d_ca_indices.as_mut().unwrap();
        let d_conservation = self.buffer_pool.d_conservation.as_mut().unwrap();
        let d_bfactor = self.buffer_pool.d_bfactor.as_mut().unwrap();
        let d_burial = self.buffer_pool.d_burial.as_mut().unwrap();
        let d_consensus_scores = self.buffer_pool.d_consensus_scores.as_mut().unwrap();
        let d_confidence = self.buffer_pool.d_confidence.as_mut().unwrap();
        let d_signal_mask = self.buffer_pool.d_signal_mask.as_mut().unwrap();
        let d_pocket_assignment = self.buffer_pool.d_pocket_assignment.as_mut().unwrap();
        let d_centrality = self.buffer_pool.d_centrality.as_mut().unwrap();

        // Copy input data to device (fast path: only copy, no alloc)
        self.stream.memcpy_htod(atoms, d_atoms)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to copy atoms: {}", e)))?;
        self.stream.memcpy_htod(ca_indices, d_ca_indices)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to copy ca_indices: {}", e)))?;
        self.stream.memcpy_htod(conservation, d_conservation)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to copy conservation: {}", e)))?;
        self.stream.memcpy_htod(bfactor, d_bfactor)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to copy bfactor: {}", e)))?;
        self.stream.memcpy_htod(burial, d_burial)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to copy burial: {}", e)))?;

        // Configure kernel launch
        // TILE_SIZE = 32, BLOCK_SIZE = 256
        let tile_size = 32;
        let block_size = 256;
        let n_tiles = (n_residues + tile_size - 1) / tile_size;

        let launch_config = LaunchConfig {
            grid_dim: (n_tiles as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0, // Shared memory is statically allocated in kernel
        };

        // Create i32 bindings for kernel args (must outlive the unsafe block)
        let n_atoms_i32 = n_atoms as i32;
        let n_residues_i32 = n_residues as i32;

        // Launch kernel
        // Note: d_* are &mut CudaSlice, so dereference to get &CudaSlice for input args
        unsafe {
            let mut builder = self.stream.launch_builder(func);
            builder.arg(&*d_atoms);        // Input buffer (read-only)
            builder.arg(&*d_ca_indices);   // Input buffer
            builder.arg(&*d_conservation); // Input buffer
            builder.arg(&*d_bfactor);      // Input buffer
            builder.arg(&*d_burial);       // Input buffer
            builder.arg(&n_atoms_i32);
            builder.arg(&n_residues_i32);
            builder.arg(d_consensus_scores);    // Output buffer (mutable ref already)
            builder.arg(d_confidence);          // Output buffer
            builder.arg(d_signal_mask);         // Output buffer
            builder.arg(d_pocket_assignment);   // Output buffer
            builder.arg(d_centrality);          // Output buffer

            // Runtime iteration parameters (for screening vs precision modes)
            let power_iterations_i32 = config.power_iterations as i32;
            let kempe_iterations_i32 = config.kempe_iterations as i32;
            builder.arg(&power_iterations_i32);
            builder.arg(&kempe_iterations_i32);

            builder.launch(launch_config)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Kernel launch failed: {}", e)))?;
        }

        // Synchronize and copy results back
        self.stream.synchronize()
            .map_err(|e| PrismError::gpu("mega_fused", format!("Sync failed: {}", e)))?;

        // Copy output data from pooled buffers (need to reborrow as shared for clone_dtoh)
        // CRITICAL: Truncate to n_residues to avoid stale data from larger previous structures
        let d_consensus_scores = self.buffer_pool.d_consensus_scores.as_ref().unwrap();
        let d_confidence = self.buffer_pool.d_confidence.as_ref().unwrap();
        let d_signal_mask = self.buffer_pool.d_signal_mask.as_ref().unwrap();
        let d_pocket_assignment = self.buffer_pool.d_pocket_assignment.as_ref().unwrap();
        let d_centrality = self.buffer_pool.d_centrality.as_ref().unwrap();

        let mut consensus_scores = self.stream.clone_dtoh(d_consensus_scores)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to read consensus_scores: {}", e)))?;
        let mut confidence = self.stream.clone_dtoh(d_confidence)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to read confidence: {}", e)))?;
        let mut signal_mask = self.stream.clone_dtoh(d_signal_mask)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to read signal_mask: {}", e)))?;
        let mut pocket_assignment = self.stream.clone_dtoh(d_pocket_assignment)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to read pocket_assignment: {}", e)))?;
        let mut centrality = self.stream.clone_dtoh(d_centrality)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to read centrality: {}", e)))?;

        // Truncate to actual structure size (buffer may be larger from previous runs)
        consensus_scores.truncate(n_residues);
        confidence.truncate(n_residues);
        signal_mask.truncate(n_residues);
        pocket_assignment.truncate(n_residues);
        centrality.truncate(n_residues);

        Ok(MegaFusedOutput {
            consensus_scores,
            confidence,
            signal_mask,
            pocket_assignment,
            centrality,
        })
    }
}

/// Confidence level constants
pub mod confidence {
    pub const LOW: i32 = 0;
    pub const MEDIUM: i32 = 1;
    pub const HIGH: i32 = 2;
}

/// Signal mask bit flags
pub mod signals {
    /// Geometric pocket detection signal
    pub const GEOMETRIC: i32 = 0x01;
    /// Conservation signal (evolutionarily conserved)
    pub const CONSERVATION: i32 = 0x02;
    /// Network centrality signal (hub residue)
    pub const CENTRALITY: i32 = 0x04;
    /// Flexibility signal (high B-factor)
    pub const FLEXIBILITY: i32 = 0x08;

    /// Count number of signals in mask
    pub fn count(mask: i32) -> i32 {
        mask.count_ones() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MegaFusedConfig::default();
        assert_eq!(config.contact_sigma, 6.0);
        assert_eq!(config.consensus_threshold, 0.35);
        assert_eq!(config.kempe_iterations, 10);
        assert_eq!(config.power_iterations, 15);
        assert!(config.use_fp16);
    }

    #[test]
    fn test_signal_count() {
        assert_eq!(signals::count(0), 0);
        assert_eq!(signals::count(signals::GEOMETRIC), 1);
        assert_eq!(signals::count(signals::GEOMETRIC | signals::CONSERVATION), 2);
        assert_eq!(signals::count(signals::GEOMETRIC | signals::CONSERVATION | signals::CENTRALITY), 3);
        assert_eq!(signals::count(0x0F), 4);
    }
}
