//! GPU Mega-Fused BATCH Pocket Detection Kernel
//!
//! Processes ALL structures in a SINGLE kernel launch for maximum throughput.
//! Uses L1 cache hints (__ldg) and register optimization (__launch_bounds__).
//!
//! Target: 221 structures in <100ms (vs 2+ seconds sequential)
//!
//! ## Architecture
//! - One CUDA block per structure (grid_dim = n_structures)
//! - Packed contiguous arrays for all structures
//! - BatchStructureDesc provides offsets for each structure
//! - 6 stages fused: Distance → Contact → Centrality → Reservoir → Consensus → Kempe

use cudarc::driver::{
    CudaContext, CudaStream, CudaFunction, CudaSlice,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use prism_core::PrismError;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use serde::{Serialize, Deserialize};

// Re-export from mega_fused.rs for shared types
pub use super::mega_fused::{
    MegaFusedParams, MegaFusedConfig, MegaFusedMode, MegaFusedOutput,
    GpuTelemetry, GpuProvenanceData, KernelTelemetryEvent,
    confidence, signals,
};

/// Structure descriptor for batch processing - MUST match CUDA BatchStructureDesc
/// Uses 16-byte alignment for coalesced memory access
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchStructureDesc {
    /// Start index in packed atoms array (atoms_packed[atom_offset * 3])
    pub atom_offset: i32,
    /// Start index in packed residue arrays
    pub residue_offset: i32,
    /// Number of atoms in this structure
    pub n_atoms: i32,
    /// Number of residues in this structure
    pub n_residues: i32,
}

/// Input data for a single structure in batch processing
#[derive(Debug, Clone)]
pub struct StructureInput {
    /// Structure identifier (e.g., PDB ID)
    pub id: String,
    /// Flat array of atom coordinates [x0, y0, z0, x1, y1, z1, ...]
    pub atoms: Vec<f32>,
    /// Indices of CA atoms for each residue
    pub ca_indices: Vec<i32>,
    /// Per-residue conservation scores (0.0 - 1.0)
    pub conservation: Vec<f32>,
    /// Per-residue B-factor / flexibility (normalized)
    pub bfactor: Vec<f32>,
    /// Per-residue burial scores (0.0 - 1.0)
    pub burial: Vec<f32>,
}

impl StructureInput {
    /// Create new structure input
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            atoms: Vec::new(),
            ca_indices: Vec::new(),
            conservation: Vec::new(),
            bfactor: Vec::new(),
            burial: Vec::new(),
        }
    }

    /// Number of atoms in this structure
    pub fn n_atoms(&self) -> usize {
        self.atoms.len() / 3
    }

    /// Number of residues in this structure
    pub fn n_residues(&self) -> usize {
        self.ca_indices.len()
    }

    /// Validate input consistency
    pub fn validate(&self) -> Result<(), String> {
        let n_res = self.n_residues();
        if self.atoms.len() % 3 != 0 {
            return Err(format!("atoms array length {} not divisible by 3", self.atoms.len()));
        }
        if self.conservation.len() != n_res {
            return Err(format!("conservation length {} != n_residues {}", self.conservation.len(), n_res));
        }
        if self.bfactor.len() != n_res {
            return Err(format!("bfactor length {} != n_residues {}", self.bfactor.len(), n_res));
        }
        if self.burial.len() != n_res {
            return Err(format!("burial length {} != n_residues {}", self.burial.len(), n_res));
        }
        Ok(())
    }
}

//=============================================================================
// v2.0 FINAL: BATCH METRICS STRUCTURES (2025-12-05)
//=============================================================================

/// Structure input WITH ground truth for validation batches
#[derive(Debug, Clone)]
pub struct StructureInputWithGT {
    pub base: StructureInput,
    pub gt_pocket_mask: Vec<u8>,
}

/// Structure offset for batch mapping - MUST match CUDA StructureOffset
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default)]
pub struct StructureOffset {
    pub structure_id: i32,
    pub residue_start: i32,
    pub residue_count: i32,
    pub padding: i32,
}

// Implement bytemuck traits for safe casting
unsafe impl bytemuck::Pod for StructureOffset {}
unsafe impl bytemuck::Zeroable for StructureOffset {}

/// Per-structure metrics from GPU - MUST match CUDA BatchMetricsOutput
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchMetricsOutput {
    pub structure_id: i32,
    pub n_residues: i32,
    pub true_positives: i32,
    pub false_positives: i32,
    pub true_negatives: i32,
    pub false_negatives: i32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub mcc: f32,
    pub auc_roc: f32,
    pub auprc: f32,
    pub avg_druggability: f32,
    pub n_pockets_detected: i32,
}

unsafe impl bytemuck::Pod for BatchMetricsOutput {}
unsafe impl bytemuck::Zeroable for BatchMetricsOutput {}
unsafe impl cudarc::driver::DeviceRepr for BatchMetricsOutput {}
unsafe impl cudarc::driver::ValidAsZeroBits for BatchMetricsOutput {}

const N_BINS: usize = 100;

/// Packed batch WITH ground truth for validation
#[derive(Debug)]
pub struct PackedBatchWithGT {
    pub base: PackedBatch,
    pub gt_pocket_mask_packed: Vec<u8>,
    pub offsets: Vec<StructureOffset>,
    pub tile_prefix_sum: Vec<i32>,
    pub total_tiles: i32,
}

impl PackedBatchWithGT {
    pub fn from_structures_with_gt(structures: &[StructureInputWithGT]) -> Result<Self, PrismError> {
        // FIX 3: GT mask validation
        for s in structures {
            assert_eq!(
                s.gt_pocket_mask.len(),
                s.base.n_residues(),
                "GT mask length mismatch for structure {}",
                s.base.id
            );
        }

        let base_inputs: Vec<StructureInput> = structures.iter().map(|s| s.base.clone()).collect();
        let base = PackedBatch::from_structures(&base_inputs)?;

        let mut gt_packed: Vec<u8> = Vec::with_capacity(base.total_residues);
        let mut offsets: Vec<StructureOffset> = Vec::with_capacity(structures.len());
        let mut tile_prefix_sum: Vec<i32> = vec![0; structures.len() + 1];

        let mut residue_offset = 0i32;
        let tile_size = 32i32;

        for (idx, s) in structures.iter().enumerate() {
            let n_res = s.base.n_residues() as i32;
            let n_tiles = (n_res + tile_size - 1) / tile_size;

            offsets.push(StructureOffset {
                structure_id: idx as i32,
                residue_start: residue_offset,
                residue_count: n_res,
                padding: 0,
            });

            tile_prefix_sum[idx + 1] = tile_prefix_sum[idx] + n_tiles;
            gt_packed.extend_from_slice(&s.gt_pocket_mask);
            residue_offset += n_res;
        }

        let total_tiles = tile_prefix_sum[structures.len()];

        log::info!(
            "PackedBatchWithGT: {} structures, {} residues, {} tiles",
            structures.len(), base.total_residues, total_tiles
        );

        Ok(Self {
            base,
            gt_pocket_mask_packed: gt_packed,
            offsets,
            tile_prefix_sum,
            total_tiles,
        })
    }
}

/// Batch output including per-structure metrics
#[derive(Debug)]
pub struct BatchOutputWithMetrics {
    pub structures: Vec<BatchStructureOutput>,
    pub metrics: Vec<BatchMetricsOutput>,
    pub kernel_time_us: u64,
    pub aggregate: AggregateMetrics,
}

#[derive(Debug, Clone, Default)]
pub struct AggregateMetrics {
    pub mean_f1: f32,
    pub mean_mcc: f32,
    pub mean_auc_roc: f32,
    pub mean_auprc: f32,
    pub mean_precision: f32,
    pub mean_recall: f32,
}

/// Packed batch of structures ready for GPU transfer
#[derive(Debug)]
pub struct PackedBatch {
    /// Structure descriptors with offsets
    pub descriptors: Vec<BatchStructureDesc>,
    /// Structure IDs in order
    pub ids: Vec<String>,
    /// Packed atom coordinates for all structures
    pub atoms_packed: Vec<f32>,
    /// Packed CA indices for all structures
    pub ca_indices_packed: Vec<i32>,
    /// Packed conservation scores for all structures
    pub conservation_packed: Vec<f32>,
    /// Packed B-factors for all structures
    pub bfactor_packed: Vec<f32>,
    /// Packed burial scores for all structures
    pub burial_packed: Vec<f32>,
    /// Total atoms across all structures
    pub total_atoms: usize,
    /// Total residues across all structures
    pub total_residues: usize,
}

impl PackedBatch {
    /// Pack multiple structures into contiguous arrays
    pub fn from_structures(structures: &[StructureInput]) -> Result<Self, PrismError> {
        // Validate all structures first
        for (i, s) in structures.iter().enumerate() {
            if let Err(e) = s.validate() {
                return Err(PrismError::gpu(
                    "batch_pack",
                    format!("Structure {} ({}): {}", i, s.id, e),
                ));
            }
        }

        let n_structures = structures.len();
        if n_structures == 0 {
            return Err(PrismError::gpu("batch_pack", "No structures provided"));
        }

        // Calculate total sizes
        let total_atoms: usize = structures.iter().map(|s| s.n_atoms()).sum();
        let total_residues: usize = structures.iter().map(|s| s.n_residues()).sum();

        // Pre-allocate packed arrays
        let mut atoms_packed = Vec::with_capacity(total_atoms * 3);
        let mut ca_indices_packed = Vec::with_capacity(total_residues);
        let mut conservation_packed = Vec::with_capacity(total_residues);
        let mut bfactor_packed = Vec::with_capacity(total_residues);
        let mut burial_packed = Vec::with_capacity(total_residues);
        let mut descriptors = Vec::with_capacity(n_structures);
        let mut ids = Vec::with_capacity(n_structures);

        let mut atom_offset = 0i32;
        let mut residue_offset = 0i32;

        for s in structures {
            // Record descriptor
            descriptors.push(BatchStructureDesc {
                atom_offset,
                residue_offset,
                n_atoms: s.n_atoms() as i32,
                n_residues: s.n_residues() as i32,
            });
            ids.push(s.id.clone());

            // Pack atoms
            atoms_packed.extend_from_slice(&s.atoms);

            // Pack residue data (CA indices need offset adjustment)
            for &ca_idx in &s.ca_indices {
                ca_indices_packed.push(ca_idx + atom_offset);
            }
            conservation_packed.extend_from_slice(&s.conservation);
            bfactor_packed.extend_from_slice(&s.bfactor);
            burial_packed.extend_from_slice(&s.burial);

            // Update offsets for next structure
            atom_offset += s.n_atoms() as i32;
            residue_offset += s.n_residues() as i32;
        }

        log::info!(
            "Packed {} structures: {} atoms, {} residues",
            n_structures, total_atoms, total_residues
        );

        Ok(Self {
            descriptors,
            ids,
            atoms_packed,
            ca_indices_packed,
            conservation_packed,
            bfactor_packed,
            burial_packed,
            total_atoms,
            total_residues,
        })
    }

    /// Number of structures in batch
    pub fn n_structures(&self) -> usize {
        self.descriptors.len()
    }
}

/// Output from batch pocket detection for a single structure
#[derive(Debug, Clone)]
pub struct BatchStructureOutput {
    /// Structure ID
    pub id: String,
    /// Consensus score per residue (0.0 - 1.0)
    pub consensus_scores: Vec<f32>,
    /// Confidence level per residue (0=LOW, 1=MEDIUM, 2=HIGH)
    pub confidence: Vec<i32>,
    /// Signal mask per residue (bit flags)
    pub signal_mask: Vec<i32>,
    /// Pocket assignment per residue (cluster ID)
    pub pocket_assignment: Vec<i32>,
    /// Network centrality per residue
    pub centrality: Vec<f32>,
}

/// Complete output from batch processing
#[derive(Debug)]
pub struct BatchOutput {
    /// Per-structure outputs
    pub structures: Vec<BatchStructureOutput>,
    /// GPU telemetry for provenance
    pub gpu_telemetry: Option<GpuProvenanceData>,
    /// Total batch processing time (microseconds)
    pub batch_time_us: u64,
    /// Kernel launch time only (microseconds)
    pub kernel_time_us: u64,
}

/// GPU buffer pool for batch processing
struct BatchBufferPool {
    // Packed input buffers
    atoms_capacity: usize,
    d_atoms: Option<CudaSlice<f32>>,

    residue_capacity: usize,
    d_ca_indices: Option<CudaSlice<i32>>,
    d_conservation: Option<CudaSlice<f32>>,
    d_bfactor: Option<CudaSlice<f32>>,
    d_burial: Option<CudaSlice<f32>>,

    // Structure descriptors
    descriptors_capacity: usize,
    d_descriptors: Option<CudaSlice<u8>>,

    // Output buffers (per-residue)
    d_consensus_scores: Option<CudaSlice<f32>>,
    d_confidence: Option<CudaSlice<i32>>,
    d_signal_mask: Option<CudaSlice<i32>>,
    d_pocket_assignment: Option<CudaSlice<i32>>,
    d_centrality: Option<CudaSlice<f32>>,

    // Params buffer
    d_params: Option<CudaSlice<u8>>,

    // Statistics
    allocations: usize,
    reuses: usize,
}

impl BatchBufferPool {
    fn new() -> Self {
        Self {
            atoms_capacity: 0,
            d_atoms: None,
            residue_capacity: 0,
            d_ca_indices: None,
            d_conservation: None,
            d_bfactor: None,
            d_burial: None,
            descriptors_capacity: 0,
            d_descriptors: None,
            d_consensus_scores: None,
            d_confidence: None,
            d_signal_mask: None,
            d_pocket_assignment: None,
            d_centrality: None,
            d_params: None,
            allocations: 0,
            reuses: 0,
        }
    }

    fn stats(&self) -> (usize, usize) {
        (self.allocations, self.reuses)
    }
}

/// GPU executor for mega-fused BATCH pocket detection
///
/// Processes all structures in a SINGLE kernel launch.
/// Uses L1 cache optimization and register allocation hints.
pub struct MegaFusedBatchGpu {
    #[allow(dead_code)]
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Batch kernel function
    batch_func: Option<CudaFunction>,
    /// v2.0: Batch kernel with metrics
    batch_metrics_func: Option<CudaFunction>,
    /// v2.0: Finalize metrics kernel
    finalize_func: Option<CudaFunction>,
    /// Buffer pool for batch processing
    buffer_pool: BatchBufferPool,
    /// Telemetry collector
    telemetry: GpuTelemetry,
}

impl MegaFusedBatchGpu {
    /// Load batch mega-fused PTX from `ptx_dir`. Expects:
    /// - mega_fused_batch.ptx with `mega_fused_batch_detection`
    pub fn new(context: Arc<CudaContext>, ptx_dir: &Path) -> Result<Self, PrismError> {
        let stream = context.default_stream();

        // Load batch kernel
        let batch_path = ptx_dir.join("mega_fused_batch.ptx");
        let batch_func = if batch_path.exists() {
            let ptx_src = std::fs::read_to_string(&batch_path)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Failed to read PTX: {}", e)))?;
            let module = context.load_module(Ptx::from_src(ptx_src))
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Failed to load PTX: {}", e)))?;
            let func = module.load_function("mega_fused_batch_detection")
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Failed to load kernel: {}", e)))?;
            log::info!("Loaded mega_fused_batch.ptx (L1/Register optimized batch kernel)");
            Some(func)
        } else {
            log::warn!("mega_fused_batch.ptx not found at {:?}", batch_path);
            None
        };

        if batch_func.is_none() {
            return Err(PrismError::gpu(
                "mega_fused_batch",
                "Batch kernel not available (mega_fused_batch.ptx not found)",
            ));
        }

        // Load v2.0 metrics kernels from the same PTX
        let batch_metrics_func = if batch_path.exists() {
            let ptx_src = std::fs::read_to_string(&batch_path)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Failed to read PTX: {}", e)))?;
            let module = context.load_module(Ptx::from_src(ptx_src))
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Failed to load PTX: {}", e)))?;
            match module.load_function("mega_fused_pocket_detection_batch_with_metrics") {
                Ok(func) => {
                    log::info!("Loaded mega_fused_pocket_detection_batch_with_metrics (v2.0 metrics kernel)");
                    Some(func)
                }
                Err(e) => {
                    log::warn!("v2.0 batch_with_metrics kernel not found: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let finalize_func = if batch_path.exists() {
            let ptx_src = std::fs::read_to_string(&batch_path)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Failed to read PTX: {}", e)))?;
            let module = context.load_module(Ptx::from_src(ptx_src))
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Failed to load PTX: {}", e)))?;
            match module.load_function("finalize_batch_metrics") {
                Ok(func) => {
                    log::info!("Loaded finalize_batch_metrics (v2.0 finalize kernel)");
                    Some(func)
                }
                Err(e) => {
                    log::warn!("v2.0 finalize kernel not found: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            context,
            stream,
            batch_func,
            batch_metrics_func,
            finalize_func,
            buffer_pool: BatchBufferPool::new(),
            telemetry: GpuTelemetry::new(),
        })
    }

    /// Check if batch kernel is available
    pub fn is_available(&self) -> bool {
        self.batch_func.is_some()
    }

    /// Get buffer pool statistics
    pub fn buffer_pool_stats(&self) -> (usize, usize) {
        self.buffer_pool.stats()
    }

    /// Run batch pocket detection on multiple structures
    ///
    /// All structures are processed in a SINGLE kernel launch.
    /// Returns per-structure outputs with GPU telemetry.
    pub fn detect_pockets_batch(
        &mut self,
        batch: &PackedBatch,
        config: &MegaFusedConfig,
    ) -> Result<BatchOutput, PrismError> {
        let batch_start = Instant::now();
        let n_structures = batch.n_structures();
        let total_atoms = batch.total_atoms;
        let total_residues = batch.total_residues;

        if n_structures == 0 {
            return Ok(BatchOutput {
                structures: Vec::new(),
                gpu_telemetry: None,
                batch_time_us: 0,
                kernel_time_us: 0,
            });
        }

        let func = self.batch_func.as_ref().ok_or_else(|| {
            PrismError::gpu("mega_fused_batch", "Batch kernel not loaded")
        })?;

        // === ALLOCATE/REUSE BUFFERS ===

        // Atoms buffer
        let atoms_size = total_atoms * 3;
        if atoms_size > self.buffer_pool.atoms_capacity || self.buffer_pool.d_atoms.is_none() {
            let new_cap = (atoms_size * 6 / 5).max(atoms_size);
            self.buffer_pool.d_atoms = Some(self.stream.alloc_zeros::<f32>(new_cap)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Failed to allocate atoms: {}", e)))?);
            self.buffer_pool.atoms_capacity = new_cap;
            self.buffer_pool.allocations += 1;
        } else {
            self.buffer_pool.reuses += 1;
        }

        // Residue buffers
        if total_residues > self.buffer_pool.residue_capacity || self.buffer_pool.d_ca_indices.is_none() {
            let new_cap = (total_residues * 6 / 5).max(total_residues);

            self.buffer_pool.d_ca_indices = Some(self.stream.alloc_zeros::<i32>(new_cap)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc ca_indices: {}", e)))?);
            self.buffer_pool.d_conservation = Some(self.stream.alloc_zeros::<f32>(new_cap)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc conservation: {}", e)))?);
            self.buffer_pool.d_bfactor = Some(self.stream.alloc_zeros::<f32>(new_cap)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc bfactor: {}", e)))?);
            self.buffer_pool.d_burial = Some(self.stream.alloc_zeros::<f32>(new_cap)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc burial: {}", e)))?);

            // Output buffers
            self.buffer_pool.d_consensus_scores = Some(self.stream.alloc_zeros::<f32>(new_cap)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc consensus: {}", e)))?);
            self.buffer_pool.d_confidence = Some(self.stream.alloc_zeros::<i32>(new_cap)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc confidence: {}", e)))?);
            self.buffer_pool.d_signal_mask = Some(self.stream.alloc_zeros::<i32>(new_cap)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc signal_mask: {}", e)))?);
            self.buffer_pool.d_pocket_assignment = Some(self.stream.alloc_zeros::<i32>(new_cap)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc pocket_assign: {}", e)))?);
            self.buffer_pool.d_centrality = Some(self.stream.alloc_zeros::<f32>(new_cap)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc centrality: {}", e)))?);

            self.buffer_pool.residue_capacity = new_cap;
            self.buffer_pool.allocations += 9;
        }

        // Descriptors buffer
        let desc_size = n_structures * std::mem::size_of::<BatchStructureDesc>();
        if desc_size > self.buffer_pool.descriptors_capacity || self.buffer_pool.d_descriptors.is_none() {
            let new_cap = (desc_size * 6 / 5).max(desc_size);
            self.buffer_pool.d_descriptors = Some(self.stream.alloc_zeros::<u8>(new_cap)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc descriptors: {}", e)))?);
            self.buffer_pool.descriptors_capacity = new_cap;
            self.buffer_pool.allocations += 1;
        }

        // Params buffer
        if self.buffer_pool.d_params.is_none() {
            let params_size = std::mem::size_of::<MegaFusedParams>();
            self.buffer_pool.d_params = Some(self.stream.alloc_zeros::<u8>(params_size)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc params: {}", e)))?);
            self.buffer_pool.allocations += 1;
        }

        // === COPY DATA TO GPU ===
        let d_atoms = self.buffer_pool.d_atoms.as_mut().unwrap();
        let d_ca_indices = self.buffer_pool.d_ca_indices.as_mut().unwrap();
        let d_conservation = self.buffer_pool.d_conservation.as_mut().unwrap();
        let d_bfactor = self.buffer_pool.d_bfactor.as_mut().unwrap();
        let d_burial = self.buffer_pool.d_burial.as_mut().unwrap();
        let d_descriptors = self.buffer_pool.d_descriptors.as_mut().unwrap();

        self.stream.memcpy_htod(&batch.atoms_packed, d_atoms)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy atoms: {}", e)))?;
        self.stream.memcpy_htod(&batch.ca_indices_packed, d_ca_indices)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy ca_indices: {}", e)))?;
        self.stream.memcpy_htod(&batch.conservation_packed, d_conservation)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy conservation: {}", e)))?;
        self.stream.memcpy_htod(&batch.bfactor_packed, d_bfactor)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy bfactor: {}", e)))?;
        self.stream.memcpy_htod(&batch.burial_packed, d_burial)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy burial: {}", e)))?;

        // Copy descriptors as bytes
        let desc_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                batch.descriptors.as_ptr() as *const u8,
                batch.descriptors.len() * std::mem::size_of::<BatchStructureDesc>(),
            )
        };
        self.stream.memcpy_htod(desc_bytes, d_descriptors)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy descriptors: {}", e)))?;

        // Copy params
        let params = MegaFusedParams::from_config(config);
        let params_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &params as *const MegaFusedParams as *const u8,
                std::mem::size_of::<MegaFusedParams>(),
            )
        };
        let d_params = self.buffer_pool.d_params.as_mut().unwrap();
        self.stream.memcpy_htod(params_bytes, d_params)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy params: {}", e)))?;

        // === LAUNCH KERNEL ===
        // One block per structure, 256 threads per block
        let block_size = 256u32;
        let launch_config = LaunchConfig {
            grid_dim: (n_structures as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0, // Shared memory statically allocated in kernel
        };

        let n_structures_i32 = n_structures as i32;

        // Capture telemetry
        let clock_before = self.telemetry.get_clock_mhz();
        let memory_clock_before = self.telemetry.get_memory_clock_mhz();
        let kernel_start = Instant::now();

        // Get mutable refs to output buffers for kernel args
        let d_consensus_scores = self.buffer_pool.d_consensus_scores.as_mut().unwrap();
        let d_confidence = self.buffer_pool.d_confidence.as_mut().unwrap();
        let d_signal_mask = self.buffer_pool.d_signal_mask.as_mut().unwrap();
        let d_pocket_assignment = self.buffer_pool.d_pocket_assignment.as_mut().unwrap();
        let d_centrality = self.buffer_pool.d_centrality.as_mut().unwrap();

        // Need to reborrow as shared for kernel args
        let d_atoms = self.buffer_pool.d_atoms.as_ref().unwrap();
        let d_ca_indices = self.buffer_pool.d_ca_indices.as_ref().unwrap();
        let d_conservation = self.buffer_pool.d_conservation.as_ref().unwrap();
        let d_bfactor = self.buffer_pool.d_bfactor.as_ref().unwrap();
        let d_burial = self.buffer_pool.d_burial.as_ref().unwrap();
        let d_descriptors = self.buffer_pool.d_descriptors.as_ref().unwrap();
        let d_params = self.buffer_pool.d_params.as_ref().unwrap();

        unsafe {
            let mut builder = self.stream.launch_builder(func);
            // Packed input arrays
            builder.arg(d_atoms);
            builder.arg(d_ca_indices);
            builder.arg(d_conservation);
            builder.arg(d_bfactor);
            builder.arg(d_burial);
            // Structure descriptors
            builder.arg(d_descriptors);
            builder.arg(&n_structures_i32);
            // Output arrays
            builder.arg(d_consensus_scores);
            builder.arg(d_confidence);
            builder.arg(d_signal_mask);
            builder.arg(d_pocket_assignment);
            builder.arg(d_centrality);
            // Params
            builder.arg(d_params);

            builder.launch(launch_config)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Kernel launch failed: {}", e)))?;
        }

        // Synchronize
        self.stream.synchronize()
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Sync failed: {}", e)))?;

        let kernel_elapsed = kernel_start.elapsed();
        let clock_after = self.telemetry.get_clock_mhz();
        let memory_clock_after = self.telemetry.get_memory_clock_mhz();

        // === COPY RESULTS BACK ===
        let d_consensus_scores = self.buffer_pool.d_consensus_scores.as_ref().unwrap();
        let d_confidence = self.buffer_pool.d_confidence.as_ref().unwrap();
        let d_signal_mask = self.buffer_pool.d_signal_mask.as_ref().unwrap();
        let d_pocket_assignment = self.buffer_pool.d_pocket_assignment.as_ref().unwrap();
        let d_centrality = self.buffer_pool.d_centrality.as_ref().unwrap();

        let all_consensus: Vec<f32> = self.stream.clone_dtoh(d_consensus_scores)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read consensus: {}", e)))?;
        let all_confidence: Vec<i32> = self.stream.clone_dtoh(d_confidence)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read confidence: {}", e)))?;
        let all_signal_mask: Vec<i32> = self.stream.clone_dtoh(d_signal_mask)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read signal_mask: {}", e)))?;
        let all_pocket_assignment: Vec<i32> = self.stream.clone_dtoh(d_pocket_assignment)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read pocket_assign: {}", e)))?;
        let all_centrality: Vec<f32> = self.stream.clone_dtoh(d_centrality)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read centrality: {}", e)))?;

        // === UNPACK RESULTS ===
        let mut structures = Vec::with_capacity(n_structures);
        for (i, desc) in batch.descriptors.iter().enumerate() {
            let start = desc.residue_offset as usize;
            let end = start + desc.n_residues as usize;

            structures.push(BatchStructureOutput {
                id: batch.ids[i].clone(),
                consensus_scores: all_consensus[start..end].to_vec(),
                confidence: all_confidence[start..end].to_vec(),
                signal_mask: all_signal_mask[start..end].to_vec(),
                pocket_assignment: all_pocket_assignment[start..end].to_vec(),
                centrality: all_centrality[start..end].to_vec(),
            });
        }

        // Build telemetry event
        let kernel_event = KernelTelemetryEvent {
            file: format!("batch_{}_structures", n_structures),
            kernel: "mega_fused_batch_detection".to_string(),
            clock_before_mhz: clock_before,
            clock_after_mhz: clock_after,
            memory_clock_before_mhz: memory_clock_before,
            memory_clock_after_mhz: memory_clock_after,
            temperature_c: self.telemetry.get_temperature(),
            memory_used_bytes: self.telemetry.get_memory_used(),
            execution_time_us: kernel_elapsed.as_micros() as u64,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let gpu_provenance = GpuProvenanceData {
            gpu_name: self.telemetry.get_gpu_name(),
            driver_version: self.telemetry.get_driver_version(),
            kernel_events: vec![kernel_event],
            total_gpu_time_us: kernel_elapsed.as_micros() as u64,
            avg_clock_mhz: clock_before.or(clock_after),
            peak_temperature_c: self.telemetry.get_temperature(),
        };

        let batch_elapsed = batch_start.elapsed();

        log::info!(
            "Batch processed {} structures in {:.2}ms (kernel: {:.2}ms)",
            n_structures,
            batch_elapsed.as_secs_f64() * 1000.0,
            kernel_elapsed.as_secs_f64() * 1000.0
        );

        Ok(BatchOutput {
            structures,
            gpu_telemetry: Some(gpu_provenance),
            batch_time_us: batch_elapsed.as_micros() as u64,
            kernel_time_us: kernel_elapsed.as_micros() as u64,
        })
    }

    //=========================================================================
    // v2.0 FINAL: BATCH WITH METRICS (ALL 5 FIXES APPLIED)
    //=========================================================================

    /// Check if metrics kernel is available
    pub fn is_metrics_available(&self) -> bool {
        self.batch_metrics_func.is_some() && self.finalize_func.is_some()
    }

    /// Run batch pocket detection WITH ground truth metrics
    /// All accuracy metrics computed on GPU - no Python scripts needed
    pub fn detect_pockets_batch_with_metrics(
        &mut self,
        batch: &PackedBatchWithGT,
        config: &MegaFusedConfig,
    ) -> Result<BatchOutputWithMetrics, PrismError> {
        let kernel_start = Instant::now();
        let n_structures = batch.offsets.len();
        let total_residues = batch.base.total_residues;
        let total_tiles = batch.total_tiles as usize;

        if n_structures == 0 {
            return Ok(BatchOutputWithMetrics {
                structures: Vec::new(),
                metrics: Vec::new(),
                kernel_time_us: 0,
                aggregate: AggregateMetrics::default(),
            });
        }

        let metrics_func = self.batch_metrics_func.as_ref().ok_or_else(|| {
            PrismError::gpu("mega_fused_batch", "v2.0 metrics kernel not loaded")
        })?;
        let finalize_func = self.finalize_func.as_ref().ok_or_else(|| {
            PrismError::gpu("mega_fused_batch", "v2.0 finalize kernel not loaded")
        })?;

        // === ALLOCATE INPUT BUFFERS ===
        let mut d_atoms = self.stream.alloc_zeros::<f32>(batch.base.atoms_packed.len().max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc atoms: {}", e)))?;
        let mut d_ca_indices = self.stream.alloc_zeros::<i32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc ca_indices: {}", e)))?;
        let mut d_conservation = self.stream.alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc conservation: {}", e)))?;
        let mut d_bfactor = self.stream.alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc bfactor: {}", e)))?;
        let mut d_burial = self.stream.alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc burial: {}", e)))?;
        let mut d_gt_mask = self.stream.alloc_zeros::<u8>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc gt_mask: {}", e)))?;

        // FIX 1: StructureOffset upload using bytemuck::cast_slice
        let mut d_offsets = self.stream.alloc_zeros::<u8>(n_structures * std::mem::size_of::<StructureOffset>())
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc offsets: {}", e)))?;

        let mut d_tile_prefix = self.stream.alloc_zeros::<i32>(batch.tile_prefix_sum.len())
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc tile_prefix: {}", e)))?;

        // === ALLOCATE OUTPUT BUFFERS ===
        let mut d_consensus_scores = self.stream.alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc consensus: {}", e)))?;
        let mut d_confidence = self.stream.alloc_zeros::<i32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc confidence: {}", e)))?;
        let mut d_signal_mask = self.stream.alloc_zeros::<i32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc signal_mask: {}", e)))?;
        let mut d_pocket_assignment = self.stream.alloc_zeros::<i32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc pocket_assign: {}", e)))?;
        let mut d_centrality = self.stream.alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc centrality: {}", e)))?;

        // === ALLOCATE METRICS BUFFERS ===
        let mut d_tp_counts = self.stream.alloc_zeros::<i32>(n_structures)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc tp: {}", e)))?;
        let mut d_fp_counts = self.stream.alloc_zeros::<i32>(n_structures)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc fp: {}", e)))?;
        let mut d_tn_counts = self.stream.alloc_zeros::<i32>(n_structures)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc tn: {}", e)))?;
        let mut d_fn_counts = self.stream.alloc_zeros::<i32>(n_structures)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc fn: {}", e)))?;
        let mut d_score_sums = self.stream.alloc_zeros::<f32>(n_structures)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc score_sums: {}", e)))?;
        let mut d_pocket_counts = self.stream.alloc_zeros::<i32>(n_structures)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc pocket_counts: {}", e)))?;

        // Histogram buffers (100 bins per structure)
        let mut d_hist_pos = self.stream.alloc_zeros::<u64>(n_structures * N_BINS)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc hist_pos: {}", e)))?;
        let mut d_hist_neg = self.stream.alloc_zeros::<u64>(n_structures * N_BINS)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc hist_neg: {}", e)))?;

        // FIX 2: Metrics buffer as alloc_zeros::<BatchMetricsOutput>(n_structures) NOT u8
        let mut d_metrics_out = self.stream.alloc_zeros::<BatchMetricsOutput>(n_structures)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc metrics_out: {}", e)))?;

        // Params buffer
        let mut d_params = self.stream.alloc_zeros::<u8>(std::mem::size_of::<MegaFusedParams>())
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc params: {}", e)))?;

        // === COPY DATA TO GPU ===
        self.stream.memcpy_htod(&batch.base.atoms_packed, &mut d_atoms)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy atoms: {}", e)))?;
        self.stream.memcpy_htod(&batch.base.ca_indices_packed, &mut d_ca_indices)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy ca_indices: {}", e)))?;
        self.stream.memcpy_htod(&batch.base.conservation_packed, &mut d_conservation)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy conservation: {}", e)))?;
        self.stream.memcpy_htod(&batch.base.bfactor_packed, &mut d_bfactor)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy bfactor: {}", e)))?;
        self.stream.memcpy_htod(&batch.base.burial_packed, &mut d_burial)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy burial: {}", e)))?;
        self.stream.memcpy_htod(&batch.gt_pocket_mask_packed, &mut d_gt_mask)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy gt_mask: {}", e)))?;
        self.stream.memcpy_htod(&batch.tile_prefix_sum, &mut d_tile_prefix)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy tile_prefix: {}", e)))?;

        // FIX 1: Use bytemuck::cast_slice for StructureOffset
        let offsets_bytes: &[u8] = bytemuck::cast_slice(&batch.offsets);
        self.stream.memcpy_htod(offsets_bytes, &mut d_offsets)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy offsets: {}", e)))?;

        // Copy params
        let params = MegaFusedParams::from_config(config);
        let params_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &params as *const MegaFusedParams as *const u8,
                std::mem::size_of::<MegaFusedParams>(),
            )
        };
        self.stream.memcpy_htod(params_bytes, &mut d_params)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy params: {}", e)))?;

        // === LAUNCH BATCH METRICS KERNEL ===
        let block_size = 256u32;
        let launch_config = LaunchConfig {
            grid_dim: (total_tiles as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_structures_i32 = n_structures as i32;
        let total_tiles_i32 = total_tiles as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(metrics_func);
            builder.arg(&d_atoms);
            builder.arg(&d_ca_indices);
            builder.arg(&d_conservation);
            builder.arg(&d_bfactor);
            builder.arg(&d_burial);
            builder.arg(&d_gt_mask);
            builder.arg(&d_offsets);
            builder.arg(&d_tile_prefix);
            builder.arg(&n_structures_i32);
            builder.arg(&total_tiles_i32);
            builder.arg(&d_consensus_scores);
            builder.arg(&d_confidence);
            builder.arg(&d_signal_mask);
            builder.arg(&d_pocket_assignment);
            builder.arg(&d_centrality);
            builder.arg(&d_tp_counts);
            builder.arg(&d_fp_counts);
            builder.arg(&d_tn_counts);
            builder.arg(&d_fn_counts);
            builder.arg(&d_score_sums);
            builder.arg(&d_pocket_counts);
            builder.arg(&d_hist_pos);
            builder.arg(&d_hist_neg);
            builder.arg(&d_params);

            builder.launch(launch_config)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Metrics kernel launch failed: {}", e)))?;
        }

        // === LAUNCH FINALIZE KERNEL ===
        let finalize_blocks = (n_structures as u32 + 255) / 256;
        let finalize_config = LaunchConfig {
            grid_dim: (finalize_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(finalize_func);
            builder.arg(&d_tp_counts);
            builder.arg(&d_fp_counts);
            builder.arg(&d_tn_counts);
            builder.arg(&d_fn_counts);
            builder.arg(&d_score_sums);
            builder.arg(&d_pocket_counts);
            builder.arg(&d_hist_pos);
            builder.arg(&d_hist_neg);
            builder.arg(&d_metrics_out);
            builder.arg(&d_offsets);
            builder.arg(&n_structures_i32);

            builder.launch(finalize_config)
                .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Finalize kernel launch failed: {}", e)))?;
        }

        // Synchronize
        self.stream.synchronize()
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Sync failed: {}", e)))?;

        let kernel_elapsed = kernel_start.elapsed();

        // === COPY RESULTS BACK ===
        let all_consensus: Vec<f32> = self.stream.clone_dtoh(&d_consensus_scores)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read consensus: {}", e)))?;
        let all_confidence: Vec<i32> = self.stream.clone_dtoh(&d_confidence)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read confidence: {}", e)))?;
        let all_signal_mask: Vec<i32> = self.stream.clone_dtoh(&d_signal_mask)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read signal_mask: {}", e)))?;
        let all_pocket_assignment: Vec<i32> = self.stream.clone_dtoh(&d_pocket_assignment)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read pocket_assign: {}", e)))?;
        let all_centrality: Vec<f32> = self.stream.clone_dtoh(&d_centrality)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read centrality: {}", e)))?;

        // Read metrics
        let metrics: Vec<BatchMetricsOutput> = self.stream.clone_dtoh(&d_metrics_out)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read metrics: {}", e)))?;

        // === UNPACK STRUCTURE OUTPUTS (FIX 5: Return real BatchStructureOutput) ===
        let mut structures = Vec::with_capacity(n_structures);
        for (i, offset) in batch.offsets.iter().enumerate() {
            let start = offset.residue_start as usize;
            let end = start + offset.residue_count as usize;

            structures.push(BatchStructureOutput {
                id: batch.base.ids[i].clone(),
                consensus_scores: all_consensus[start..end].to_vec(),
                confidence: all_confidence[start..end].to_vec(),
                signal_mask: all_signal_mask[start..end].to_vec(),
                pocket_assignment: all_pocket_assignment[start..end].to_vec(),
                centrality: all_centrality[start..end].to_vec(),
            });
        }

        // === COMPUTE AGGREGATE METRICS ===
        let mut aggregate = AggregateMetrics::default();
        if !metrics.is_empty() {
            let n = metrics.len() as f32;
            aggregate.mean_f1 = metrics.iter().map(|m| m.f1_score).sum::<f32>() / n;
            aggregate.mean_mcc = metrics.iter().map(|m| m.mcc).sum::<f32>() / n;
            aggregate.mean_auc_roc = metrics.iter().map(|m| m.auc_roc).sum::<f32>() / n;
            aggregate.mean_auprc = metrics.iter().map(|m| m.auprc).sum::<f32>() / n;
            aggregate.mean_precision = metrics.iter().map(|m| m.precision).sum::<f32>() / n;
            aggregate.mean_recall = metrics.iter().map(|m| m.recall).sum::<f32>() / n;
        }

        log::info!(
            "Batch with metrics: {} structures in {:.2}ms | F1={:.4} MCC={:.4} AUC-ROC={:.4} AUPRC={:.4}",
            n_structures,
            kernel_elapsed.as_secs_f64() * 1000.0,
            aggregate.mean_f1,
            aggregate.mean_mcc,
            aggregate.mean_auc_roc,
            aggregate.mean_auprc
        );

        Ok(BatchOutputWithMetrics {
            structures,
            metrics,
            kernel_time_us: kernel_elapsed.as_micros() as u64,
            aggregate,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_structure_desc_alignment() {
        assert_eq!(std::mem::size_of::<BatchStructureDesc>(), 16);
        assert_eq!(std::mem::align_of::<BatchStructureDesc>(), 16);
    }

    #[test]
    fn test_structure_input_validation() {
        let mut input = StructureInput::new("test");
        input.atoms = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 2 atoms
        input.ca_indices = vec![0, 1]; // 2 residues
        input.conservation = vec![0.5, 0.6];
        input.bfactor = vec![0.1, 0.2];
        input.burial = vec![0.3, 0.4];

        assert!(input.validate().is_ok());
        assert_eq!(input.n_atoms(), 2);
        assert_eq!(input.n_residues(), 2);
    }

    #[test]
    fn test_pack_batch() {
        let mut s1 = StructureInput::new("pdb1");
        s1.atoms = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        s1.ca_indices = vec![0, 1];
        s1.conservation = vec![0.5, 0.6];
        s1.bfactor = vec![0.1, 0.2];
        s1.burial = vec![0.3, 0.4];

        let mut s2 = StructureInput::new("pdb2");
        s2.atoms = vec![2.0, 2.0, 2.0];
        s2.ca_indices = vec![0];
        s2.conservation = vec![0.7];
        s2.bfactor = vec![0.3];
        s2.burial = vec![0.5];

        let batch = PackedBatch::from_structures(&[s1, s2]).unwrap();

        assert_eq!(batch.n_structures(), 2);
        assert_eq!(batch.total_atoms, 3);
        assert_eq!(batch.total_residues, 3);
        assert_eq!(batch.atoms_packed.len(), 9); // 3 atoms * 3 coords
        assert_eq!(batch.descriptors[0].atom_offset, 0);
        assert_eq!(batch.descriptors[0].n_atoms, 2);
        assert_eq!(batch.descriptors[1].atom_offset, 2);
        assert_eq!(batch.descriptors[1].n_atoms, 1);
    }
}
