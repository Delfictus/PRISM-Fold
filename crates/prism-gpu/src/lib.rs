//! # prism-gpu
//!
//! GPU acceleration layer for PRISM v2.
//!
//! Provides CUDA kernel wrappers and GPU context management.
//! Implements PRISM GPU Plan §4: GPU Integration.
//!
//! ## Resolved TODOs
//!
//! - ✅ RESOLVED(GPU-Context): GpuContext with CudaDevice initialization, PTX loading, security, and telemetry
//! - ✅ DONE(GPU-Phase0): Dendritic reservoir kernel integration
//! - ✅ DONE(GPU-Phase1): Active Inference kernel integration
//! - ✅ DONE(GPU-Phase3): Quantum evolution kernel integration
//! - ✅ DONE(GPU-Phase6): TDA persistent homology kernel integration

pub mod aatgs;
pub mod active_inference;
pub mod cma;
pub mod cma_es;
pub mod context;
pub mod dendritic_reservoir;
pub mod dendritic_whcr;
pub mod floyd_warshall;
pub mod lbs;
pub mod molecular;
pub mod multi_device_pool;
pub mod multi_gpu;
pub mod pimc;
pub mod quantum;
pub mod stream_manager;
pub mod tda;
pub mod thermodynamic;
pub mod transfer_entropy;
pub mod whcr;

// Re-export commonly used items
pub use aatgs::{AATGSBuffers, AATGSScheduler, AsyncPipeline};
pub use active_inference::{ActiveInferenceGpu, ActiveInferencePolicy};
pub use cma::{CmaEnsembleGpu, CmaEnsembleParams, CmaMetrics};
pub use cma_es::{CmaOptimizer, CmaParams, CmaState};
pub use context::{GpuContext, GpuInfo, GpuSecurityConfig};
pub use dendritic_reservoir::DendriticReservoirGpu;
pub use dendritic_whcr::DendriticReservoirGpu as DendriticWhcrGpu;
pub use floyd_warshall::FloydWarshallGpu;
pub use lbs::LbsGpu;
pub use molecular::{MDParams, MDResults, MolecularDynamicsGpu, Particle};
pub use multi_device_pool::{MultiGpuDevicePool, P2PCapability, ReplicaExchangeCoordinator};
pub use multi_gpu::{GpuMetrics, MultiGpuManager, SchedulingPolicy};
pub use pimc::{PimcGpu, PimcMetrics, PimcObservables, PimcParams};
pub use quantum::QuantumEvolutionGpu;
pub use stream_manager::{
    AsyncPipelineCoordinator, ManagedStream, PipelineStage, StreamPool, StreamPurpose,
    TripleBuffer,
};
pub use tda::TdaGpu;
pub use thermodynamic::ThermodynamicGpu;
pub use transfer_entropy::{CausalGraph, TEMatrix, TEParams, TransferEntropyGpu};
pub use whcr::{RepairResult, WhcrGpu};
