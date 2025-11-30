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
pub mod aatgs_integration;
pub mod active_inference;
pub mod cma;
pub mod cma_es;
pub mod context;
pub mod cryptic_gpu;
pub mod dendritic_reservoir;
pub mod dendritic_whcr;
pub mod floyd_warshall;
pub mod lbs;
pub mod molecular;
pub mod multi_device_pool;
pub mod multi_gpu;
pub mod multi_gpu_integration;
pub mod pimc;
pub mod quantum;
pub mod stream_integration;
pub mod stream_manager;
pub mod tda;
pub mod thermodynamic;
pub mod transfer_entropy;
pub mod ultra_kernel;
pub mod whcr;

// Re-export commonly used items
pub use aatgs::{AATGSBuffers, AATGSScheduler, AsyncPipeline};
pub use aatgs_integration::{ExecutionStats, GpuExecutionContext, GpuExecutionContextBuilder};
pub use active_inference::{ActiveInferenceGpu, ActiveInferencePolicy};
pub use cma::{CmaEnsembleGpu, CmaEnsembleParams, CmaMetrics};
pub use cma_es::{CmaOptimizer, CmaParams, CmaState};
pub use context::{GpuContext, GpuInfo, GpuSecurityConfig};
pub use dendritic_reservoir::DendriticReservoirGpu;
pub use dendritic_whcr::DendriticReservoirGpu as DendriticWhcrGpu;
pub use floyd_warshall::FloydWarshallGpu;
pub use lbs::LbsGpu;
pub use molecular::{MDParams, MDResults, MolecularDynamicsGpu, Particle};
pub use multi_device_pool::{
    CrossGpuReplicaManager, DeviceCapability, ExchangePair, ExchangeResult, GpuLoadBalancer,
    MigrationPlan, MultiGpuDevicePool, P2PCapability, P2PMemoryManager, ReduceOp,
    ReplicaExchangeCoordinator, ReplicaHandle, UnifiedBuffer,
};
pub use multi_gpu::{GpuMetrics, MultiGpuManager, SchedulingPolicy};
pub use multi_gpu_integration::MultiGpuContext;
pub use pimc::{PimcGpu, PimcMetrics, PimcObservables, PimcParams};
pub use quantum::QuantumEvolutionGpu;
pub use stream_integration::{
    AsyncCoordinator, CompletedOp, ManagedGpuContext, PipelineStage as GpuPipelineStage,
    PipelineStageManager, PipelineStats, TripleBuffer as GpuTripleBuffer,
};
pub use stream_manager::{
    AsyncPipelineCoordinator, ManagedStream, PipelineStage as CpuPipelineStage, StreamPool,
    StreamPurpose, TripleBuffer as CpuTripleBuffer,
};
pub use tda::TdaGpu;
pub use thermodynamic::ThermodynamicGpu;
pub use transfer_entropy::{CausalGraph, TEMatrix, TEParams, TransferEntropyGpu};
pub use ultra_kernel::UltraKernelGpu;
pub use whcr::{RepairResult, WhcrGpu};
pub use cryptic_gpu::{CrypticGpu, CrypticGpuConfig, CrypticGpuResult, CrypticCluster};
