//! Pipeline GPU State Management
//!
//! Centralized GPU resources for the world-record pipeline:
//! - Single Arc<CudaDevice> (constitutional requirement)
//! - Stream pool for parallel phase execution
//! - Event registry for cross-phase dependencies
//! - Memory pool for efficient buffer reuse
//!
//! This module ensures proper GPU resource lifecycle and prevents
//! silent CPU fallbacks when GPU is required.

use crate::errors::*;
use crate::gpu::event::EventRegistry;
use crate::gpu::stream_pool::CudaStreamPool;
use cudarc::driver::CudaContext;
use std::sync::Arc;

/// Centralized GPU state for pipeline execution
pub struct PipelineGpuState {
    /// Single shared device (constitutional requirement)
    context: Arc<CudaContext>,

    /// Stream pool for parallel phase execution
    stream_pool: Arc<CudaStreamPool>,

    /// Event registry for cross-phase dependencies
    event_registry: Arc<EventRegistry>,

    /// Stream execution mode
    mode: StreamMode,
}

/// Stream execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// All phases use default stream (sequential)
    Sequential,

    /// Phases use separate streams (parallel)
    Parallel,
}

impl PipelineGpuState {
    /// Create new pipeline GPU state
    ///
    /// # Arguments
    /// - `device_id`: CUDA device index (typically 0)
    /// - `num_streams`: Number of streams (clamped to 1..=32)
    /// - `mode`: Stream execution mode
    ///
    /// # Returns
    /// Initialized GPU state with streams and events
    ///
    /// # Errors
    /// - `PRCTError::GpuError` if device initialization fails
    pub fn new(device_id: usize, num_streams: usize, mode: StreamMode) -> Result<Self> {
        // CudaContext::new() returns Arc<CudaContext>, don't double-wrap
        let context = CudaContext::new(device_id).map_err(|e| {
            PRCTError::GpuError(format!(
                "Failed to initialize CUDA device {}: {}",
                device_id, e
            ))
        })?;

        let stream_pool = CudaStreamPool::new(&context, num_streams)?;
        let stream_pool = Arc::new(stream_pool);

        let event_registry = EventRegistry::new(context.clone());
        let event_registry = Arc::new(event_registry);

        Ok(Self {
            context,
            stream_pool,
            event_registry,
            mode,
        })
    }

    /// Get shared device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.context
    }

    /// Get stream pool
    pub fn stream_pool(&self) -> &Arc<CudaStreamPool> {
        &self.stream_pool
    }

    /// Get event registry
    pub fn event_registry(&self) -> &Arc<EventRegistry> {
        &self.event_registry
    }

    /// Get stream mode
    pub fn mode(&self) -> StreamMode {
        self.mode
    }

    /// Get stream for specific phase
    ///
    /// In Sequential mode, always returns default stream.
    /// In Parallel mode, returns fixed stream per phase:
    /// - Phase 0 (Reservoir): stream 0
    /// - Phase 1 (TE + AI): stream 1
    /// - Phase 2 (Thermo): stream 2
    /// - Phase 3 (Quantum): stream 3
    pub fn stream_for_phase(&self, phase_index: usize) -> &cudarc::driver::CudaStream {
        match self.mode {
            StreamMode::Sequential => self.stream_pool.default_stream(),
            StreamMode::Parallel => self.stream_pool.get_fixed(phase_index),
        }
    }

    /// Synchronize all streams
    pub fn synchronize_all(&self) -> Result<()> {
        self.stream_pool.synchronize_all()
    }

    /// Clear event registry (for new run)
    pub fn reset_events(&self) {
        self.event_registry.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_pipeline_gpu_state_creation() {
        let state =
            PipelineGpuState::new(0, 4, StreamMode::Parallel).expect("Failed to create GPU state");

        assert_eq!(state.stream_pool().count(), 4);
        assert_eq!(state.mode(), StreamMode::Parallel);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_sequential_mode() {
        let state = PipelineGpuState::new(0, 4, StreamMode::Sequential)
            .expect("Failed to create GPU state");

        // All phases should get same stream
        let s0 = state.stream_for_phase(0) as *const _;
        let s1 = state.stream_for_phase(1) as *const _;
        let s2 = state.stream_for_phase(2) as *const _;
        let s3 = state.stream_for_phase(3) as *const _;

        assert_eq!(s0, s1);
        assert_eq!(s1, s2);
        assert_eq!(s2, s3);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_parallel_mode() {
        let state =
            PipelineGpuState::new(0, 4, StreamMode::Parallel).expect("Failed to create GPU state");

        // Phases should get different streams
        let s0 = state.stream_for_phase(0) as *const _;
        let s1 = state.stream_for_phase(1) as *const _;
        let s2 = state.stream_for_phase(2) as *const _;
        let s3 = state.stream_for_phase(3) as *const _;

        // All should be different (if we have 4 streams)
        assert_ne!(s0, s1);
        assert_ne!(s1, s2);
        assert_ne!(s2, s3);
    }
}
