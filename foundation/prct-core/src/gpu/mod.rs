//! GPU Infrastructure for PRISM Pipeline
//!
//! Provides stream management, event synchronization, centralized
//! GPU state, and multi-GPU device pooling for the world-record pipeline.

pub mod event;
pub mod multi_device_pool;
pub mod state;
pub mod stream_pool;

pub use event::{event_names, EventRegistry};
pub use multi_device_pool::MultiGpuDevicePool;
pub use state::{PipelineGpuState, StreamMode};
pub use stream_pool::CudaStreamPool;
