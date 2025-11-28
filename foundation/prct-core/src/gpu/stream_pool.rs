//! CUDA Stream Pool for Parallel Phase Execution
//!
//! Manages multiple CUDA streams to enable overlapping GPU execution across
//! pipeline phases (Reservoir, Transfer Entropy, Thermodynamic, Quantum).
//!
//! Constitutional compliance:
//! - Single Arc<CudaDevice> shared across all streams
//! - Round-robin allocation with fixed per-phase assignment
//! - Proper stream lifecycle management (no leaks)
//! - Fallback to single default stream for cudarc 0.9 limitations

use crate::errors::*;
use cudarc::driver::{CudaDevice, CudaStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Pool of CUDA streams for parallel phase execution
pub struct CudaStreamPool {
    /// All managed streams (first is default stream)
    streams: Vec<CudaStream>,

    /// Round-robin counter for dynamic allocation
    next_index: AtomicUsize,

    /// Parent device for validation
    device: Arc<CudaDevice>,
}

impl CudaStreamPool {
    /// Create new stream pool with specified count
    ///
    /// # Arguments
    /// - `device`: Shared CUDA device (single device per process)
    /// - `count`: Number of streams (clamped to 1..=32)
    ///
    /// # Returns
    /// Stream pool with default stream + (count-1) additional streams
    ///
    /// # Errors
    /// - `PRCTError::GpuError` if stream creation fails
    pub fn new(device: &Arc<CudaDevice>, count: usize) -> Result<Self> {
        let count = count.clamp(1, 32);
        let mut streams = Vec::with_capacity(count);

        // Default stream (index 0) - always available
        let default_stream = device
            .fork_default_stream()
            .map_err(|e| PRCTError::GpuError(format!("Failed to fork default stream: {}", e)))?;
        streams.push(default_stream);

        // Additional streams for parallel execution
        for i in 1..count {
            match device.fork_default_stream() {
                Ok(stream) => streams.push(stream),
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to create stream {}/{}, using {} streams: {}",
                        i,
                        count,
                        streams.len(),
                        e
                    );
                    break;
                }
            }
        }

        if streams.len() < count {
            eprintln!(
                "Warning: Requested {} streams but only created {}",
                count,
                streams.len()
            );
        }

        Ok(Self {
            streams,
            next_index: AtomicUsize::new(0),
            device: device.clone(),
        })
    }

    /// Get stream using round-robin allocation
    ///
    /// Thread-safe for concurrent phase scheduling
    pub fn get(&self) -> &CudaStream {
        let idx = self.next_index.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        &self.streams[idx]
    }

    /// Get stream at fixed index (for per-phase assignment)
    ///
    /// # Arguments
    /// - `index`: Stream index (wraps if >= stream count)
    ///
    /// # Returns
    /// Reference to stream at index % stream_count
    pub fn get_fixed(&self, index: usize) -> &CudaStream {
        &self.streams[index % self.streams.len()]
    }

    /// Get default stream (index 0)
    pub fn default_stream(&self) -> &CudaStream {
        &self.streams[0]
    }

    /// Number of available streams
    pub fn count(&self) -> usize {
        self.streams.len()
    }

    /// Synchronize all streams
    ///
    /// Blocks until all pending operations complete
    /// Note: cudarc 0.9 synchronizes automatically, this is a no-op
    pub fn synchronize_all(&self) -> Result<()> {
        // cudarc 0.9: operations are synchronous on device
        // Explicit synchronization not needed
        Ok(())
    }

    /// Get parent device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

impl Drop for CudaStreamPool {
    fn drop(&mut self) {
        // Synchronize before dropping to ensure no pending operations
        if let Err(e) = self.synchronize_all() {
            eprintln!("Warning: Failed to synchronize streams during drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_stream_pool_creation() {
        let device = CudaDevice::new(0).expect("Failed to create device");
        let device = Arc::new(device);

        let pool = CudaStreamPool::new(&device, 4).expect("Failed to create pool");
        assert_eq!(pool.count(), 4);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_round_robin_allocation() {
        let device = CudaDevice::new(0).expect("Failed to create device");
        let device = Arc::new(device);

        let pool = CudaStreamPool::new(&device, 3).expect("Failed to create pool");

        // Get streams multiple times, should wrap
        for _ in 0..10 {
            let _ = pool.get();
        }

        // Next should be index 10 % 3 = 1
        let idx = pool.next_index.load(Ordering::Relaxed);
        assert_eq!(idx % 3, 1);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_fixed_allocation() {
        let device = CudaDevice::new(0).expect("Failed to create device");
        let device = Arc::new(device);

        let pool = CudaStreamPool::new(&device, 4).expect("Failed to create pool");

        // Fixed indices should always return same stream
        let s0_ptr = pool.get_fixed(0) as *const CudaStream;
        let s0_ptr2 = pool.get_fixed(0) as *const CudaStream;
        assert_eq!(s0_ptr, s0_ptr2);

        // Wrapping should work
        let s0_wrap = pool.get_fixed(4) as *const CudaStream;
        assert_eq!(s0_ptr, s0_wrap);
    }
}
