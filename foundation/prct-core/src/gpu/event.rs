//! CUDA Event System for Cross-Phase Dependencies
//!
//! Enables proper synchronization between phases executing on different streams:
//! - Phase 1 (TE) completes → record event
//! - Phase 2 (Thermo) waits for event before using EFE values
//! - Phase 3 (Quantum) waits for thermo results
//!
//! Note: cudarc 0.9 doesn't support explicit events, so this is a no-op
//! implementation for API compatibility. Synchronization happens automatically.

use crate::errors::*;
use cudarc::driver::{CudaDevice, CudaStream};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// CUDA event wrapper for cross-stream synchronization
/// Note: cudarc 0.9 doesn't support events, this is a stub
pub struct CudaEvent {
    /// Debug name for telemetry
    name: String,
}

impl CudaEvent {
    /// Create new event (no-op in cudarc 0.9)
    fn new(_device: &Arc<CudaDevice>, name: String) -> Result<Self> {
        Ok(Self { name })
    }

    /// Record event on stream (no-op in cudarc 0.9)
    fn record(&self, _stream: &CudaStream) -> Result<()> {
        // cudarc 0.9: no explicit events, automatic sync
        Ok(())
    }

    /// Wait for event on stream (no-op in cudarc 0.9)
    fn wait(&self, _stream: &CudaStream) -> Result<()> {
        // cudarc 0.9: no explicit events, automatic sync
        Ok(())
    }
}

/// Registry of named events for pipeline synchronization
pub struct EventRegistry {
    /// Named events for cross-phase dependencies
    events: Arc<Mutex<HashMap<String, CudaEvent>>>,

    /// Parent device
    context: Arc<CudaContext>,
}

impl EventRegistry {
    /// Create new event registry
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            events: Arc::new(Mutex::new(HashMap::new())),
            device,
        }
    }

    /// Record event on stream
    ///
    /// Creates event if it doesn't exist
    ///
    /// # Arguments
    /// - `name`: Unique event identifier (e.g., "ai_complete", "te_ready")
    /// - `stream`: Stream to record on
    pub fn record(&self, name: &str, stream: &CudaStream) -> Result<()> {
        let mut events = self
            .events
            .lock()
            .map_err(|e| PRCTError::GpuError(format!("Failed to lock event registry: {}", e)))?;

        let event = events.entry(name.to_string()).or_insert_with(|| {
            CudaEvent::new(&self.context, name.to_string()).expect("Failed to create event")
        });

        event.record(stream)
    }

    /// Wait for event on stream
    ///
    /// # Errors
    /// - `PRCTError::GpuError` if event doesn't exist
    pub fn wait(&self, name: &str, stream: &CudaStream) -> Result<()> {
        let events = self
            .events
            .lock()
            .map_err(|e| PRCTError::GpuError(format!("Failed to lock event registry: {}", e)))?;

        let event = events
            .get(name)
            .ok_or_else(|| PRCTError::GpuError(format!("Event '{}' not found", name)))?;

        event.wait(stream)
    }

    /// Check if event exists
    pub fn has_event(&self, name: &str) -> bool {
        self.events
            .lock()
            .map(|events| events.contains_key(name))
            .unwrap_or(false)
    }

    /// Clear all events (for new pipeline run)
    pub fn clear(&self) {
        if let Ok(mut events) = self.events.lock() {
            events.clear();
        }
    }
}

/// Common event names for pipeline phases
pub mod event_names {
    pub const RESERVOIR_COMPLETE: &str = "reservoir_complete";
    pub const TE_COMPLETE: &str = "te_complete";
    pub const AI_COMPLETE: &str = "ai_complete";
    pub const THERMO_COMPLETE: &str = "thermo_complete";
    pub const QUANTUM_COMPLETE: &str = "quantum_complete";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_event_registry() {
        let device = CudaDevice::new(0).expect("Failed to create device");
        let device = Arc::new(device);

        let registry = EventRegistry::new(device.clone());
        let stream = device
            .fork_default_stream()
            .expect("Failed to create stream");

        // Record and wait should work
        registry
            .record("test_event", &stream)
            .expect("Failed to record");
        assert!(registry.has_event("test_event"));

        registry
            .wait("test_event", &stream)
            .expect("Failed to wait");
    }
}
