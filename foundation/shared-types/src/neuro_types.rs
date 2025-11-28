//! Neuromorphic Domain Types
//!
//! Pure data structures representing neuromorphic system states

/// Neuromorphic system state snapshot
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NeuroState {
    /// Reservoir neuron states (voltage levels)
    pub neuron_states: alloc::vec::Vec<f64>,

    /// Recent spike pattern (binary: 1 = spike, 0 = no spike)
    pub spike_pattern: alloc::vec::Vec<u8>,

    /// Temporal coherence measure (0.0 to 1.0)
    pub coherence: f64,

    /// Pattern strength detected (0.0 to 1.0)
    pub pattern_strength: f64,

    /// Timestamp of this state (nanoseconds)
    pub timestamp_ns: u64,
}

/// Single spike event
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Spike {
    /// Neuron ID that spiked
    pub neuron_id: usize,

    /// Spike time (milliseconds)
    pub time_ms: f64,

    /// Spike amplitude (mV)
    pub amplitude: f64,
}

/// Temporal spike pattern
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpikePattern {
    /// All spikes in this pattern
    pub spikes: alloc::vec::Vec<Spike>,

    /// Pattern duration (milliseconds)
    pub duration_ms: f64,

    /// Number of neurons involved
    pub num_neurons: usize,
}

/// Detected neuromorphic pattern
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectedPattern {
    /// Pattern ID
    pub id: usize,

    /// Pattern type classification
    pub pattern_type: PatternType,

    /// Confidence in detection (0.0 to 1.0)
    pub confidence: f64,

    /// Temporal frequency (Hz)
    pub frequency_hz: f64,

    /// Neurons participating in pattern
    pub neuron_ids: alloc::vec::Vec<usize>,
}

/// Pattern classification types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PatternType {
    /// Synchronous spiking across neurons
    Synchronous,

    /// Traveling wave pattern
    TravelingWave,

    /// Burst pattern (rapid succession)
    Burst,

    /// Oscillatory pattern
    Oscillatory,

    /// Chaotic/irregular
    Chaotic,

    /// Emergent complex pattern
    Emergent,
}

extern crate alloc;
