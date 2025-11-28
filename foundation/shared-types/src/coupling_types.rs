//! Neuromorphic-Quantum Coupling Types
//!
//! Data structures representing bidirectional coupling between systems

/// Coupling strength between neuromorphic and quantum systems
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CouplingStrength {
    /// Neuromorphic → Quantum influence (0.0 to 1.0)
    pub neuro_to_quantum: f64,

    /// Quantum → Neuromorphic influence (0.0 to 1.0)
    pub quantum_to_neuro: f64,

    /// Bidirectional coherence (0.0 to 1.0)
    pub bidirectional_coherence: f64,

    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
}

/// Kuramoto phase synchronization state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KuramotoState {
    /// Phase angles for each oscillator (radians)
    pub phases: alloc::vec::Vec<f64>,

    /// Natural frequencies for each oscillator (Hz)
    pub natural_frequencies: alloc::vec::Vec<f64>,

    /// Coupling strength between oscillators
    pub coupling_matrix: alloc::vec::Vec<f64>, // Flattened n×n

    /// Global order parameter (synchronization measure, 0.0 to 1.0)
    pub order_parameter: f64,

    /// Mean phase angle (radians)
    pub mean_phase: f64,
}

/// Transfer entropy measurement
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TransferEntropy {
    /// Information transfer from source to target (bits)
    pub entropy_bits: f64,

    /// Confidence in measurement (0.0 to 1.0)
    pub confidence: f64,

    /// Time lag used in calculation (milliseconds)
    pub lag_ms: f64,
}

/// Bidirectional coupling state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BidirectionalCoupling {
    /// Neuromorphic → Quantum transfer entropy
    pub neuro_to_quantum_entropy: TransferEntropy,

    /// Quantum → Neuromorphic transfer entropy
    pub quantum_to_neuro_entropy: TransferEntropy,

    /// Kuramoto synchronization state
    pub kuramoto_state: KuramotoState,

    /// Overall coupling quality (0.0 to 1.0)
    pub coupling_quality: f64,
}

extern crate alloc;
