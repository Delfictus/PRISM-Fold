//! Quantum Domain Types
//!
//! Pure data structures representing quantum system states

/// Quantum system state snapshot
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuantumState {
    /// Quantum state amplitudes (complex numbers represented as (real, imag) pairs)
    pub amplitudes: alloc::vec::Vec<(f64, f64)>,

    /// Phase coherence measure (0.0 to 1.0)
    pub phase_coherence: f64,

    /// System energy (arbitrary units)
    pub energy: f64,

    /// Entanglement measure (von Neumann entropy)
    pub entanglement: f64,

    /// Timestamp of this state (nanoseconds)
    pub timestamp_ns: u64,
}

/// Phase resonance field state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PhaseField {
    /// Phase angles (radians) for each quantum element
    pub phases: alloc::vec::Vec<f64>,

    /// Phase coherence between elements
    pub coherence_matrix: alloc::vec::Vec<f64>, // Flattened n×n matrix

    /// Global phase order parameter (0.0 to 1.0)
    pub order_parameter: f64,

    /// Resonance frequency (Hz)
    pub resonance_frequency: f64,
}

/// Hamiltonian operator state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HamiltonianState {
    /// Hamiltonian matrix elements (complex, flattened)
    pub matrix_elements: alloc::vec::Vec<(f64, f64)>,

    /// Eigenvalues (energy levels)
    pub eigenvalues: alloc::vec::Vec<f64>,

    /// Ground state energy
    pub ground_state_energy: f64,

    /// Hilbert space dimension
    pub dimension: usize,
}

/// Quantum evolution parameters
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EvolutionParams {
    /// Evolution time step (seconds)
    pub dt: f64,

    /// Hamiltonian strength multiplier
    pub strength: f64,

    /// Damping/decoherence rate
    pub damping: f64,

    /// Temperature (Kelvin)
    pub temperature: f64,
}

/// Force modulation parameters for thermodynamic evolution
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForceModulationParams {
    /// Temperature at which conflict forces start activating (default: 5.0)
    pub force_start_temp: f64,

    /// Temperature at which conflict forces reach full strength (default: 1.0)
    pub force_full_strength_temp: f64,
}

extern crate alloc;
