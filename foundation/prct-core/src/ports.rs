//! Port Interfaces (Hexagonal Architecture)
//!
//! These traits define the boundaries between domain and infrastructure.
//! Domain logic depends on these abstractions, NOT on concrete implementations.
//!
//! Adapters (infrastructure) implement these ports.

use crate::errors::Result;
use shared_types::*;

/// Neuromorphic Processing Port
///
/// Abstracts spike encoding, reservoir computing, and pattern detection.
pub trait NeuromorphicPort: Send + Sync {
    /// Encode graph structure as temporal spike patterns
    ///
    /// # Arguments
    /// * `graph` - Graph to encode
    /// * `encoding_params` - Encoding parameters (frequency, duration, etc.)
    ///
    /// # Returns
    /// Spike pattern representing graph structure
    fn encode_graph_as_spikes(
        &self,
        graph: &Graph,
        encoding_params: &NeuromorphicEncodingParams,
    ) -> Result<SpikePattern>;

    /// Process spikes through reservoir and detect patterns
    ///
    /// # Arguments
    /// * `spikes` - Input spike pattern
    ///
    /// # Returns
    /// Neuromorphic state with detected patterns
    fn process_and_detect_patterns(&self, spikes: &SpikePattern) -> Result<NeuroState>;

    /// Get detected patterns from neuromorphic processing
    ///
    /// # Returns
    /// List of detected patterns with confidence scores
    fn get_detected_patterns(&self) -> Result<Vec<DetectedPattern>>;
}

/// Quantum Processing Port
///
/// Abstracts Hamiltonian evolution, phase resonance, and quantum optimization.
pub trait QuantumPort: Send + Sync {
    /// Build Hamiltonian from graph coupling matrix
    ///
    /// # Arguments
    /// * `graph` - Graph with coupling matrix
    /// * `params` - Evolution parameters
    ///
    /// # Returns
    /// Hamiltonian state ready for evolution
    fn build_hamiltonian(
        &self,
        graph: &Graph,
        params: &EvolutionParams,
    ) -> Result<HamiltonianState>;

    /// Evolve quantum state using Hamiltonian
    ///
    /// # Arguments
    /// * `hamiltonian` - Hamiltonian operator
    /// * `initial_state` - Starting quantum state
    /// * `evolution_time` - Time to evolve (seconds)
    ///
    /// # Returns
    /// Evolved quantum state
    fn evolve_state(
        &self,
        hamiltonian: &HamiltonianState,
        initial_state: &QuantumState,
        evolution_time: f64,
    ) -> Result<QuantumState>;

    /// Get phase resonance field from quantum state
    ///
    /// # Arguments
    /// * `state` - Quantum state
    ///
    /// # Returns
    /// Phase field with coherence information
    fn get_phase_field(&self, state: &QuantumState) -> Result<PhaseField>;

    /// Compute ground state for optimization
    ///
    /// # Arguments
    /// * `hamiltonian` - Hamiltonian operator
    ///
    /// # Returns
    /// Ground state quantum state
    fn compute_ground_state(&self, hamiltonian: &HamiltonianState) -> Result<QuantumState>;
}

/// Physics Coupling Port
///
/// Abstracts Kuramoto synchronization and neuromorphic-quantum coupling.
pub trait PhysicsCouplingPort: Send + Sync {
    /// Compute coupling strength between neuromorphic and quantum systems
    ///
    /// # Arguments
    /// * `neuro_state` - Neuromorphic system state
    /// * `quantum_state` - Quantum system state
    ///
    /// # Returns
    /// Bidirectional coupling strength and quality
    fn compute_coupling(
        &self,
        neuro_state: &NeuroState,
        quantum_state: &QuantumState,
    ) -> Result<CouplingStrength>;

    /// Update Kuramoto phase synchronization
    ///
    /// # Arguments
    /// * `neuro_phases` - Neuromorphic oscillator phases
    /// * `quantum_phases` - Quantum phase field
    /// * `dt` - Time step (seconds)
    ///
    /// # Returns
    /// Updated Kuramoto synchronization state
    fn update_kuramoto_sync(
        &self,
        neuro_phases: &[f64],
        quantum_phases: &[f64],
        dt: f64,
    ) -> Result<KuramotoState>;

    /// Calculate transfer entropy (information flow)
    ///
    /// # Arguments
    /// * `source` - Source time series
    /// * `target` - Target time series
    /// * `lag` - Time lag (milliseconds)
    ///
    /// # Returns
    /// Transfer entropy measurement
    fn calculate_transfer_entropy(
        &self,
        source: &[f64],
        target: &[f64],
        lag: f64,
    ) -> Result<TransferEntropy>;

    /// Get full bidirectional coupling state
    ///
    /// # Arguments
    /// * `neuro_state` - Neuromorphic state
    /// * `quantum_state` - Quantum state
    ///
    /// # Returns
    /// Complete bidirectional coupling information
    fn get_bidirectional_coupling(
        &self,
        neuro_state: &NeuroState,
        quantum_state: &QuantumState,
    ) -> Result<BidirectionalCoupling>;
}

/// Neuromorphic encoding parameters
#[derive(Debug, Clone)]
pub struct NeuromorphicEncodingParams {
    /// Base spike frequency (Hz)
    pub base_frequency: f64,

    /// Encoding time window (milliseconds)
    pub time_window: f64,

    /// Number of reservoir neurons
    pub num_neurons: usize,

    /// Enable burst encoding for high clustering
    pub enable_burst_encoding: bool,
}

impl Default for NeuromorphicEncodingParams {
    fn default() -> Self {
        Self {
            base_frequency: 20.0,
            time_window: 100.0,
            num_neurons: 1000,
            enable_burst_encoding: true,
        }
    }
}
