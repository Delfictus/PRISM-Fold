//! PRCT Algorithm - Phase Resonance Chromatic-TSP
//!
//! Pure domain logic implementing the complete 3-layer algorithm:
//! 1. Neuromorphic spike encoding and pattern detection
//! 2. Quantum phase resonance and Hamiltonian evolution
//! 3. Kuramoto-synchronized chromatic coloring + TSP

use crate::coloring::phase_guided_coloring;
use crate::errors::*;
use crate::ports::*;
use crate::tsp::phase_guided_tsp;
use shared_types::*;
use std::sync::Arc;

/// PRCT Algorithm Configuration
#[derive(Debug, Clone)]
pub struct PRCTConfig {
    /// Target number of colors for graph coloring
    pub target_colors: usize,

    /// Quantum evolution time (seconds)
    pub quantum_evolution_time: f64,

    /// Kuramoto coupling strength
    pub kuramoto_coupling: f64,

    /// Neuromorphic encoding parameters
    pub neuro_encoding: NeuromorphicEncodingParams,

    /// Quantum evolution parameters
    pub quantum_params: EvolutionParams,
}

impl Default for PRCTConfig {
    fn default() -> Self {
        Self {
            target_colors: 10,
            quantum_evolution_time: 0.1,
            kuramoto_coupling: 0.5,
            neuro_encoding: NeuromorphicEncodingParams::default(),
            quantum_params: EvolutionParams {
                dt: 0.01,
                strength: 1.0,
                damping: 0.1,
                temperature: 300.0,
            },
        }
    }
}

/// PRCT Algorithm - The complete integrated system
///
/// Uses dependency injection (ports) to remain infrastructure-agnostic.
pub struct PRCTAlgorithm {
    /// Neuromorphic processing port
    neuro_port: Arc<dyn NeuromorphicPort>,

    /// Quantum processing port
    quantum_port: Arc<dyn QuantumPort>,

    /// Physics coupling port
    coupling_port: Arc<dyn PhysicsCouplingPort>,

    /// Algorithm configuration
    config: PRCTConfig,
}

impl PRCTAlgorithm {
    /// Create new PRCT algorithm with injected dependencies
    ///
    /// # Arguments
    /// * `neuro_port` - Neuromorphic processing implementation
    /// * `quantum_port` - Quantum processing implementation
    /// * `coupling_port` - Physics coupling implementation
    /// * `config` - Algorithm configuration
    pub fn new(
        neuro_port: Arc<dyn NeuromorphicPort>,
        quantum_port: Arc<dyn QuantumPort>,
        coupling_port: Arc<dyn PhysicsCouplingPort>,
        config: PRCTConfig,
    ) -> Self {
        Self {
            neuro_port,
            quantum_port,
            coupling_port,
            config,
        }
    }

    /// Solve graph coloring problem using full PRCT pipeline
    ///
    /// # Arguments
    /// * `graph` - Graph to color
    ///
    /// # Returns
    /// Complete PRCT solution with coloring and TSP tours
    pub fn solve(&self, graph: &Graph) -> Result<PRCTSolution> {
        let start_time = std::time::Instant::now();

        // LAYER 1: NEUROMORPHIC PROCESSING
        // Encode graph as spikes, process through reservoir, detect patterns
        let spikes = self
            .neuro_port
            .encode_graph_as_spikes(graph, &self.config.neuro_encoding)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("Spike encoding failed: {}", e)))?;

        let neuro_state = self
            .neuro_port
            .process_and_detect_patterns(&spikes)
            .map_err(|e| {
                PRCTError::NeuromorphicFailed(format!("Pattern detection failed: {}", e))
            })?;

        // LAYER 2: QUANTUM PROCESSING
        // Build Hamiltonian, evolve state, extract phase field
        let hamiltonian = self
            .quantum_port
            .build_hamiltonian(graph, &self.config.quantum_params)
            .map_err(|e| {
                PRCTError::QuantumFailed(format!("Hamiltonian construction failed: {}", e))
            })?;

        // Create initial quantum state with correct dimension
        let dim = hamiltonian.dimension;
        let initial_state = QuantumState {
            amplitudes: vec![(1.0 / (dim as f64).sqrt(), 0.0); dim],
            phase_coherence: 0.0,
            energy: 0.0,
            entanglement: 0.0,
            timestamp_ns: 0,
        };

        let quantum_state = self
            .quantum_port
            .evolve_state(
                &hamiltonian,
                &initial_state,
                self.config.quantum_evolution_time,
            )
            .map_err(|e| PRCTError::QuantumFailed(format!("Quantum evolution failed: {}", e)))?;

        let phase_field = self
            .quantum_port
            .get_phase_field(&quantum_state)
            .map_err(|e| {
                PRCTError::QuantumFailed(format!("Phase field extraction failed: {}", e))
            })?;

        // LAYER 2.5: PHYSICS COUPLING (Kuramoto Synchronization)
        // Synchronize neuromorphic and quantum phases
        let coupling = self
            .coupling_port
            .get_bidirectional_coupling(&neuro_state, &quantum_state)
            .map_err(|e| {
                PRCTError::CouplingFailed(format!("Coupling computation failed: {}", e))
            })?;

        // LAYER 3: OPTIMIZATION (Coloring + TSP)
        // Use synchronized phases to guide graph coloring
        let coloring = phase_guided_coloring(
            graph,
            &phase_field,
            &coupling.kuramoto_state,
            self.config.target_colors,
        )?;

        // Build TSP tours within each color class
        let color_class_tours = phase_guided_tsp(graph, &coloring, &phase_field)?;

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let overall_quality = self.compute_solution_quality(&coloring, &color_class_tours);

        Ok(PRCTSolution {
            coloring,
            color_class_tours,
            phase_coherence: phase_field.order_parameter,
            kuramoto_order: coupling.kuramoto_state.order_parameter,
            overall_quality,
            total_time_ms: total_time,
        })
    }

    /// Compute overall solution quality metric
    fn compute_solution_quality(&self, coloring: &ColoringSolution, tours: &[TSPSolution]) -> f64 {
        // Quality = (coloring quality + average TSP quality) / 2
        let tsp_quality: f64 =
            tours.iter().map(|t| t.quality_score).sum::<f64>() / tours.len().max(1) as f64;

        (coloring.quality_score + tsp_quality) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementations for testing (dependency injection)
    struct MockNeuromorphicPort;
    impl NeuromorphicPort for MockNeuromorphicPort {
        fn encode_graph_as_spikes(
            &self,
            _graph: &Graph,
            _params: &NeuromorphicEncodingParams,
        ) -> Result<SpikePattern> {
            Ok(SpikePattern {
                spikes: vec![],
                duration_ms: 100.0,
                num_neurons: 100,
            })
        }

        fn process_and_detect_patterns(&self, _spikes: &SpikePattern) -> Result<NeuroState> {
            Ok(NeuroState {
                neuron_states: vec![0.5; 100],
                spike_pattern: vec![0; 100],
                coherence: 0.8,
                pattern_strength: 0.7,
                timestamp_ns: 0,
            })
        }

        fn get_detected_patterns(&self) -> Result<Vec<DetectedPattern>> {
            Ok(vec![])
        }
    }

    struct MockQuantumPort;
    impl QuantumPort for MockQuantumPort {
        fn build_hamiltonian(
            &self,
            graph: &Graph,
            _params: &EvolutionParams,
        ) -> Result<HamiltonianState> {
            Ok(HamiltonianState {
                matrix_elements: vec![(0.0, 0.0); graph.num_vertices * graph.num_vertices],
                eigenvalues: vec![0.0; graph.num_vertices],
                ground_state_energy: -1.0,
                dimension: graph.num_vertices,
            })
        }

        fn evolve_state(
            &self,
            _h: &HamiltonianState,
            _init: &QuantumState,
            _t: f64,
        ) -> Result<QuantumState> {
            Ok(QuantumState {
                amplitudes: vec![(0.7, 0.0); 10],
                phase_coherence: 0.9,
                energy: -0.5,
                entanglement: 0.3,
                timestamp_ns: 0,
            })
        }

        fn get_phase_field(&self, _state: &QuantumState) -> Result<PhaseField> {
            Ok(PhaseField {
                phases: vec![0.0; 10],
                coherence_matrix: vec![0.8; 100],
                order_parameter: 0.9,
                resonance_frequency: 50.0,
            })
        }

        fn compute_ground_state(&self, _h: &HamiltonianState) -> Result<QuantumState> {
            Ok(QuantumState {
                amplitudes: vec![(1.0, 0.0); 10],
                phase_coherence: 1.0,
                energy: -1.0,
                entanglement: 0.0,
                timestamp_ns: 0,
            })
        }
    }

    struct MockCouplingPort;
    impl PhysicsCouplingPort for MockCouplingPort {
        fn compute_coupling(&self, _n: &NeuroState, _q: &QuantumState) -> Result<CouplingStrength> {
            Ok(CouplingStrength {
                neuro_to_quantum: 0.7,
                quantum_to_neuro: 0.6,
                bidirectional_coherence: 0.8,
                timestamp_ns: 0,
            })
        }

        fn update_kuramoto_sync(
            &self,
            _np: &[f64],
            _qp: &[f64],
            _dt: f64,
        ) -> Result<KuramotoState> {
            Ok(KuramotoState {
                phases: vec![0.0; 10],
                natural_frequencies: vec![1.0; 10],
                coupling_matrix: vec![0.5; 100],
                order_parameter: 0.95,
                mean_phase: 0.0,
            })
        }

        fn calculate_transfer_entropy(
            &self,
            _s: &[f64],
            _t: &[f64],
            _lag: f64,
        ) -> Result<TransferEntropy> {
            Ok(TransferEntropy {
                entropy_bits: 0.5,
                confidence: 0.9,
                lag_ms: 10.0,
            })
        }

        fn get_bidirectional_coupling(
            &self,
            _n: &NeuroState,
            _q: &QuantumState,
        ) -> Result<BidirectionalCoupling> {
            Ok(BidirectionalCoupling {
                neuro_to_quantum_entropy: TransferEntropy {
                    entropy_bits: 0.5,
                    confidence: 0.9,
                    lag_ms: 10.0,
                },
                quantum_to_neuro_entropy: TransferEntropy {
                    entropy_bits: 0.4,
                    confidence: 0.9,
                    lag_ms: 10.0,
                },
                kuramoto_state: KuramotoState {
                    phases: vec![0.0; 10],
                    natural_frequencies: vec![1.0; 10],
                    coupling_matrix: vec![0.5; 100],
                    order_parameter: 0.95,
                    mean_phase: 0.0,
                },
                coupling_quality: 0.85,
            })
        }
    }

    #[test]
    fn test_prct_algorithm_construction() {
        let neuro = Arc::new(MockNeuromorphicPort) as Arc<dyn NeuromorphicPort>;
        let quantum = Arc::new(MockQuantumPort) as Arc<dyn QuantumPort>;
        let coupling = Arc::new(MockCouplingPort) as Arc<dyn PhysicsCouplingPort>;

        let _algorithm = PRCTAlgorithm::new(neuro, quantum, coupling, PRCTConfig::default());

        // If this compiles and runs, dependency injection works!
    }
}
