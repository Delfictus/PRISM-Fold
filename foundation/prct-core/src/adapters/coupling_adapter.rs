//! Coupling Adapter - Kuramoto Synchronization & Transfer Entropy
//!
//! Connects PRCT domain logic to physics coupling computations.
//! Implements Kuramoto dynamics and transfer entropy measurements.

use crate::coupling::PhysicsCouplingService;
use crate::errors::{PRCTError, Result};
use crate::ports::PhysicsCouplingPort;
use shared_types::*;

#[cfg(feature = "cuda")]
use crate::gpu_kuramoto::GpuKuramotoSolver;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

/// Physics coupling adapter
pub struct CouplingAdapter {
    service: PhysicsCouplingService,
    coupling_strength: f64,
    #[cfg(feature = "cuda")]
    gpu_solver: Option<GpuKuramotoSolver>,
}

impl CouplingAdapter {
    /// Create new coupling adapter
    ///
    /// # Arguments
    /// * `coupling_strength` - Kuramoto coupling strength (typically 0.1 to 1.0)
    pub fn new(coupling_strength: f64) -> Result<Self> {
        if coupling_strength < 0.0 {
            return Err(PRCTError::CouplingFailed(
                "Coupling strength must be non-negative".to_string(),
            ));
        }

        let service = PhysicsCouplingService::new(coupling_strength);

        #[cfg(feature = "cuda")]
        let gpu_solver = {
            // Try to initialize GPU solver
            match CudaDevice::new(0) {
                Ok(device) => {
                    // CudaDevice::new() returns Arc<CudaDevice>, pass directly
                    match GpuKuramotoSolver::new(device) {
                        Ok(solver) => {
                            println!("[KURAMOTO-GPU] GPU acceleration enabled");
                            Some(solver)
                        }
                        Err(e) => {
                            println!("[KURAMOTO-GPU] GPU solver initialization failed: {}, falling back to CPU", e);
                            None
                        }
                    }
                }
                Err(e) => {
                    println!("[KURAMOTO-GPU] No GPU detected: {:?}, using CPU", e);
                    None
                }
            }
        };

        Ok(Self {
            service,
            coupling_strength,
            #[cfg(feature = "cuda")]
            gpu_solver,
        })
    }

    /// Extract time series from neuromorphic state
    fn extract_neuro_timeseries(&self, neuro_state: &NeuroState) -> Vec<f64> {
        neuro_state.neuron_states.clone()
    }

    /// Extract time series from quantum state
    fn extract_quantum_timeseries(&self, quantum_state: &QuantumState) -> Vec<f64> {
        // Extract magnitude of quantum amplitudes as time series
        quantum_state
            .amplitudes
            .iter()
            .map(|(re, im)| (re * re + im * im).sqrt())
            .collect()
    }

    /// Extract phases from neuromorphic state
    fn extract_neuro_phases(&self, neuro_state: &NeuroState) -> Vec<f64> {
        // Map neuron states to phases using arctan scaling
        neuro_state
            .neuron_states
            .iter()
            .map(|&v| {
                // Map [-∞, ∞] → [-π/2, π/2] → [0, 2π]
                let scaled = v.atan();
                (scaled + std::f64::consts::PI / 2.0) * 2.0
            })
            .collect()
    }

    /// Extract phases from quantum state
    fn extract_quantum_phases(&self, quantum_state: &QuantumState) -> Vec<f64> {
        quantum_state
            .amplitudes
            .iter()
            .map(|(re, im)| im.atan2(*re))
            .collect()
    }
}

impl PhysicsCouplingPort for CouplingAdapter {
    fn compute_coupling(
        &self,
        neuro_state: &NeuroState,
        quantum_state: &QuantumState,
    ) -> Result<CouplingStrength> {
        // Neuromorphic → Quantum: Pattern strength influences quantum coherence
        let neuro_to_quantum = neuro_state.pattern_strength * quantum_state.phase_coherence;

        // Quantum → Neuromorphic: Quantum phase coherence influences spike timing
        let quantum_to_neuro = quantum_state.phase_coherence * neuro_state.coherence;

        // Bidirectional coherence: geometric mean
        let bidirectional_coherence = (neuro_to_quantum * quantum_to_neuro).sqrt();

        Ok(CouplingStrength {
            neuro_to_quantum,
            quantum_to_neuro,
            bidirectional_coherence,
            timestamp_ns: quantum_state.timestamp_ns,
        })
    }

    fn update_kuramoto_sync(
        &self,
        neuro_phases: &[f64],
        quantum_phases: &[f64],
        dt: f64,
    ) -> Result<KuramotoState> {
        // Combine neuromorphic and quantum phases
        let mut combined_phases = Vec::with_capacity(neuro_phases.len() + quantum_phases.len());
        combined_phases.extend_from_slice(neuro_phases);
        combined_phases.extend_from_slice(quantum_phases);

        let n = combined_phases.len();

        // Set natural frequencies (unity for simplicity)
        let natural_frequencies = vec![1.0; n];

        // Evolve Kuramoto dynamics
        let num_steps = 100;
        let step_dt = dt / num_steps as f64;

        let mut phases = combined_phases.clone();

        #[cfg(feature = "cuda")]
        {
            if let Some(ref gpu_solver) = self.gpu_solver {
                // Use GPU acceleration
                gpu_solver.evolve(
                    &mut phases,
                    &natural_frequencies,
                    self.coupling_strength,
                    step_dt,
                    num_steps,
                )?;
            } else {
                // Fall back to CPU
                for _ in 0..num_steps {
                    self.service
                        .kuramoto_step(&mut phases, &natural_frequencies, step_dt)?;
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // CPU-only path
            for _ in 0..num_steps {
                self.service
                    .kuramoto_step(&mut phases, &natural_frequencies, step_dt)?;
            }
        }

        // Compute order parameter
        #[cfg(feature = "cuda")]
        let order_parameter = if let Some(ref gpu_solver) = self.gpu_solver {
            gpu_solver.compute_order_parameter(&phases)?
        } else {
            PhysicsCouplingService::compute_order_parameter(&phases)
        };

        #[cfg(not(feature = "cuda"))]
        let order_parameter = PhysicsCouplingService::compute_order_parameter(&phases);

        // Compute mean phase
        let mean_phase = phases.iter().sum::<f64>() / n as f64;

        // Build coupling matrix (all-to-all uniform coupling)
        let coupling_matrix = vec![self.coupling_strength; n * n];

        Ok(KuramotoState {
            phases,
            natural_frequencies,
            coupling_matrix,
            order_parameter,
            mean_phase,
        })
    }

    fn calculate_transfer_entropy(
        &self,
        source: &[f64],
        target: &[f64],
        lag_ms: f64,
    ) -> Result<TransferEntropy> {
        if source.is_empty() || target.is_empty() {
            return Err(PRCTError::CouplingFailed(
                "Source and target must be non-empty".to_string(),
            ));
        }

        if source.len() != target.len() {
            return Err(PRCTError::CouplingFailed(
                "Source and target must have same length".to_string(),
            ));
        }

        // Convert lag from milliseconds to steps (assume 1ms per step)
        let lag_steps = (lag_ms as usize).max(1).min(source.len() / 2);

        // Calculate transfer entropy
        let te = PhysicsCouplingService::calculate_transfer_entropy(source, target, lag_steps)?;

        // Confidence based on signal length and lag
        let confidence = if source.len() > lag_steps * 10 {
            0.9
        } else if source.len() > lag_steps * 5 {
            0.7
        } else {
            0.5
        };

        Ok(TransferEntropy {
            entropy_bits: te,
            confidence,
            lag_ms,
        })
    }

    fn get_bidirectional_coupling(
        &self,
        neuro_state: &NeuroState,
        quantum_state: &QuantumState,
    ) -> Result<BidirectionalCoupling> {
        // Extract time series for transfer entropy
        let neuro_series = self.extract_neuro_timeseries(neuro_state);
        let quantum_series = self.extract_quantum_timeseries(quantum_state);

        // Ensure same length
        let min_len = neuro_series.len().min(quantum_series.len());
        let neuro_series = &neuro_series[..min_len];
        let quantum_series = &quantum_series[..min_len];

        // Calculate bidirectional transfer entropy
        let neuro_to_quantum_entropy = self.calculate_transfer_entropy(
            neuro_series,
            quantum_series,
            10.0, // 10ms lag
        )?;

        let quantum_to_neuro_entropy =
            self.calculate_transfer_entropy(quantum_series, neuro_series, 10.0)?;

        // Extract phases for Kuramoto synchronization
        let neuro_phases = self.extract_neuro_phases(neuro_state);
        let quantum_phases = self.extract_quantum_phases(quantum_state);

        // Update Kuramoto synchronization
        let kuramoto_state = self.update_kuramoto_sync(
            &neuro_phases,
            &quantum_phases,
            0.1, // 100ms evolution
        )?;

        // Compute overall coupling quality
        let coupling_quality = (neuro_to_quantum_entropy.entropy_bits.abs()
            + quantum_to_neuro_entropy.entropy_bits.abs()
            + kuramoto_state.order_parameter)
            / 3.0;

        Ok(BidirectionalCoupling {
            neuro_to_quantum_entropy,
            quantum_to_neuro_entropy,
            kuramoto_state,
            coupling_quality,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupling_adapter_creation() {
        let adapter = CouplingAdapter::new(0.5).expect("adapter creation");
        assert!((adapter.coupling_strength - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coupling_strength_calculation() {
        let adapter = CouplingAdapter::new(0.5).expect("adapter creation");

        let neuro_state = NeuroState {
            neuron_states: vec![0.5; 100],
            spike_pattern: vec![1; 100],
            coherence: 0.8,
            pattern_strength: 0.7,
            timestamp_ns: 0,
        };

        let quantum_state = QuantumState {
            amplitudes: vec![(0.5, 0.5); 100],
            phase_coherence: 0.9,
            energy: -1.0,
            entanglement: 0.0,
            timestamp_ns: 0,
        };

        let coupling = adapter
            .compute_coupling(&neuro_state, &quantum_state)
            .expect("coupling computation");

        assert!(coupling.neuro_to_quantum > 0.0);
        assert!(coupling.quantum_to_neuro > 0.0);
        assert!(coupling.bidirectional_coherence > 0.0);
    }

    #[test]
    fn test_kuramoto_synchronization() {
        let adapter = CouplingAdapter::new(1.0).expect("adapter creation");

        let neuro_phases = vec![0.0, 1.0, 2.0, 3.0];
        let quantum_phases = vec![0.5, 1.5, 2.5, 3.5];

        let kuramoto = adapter
            .update_kuramoto_sync(&neuro_phases, &quantum_phases, 0.1)
            .expect("kuramoto update");

        assert_eq!(kuramoto.phases.len(), 8);
        assert!(kuramoto.order_parameter >= 0.0 && kuramoto.order_parameter <= 1.0);
    }

    #[test]
    fn test_transfer_entropy() {
        let adapter = CouplingAdapter::new(0.5).expect("adapter creation");

        // Correlated signals
        let source: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let target: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1 + 0.5).sin()).collect();

        let te = adapter
            .calculate_transfer_entropy(&source, &target, 5.0)
            .expect("transfer entropy");

        assert!(te.entropy_bits > 0.0);
        assert!(te.confidence >= 0.5 && te.confidence <= 1.0);
        assert_eq!(te.lag_ms, 5.0);
    }

    #[test]
    fn test_bidirectional_coupling() {
        let adapter = CouplingAdapter::new(0.5).expect("adapter creation");

        let neuro_state = NeuroState {
            neuron_states: (0..100).map(|i| (i as f64 * 0.1).sin()).collect(),
            spike_pattern: vec![1; 100],
            coherence: 0.8,
            pattern_strength: 0.7,
            timestamp_ns: 0,
        };

        let quantum_state = QuantumState {
            amplitudes: (0..100)
                .map(|i| {
                    let phase = i as f64 * 0.1;
                    (phase.cos(), phase.sin())
                })
                .collect(),
            phase_coherence: 0.9,
            energy: -1.0,
            entanglement: 0.0,
            timestamp_ns: 0,
        };

        let coupling = adapter
            .get_bidirectional_coupling(&neuro_state, &quantum_state)
            .expect("bidirectional coupling");

        assert!(coupling.neuro_to_quantum_entropy.entropy_bits >= 0.0);
        assert!(coupling.quantum_to_neuro_entropy.entropy_bits >= 0.0);
        assert!(coupling.kuramoto_state.order_parameter >= 0.0);
        assert!(coupling.coupling_quality >= 0.0);
    }
}
