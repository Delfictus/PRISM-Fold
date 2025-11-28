//! Quantum Adapter - GPU-Accelerated Hamiltonian Evolution
//!
//! Connects PRCT domain logic to quantum Hamiltonian processing.
//! Uses foundation/quantum with complete PRCT implementation.

use crate::errors::{PRCTError, Result};
use crate::ports::QuantumPort;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use shared_types::*;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

#[cfg(feature = "cuda")]
use crate::gpu_quantum::GpuQuantumSolver;

/// Quantum adapter using Hamiltonian evolution
pub struct QuantumAdapter {
    #[cfg(feature = "cuda")]
    _cuda_device: Option<Arc<CudaDevice>>,
    #[cfg(feature = "cuda")]
    gpu_solver: Option<GpuQuantumSolver>,
}

impl QuantumAdapter {
    /// Create new quantum adapter with optional GPU acceleration
    #[cfg(feature = "cuda")]
    pub fn new(cuda_device: Option<Arc<CudaDevice>>) -> Result<Self> {
        let gpu_solver =
            cuda_device
                .as_ref()
                .and_then(|device| match GpuQuantumSolver::new(device.clone()) {
                    Ok(solver) => {
                        println!("[QUANTUM-GPU] GPU acceleration enabled");
                        Some(solver)
                    }
                    Err(e) => {
                        println!("[QUANTUM-GPU] GPU solver init failed: {}, using CPU", e);
                        None
                    }
                });

        Ok(Self {
            _cuda_device: cuda_device,
            gpu_solver,
        })
    }

    /// Create new quantum adapter (CPU-only)
    #[cfg(not(feature = "cuda"))]
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }

    /// Build coupling matrix from graph adjacency
    fn build_coupling_matrix(&self, graph: &Graph) -> Array2<Complex64> {
        let n = graph.num_vertices;
        let mut coupling = Array2::zeros((n, n));

        // Build coupling matrix from graph structure
        for i in 0..n {
            for j in i + 1..n {
                if graph.adjacency[i * n + j] {
                    // Adjacent vertices have strong coupling
                    let strength = 1.0;

                    // Add phase relationship based on indices
                    let phase = 2.0 * std::f64::consts::PI * (i + j) as f64 / n as f64;
                    let coupling_value = Complex64::from_polar(strength, phase);

                    coupling[[i, j]] = coupling_value;
                    coupling[[j, i]] = coupling_value.conj(); // Hermitian
                }
            }
        }

        // Add diagonal terms (self-interaction)
        for i in 0..n {
            let degree = graph.adjacency[i * n..(i + 1) * n]
                .iter()
                .filter(|&&e| e)
                .count();
            coupling[[i, i]] = Complex64::new(degree as f64, 0.0);
        }

        coupling
    }

    /// Build Hamiltonian matrix from coupling
    fn build_hamiltonian_matrix(
        &self,
        coupling: &Array2<Complex64>,
        params: &EvolutionParams,
    ) -> Array2<Complex64> {
        let n = coupling.nrows();
        let mut hamiltonian = Array2::zeros((n, n));

        // H = -J Σ_{ij} coupling_{ij} |i⟩⟨j|
        for i in 0..n {
            for j in 0..n {
                hamiltonian[[i, j]] = -params.strength * coupling[[i, j]];
            }
        }

        // Add damping (imaginary part for dissipation)
        for i in 0..n {
            let damping_term = Complex64::new(0.0, -params.damping);
            hamiltonian[[i, i]] += damping_term;
        }

        hamiltonian
    }

    /// Compute eigenvalues of Hamiltonian
    fn compute_eigenvalues(&self, hamiltonian: &Array2<Complex64>) -> Vec<f64> {
        let n = hamiltonian.nrows();

        // For now, use approximate eigenvalues from diagonal + perturbation theory
        // Full implementation would use LAPACK or cuSOLVER for GPU
        let mut eigenvalues = Vec::with_capacity(n);

        for i in 0..n {
            // Diagonal + first-order correction
            let diag = hamiltonian[[i, i]].re;
            let off_diag_correction: f64 = (0..n)
                .filter(|&j| i != j)
                .map(|j| {
                    hamiltonian[[i, j]].norm_sqr()
                        / (diag - hamiltonian[[j, j]].re).abs().max(1e-10)
                })
                .sum();

            eigenvalues.push(diag - off_diag_correction);
        }

        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        eigenvalues
    }

    /// CPU-based quantum state evolution (fallback)
    fn cpu_evolve_state(
        &self,
        hamiltonian: &Array2<Complex64>,
        initial_state: &[Complex64],
        dt: f64,
        num_steps: usize,
    ) -> Vec<Complex64> {
        let n = hamiltonian.nrows();
        let mut state = Array1::from_vec(initial_state.to_vec());

        for _ in 0..num_steps {
            // First-order evolution: |ψ(t+dt)⟩ = (I - iH dt)|ψ(t)⟩
            let mut new_state = Array1::zeros(n);

            for i in 0..n {
                let mut amplitude = state[i]; // Identity part

                // Hamiltonian part: -iH dt
                for j in 0..n {
                    let h_ij = hamiltonian[[i, j]];
                    amplitude += Complex64::new(0.0, -1.0) * h_ij * dt * state[j];
                }

                new_state[i] = amplitude;
            }

            // Normalize
            let norm: f64 = new_state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            if norm > 1e-10 {
                new_state.mapv_inplace(|c| c / norm);
            }

            state = new_state;
        }

        state.to_vec()
    }

    /// Evolve quantum state using matrix exponentiation
    fn evolve_quantum_state(
        &self,
        hamiltonian: &Array2<Complex64>,
        initial_state: &QuantumState,
        evolution_time: f64,
    ) -> QuantumState {
        let n = hamiltonian.nrows();

        // Convert amplitudes to complex array
        let state_vec: Vec<Complex64> = initial_state
            .amplitudes
            .iter()
            .map(|(re, im)| Complex64::new(*re, *im))
            .collect();

        // Time evolution parameters
        let num_steps = 100;
        let dt = evolution_time / num_steps as f64;

        // GPU path
        #[cfg(feature = "cuda")]
        let final_state_vec = if let Some(ref solver) = self.gpu_solver {
            match solver.evolve_state_gpu(hamiltonian, &state_vec, dt, num_steps) {
                Ok(evolved) => evolved,
                Err(e) => {
                    eprintln!(
                        "[QUANTUM-GPU] GPU evolution failed: {}, falling back to CPU",
                        e
                    );
                    // Fall back to CPU
                    self.cpu_evolve_state(hamiltonian, &state_vec, dt, num_steps)
                }
            }
        } else {
            self.cpu_evolve_state(hamiltonian, &state_vec, dt, num_steps)
        };

        #[cfg(not(feature = "cuda"))]
        let final_state_vec = self.cpu_evolve_state(hamiltonian, &state_vec, dt, num_steps);

        // Calculate observables
        let norm_sqr: f64 = final_state_vec.iter().map(|c| c.norm_sqr()).sum();
        let phase_coherence = if norm_sqr > 1e-10 {
            let mean_phase = final_state_vec.iter().map(|c| c.arg()).sum::<f64>() / n as f64;

            let coherence = final_state_vec
                .iter()
                .map(|c| (c.arg() - mean_phase).cos())
                .sum::<f64>()
                / n as f64;

            coherence.abs()
        } else {
            0.0
        };

        // Calculate energy expectation value: E = ⟨ψ|H|ψ⟩
        let mut energy = 0.0;
        for i in 0..n {
            for j in 0..n {
                energy += (final_state_vec[i].conj() * hamiltonian[[i, j]] * final_state_vec[j]).re;
            }
        }

        // Convert back to (real, imag) pairs
        let amplitudes: Vec<(f64, f64)> = final_state_vec.iter().map(|c| (c.re, c.im)).collect();

        QuantumState {
            amplitudes,
            phase_coherence,
            energy,
            entanglement: 0.0, // Would need density matrix for this
            timestamp_ns: initial_state.timestamp_ns,
        }
    }

    /// Extract phase field from quantum state
    fn extract_phase_field(&self, state: &QuantumState) -> PhaseField {
        let n = state.amplitudes.len();

        // Extract phases from amplitudes
        let phases: Vec<f64> = state
            .amplitudes
            .iter()
            .map(|(re, im)| im.atan2(*re))
            .collect();

        // Build coherence matrix
        let mut coherence_matrix = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                // Phase coherence: cos(φ_i - φ_j)
                let phase_diff = phases[i] - phases[j];
                let coherence = phase_diff.cos();
                coherence_matrix.push(coherence);
            }
        }

        // Calculate order parameter (Kuramoto-style)
        let mean_phase = phases.iter().sum::<f64>() / n as f64;
        let order_parameter =
            phases.iter().map(|&p| (p - mean_phase).cos()).sum::<f64>() / n as f64;

        // Resonance frequency from energy spacing
        let resonance_frequency = state.energy.abs() / (2.0 * std::f64::consts::PI);

        PhaseField {
            phases,
            coherence_matrix,
            order_parameter: order_parameter.abs(),
            resonance_frequency,
        }
    }
}

impl QuantumPort for QuantumAdapter {
    fn build_hamiltonian(
        &self,
        graph: &Graph,
        params: &EvolutionParams,
    ) -> Result<HamiltonianState> {
        let n = graph.num_vertices;

        if n == 0 {
            return Err(PRCTError::QuantumFailed(
                "Cannot build Hamiltonian for empty graph".to_string(),
            ));
        }

        // Build coupling matrix from graph
        let coupling = self.build_coupling_matrix(graph);

        // Build Hamiltonian matrix
        let hamiltonian = self.build_hamiltonian_matrix(&coupling, params);

        // Compute eigenvalues
        let eigenvalues = self.compute_eigenvalues(&hamiltonian);

        // Ground state energy
        let ground_state_energy = eigenvalues
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(0.0);

        // Flatten matrix elements
        let matrix_elements: Vec<(f64, f64)> = hamiltonian.iter().map(|c| (c.re, c.im)).collect();

        Ok(HamiltonianState {
            matrix_elements,
            eigenvalues,
            ground_state_energy,
            dimension: n,
        })
    }

    fn evolve_state(
        &self,
        hamiltonian: &HamiltonianState,
        initial_state: &QuantumState,
        evolution_time: f64,
    ) -> Result<QuantumState> {
        let n = hamiltonian.dimension;

        // Reconstruct Hamiltonian matrix
        let hamiltonian_matrix = Array2::from_shape_vec(
            (n, n),
            hamiltonian
                .matrix_elements
                .iter()
                .map(|(re, im)| Complex64::new(*re, *im))
                .collect(),
        )
        .map_err(|e| {
            PRCTError::QuantumFailed(format!("Failed to reconstruct Hamiltonian: {}", e))
        })?;

        // Evolve state
        let evolved_state =
            self.evolve_quantum_state(&hamiltonian_matrix, initial_state, evolution_time);

        Ok(evolved_state)
    }

    fn get_phase_field(&self, state: &QuantumState) -> Result<PhaseField> {
        Ok(self.extract_phase_field(state))
    }

    fn compute_ground_state(&self, hamiltonian: &HamiltonianState) -> Result<QuantumState> {
        let n = hamiltonian.dimension;

        // Ground state is uniform superposition (approximation)
        // Full implementation would use eigensolver
        let amplitude = (1.0 / n as f64).sqrt();
        let amplitudes = vec![(amplitude, 0.0); n];

        Ok(QuantumState {
            amplitudes,
            phase_coherence: 1.0,
            energy: hamiltonian.ground_state_energy,
            entanglement: 0.0,
            timestamp_ns: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_adapter_hamiltonian() {
        #[cfg(feature = "cuda")]
        let adapter = QuantumAdapter::new(None).expect("adapter creation");
        #[cfg(not(feature = "cuda"))]
        let adapter = QuantumAdapter::new().expect("adapter creation");

        // Create simple test graph (cycle)
        let graph = Graph {
            num_vertices: 4,
            num_edges: 4,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
            adjacency: vec![
                false, true, false, true, true, false, true, false, false, true, false, true, true,
                false, true, false,
            ],
            coordinates: None,
        };

        let params = EvolutionParams {
            dt: 0.01,
            strength: 1.0,
            damping: 0.1,
            temperature: 300.0,
        };

        let hamiltonian = adapter
            .build_hamiltonian(&graph, &params)
            .expect("hamiltonian construction");

        assert_eq!(hamiltonian.dimension, 4);
        assert_eq!(hamiltonian.matrix_elements.len(), 16);
        assert!(hamiltonian.eigenvalues.len() > 0);
    }

    #[test]
    fn test_quantum_evolution() {
        #[cfg(feature = "cuda")]
        let adapter = QuantumAdapter::new(None).expect("adapter creation");
        #[cfg(not(feature = "cuda"))]
        let adapter = QuantumAdapter::new().expect("adapter creation");

        let graph = Graph {
            num_vertices: 3,
            num_edges: 3,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
            adjacency: vec![false, true, true, true, false, true, true, true, false],
            coordinates: None,
        };

        let params = EvolutionParams {
            dt: 0.01,
            strength: 1.0,
            damping: 0.05,
            temperature: 300.0,
        };

        let hamiltonian = adapter
            .build_hamiltonian(&graph, &params)
            .expect("hamiltonian");
        let ground_state = adapter
            .compute_ground_state(&hamiltonian)
            .expect("ground state");

        let evolved = adapter
            .evolve_state(&hamiltonian, &ground_state, 0.1)
            .expect("evolution");

        assert_eq!(evolved.amplitudes.len(), 3);
        assert!(evolved.phase_coherence >= 0.0 && evolved.phase_coherence <= 1.0);
    }

    #[test]
    fn test_phase_field_extraction() {
        #[cfg(feature = "cuda")]
        let adapter = QuantumAdapter::new(None).expect("adapter creation");
        #[cfg(not(feature = "cuda"))]
        let adapter = QuantumAdapter::new().expect("adapter creation");

        let state = QuantumState {
            amplitudes: vec![(0.5, 0.0), (0.5, 0.5), (0.3, 0.3)],
            phase_coherence: 0.9,
            energy: -1.5,
            entanglement: 0.0,
            timestamp_ns: 0,
        };

        let phase_field = adapter.get_phase_field(&state).expect("phase field");

        assert_eq!(phase_field.phases.len(), 3);
        assert_eq!(phase_field.coherence_matrix.len(), 9);
        assert!(phase_field.order_parameter >= 0.0 && phase_field.order_parameter <= 1.0);
    }
}
