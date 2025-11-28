//! Physics Coupling Domain Service
//!
//! Implements Kuramoto synchronization, transfer entropy, and bidirectional
//! neuromorphic-quantum coupling. This is DOMAIN LOGIC, not infrastructure.

use crate::errors::*;
use shared_types::*;

/// Physics Coupling Service (Domain Service)
///
/// This is NOT a port - it's domain business logic that happens to be shared.
pub struct PhysicsCouplingService {
    /// Kuramoto coupling strength
    coupling_strength: f64,
}

impl PhysicsCouplingService {
    /// Create new physics coupling service
    pub fn new(coupling_strength: f64) -> Self {
        Self { coupling_strength }
    }

    /// Compute Kuramoto order parameter from phases
    ///
    /// r = |⟨e^(iθ)⟩| where ⟨⟩ is ensemble average
    pub fn compute_order_parameter(phases: &[f64]) -> f64 {
        if phases.is_empty() {
            return 0.0;
        }

        let n = phases.len() as f64;
        let sum_real: f64 = phases.iter().map(|p| p.cos()).sum();
        let sum_imag: f64 = phases.iter().map(|p| p.sin()).sum();

        ((sum_real / n).powi(2) + (sum_imag / n).powi(2)).sqrt()
    }

    /// Update Kuramoto phases using coupling dynamics
    ///
    /// dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
    pub fn kuramoto_step(
        &self,
        phases: &mut [f64],
        natural_frequencies: &[f64],
        dt: f64,
    ) -> Result<()> {
        let n = phases.len();
        if n != natural_frequencies.len() {
            return Err(PRCTError::CouplingFailed(
                "Phase and frequency array size mismatch".into(),
            ));
        }

        let mut new_phases = phases.to_vec();

        for i in 0..n {
            let mut coupling_sum = 0.0;

            // Sum coupling terms: Σⱼ sin(θⱼ - θᵢ)
            for j in 0..n {
                if i != j {
                    coupling_sum += (phases[j] - phases[i]).sin();
                }
            }

            // Kuramoto equation: dθ/dt = ω + (K/N) * coupling_sum
            let dphase_dt =
                natural_frequencies[i] + (self.coupling_strength / n as f64) * coupling_sum;

            // Euler integration
            new_phases[i] = (phases[i] + dphase_dt * dt) % (2.0 * core::f64::consts::PI);
        }

        phases.copy_from_slice(&new_phases);
        Ok(())
    }

    /// Calculate transfer entropy (information flow from X to Y)
    ///
    /// TE(X→Y) = I(Y_future ; X_past | Y_past)
    ///
    /// Simplified implementation using time-delayed mutual information
    pub fn calculate_transfer_entropy(
        source: &[f64],
        target: &[f64],
        lag_steps: usize,
    ) -> Result<f64> {
        if source.len() != target.len() {
            return Err(PRCTError::CouplingFailed(
                "Source and target must have same length".into(),
            ));
        }

        if source.len() < lag_steps + 2 {
            return Err(PRCTError::CouplingFailed(
                "Time series too short for lag".into(),
            ));
        }

        let n = source.len();

        // Compute delayed mutual information as proxy for transfer entropy
        let mut te = 0.0;
        let mut count = 0;

        for i in lag_steps..(n - 1) {
            // Y_future = target[i+1]
            // X_past = source[i - lag]
            // Y_past = target[i]

            let y_future = target[i + 1];
            let x_past = source[i - lag_steps];
            let y_past = target[i];

            // Simplified TE calculation (correlation-based approximation)
            let correlation = (y_future - y_past) * x_past;
            te += correlation.abs();
            count += 1;
        }

        Ok(te / count as f64)
    }

    /// Compute coupling strength from neuromorphic and quantum states
    pub fn compute_coupling_strength(
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_parameter() {
        // Fully synchronized phases (all 0.0)
        let phases = vec![0.0; 10];
        let order = PhysicsCouplingService::compute_order_parameter(&phases);
        assert!((order - 1.0).abs() < 0.01);

        // Random phases (low synchronization)
        let random_phases: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.137) % (2.0 * core::f64::consts::PI))
            .collect();
        let order_random = PhysicsCouplingService::compute_order_parameter(&random_phases);
        assert!(order_random < 0.3);
    }

    #[test]
    fn test_kuramoto_synchronization() {
        let service = PhysicsCouplingService::new(1.0);

        // Start with random phases
        let mut phases = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let frequencies = vec![1.0; 5];

        let initial_order = PhysicsCouplingService::compute_order_parameter(&phases);

        // Evolve
        for _ in 0..1000 {
            service
                .kuramoto_step(&mut phases, &frequencies, 0.01)
                .unwrap();
        }

        let final_order = PhysicsCouplingService::compute_order_parameter(&phases);

        // Order parameter should increase (synchronization)
        assert!(final_order > initial_order);
    }

    #[test]
    fn test_transfer_entropy() {
        // Correlated signals
        let source: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let target: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1 + 0.5).sin()).collect();

        let te = PhysicsCouplingService::calculate_transfer_entropy(&source, &target, 5).unwrap();
        assert!(te > 0.0);
    }
}
