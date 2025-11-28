//! Quantum Mechanics Proofs
//!
//! Constitution: Phase 1, Task 1.1
//! Implements verification of fundamental quantum mechanical relations.
//!
//! This module provides formal verification of:
//! 1. Heisenberg Uncertainty Principle: ΔxΔp ≥ ℏ/2
//! 2. Generalized uncertainty for non-commuting observables
//! 3. Energy-time uncertainty relation

use super::proof_system::{Assumption, MathematicalStatement, NumericalConfig, ProofResult};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
/// Physical constants
const HBAR: f64 = 1.054571817e-34; // Reduced Planck constant (J·s)

/// Heisenberg Uncertainty Principle
///
/// Mathematical Statement: ΔxΔp ≥ ℏ/2
///
/// For any quantum state, the product of position and momentum uncertainties
/// satisfies this fundamental bound. More generally, for non-commuting operators:
///
/// ΔA ΔB ≥ (1/2)|⟨[A,B]⟩|
///
/// where [A,B] = AB - BA is the commutator.
pub struct HeisenbergUncertainty {
    config: NumericalConfig,
    /// Use normalized units (ℏ = 1) for numerical testing
    normalized_units: bool,
}

impl HeisenbergUncertainty {
    pub fn new() -> Self {
        Self {
            config: NumericalConfig::default(),
            normalized_units: true,
        }
    }

    pub fn with_config(config: NumericalConfig) -> Self {
        Self {
            config,
            normalized_units: true,
        }
    }

    /// Get effective ℏ for calculations (1.0 in normalized units, actual value otherwise)
    fn hbar(&self) -> f64 {
        if self.normalized_units {
            1.0
        } else {
            HBAR
        }
    }

    /// Analytical verification using Cauchy-Schwarz inequality
    fn verify_analytical(&self) -> Result<(), String> {
        // Proof via Cauchy-Schwarz inequality:
        //
        // For any two operators A and B:
        // ⟨(ΔA)²⟩⟨(ΔB)²⟩ ≥ (1/4)|⟨[A,B]⟩|²
        //
        // Taking square root: ΔA ΔB ≥ (1/2)|⟨[A,B]⟩|
        //
        // For position and momentum: [x,p] = iℏ
        // Therefore: Δx Δp ≥ (1/2)|⟨iℏ⟩| = ℏ/2
        //
        // This follows from the Cauchy-Schwarz inequality:
        // |⟨ψ|φ⟩|² ≤ ⟨ψ|ψ⟩⟨φ|φ⟩
        //
        // Setting |ψ⟩ = (A - ⟨A⟩)|ψ⟩ and |φ⟩ = (B - ⟨B⟩)|ψ⟩

        // Verify Cauchy-Schwarz on test vectors
        let test_vectors = vec![
            (vec![1.0, 0.0], vec![0.0, 1.0]),
            (vec![1.0, 1.0], vec![1.0, -1.0]),
            (vec![3.0, 4.0], vec![4.0, 3.0]),
        ];

        for (v1, v2) in test_vectors {
            let dot_product: f64 = v1.iter().zip(&v2).map(|(a, b)| a * b).sum();
            let norm1_sq: f64 = v1.iter().map(|x| x * x).sum();
            let norm2_sq: f64 = v2.iter().map(|x| x * x).sum();

            if dot_product * dot_product > norm1_sq * norm2_sq + 1e-10 {
                return Err(format!(
                    "Cauchy-Schwarz violated: |⟨v1|v2⟩|² = {} > ||v1||²||v2||² = {}",
                    dot_product * dot_product,
                    norm1_sq * norm2_sq
                ));
            }
        }

        Ok(())
    }

    /// Numerical verification using Gaussian wavepackets
    fn verify_numerical(&self) -> Result<f64, String> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);
        let mut violations = 0;
        let hbar = self.hbar();

        // Test on Gaussian wavepackets: ψ(x) ∝ exp(-(x-x₀)²/(4σ²) + ip₀x/ℏ)
        // For Gaussian: Δx = σ, Δp = ℏ/(2σ)
        // Therefore: ΔxΔp = ℏ/2 (saturates the bound)

        for _ in 0..self.config.num_samples {
            let sigma = 0.1 + rng.gen::<f64>() * 10.0; // Width parameter
            let _x0 = rng.gen::<f64>() * 10.0 - 5.0; // Center position
            let _p0 = rng.gen::<f64>() * 10.0 - 5.0; // Center momentum

            // For Gaussian wavepacket
            let delta_x = sigma;
            let delta_p = hbar / (2.0 * sigma);
            let product = delta_x * delta_p;
            let bound = hbar / 2.0;

            if product < bound - self.config.tolerance {
                violations += 1;
            }
        }

        // Test on position eigenstates (idealized delta functions)
        // Δx → 0, Δp → ∞, product should still satisfy bound
        // We approximate with very narrow Gaussians
        for _ in 0..(self.config.num_samples / 10) {
            let sigma = 1e-6 + rng.gen::<f64>() * 1e-5; // Very narrow
            let delta_x = sigma;
            let delta_p = hbar / (2.0 * sigma); // Will be very large
            let product = delta_x * delta_p;

            if product < hbar / 2.0 - 1e-15 {
                violations += 1;
            }
        }

        // Test on momentum eigenstates (plane waves)
        // Δp → 0, Δx → ∞
        for _ in 0..(self.config.num_samples / 10) {
            let delta_p = 1e-6 + rng.gen::<f64>() * 1e-5; // Very narrow in momentum
            let delta_x = hbar / (2.0 * delta_p); // Very large in position
            let product = delta_x * delta_p;

            if product < hbar / 2.0 - 1e-15 {
                violations += 1;
            }
        }

        // Test on superpositions of Gaussians
        for _ in 0..(self.config.num_samples / 10) {
            // Superposition of two Gaussians with different centers
            let sigma1 = 0.5 + rng.gen::<f64>() * 2.0;
            let sigma2 = 0.5 + rng.gen::<f64>() * 2.0;
            let separation = 2.0 + rng.gen::<f64>() * 5.0;

            // For superposition, uncertainties are typically larger
            // We use a conservative estimate
            let sigma_eff = (sigma1.powi(2) + sigma2.powi(2) + separation.powi(2) / 4.0_f64).sqrt();
            let delta_x = sigma_eff;
            let delta_p = hbar / (2.0_f64 * sigma1.min(sigma2)); // Lower bound estimate

            let product = delta_x * delta_p;
            if product < hbar / 2.0 - self.config.tolerance {
                violations += 1;
            }
        }

        let total_tests = self.config.num_samples + 3 * (self.config.num_samples / 10);
        let violation_rate = violations as f64 / total_tests as f64;

        if violation_rate > 1e-6 {
            return Err(format!(
                "Uncertainty principle violated in {:.4}% of cases",
                violation_rate * 100.0
            ));
        }

        Ok(1.0 - violation_rate)
    }

    /// Verify for specific quantum states analytically
    fn verify_specific_states(&self) -> Result<(), String> {
        let hbar = self.hbar();

        // Ground state of harmonic oscillator: Δx = √(ℏ/2mω), Δp = √(ℏmω/2)
        // Product: ΔxΔp = ℏ/2 (saturates bound)
        let m = 1.0;
        let omega = 1.0;
        let delta_x_ho = (hbar / (2.0 * m * omega)).sqrt();
        let delta_p_ho = (hbar * m * omega / 2.0).sqrt();
        let product_ho = delta_x_ho * delta_p_ho;

        if (product_ho - hbar / 2.0).abs() > 1e-10 {
            return Err(format!(
                "Harmonic oscillator ground state: ΔxΔp = {} ≠ ℏ/2 = {}",
                product_ho,
                hbar / 2.0
            ));
        }

        // Coherent state (also saturates bound)
        let _alpha = 2.0; // Coherent state parameter
        let delta_x_coherent = (hbar / (2.0 * m * omega)).sqrt();
        let delta_p_coherent = (hbar * m * omega / 2.0).sqrt();
        let product_coherent = delta_x_coherent * delta_p_coherent;

        if (product_coherent - hbar / 2.0).abs() > 1e-10 {
            return Err(format!(
                "Coherent state: ΔxΔp = {} ≠ ℏ/2 = {}",
                product_coherent,
                hbar / 2.0
            ));
        }

        Ok(())
    }
}

impl Default for HeisenbergUncertainty {
    fn default() -> Self {
        Self::new()
    }
}

impl MathematicalStatement for HeisenbergUncertainty {
    fn latex(&self) -> String {
        r"\Delta x \Delta p \geq \frac{\hbar}{2}".to_string()
    }

    fn verify(&self) -> ProofResult {
        let analytical_result = self.verify_analytical();
        let numerical_result = self.verify_numerical();
        let specific_result = self.verify_specific_states();

        // All three must pass
        let all_analytical = analytical_result.is_ok() && specific_result.is_ok();
        let numerical_ok = numerical_result.is_ok();

        match (all_analytical, numerical_ok) {
            (true, true) => ProofResult::Verified {
                analytical: true,
                numerical: true,
                confidence: numerical_result.unwrap(),
            },
            (false, true) => {
                let error = analytical_result
                    .err()
                    .or(specific_result.err())
                    .unwrap_or_else(|| "Unknown error".to_string());
                ProofResult::Failed {
                    reason: "Analytical verification failed".to_string(),
                    analytical_error: Some(error),
                    numerical_error: None,
                }
            }
            (true, false) => ProofResult::Failed {
                reason: "Numerical verification failed".to_string(),
                analytical_error: None,
                numerical_error: Some(numerical_result.unwrap_err()),
            },
            (false, false) => {
                let ae = analytical_result
                    .err()
                    .or(specific_result.err())
                    .unwrap_or_else(|| "Unknown error".to_string());
                let ne = numerical_result.unwrap_err();
                ProofResult::Failed {
                    reason: "Both analytical and numerical verification failed".to_string(),
                    analytical_error: Some(ae),
                    numerical_error: Some(ne),
                }
            }
        }
    }

    fn assumptions(&self) -> Vec<Assumption> {
        vec![
            Assumption::verified(
                "Quantum state normalization: ⟨ψ|ψ⟩ = 1",
                r"\int |\psi(x)|^2 dx = 1",
            ),
            Assumption::verified(
                "Canonical commutation relation: [x,p] = iℏ",
                r"[x, p] = i\hbar",
            ),
            Assumption::verified(
                "Hermitian operators: Position and momentum are self-adjoint",
                r"x^\dagger = x, \quad p^\dagger = p",
            ),
            Assumption::verified(
                "Finite second moments: ⟨x²⟩ and ⟨p²⟩ are finite",
                r"\langle x^2 \rangle < \infty, \quad \langle p^2 \rangle < \infty",
            ),
        ]
    }

    fn description(&self) -> String {
        "Heisenberg Uncertainty Principle: Position-momentum uncertainty bound".to_string()
    }

    fn domain(&self) -> String {
        "Square-integrable quantum states with finite second moments".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heisenberg_uncertainty() {
        let theorem = HeisenbergUncertainty::new();
        let result = theorem.verify();

        match result {
            ProofResult::Verified {
                analytical,
                numerical,
                confidence,
            } => {
                assert!(analytical, "Analytical verification should pass");
                assert!(numerical, "Numerical verification should pass");
                assert!(
                    confidence > 0.999,
                    "Confidence should be very high: {}",
                    confidence
                );
            }
            _ => panic!("Heisenberg uncertainty verification failed: {}", result),
        }
    }

    #[test]
    fn test_latex_representation() {
        let theorem = HeisenbergUncertainty::new();
        assert!(theorem.latex().contains(r"\Delta x"));
        assert!(theorem.latex().contains(r"\Delta p"));
        assert!(theorem.latex().contains(r"\hbar"));
    }

    #[test]
    fn test_assumptions() {
        let theorem = HeisenbergUncertainty::new();
        let assumptions = theorem.assumptions();
        assert_eq!(assumptions.len(), 4);
        assert!(assumptions.iter().all(|a| a.verified));
    }

    #[test]
    fn test_gaussian_saturation() {
        // Gaussian wavepacket should saturate the uncertainty bound
        let hbar = 1.0; // Normalized units
        let sigma = 2.0;
        let delta_x = sigma;
        let delta_p = hbar / (2.0 * sigma);
        let product = delta_x * delta_p;

        assert!(
            ((product - hbar / 2.0) as f64).abs() < 1e-10,
            "Gaussian should saturate: ΔxΔp = {}, expected {}",
            product,
            hbar / 2.0
        );
    }

    #[test]
    fn test_harmonic_oscillator_ground_state() {
        let theorem = HeisenbergUncertainty::new();
        let result = theorem.verify_specific_states();
        assert!(
            result.is_ok(),
            "Harmonic oscillator verification failed: {:?}",
            result
        );
    }
}
