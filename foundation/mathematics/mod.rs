//! Mathematics Module
//!
//! Constitution: Phase 1, Task 1.1
//!
//! This module provides the mathematical proof infrastructure for the
//! Active Inference Platform. All algorithms must be mathematically proven
//! before implementation.
//!
//! ## Implemented Proofs
//!
//! 1. **Thermodynamics**: Second Law (dS/dt ≥ 0)
//! 2. **Information Theory**: Entropy non-negativity (H(X) ≥ 0)
//! 3. **Quantum Mechanics**: Heisenberg Uncertainty (ΔxΔp ≥ ℏ/2)
//!
//! ## Usage
//!
//! ```rust
//! use mathematics::proof_system::MathematicalStatement;
//! use mathematics::thermodynamics::EntropyProductionTheorem;
//!
//! let theorem = EntropyProductionTheorem::new();
//! let result = theorem.verify();
//! println!("Verification: {}", result);
//! ```

pub mod information_theory;
pub mod proof_system;
pub mod quantum_mechanics;
pub mod thermodynamics;

// Re-export main types for convenience
pub use proof_system::{Assumption, MathematicalStatement, NumericalConfig, ProofResult};

/// Verify all fundamental theorems required by the constitution
///
/// Constitution Reference: Phase 1, Task 1.1 - Validation Criteria
///
/// This function verifies:
/// - Entropy production theorem (thermodynamics)
/// - Information bounds (information theory)
/// - Quantum relations (quantum mechanics)
///
/// All proofs must pass both analytical and numerical verification.
pub fn verify_all_theorems() -> Result<(), String> {
    use information_theory::{EntropyNonNegativity, MutualInformationNonNegativity};
    use quantum_mechanics::HeisenbergUncertainty;
    use thermodynamics::EntropyProductionTheorem;

    let mut failures = Vec::new();

    // 1. Thermodynamics: Second Law
    let thermo = EntropyProductionTheorem::new();
    match thermo.verify() {
        ProofResult::Verified { .. } => {
            println!("✓ Thermodynamics: {}", thermo.description());
        }
        result => {
            let msg = format!("✗ Thermodynamics failed: {}", result);
            println!("{}", msg);
            failures.push(msg);
        }
    }

    // 2. Information Theory: Entropy Non-Negativity
    let info_entropy = EntropyNonNegativity::new();
    match info_entropy.verify() {
        ProofResult::Verified { .. } => {
            println!("✓ Information Theory: {}", info_entropy.description());
        }
        result => {
            let msg = format!("✗ Information entropy failed: {}", result);
            println!("{}", msg);
            failures.push(msg);
        }
    }

    // 3. Information Theory: Mutual Information Non-Negativity
    let info_mi = MutualInformationNonNegativity::new();
    match info_mi.verify() {
        ProofResult::Verified { .. } => {
            println!("✓ Information Theory: {}", info_mi.description());
        }
        result => {
            let msg = format!("✗ Mutual information failed: {}", result);
            println!("{}", msg);
            failures.push(msg);
        }
    }

    // 4. Quantum Mechanics: Heisenberg Uncertainty
    let quantum = HeisenbergUncertainty::new();
    match quantum.verify() {
        ProofResult::Verified { .. } => {
            println!("✓ Quantum Mechanics: {}", quantum.description());
        }
        result => {
            let msg = format!("✗ Quantum uncertainty failed: {}", result);
            println!("{}", msg);
            failures.push(msg);
        }
    }

    if failures.is_empty() {
        println!("\n✓ All fundamental theorems verified successfully");
        Ok(())
    } else {
        Err(format!(
            "Verification failed for {} theorem(s):\n{}",
            failures.len(),
            failures.join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_theorems_verify() {
        let result = verify_all_theorems();
        assert!(
            result.is_ok(),
            "All fundamental theorems must verify: {:?}",
            result
        );
    }

    #[test]
    fn test_theorem_latex_representations() {
        use information_theory::EntropyNonNegativity;
        use quantum_mechanics::HeisenbergUncertainty;
        use thermodynamics::EntropyProductionTheorem;

        let thermo = EntropyProductionTheorem::new();
        assert!(!thermo.latex().is_empty());

        let info = EntropyNonNegativity::new();
        assert!(!info.latex().is_empty());

        let quantum = HeisenbergUncertainty::new();
        assert!(!quantum.latex().is_empty());
    }

    #[test]
    fn test_theorem_assumptions() {
        use information_theory::EntropyNonNegativity;
        use quantum_mechanics::HeisenbergUncertainty;
        use thermodynamics::EntropyProductionTheorem;

        let thermo = EntropyProductionTheorem::new();
        assert!(!thermo.assumptions().is_empty());

        let info = EntropyNonNegativity::new();
        assert!(!info.assumptions().is_empty());

        let quantum = HeisenbergUncertainty::new();
        assert!(!quantum.assumptions().is_empty());
    }
}
