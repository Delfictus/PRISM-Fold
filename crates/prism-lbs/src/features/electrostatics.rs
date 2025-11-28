//! Electrostatic feature helpers

use crate::structure::Atom;

/// Returns pre-computed partial charge; placeholder for future Poisson-Boltzmann integration
pub fn electrostatic_feature(atom: &Atom) -> f64 {
    atom.partial_charge
}
