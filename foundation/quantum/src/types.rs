//! Core types for quantum computing engine
//! Simplified from PRCT engine for universal optimization

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Force field parameters for quantum optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceFieldParams {
    /// Lennard-Jones parameters by atom type
    pub lj_params: HashMap<String, LJParams>,
    /// Partial charges by atom type
    pub charges: HashMap<String, f64>,
    /// van der Waals correction coefficients
    pub vdw_coeffs: HashMap<String, VdWParams>,
}

/// Lennard-Jones parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LJParams {
    /// Well depth (kcal/mol)
    pub epsilon: f64,
    /// van der Waals radius (Å)
    pub sigma: f64,
}

/// van der Waals dispersion parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VdWParams {
    /// C6 coefficient
    pub c6: f64,
    /// C8 coefficient
    pub c8: f64,
}

impl ForceFieldParams {
    /// Create default force field parameters
    pub fn new() -> Self {
        let mut lj_params = HashMap::new();
        let mut charges = HashMap::new();
        let mut vdw_coeffs = HashMap::new();

        // Default carbon parameters (CHARMM36-like)
        lj_params.insert(
            "CA".to_string(),
            LJParams {
                epsilon: 0.07, // kcal/mol
                sigma: 3.55,   // Å
            },
        );

        charges.insert("CA".to_string(), -0.1);

        vdw_coeffs.insert(
            "CA".to_string(),
            VdWParams {
                c6: 100.0,
                c8: 1000.0,
            },
        );

        Self {
            lj_params,
            charges,
            vdw_coeffs,
        }
    }

    /// Get LJ parameters for atom type
    pub fn get_lj_params(&self, atom_type: &str) -> Option<&LJParams> {
        self.lj_params.get(atom_type)
    }

    /// Get partial charge for atom type
    pub fn get_charge(&self, atom_type: &str) -> Option<f64> {
        self.charges.get(atom_type).copied()
    }

    /// Get vdW parameters for atom type
    pub fn get_vdw_params(&self, atom_type: &str) -> Option<&VdWParams> {
        self.vdw_coeffs.get(atom_type)
    }
}

impl Default for ForceFieldParams {
    fn default() -> Self {
        Self::new()
    }
}
