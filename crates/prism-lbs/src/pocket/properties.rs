//! Pocket representation and derived properties

use crate::scoring::DruggabilityScore;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Pocket {
    pub atom_indices: Vec<usize>,
    pub residue_indices: Vec<usize>,
    pub centroid: [f64; 3],
    pub volume: f64,
    pub enclosure_ratio: f64,
    pub mean_hydrophobicity: f64,
    pub mean_sasa: f64,
    pub mean_depth: f64,
    pub mean_flexibility: f64,
    pub mean_conservation: f64,
    pub persistence_score: f64,
    pub hbond_donors: usize,
    pub hbond_acceptors: usize,
    pub druggability_score: DruggabilityScore,
    pub boundary_atoms: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PocketProperties {
    pub pockets: Vec<Pocket>,
}
