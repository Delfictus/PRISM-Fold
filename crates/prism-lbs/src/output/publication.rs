//! Publication-ready output format for PRISM-LBS
//!
//! Produces JSON output conforming to Nature Communications standards with:
//! - Schema versioning
//! - Per-residue ligandability scores
//! - Pocket confidence scores
//! - Alpha sphere geometry
//! - Detection metadata
//! - Runtime metadata

use crate::pocket::Pocket;
use crate::scoring::DruggabilityScore;
use crate::structure::{Atom, ProteinStructure};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Result, Write};
use std::path::Path;

/// Current schema version
pub const SCHEMA_VERSION: &str = "2.0.0";

/// PRISM version
pub const PRISM_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Complete PRISM-LBS output format for publication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismOutput {
    pub schema_version: String,
    pub prism_version: String,
    pub pockets: Vec<PocketOutput>,
    pub protein_metadata: ProteinMetadata,
    pub runtime_metadata: RuntimeMetadata,
}

/// Publication-ready pocket output with all required fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketOutput {
    pub id: usize,
    pub rank: usize,
    pub chain_id: String,
    pub residue_indices: Vec<i32>,
    pub residue_names: Vec<String>,
    pub residue_scores: HashMap<String, f64>,
    pub atom_indices: Vec<usize>,
    pub geometry: PocketGeometry,
    pub scores: PocketScores,
    pub detection_metadata: DetectionMetadata,
}

/// Comprehensive pocket geometry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketGeometry {
    pub centroid: [f64; 3],
    pub volume_angstrom3: f64,
    pub surface_area_angstrom2: f64,
    pub bounding_box: BoundingBox,
    pub alpha_sphere_centers: Vec<[f64; 3]>,
    pub alpha_sphere_radii: Vec<f64>,
}

/// Axis-aligned bounding box
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

/// All pocket-level scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketScores {
    pub confidence: f64,
    pub druggability_total: f64,
    pub druggability_hydrophobicity: f64,
    pub druggability_enclosure: f64,
    pub druggability_depth: f64,
    pub enclosure_ratio: f64,
    pub mean_depth_angstrom: f64,
    pub mean_flexibility: f64,
}

/// Metadata about how the pocket was detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionMetadata {
    pub detection_type: String,
    pub merge_history: Vec<String>,
    pub merge_phase: usize,
}

/// Metadata about the input protein
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinMetadata {
    pub pdb_id: String,
    pub pdb_file: String,
    pub num_atoms: usize,
    pub num_residues: usize,
    pub num_chains: usize,
    pub chains: Vec<String>,
    pub resolution_angstrom: Option<f64>,
}

/// Runtime information for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMetadata {
    pub processing_time_ms: u64,
    pub gpu_used: bool,
    pub gpu_name: Option<String>,
    pub timestamp_utc: String,
    pub prism_commit: String,
}

impl PrismOutput {
    /// Create publication output from prediction results
    pub fn from_prediction(
        pockets: &[Pocket],
        structure: &ProteinStructure,
        processing_time_ms: u64,
        gpu_used: bool,
        gpu_name: Option<String>,
    ) -> Self {
        let pocket_outputs: Vec<PocketOutput> = pockets
            .iter()
            .enumerate()
            .map(|(i, p)| PocketOutput::from_pocket(p, i + 1, i + 1, &structure.atoms))
            .collect();

        let protein_metadata = ProteinMetadata::from_structure(structure);
        let runtime_metadata = RuntimeMetadata {
            processing_time_ms,
            gpu_used,
            gpu_name,
            timestamp_utc: Utc::now().to_rfc3339(),
            prism_commit: option_env!("GIT_HASH").unwrap_or("unknown").to_string(),
        };

        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            prism_version: PRISM_VERSION.to_string(),
            pockets: pocket_outputs,
            protein_metadata,
            runtime_metadata,
        }
    }

    /// Write to JSON file
    pub fn write_json(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// Write to JSON string
    pub fn to_json_string(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

impl PocketOutput {
    /// Create from internal Pocket representation
    pub fn from_pocket(pocket: &Pocket, id: usize, rank: usize, atoms: &[Atom]) -> Self {
        // Extract residue information
        let (residue_names, chain_id) = Self::extract_residue_info(pocket, atoms);

        // Calculate per-residue scores
        let residue_scores = Self::calculate_residue_scores(pocket, atoms);

        // Calculate geometry
        let geometry = PocketGeometry::from_pocket(pocket, atoms);

        // Calculate scores
        let scores = PocketScores::from_pocket(pocket);

        // Detection metadata (default - can be enhanced with tracking)
        let detection_metadata = DetectionMetadata {
            detection_type: "consensus".to_string(),
            merge_history: Vec::new(),
            merge_phase: 0,
        };

        Self {
            id,
            rank,
            chain_id,
            residue_indices: pocket.residue_indices.iter().map(|&r| r as i32).collect(),
            residue_names,
            residue_scores,
            atom_indices: pocket.atom_indices.clone(),
            geometry,
            scores,
            detection_metadata,
        }
    }

    /// Extract residue names and primary chain from atoms
    fn extract_residue_info(pocket: &Pocket, atoms: &[Atom]) -> (Vec<String>, String) {
        use std::collections::HashSet;

        let pocket_residues: HashSet<usize> = pocket.residue_indices.iter().copied().collect();
        let mut residue_names = Vec::new();
        let mut chain_counts: HashMap<String, usize> = HashMap::new();
        let mut seen_residues: HashSet<usize> = HashSet::new();

        for atom in atoms {
            let res_idx = atom.residue_seq as usize;
            if pocket_residues.contains(&res_idx) && !seen_residues.contains(&res_idx) {
                seen_residues.insert(res_idx);
                residue_names.push(format!("{}{}", atom.residue_name, atom.residue_seq));
                *chain_counts.entry(atom.chain_id.to_string()).or_insert(0) += 1;
            }
        }

        let chain_id = chain_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(chain, _)| chain)
            .unwrap_or_else(|| "A".to_string());

        (residue_names, chain_id)
    }

    /// Calculate per-residue ligandability scores
    ///
    /// Score based on:
    /// 1. Burial depth of residue atoms
    /// 2. Hydrophobicity contribution
    /// 3. Distance to pocket centroid
    /// 4. Local atom density
    fn calculate_residue_scores(pocket: &Pocket, atoms: &[Atom]) -> HashMap<String, f64> {
        let mut scores = HashMap::new();
        let pocket_residues: std::collections::HashSet<usize> =
            pocket.residue_indices.iter().copied().collect();

        // Group atoms by residue
        let mut residue_atoms: HashMap<usize, Vec<&Atom>> = HashMap::new();
        for atom in atoms {
            let res_idx = atom.residue_seq as usize;
            if pocket_residues.contains(&res_idx) {
                residue_atoms.entry(res_idx).or_default().push(atom);
            }
        }

        for (&res_idx, res_atoms) in &residue_atoms {
            if res_atoms.is_empty() {
                continue;
            }

            // Calculate residue centroid
            let res_centroid = Self::calculate_centroid(res_atoms);

            // Distance to pocket centroid (normalized)
            let dist_to_center = Self::euclidean_distance(&res_centroid, &pocket.centroid);
            let dist_score = 1.0 - (dist_to_center / 20.0).min(1.0);

            // Hydrophobicity from residue type
            let hydro_score = Self::residue_hydrophobicity(&res_atoms[0].residue_name);

            // Local density (atoms within 5Å)
            let density = Self::local_atom_density(&res_centroid, atoms, 5.0);
            let density_score = (density / 50.0).min(1.0);

            // SASA contribution (inverse - buried residues score higher)
            let sasa_score = if pocket.mean_sasa > 0.0 {
                1.0 - (pocket.mean_sasa / 100.0).min(1.0)
            } else {
                0.5
            };

            // Combine scores
            let final_score = 0.30 * dist_score
                + 0.25 * hydro_score
                + 0.25 * density_score
                + 0.20 * sasa_score;

            scores.insert(res_idx.to_string(), final_score.clamp(0.0, 1.0));
        }

        scores
    }

    fn calculate_centroid(atoms: &[&Atom]) -> [f64; 3] {
        if atoms.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        let n = atoms.len() as f64;
        let sum: [f64; 3] = atoms.iter().fold([0.0, 0.0, 0.0], |acc, a| {
            [acc[0] + a.coord[0], acc[1] + a.coord[1], acc[2] + a.coord[2]]
        });
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }

    fn euclidean_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    fn local_atom_density(center: &[f64; 3], atoms: &[Atom], radius: f64) -> f64 {
        atoms
            .iter()
            .filter(|a| Self::euclidean_distance(center, &a.coord) <= radius)
            .count() as f64
    }

    /// Kyte-Doolittle hydrophobicity scale (normalized to 0-1)
    fn residue_hydrophobicity(res_name: &str) -> f64 {
        match res_name.to_uppercase().as_str() {
            "ILE" => 1.000,
            "VAL" => 0.966,
            "LEU" => 0.918,
            "PHE" => 0.795,
            "CYS" => 0.740,
            "MET" => 0.685,
            "ALA" => 0.644,
            "GLY" => 0.397,
            "THR" => 0.356,
            "SER" => 0.342,
            "TRP" => 0.315,
            "TYR" => 0.274,
            "PRO" => 0.260,
            "HIS" => 0.178,
            "GLU" | "GLN" => 0.137,
            "ASP" | "ASN" => 0.123,
            "LYS" => 0.082,
            "ARG" => 0.000,
            _ => 0.5,
        }
    }
}

impl PocketGeometry {
    /// Create from internal Pocket representation
    pub fn from_pocket(pocket: &Pocket, atoms: &[Atom]) -> Self {
        let bounding_box = Self::calculate_bounding_box(pocket, atoms);
        let surface_area = Self::estimate_surface_area(pocket.volume);

        // Extract alpha sphere data if available (placeholder for now)
        let (alpha_centers, alpha_radii) = Self::extract_alpha_spheres(pocket, atoms);

        Self {
            centroid: pocket.centroid,
            volume_angstrom3: pocket.volume,
            surface_area_angstrom2: surface_area,
            bounding_box,
            alpha_sphere_centers: alpha_centers,
            alpha_sphere_radii: alpha_radii,
        }
    }

    fn calculate_bounding_box(pocket: &Pocket, atoms: &[Atom]) -> BoundingBox {
        let pocket_residues: std::collections::HashSet<usize> =
            pocket.residue_indices.iter().copied().collect();

        let pocket_atoms: Vec<&Atom> = atoms
            .iter()
            .filter(|a| pocket_residues.contains(&(a.residue_seq as usize)))
            .collect();

        if pocket_atoms.is_empty() {
            return BoundingBox {
                min: pocket.centroid,
                max: pocket.centroid,
            };
        }

        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];

        for atom in pocket_atoms {
            for i in 0..3 {
                min[i] = min[i].min(atom.coord[i]);
                max[i] = max[i].max(atom.coord[i]);
            }
        }

        BoundingBox { min, max }
    }

    /// Estimate surface area from volume (assuming roughly spherical pocket)
    fn estimate_surface_area(volume: f64) -> f64 {
        // SA = 4πr², V = (4/3)πr³
        // r = (3V/4π)^(1/3)
        // SA = 4π * (3V/4π)^(2/3)
        let r = (3.0 * volume / (4.0 * std::f64::consts::PI)).cbrt();
        4.0 * std::f64::consts::PI * r.powi(2)
    }

    /// Extract or generate alpha sphere approximation
    fn extract_alpha_spheres(pocket: &Pocket, atoms: &[Atom]) -> (Vec<[f64; 3]>, Vec<f64>) {
        // Generate representative alpha spheres based on pocket geometry
        // In production, these would come from the Voronoi/Delaunay detector

        let pocket_residues: std::collections::HashSet<usize> =
            pocket.residue_indices.iter().copied().collect();

        let pocket_atoms: Vec<&Atom> = atoms
            .iter()
            .filter(|a| pocket_residues.contains(&(a.residue_seq as usize)))
            .collect();

        if pocket_atoms.is_empty() {
            return (vec![pocket.centroid], vec![pocket.volume.cbrt()]);
        }

        // Sample points within the pocket region
        let mut centers = Vec::new();
        let mut radii = Vec::new();

        // Add centroid as primary sphere
        let primary_radius = (3.0 * pocket.volume / (4.0 * std::f64::consts::PI)).cbrt();
        centers.push(pocket.centroid);
        radii.push(primary_radius);

        // Add additional spheres at boundary atoms
        for &atom_idx in pocket.boundary_atoms.iter().take(10) {
            if let Some(atom) = atoms.get(atom_idx) {
                centers.push(atom.coord);
                radii.push(2.0); // Standard probe radius
            }
        }

        (centers, radii)
    }
}

impl PocketScores {
    /// Create from internal Pocket representation
    pub fn from_pocket(pocket: &Pocket) -> Self {
        let confidence = Self::calculate_confidence(pocket);

        Self {
            confidence,
            druggability_total: pocket.druggability_score.total,
            druggability_hydrophobicity: pocket.druggability_score.components.hydro,
            druggability_enclosure: pocket.druggability_score.components.enclosure,
            druggability_depth: pocket.druggability_score.components.depth,
            enclosure_ratio: pocket.enclosure_ratio,
            mean_depth_angstrom: pocket.mean_depth,
            mean_flexibility: pocket.mean_flexibility,
        }
    }

    /// Calculate pocket confidence score
    ///
    /// Based on:
    /// 1. Volume in reasonable range (100-3000 Å³)
    /// 2. Druggability score
    /// 3. Enclosure ratio
    /// 4. Residue count in reasonable range
    fn calculate_confidence(pocket: &Pocket) -> f64 {
        // Volume quality score
        let volume_score = if pocket.volume >= 100.0 && pocket.volume <= 3000.0 {
            1.0
        } else if pocket.volume >= 50.0 && pocket.volume <= 5000.0 {
            0.7
        } else {
            0.3
        };

        // Residue count quality
        let residue_count = pocket.residue_indices.len();
        let residue_score = if residue_count >= 10 && residue_count <= 60 {
            1.0
        } else if residue_count >= 5 && residue_count <= 100 {
            0.7
        } else {
            0.3
        };

        // Druggability contribution
        let drug_score = pocket.druggability_score.total.clamp(0.0, 1.0);

        // Enclosure contribution
        let enclosure_score = pocket.enclosure_ratio.clamp(0.0, 1.0);

        // Combine with weights
        let confidence = 0.30 * volume_score
            + 0.25 * drug_score
            + 0.25 * residue_score
            + 0.20 * enclosure_score;

        confidence.clamp(0.0, 1.0)
    }
}

impl ProteinMetadata {
    /// Create from ProteinStructure
    pub fn from_structure(structure: &ProteinStructure) -> Self {
        use std::collections::HashSet;

        // Extract unique chains
        let chains: HashSet<String> = structure.atoms.iter().map(|a| a.chain_id.to_string()).collect();
        let chains: Vec<String> = chains.into_iter().collect();

        // Extract unique residues
        let residues: HashSet<i32> = structure.atoms.iter().map(|a| a.residue_seq).collect();

        Self {
            pdb_id: Self::extract_pdb_id(&structure.title),
            pdb_file: structure.title.clone(),
            num_atoms: structure.atoms.len(),
            num_residues: residues.len(),
            num_chains: chains.len(),
            chains,
            resolution_angstrom: structure.resolution,
        }
    }

    fn extract_pdb_id(title: &str) -> String {
        // Try to extract 4-character PDB ID from title
        let path = std::path::Path::new(title);
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(title);

        // Extract first 4 alphanumeric characters
        let chars: String = stem.chars().filter(|c| c.is_alphanumeric()).take(4).collect();
        if chars.len() == 4 {
            chars.to_uppercase()
        } else {
            stem.to_string()
        }
    }
}

/// Helper to write publication-ready output
pub fn write_publication_json(
    path: &Path,
    pockets: &[Pocket],
    structure: &ProteinStructure,
    processing_time_ms: u64,
    gpu_used: bool,
    gpu_name: Option<String>,
) -> Result<()> {
    let output = PrismOutput::from_prediction(
        pockets,
        structure,
        processing_time_ms,
        gpu_used,
        gpu_name,
    );
    output.write_json(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_version() {
        assert_eq!(SCHEMA_VERSION, "2.0.0");
    }

    #[test]
    fn test_residue_hydrophobicity() {
        assert!(PocketOutput::residue_hydrophobicity("ILE") > 0.9);
        assert!(PocketOutput::residue_hydrophobicity("ARG") < 0.1);
    }

    #[test]
    fn test_surface_area_estimation() {
        // For a sphere of volume 1000 Å³
        let sa = PocketGeometry::estimate_surface_area(1000.0);
        // r ≈ 6.2 Å, SA ≈ 483 Å²
        assert!(sa > 400.0 && sa < 600.0);
    }

    #[test]
    fn test_confidence_calculation() {
        let mut pocket = Pocket::default();
        pocket.volume = 500.0;
        pocket.residue_indices = (0..30).collect();
        pocket.druggability_score.total = 0.8;
        pocket.enclosure_ratio = 0.7;

        let scores = PocketScores::from_pocket(&pocket);
        assert!(scores.confidence > 0.6);
    }
}
