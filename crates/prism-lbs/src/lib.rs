//! PRISM-LBS: Ligand Binding Site Prediction System
//!
//! Reframes binding site prediction as a graph coloring optimization problem
//! leveraging PRISM's quantum-neuromorphic-GPU architecture.
//!
//! ## Features
//! - GPU-accelerated pocket detection (4 CUDA kernels)
//! - FluxNet RL for druggability weight optimization
//! - GNN embeddings for enhanced pocket features
//! - Ensemble prediction with multi-method voting
//! - PDBBind training integration
//! - World-class allosteric site detection (4-stage pipeline)

pub mod allosteric;
pub mod features;
pub mod graph;
pub mod output;
pub mod phases;
pub mod pipeline_integration;
pub mod pocket;
pub mod scoring;
pub mod softspot;
pub mod structure;
pub mod training;
pub mod unified;
pub mod validation;

// Re-exports
pub use allosteric::{
    AllostericDetectionConfig, AllostericDetectionOutput, AllostericDetector, AllostericPocket,
    AllostericDetectionType, ConfidenceAssessment, CoverageGap, Domain, DomainInterface,
    HingeRegion, MultiModuleEvidence,
};
pub use graph::{GraphConfig, ProteinGraph, ProteinGraphBuilder};
pub use pocket::{Pocket, PocketDetector, PocketProperties, PrecisionMode};
pub use scoring::{DrugabilityClass, DruggabilityScore, DruggabilityScorer};
pub use softspot::{CrypticCandidate, CrypticConfidence, SoftSpotDetector};
pub use structure::{Atom, PdbParseOptions, ProteinStructure, Residue};
pub use unified::{
    Confidence, DetectionType, Evidence, UnifiedDetector, UnifiedOutput, UnifiedPocket,
};

use anyhow::Result;
#[cfg(feature = "cuda")]
use prism_gpu::context::GpuContext;
#[cfg(feature = "cuda")]
use prism_gpu::global_context::GlobalGpuContext;
use serde::{Deserialize, Serialize};
use std::path::Path;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Main configuration for PRISM-LBS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LbsConfig {
    /// Graph construction parameters
    pub graph: GraphConfig,
    /// GPU acceleration toggle for LBS-specific kernels
    pub use_gpu: bool,
    /// Pocket geometry parameters
    pub geometry: pocket::GeometryConfig,

    /// Phase-specific configurations
    pub phase0: phases::SurfaceReservoirConfig,
    pub phase1: phases::PocketBeliefConfig,
    pub phase2: phases::PocketSamplingConfig,
    pub phase4: phases::CavityAnalysisConfig,
    pub phase6: phases::TopologicalPocketConfig,

    /// Scoring weights
    pub scoring: scoring::ScoringWeights,

    /// Output options
    pub output: OutputConfig,

    /// Maximum number of pockets to return
    pub top_n: usize,

    /// Pure GPU screening mode: bypass all CPU geometry (Voronoi, Delaunay, fpocket)
    /// Uses only mega-fused GPU kernel for ultra-fast batch processing
    #[cfg(feature = "cuda")]
    #[serde(default)]
    pub pure_gpu_mode: bool,
}

impl Default for LbsConfig {
    fn default() -> Self {
        Self {
            graph: GraphConfig::default(),
            use_gpu: true,
            geometry: pocket::GeometryConfig::default(),
            phase0: phases::SurfaceReservoirConfig::default(),
            phase1: phases::PocketBeliefConfig::default(),
            phase2: phases::PocketSamplingConfig::default(),
            phase4: phases::CavityAnalysisConfig::default(),
            phase6: phases::TopologicalPocketConfig::default(),
            scoring: scoring::ScoringWeights::default(),
            output: OutputConfig::default(),
            top_n: 20,  // Increased from 10 to catch more binding sites
            #[cfg(feature = "cuda")]
            pure_gpu_mode: false,
        }
    }
}

impl LbsConfig {
    /// Load configuration from TOML file
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let config_str = std::fs::read_to_string(path)?;
        let config: LbsConfig = toml::from_str(&config_str)?;
        Ok(config)
    }
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub formats: Vec<OutputFormat>,
    pub include_pymol_script: bool,
    pub include_json: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            formats: vec![OutputFormat::Pdb, OutputFormat::Json],
            include_pymol_script: true,
            include_json: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Pdb,
    Json,
    Csv,
}

/// Main PRISM-LBS predictor
pub struct PrismLbs {
    config: LbsConfig,
    detector: PocketDetector,
    scorer: DruggabilityScorer,
    #[cfg(feature = "cuda")]
    gpu_ctx: Option<Arc<GpuContext>>,
}

impl PrismLbs {
    /// Create new predictor with given configuration
    pub fn new(config: LbsConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            return Self::new_with_gpu(config, None);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let detector = PocketDetector::new(config.clone())?;
            let scorer = DruggabilityScorer::new(config.scoring.clone());

            Ok(Self {
                config,
                detector,
                scorer,
            })
        }
    }

    /// Create predictor while reusing an existing GPU context (when CUDA is available).
    #[cfg(feature = "cuda")]
    pub fn new_with_gpu(config: LbsConfig, gpu_ctx: Option<Arc<GpuContext>>) -> Result<Self> {
        let detector = PocketDetector::new(config.clone())?;
        let scorer = DruggabilityScorer::new(config.scoring.clone());

        let gpu_ctx = if config.use_gpu {
            gpu_ctx.or_else(|| Self::init_gpu_context_from_env().ok())
        } else {
            None
        };

        Ok(Self {
            config,
            detector,
            scorer,
            gpu_ctx,
        })
    }

    #[cfg(feature = "cuda")]
    fn init_gpu_context_from_env() -> Result<Arc<GpuContext>, LbsError> {
        // Trigger GlobalGpuContext initialization ONCE - this loads all PTX files
        // and pre-initializes mega-fused and LBS kernels. The detector will use
        // GlobalGpuContext directly for pocket detection (the main compute path).
        // We return None here to avoid creating a duplicate context.
        match GlobalGpuContext::try_get() {
            Ok(_global_gpu) => {
                log::info!("GlobalGpuContext initialized - detector will use pre-loaded kernels");
                // GlobalGpuContext is initialized; detector will use it directly.
                // Don't create a separate context to avoid duplicate PTX loading.
                // Surface/graph ops will use CPU fallback (minimal perf impact).
                Err(LbsError::Gpu("Using GlobalGpuContext directly (no separate Arc<GpuContext>)".to_string()))
            }
            Err(e) => {
                log::warn!("GlobalGpuContext initialization failed: {}. GPU disabled.", e);
                Err(LbsError::Gpu(format!("GlobalGpuContext init failed: {}", e)))
            }
        }
    }

    /// Load configuration from TOML file
    pub fn from_config_file(path: &Path) -> Result<Self> {
        let config_str = std::fs::read_to_string(path)?;
        let config: LbsConfig = toml::from_str(&config_str)?;
        Self::new(config)
    }

    /// Predict binding sites for a protein structure
    pub fn predict(&self, structure: &ProteinStructure) -> Result<Vec<Pocket>> {
        log::info!("Starting PRISM-LBS prediction for {}", structure.title);

        // 1. Compute surface accessibility (GPU when available/configured)
        let mut structure = structure.clone();
        #[cfg(feature = "cuda")]
        {
            if self.config.use_gpu {
                // Try gpu_ctx first, then fall back to GlobalGpuContext
                let gpu_ctx_ref = self.gpu_ctx.as_ref().map(|arc| arc.as_ref());
                let global_ctx_ref = GlobalGpuContext::try_get().ok().map(|g| g.context());

                if let Some(ctx) = gpu_ctx_ref.or(global_ctx_ref) {
                    let computer = structure::SurfaceComputer::default();
                    computer.compute_gpu(&mut structure, ctx)?;
                } else {
                    log::warn!(
                        "GPU requested for surface computation but no GPU context available; falling back to CPU"
                    );
                    structure.compute_surface_accessibility()?;
                }
            } else {
                structure.compute_surface_accessibility()?;
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            structure.compute_surface_accessibility()?;
        }

        // 2. Build protein graph
        let graph_builder = ProteinGraphBuilder::new(self.config.graph.clone());
        #[cfg(feature = "cuda")]
        let graph = if self.config.graph.use_gpu {
            // Try gpu_ctx first, then fall back to GlobalGpuContext
            let gpu_ctx_ref = self.gpu_ctx.as_ref().map(|arc| arc.as_ref());
            let global_ctx_ref = GlobalGpuContext::try_get().ok().map(|g| g.context());
            graph_builder.build_with_gpu(&structure, gpu_ctx_ref.or(global_ctx_ref))?
        } else {
            graph_builder.build(&structure)?
        };
        #[cfg(not(feature = "cuda"))]
        let graph = graph_builder.build(&structure)?;

        // 3. Run pocket detection through phases (with GPU when available)
        #[cfg(feature = "cuda")]
        let mut pockets = self.detector.detect_with_gpu(&graph, self.gpu_ctx.as_ref())?;
        #[cfg(not(feature = "cuda"))]
        let mut pockets = self.detector.detect(&graph)?;

        // 3.5 Progressive pocket merging for fragmented binding sites
        // Reduced merge distances to prevent mega-pockets (was 15/20/25, now 12/16/20)
        // Phase 1: Standard merge (kinases, proteases)
        pockets = merge_adjacent_pockets_with_seq(pockets, 12.0, 1, 10);
        // Phase 2: Channel merge (substrate channels like aldose reductase)
        pockets = merge_adjacent_pockets_with_seq(pockets, 16.0, 1, 50);
        // Phase 3: GPCR/membrane protein merge (multi-helix sites)
        pockets = merge_adjacent_pockets_with_seq(pockets, 20.0, 1, 100);

        // 3.6 Expand pocket residues to capture nearby interaction-capable residues
        // This catches residues not directly pocket-lining but within binding distance
        // Use 6Å radius with 80-residue cap (balanced: capture more cryptic site residues)
        // With max_pockets=4 in precision filter, larger pockets improve per-pocket recall
        for pocket in &mut pockets {
            expand_pocket_residues(pocket, &structure.atoms, 6.0, 80);
        }

        // 4. Score pockets (GPU when available)
        #[cfg(feature = "cuda")]
        {
            // Try gpu_ctx first, then fall back to GlobalGpuContext
            let gpu_ctx_ref = self.gpu_ctx.as_ref().map(|arc| arc.as_ref());
            let global_ctx_ref = GlobalGpuContext::try_get().ok().map(|g| g.context());

            if let Some(ctx) = gpu_ctx_ref.or(global_ctx_ref) {
                match self.scorer.score_batch_gpu(&pockets, ctx) {
                    Ok(scores) => {
                        for (pocket, score) in pockets.iter_mut().zip(scores) {
                            pocket.druggability_score = score;
                        }
                    }
                    Err(e) => {
                        log::warn!("GPU batch scoring failed, falling back to CPU: {}", e);
                        for pocket in &mut pockets {
                            pocket.druggability_score = self.scorer.score(pocket);
                        }
                    }
                }
            } else {
                for pocket in &mut pockets {
                    pocket.druggability_score = self.scorer.score(pocket);
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            for pocket in &mut pockets {
                pocket.druggability_score = self.scorer.score(pocket);
            }
        }

        // 5. Sort by druggability score
        pockets.sort_by(|a, b| {
            b.druggability_score
                .total
                .partial_cmp(&a.druggability_score.total)
                .unwrap()
        });

        // 6. Return top N
        pockets.truncate(self.config.top_n);

        log::info!("Found {} pockets", pockets.len());
        Ok(pockets)
    }

    /// Batch prediction for multiple structures
    pub async fn predict_batch(
        &self,
        structures: Vec<ProteinStructure>,
    ) -> Result<Vec<Vec<Pocket>>> {
        use rayon::prelude::*;

        structures.par_iter().map(|s| self.predict(s)).collect()
    }

    /// ULTRA-FAST PURE GPU DIRECT PATH: No graph construction, no CPU geometry
    ///
    /// This method bypasses ALL CPU-heavy operations:
    /// - NO ProteinGraphBuilder (saves 10.8s per large structure)
    /// - NO surface accessibility computation
    /// - NO Voronoi/Delaunay triangulation
    /// - NO fpocket
    /// - NO belief propagation
    ///
    /// Uses ONLY the mega-fused GPU kernel with 5 flat arrays:
    /// - atom coordinates
    /// - CA indices
    /// - conservation scores
    /// - B-factors
    /// - burial estimates
    ///
    /// Target: 219 structures in under 3 seconds
    #[cfg(feature = "cuda")]
    pub fn predict_pure_gpu(&self, structure: &ProteinStructure) -> Result<Vec<Pocket>> {
        log::info!("PURE GPU DIRECT: Bypassing graph construction for {}", structure.title);

        // Call detector's pure GPU direct method - NO graph construction
        let pockets = self.detector.detect_pure_gpu_direct(structure)?;

        Ok(pockets)
    }
}

/// Error types for PRISM-LBS
#[derive(Debug, thiserror::Error)]
pub enum LbsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("PDB parsing error: {0}")]
    PdbParse(String),

    #[error("Graph construction error: {0}")]
    GraphConstruction(String),

    #[error("Phase execution error: {0}")]
    PhaseExecution(String),

    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Training error: {0}")]
    Training(String),
}

impl From<anyhow::Error> for LbsError {
    fn from(err: anyhow::Error) -> Self {
        LbsError::Training(err.to_string())
    }
}

/// Merge adjacent pockets with configurable sequential range.
///
/// Kinase ATP sites, substrate channels, and GPCR sites often get split into
/// multiple pockets. This function iteratively merges pockets that are:
/// - Within `merge_distance` Angstroms (centroid-to-centroid)
/// - Share at least `min_shared_residues` residues
/// - Have residues within `seq_range` in sequence number
///
/// Uses iterative merging until convergence to handle chain merging (A→B→C).
fn merge_adjacent_pockets_with_seq(
    pockets: Vec<Pocket>,
    merge_distance: f64,
    min_shared_residues: usize,
    seq_range: i32,
) -> Vec<Pocket> {
    use std::collections::HashSet;

    if pockets.len() < 2 {
        return pockets;
    }

    let mut current_pockets = pockets;
    let max_iterations = 5;  // Prevent infinite loops

    for iteration in 0..max_iterations {
        let start_count = current_pockets.len();

        let mut merged: Vec<Pocket> = Vec::new();
        let mut used = vec![false; current_pockets.len()];

        for i in 0..current_pockets.len() {
            if used[i] {
                continue;
            }

            let mut current = current_pockets[i].clone();
            used[i] = true;

            // Try to merge with other pockets
            for j in (i + 1)..current_pockets.len() {
                if used[j] {
                    continue;
                }

                // Check centroid distance
                let dist = centroid_distance(&current.centroid, &current_pockets[j].centroid);

                // Check shared residues
                let current_res: HashSet<_> = current.residue_indices.iter().collect();
                let other_res: HashSet<_> = current_pockets[j].residue_indices.iter().collect();
                let shared = current_res.intersection(&other_res).count();

                // Check for sequential neighbor residues (within seq_range in sequence)
                let has_sequential_neighbor = current.residue_indices.iter().any(|&r1| {
                    current_pockets[j].residue_indices.iter().any(|&r2| {
                        let diff = (r1 as i32 - r2 as i32).abs();
                        diff > 0 && diff <= seq_range
                    })
                });

                // Max pocket size limit (prevent mega-pockets)
                let combined_residues = current.residue_indices.len() + current_pockets[j].residue_indices.len() - shared;
                const MAX_POCKET_RESIDUES: usize = 80;  // Allow larger pockets (max_pockets=4 controls count)

                // Merge if: close distance AND (sharing residues OR sequential neighbors)
                // Also enforce max size limit
                let should_merge = dist < merge_distance
                    && (shared >= min_shared_residues || has_sequential_neighbor)
                    && combined_residues <= MAX_POCKET_RESIDUES;

                if should_merge {
                    current = merge_two_pockets(current, current_pockets[j].clone());
                    used[j] = true;
                }
            }

            merged.push(current);
        }

        let end_count = merged.len();
        current_pockets = merged;

        // Converged - no more merges possible
        if end_count == start_count {
            log::debug!(
                "Pocket merging (dist={}, seq={}) converged after {} iterations: {} pockets",
                merge_distance, seq_range, iteration + 1, end_count
            );
            break;
        }
    }

    current_pockets
}

/// Calculate Euclidean distance between two centroids
fn centroid_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Merge two pockets into one
fn merge_two_pockets(mut a: Pocket, b: Pocket) -> Pocket {
    use std::collections::HashSet;

    // Calculate weights before moving values
    let a_atom_count = a.atom_indices.len();
    let b_atom_count = b.atom_indices.len();
    let total_atoms = (a_atom_count + b_atom_count) as f64;
    let a_weight = a_atom_count as f64 / total_atoms;
    let b_weight = b_atom_count as f64 / total_atoms;

    // Merge atom indices (deduplicate)
    let mut atom_set: HashSet<_> = a.atom_indices.into_iter().collect();
    atom_set.extend(b.atom_indices);
    a.atom_indices = atom_set.into_iter().collect();

    // Merge residue indices (deduplicate)
    let mut res_set: HashSet<_> = a.residue_indices.into_iter().collect();
    res_set.extend(b.residue_indices);
    a.residue_indices = res_set.into_iter().collect();

    // Merge boundary atoms
    let mut bound_set: HashSet<_> = a.boundary_atoms.into_iter().collect();
    bound_set.extend(b.boundary_atoms);
    a.boundary_atoms = bound_set.into_iter().collect();

    // Weighted average for centroid
    a.centroid = [
        a.centroid[0] * a_weight + b.centroid[0] * b_weight,
        a.centroid[1] * a_weight + b.centroid[1] * b_weight,
        a.centroid[2] * a_weight + b.centroid[2] * b_weight,
    ];

    // Combine volumes (approximate - may overlap)
    a.volume += b.volume * 0.8; // Assume 20% overlap

    // Average other properties
    a.enclosure_ratio = (a.enclosure_ratio + b.enclosure_ratio) / 2.0;
    a.mean_hydrophobicity = (a.mean_hydrophobicity + b.mean_hydrophobicity) / 2.0;
    a.mean_sasa = (a.mean_sasa + b.mean_sasa) / 2.0;
    a.mean_depth = a.mean_depth.max(b.mean_depth);
    a.mean_flexibility = (a.mean_flexibility + b.mean_flexibility) / 2.0;
    a.mean_conservation = (a.mean_conservation + b.mean_conservation) / 2.0;
    a.persistence_score = a.persistence_score.max(b.persistence_score);
    a.hbond_donors += b.hbond_donors;
    a.hbond_acceptors += b.hbond_acceptors;
    a.mean_electrostatic = (a.mean_electrostatic + b.mean_electrostatic) / 2.0;

    // Keep the better GNN score
    if b.gnn_druggability > a.gnn_druggability {
        a.gnn_druggability = b.gnn_druggability;
        a.gnn_embedding = b.gnn_embedding;
    }

    a
}

/// Expand pocket to include residues near existing pocket atoms.
///
/// This works better for elongated/multi-center binding sites by expanding
/// from each pocket atom rather than just the centroid.
///
/// Parameters:
/// - `expansion_radius`: Maximum distance (Å) to expand from pocket atoms
/// - `max_residues`: Cap on total residues to prevent mega-pockets
fn expand_pocket_residues(pocket: &mut Pocket, atoms: &[Atom], expansion_radius: f64, max_residues: usize) {
    use std::collections::HashSet;

    let current_residues: HashSet<usize> = pocket.residue_indices.iter().copied().collect();
    let mut all_residues = current_residues.clone();
    let initial_count = all_residues.len();

    // Skip expansion if pocket is already at max size
    if initial_count >= max_residues {
        return;
    }

    // Build set of current pocket atom coordinates
    let pocket_atom_coords: Vec<[f64; 3]> = atoms
        .iter()
        .filter(|a| current_residues.contains(&(a.residue_seq as usize)))
        .map(|a| a.coord)
        .collect();

    // Collect candidate residues with their distances
    let mut candidates: Vec<(usize, f64)> = Vec::new();

    for atom in atoms {
        let res_idx = atom.residue_seq as usize;
        if all_residues.contains(&res_idx) {
            continue;
        }

        // Check distance to nearest pocket atom
        let mut min_dist = f64::MAX;
        for pocket_coord in &pocket_atom_coords {
            let dx = atom.coord[0] - pocket_coord[0];
            let dy = atom.coord[1] - pocket_coord[1];
            let dz = atom.coord[2] - pocket_coord[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < min_dist {
                min_dist = dist;
            }
        }

        if min_dist < expansion_radius {
            // Track best (closest) distance for each residue
            if let Some(existing) = candidates.iter_mut().find(|(r, _)| *r == res_idx) {
                if min_dist < existing.1 {
                    existing.1 = min_dist;
                }
            } else {
                candidates.push((res_idx, min_dist));
            }
        }
    }

    // Sort by distance (closest first) and add up to cap
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let slots_available = max_residues.saturating_sub(all_residues.len());
    for (res_idx, _) in candidates.into_iter().take(slots_available) {
        all_residues.insert(res_idx);
    }

    pocket.residue_indices = all_residues.into_iter().collect();
    pocket.residue_indices.sort();

    let added = pocket.residue_indices.len() - initial_count;
    if added > 0 {
        log::debug!(
            "Pocket expansion: added {} residues (now {}, max {})",
            added,
            pocket.residue_indices.len(),
            max_residues
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LbsConfig::default();
        assert!(config.use_gpu);
        assert_eq!(config.top_n, 20);  // Updated from 10 to 20
    }
}
