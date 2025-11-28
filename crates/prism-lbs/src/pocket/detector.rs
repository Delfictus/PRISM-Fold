//! Pocket detector orchestrating phases

use crate::graph::ProteinGraph;
use crate::phases::{
    CavityAnalysisConfig, CavityAnalysisPhase, PocketBeliefConfig, PocketBeliefPhase,
    PocketRefinementConfig, PocketRefinementPhase, PocketSamplingConfig, PocketSamplingPhase,
    SurfaceReservoirConfig, SurfaceReservoirPhase, TopologicalPocketConfig, TopologicalPocketPhase,
};
use crate::pocket::geometry::{
    alpha_shape_volume, boundary_enclosure, bounding_box_volume, convex_hull_volume,
    enclosure_ratio, voxel_volume,
};
use crate::pocket::properties::Pocket;
use crate::scoring::DruggabilityScore;
use crate::LbsError;

#[derive(Debug, Clone)]
pub struct PocketDetectorConfig {
    pub max_pockets: usize,
    pub reservoir: SurfaceReservoirConfig,
    pub beliefs: PocketBeliefConfig,
    pub sampling: PocketSamplingConfig,
    pub cavity: CavityAnalysisConfig,
    pub topology: TopologicalPocketConfig,
    pub refinement: PocketRefinementConfig,
    pub geometry: crate::pocket::GeometryConfig,
}

impl Default for PocketDetectorConfig {
    fn default() -> Self {
        Self {
            max_pockets: 20,
            reservoir: SurfaceReservoirConfig::default(),
            beliefs: PocketBeliefConfig::default(),
            sampling: PocketSamplingConfig::default(),
            cavity: CavityAnalysisConfig::default(),
            topology: TopologicalPocketConfig::default(),
            refinement: PocketRefinementConfig::default(),
            geometry: crate::pocket::GeometryConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PocketDetector {
    pub config: PocketDetectorConfig,
}

impl PocketDetector {
    pub fn new(config: crate::LbsConfig) -> Result<Self, LbsError> {
        Ok(Self {
            config: PocketDetectorConfig {
                max_pockets: config.phase1.max_pockets,
                reservoir: config.phase0.clone(),
                beliefs: config.phase1.clone(),
                sampling: config.phase2.clone(),
                cavity: config.phase4.clone(),
                topology: config.phase6.clone(),
                refinement: PocketRefinementConfig::default(),
                geometry: config.geometry.clone(),
            },
        })
    }

    pub fn detect(&self, graph: &ProteinGraph) -> Result<Vec<Pocket>, LbsError> {
        let reservoir_phase = SurfaceReservoirPhase::new(self.config.reservoir.clone());
        let reservoir_output = reservoir_phase.execute(graph);

        let belief_phase = PocketBeliefPhase::new(self.config.beliefs.clone());
        let belief_output = belief_phase.execute(graph, &reservoir_output);

        let sampling_phase = PocketSamplingPhase::new(self.config.sampling.clone());
        let sampling_output = sampling_phase.execute(graph, &belief_output);

        let cavity_phase = CavityAnalysisPhase::new(self.config.cavity.clone());
        let cavity_output = cavity_phase.execute(graph);

        let topology_phase = TopologicalPocketPhase::new(self.config.topology.clone());
        let topology_output = topology_phase.execute(graph);

        let refinement_phase = PocketRefinementPhase::new(self.config.refinement.clone());
        let refinement_output = refinement_phase.execute(
            graph,
            &sampling_output.coloring,
            &cavity_output,
            &topology_output,
            &reservoir_output,
        );

        let pockets = self.assemble_pockets(
            graph,
            &sampling_output.coloring,
            &refinement_output.boundary_vertices,
        );

        let mut pockets_with_scores = Vec::new();
        for p in pockets {
            let mut pocket = p;
            pocket.druggability_score = DruggabilityScore::default();
            pockets_with_scores.push(pocket);
        }

        Ok(pockets_with_scores)
    }

    fn assemble_pockets(
        &self,
        graph: &ProteinGraph,
        coloring: &[usize],
        boundaries: &[usize],
    ) -> Vec<Pocket> {
        let boundary_vertices =
            if boundaries.is_empty() && self.config.geometry.use_boundary_enclosure {
                crate::pocket::boundary::boundary_vertices(coloring, graph)
            } else {
                boundaries.to_vec()
            };

        let mut pocket_atoms: Vec<Vec<usize>> = vec![Vec::new(); self.config.max_pockets];
        for (idx, &color) in coloring.iter().enumerate() {
            if color < pocket_atoms.len() {
                pocket_atoms[color].push(idx);
            }
        }

        pocket_atoms
            .into_iter()
            .enumerate()
            .filter(|(_, atoms)| !atoms.is_empty())
            .map(|(color, atoms)| {
                let mut centroid = [0.0, 0.0, 0.0];
                let mut total_hydro = 0.0;
                let mut total_depth = 0.0;
                let mut total_sasa = 0.0;
                let mut total_flex = 0.0;
                let mut total_cons = 0.0;
                let mut donors = 0usize;
                let mut acceptors = 0usize;

                for &v_idx in &atoms {
                    let atom_idx = graph.atom_indices[v_idx];
                    let atom = &graph.structure_ref.atoms[atom_idx];
                    centroid[0] += atom.coord[0];
                    centroid[1] += atom.coord[1];
                    centroid[2] += atom.coord[2];
                    total_hydro += atom.hydrophobicity;
                    total_depth += atom.depth;
                    total_sasa += atom.sasa;
                    donors += usize::from(atom.is_hbond_donor());
                    acceptors += usize::from(atom.is_hbond_acceptor());
                    total_flex += atom.b_factor;
                    if let Some(res_idx) = graph.structure_ref.residues.iter().position(|r| {
                        r.seq_number == atom.residue_seq && r.chain_id == atom.chain_id
                    }) {
                        let res = &graph.structure_ref.residues[res_idx];
                        total_cons += res.conservation_score;
                    }
                }
                let count = atoms.len() as f64;
                centroid[0] /= count;
                centroid[1] /= count;
                centroid[2] /= count;
                let atom_indices: Vec<usize> = atoms
                    .iter()
                    .map(|&v_idx| graph.atom_indices[v_idx])
                    .collect();
                let bbox = bounding_box_volume(&graph.structure_ref, &atom_indices);
                let voxel = if self.config.geometry.use_voxel_volume {
                    voxel_volume(
                        &graph.structure_ref,
                        &atom_indices,
                        Some(self.config.geometry.voxel_resolution),
                        Some(self.config.geometry.probe_radius),
                    )
                } else {
                    0.0
                };
                let alpha = if self.config.geometry.use_alpha_shape_volume {
                    alpha_shape_volume(
                        &graph.structure_ref,
                        &atom_indices,
                        self.config.geometry.alpha_shape_resolution,
                        self.config.geometry.alpha_shape_shrink,
                    )
                } else {
                    0.0
                };
                let hull = if self.config.geometry.use_convex_hull_volume {
                    convex_hull_volume(
                        &graph.structure_ref,
                        &atom_indices,
                        self.config.geometry.convex_hull_epsilon,
                    )
                } else {
                    0.0
                };

                let mut volume = if self.config.geometry.use_alpha_shape_volume && alpha > 0.0 {
                    alpha
                } else if self.config.geometry.use_convex_hull_volume && hull > 0.0 {
                    hull
                } else if self.config.geometry.use_voxel_volume && voxel > 0.0 {
                    voxel
                } else {
                    bbox
                };

                if volume <= 0.0 {
                    volume = voxel.max(bbox).max(hull);
                }
                volume = volume.max(bbox);

                if self.config.geometry.use_flood_fill_cavity {
                    let cavity_vol = crate::pocket::geometry::flood_fill_cavity_volume(
                        &graph.structure_ref,
                        &atom_indices,
                        self.config.geometry.cavity_resolution,
                        self.config.geometry.probe_radius,
                    );
                    volume = volume.max(cavity_vol);
                }

                let pocket_boundary_atoms: Vec<usize> = boundary_vertices
                    .iter()
                    .copied()
                    .filter(|&v| coloring.get(v) == Some(&color))
                    .filter_map(|v| graph.atom_indices.get(v).copied())
                    .collect();

                let enclosure = if self.config.geometry.use_neighbor_enclosure {
                    crate::pocket::geometry::neighbor_enclosure(
                        &graph.structure_ref,
                        &atom_indices,
                        self.config.geometry.boundary_cutoff,
                    )
                } else if self.config.geometry.use_boundary_enclosure {
                    boundary_enclosure(&atom_indices, &pocket_boundary_atoms)
                } else {
                    enclosure_ratio(&graph.structure_ref, &atom_indices)
                };

                let residue_indices = atoms
                    .iter()
                    .filter_map(|&v_idx| {
                        let atom_idx = graph.atom_indices[v_idx];
                        let atom = &graph.structure_ref.atoms[atom_idx];
                        graph.structure_ref.residues.iter().position(|r| {
                            r.seq_number == atom.residue_seq && r.chain_id == atom.chain_id
                        })
                    })
                    .collect::<Vec<_>>();

                Pocket {
                    atom_indices,
                    residue_indices,
                    centroid,
                    volume,
                    enclosure_ratio: enclosure,
                    mean_hydrophobicity: if count > 0.0 {
                        total_hydro / count
                    } else {
                        0.0
                    },
                    mean_sasa: if count > 0.0 { total_sasa / count } else { 0.0 },
                    mean_depth: if count > 0.0 {
                        total_depth / count
                    } else {
                        0.0
                    },
                    mean_flexibility: if count > 0.0 { total_flex / count } else { 0.0 },
                    mean_conservation: if count > 0.0 { total_cons / count } else { 0.0 },
                    persistence_score: 0.0,
                    hbond_donors: donors,
                    hbond_acceptors: acceptors,
                    druggability_score: DruggabilityScore::default(),
                    boundary_atoms: pocket_boundary_atoms,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ProteinGraph, VertexFeatures};
    use crate::structure::ProteinStructure;
    use crate::LbsConfig;

    #[test]
    fn fills_boundary_atoms_from_detected_vertices() {
        let pdb = r#"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   2       2.000   0.000   0.000  1.00 10.00           C
END
"#;
        let mut structure = ProteinStructure::from_pdb_str(pdb).expect("parse pdb");
        for atom in &mut structure.atoms {
            atom.is_surface = true;
            atom.sasa = 8.0;
            atom.depth = 1.0;
            atom.hydrophobicity = 1.0;
        }
        structure.refresh_residue_properties();

        let mut features = VertexFeatures::new(2);
        features.hydrophobicity = vec![1.0, 1.0];
        features.depth = vec![1.0, 1.0];

        let graph = ProteinGraph {
            atom_indices: vec![0, 1],
            adjacency: vec![vec![1], vec![0]],
            edge_weights: vec![vec![1.0], vec![1.0]],
            vertex_features: features,
            structure_ref: structure,
        };

        let coloring = vec![0, 1];
        let detector = PocketDetector::new(LbsConfig::default()).expect("detector");
        let pockets = detector.assemble_pockets(&graph, &coloring, &[]);

        assert_eq!(pockets.len(), 2);
        let mut boundary_atoms: Vec<Vec<usize>> =
            pockets.iter().map(|p| p.boundary_atoms.clone()).collect();
        boundary_atoms.sort_by_key(|v| v[0]);
        assert_eq!(boundary_atoms[0], vec![0]);
        assert_eq!(boundary_atoms[1], vec![1]);
    }
}
