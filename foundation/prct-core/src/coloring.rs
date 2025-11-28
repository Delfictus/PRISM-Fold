//! Phase-Guided Graph Coloring
//!
//! Pure algorithm: Uses quantum phase field and Kuramoto synchronization
//! to guide graph coloring decisions.

use crate::errors::*;
use ndarray::Array2;
use rayon::prelude::*;
use shared_types::*;
use std::collections::{HashMap, HashSet};

/// Phase-guided graph coloring algorithm
///
/// Uses quantum phase field coherence and Kuramoto synchronization
/// to guide color assignment decisions.
pub fn phase_guided_coloring(
    graph: &Graph,
    phase_field: &PhaseField,
    kuramoto_state: &KuramotoState,
    target_colors: usize,
) -> Result<ColoringSolution> {
    let start = std::time::Instant::now();
    let n = graph.num_vertices;

    if n == 0 {
        return Err(PRCTError::InvalidGraph("Empty graph".into()));
    }

    // Build adjacency matrix
    let adjacency = build_adjacency_matrix(graph);

    // Order vertices by Kuramoto phase (synchronized vertices colored together)
    // NOTE: Kuramoto state may have more phases than vertices (includes neuro+quantum),
    // so we only take the first n phases corresponding to graph vertices
    let mut vertices_by_phase: Vec<(usize, f64)> = kuramoto_state
        .phases
        .iter()
        .take(n) // Only use first n phases for n vertices
        .enumerate()
        .map(|(i, &phase)| (i, phase))
        .collect();

    vertices_by_phase.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

    // Color vertices in phase order
    let mut coloring = vec![usize::MAX; n];

    for (vertex, _phase) in vertices_by_phase {
        if coloring[vertex] != usize::MAX {
            continue; // Already colored
        }

        // Find best color using phase coherence
        let color =
            find_phase_coherent_color(vertex, &coloring, &adjacency, phase_field, target_colors)?;

        coloring[vertex] = color;
    }

    // Count conflicts
    let conflicts = count_conflicts(&coloring, &adjacency);

    // Compute quality score (lower chromatic number = better)
    let colors_used = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);
    let quality_score = 1.0 - (colors_used as f64 / target_colors as f64).min(1.0);

    let computation_time = start.elapsed().as_secs_f64() * 1000.0;

    Ok(ColoringSolution {
        colors: coloring,
        chromatic_number: colors_used,
        conflicts,
        quality_score,
        computation_time_ms: computation_time,
    })
}

/// Find color that maximizes phase coherence
fn find_phase_coherent_color(
    vertex: usize,
    coloring: &[usize],
    adjacency: &Array2<bool>,
    phase_field: &PhaseField,
    max_colors: usize,
) -> Result<usize> {
    let n = coloring.len();

    // Forbidden colors (used by neighbors)
    let forbidden: HashSet<usize> = (0..n)
        .filter(|&u| adjacency[[vertex, u]] && coloring[u] != usize::MAX)
        .map(|u| coloring[u])
        .collect();

    // Score each available color by phase coherence
    let mut color_scores: Vec<(usize, f64)> = Vec::new();

    for color in 0..max_colors {
        if forbidden.contains(&color) {
            continue;
        }

        // Compute phase coherence with vertices of this color
        let score = compute_color_coherence_score(vertex, color, coloring, phase_field);
        color_scores.push((color, score));
    }

    if color_scores.is_empty() {
        return Err(PRCTError::ColoringFailed(format!(
            "Vertex {} has no available colors (need > {} colors)",
            vertex, max_colors
        )));
    }

    // Select color with highest coherence
    color_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    Ok(color_scores[0].0)
}

/// Compute phase coherence score for assigning a vertex to a color
fn compute_color_coherence_score(
    vertex: usize,
    color: usize,
    coloring: &[usize],
    phase_field: &PhaseField,
) -> f64 {
    let n = coloring.len();

    // Find vertices already assigned this color
    let same_color_vertices: Vec<usize> = (0..n).filter(|&u| coloring[u] == color).collect();

    if same_color_vertices.is_empty() {
        return 1.0; // New color - neutral score
    }

    // Average phase coherence with same-colored vertices
    let mut total_coherence = 0.0;

    for &u in &same_color_vertices {
        // Phase coherence from field
        let coherence = get_phase_coherence(phase_field, vertex, u);
        total_coherence += coherence;
    }

    total_coherence / same_color_vertices.len() as f64
}

/// Get phase coherence between two vertices
fn get_phase_coherence(phase_field: &PhaseField, i: usize, j: usize) -> f64 {
    let n = (phase_field.coherence_matrix.len() as f64).sqrt() as usize;

    if i >= n || j >= n {
        return 0.0;
    }

    phase_field.coherence_matrix[i * n + j]
}

/// Build adjacency matrix from graph edge list
fn build_adjacency_matrix(graph: &Graph) -> Array2<bool> {
    let n = graph.num_vertices;
    let mut adjacency = Array2::from_elem((n, n), false);

    for i in 0..n {
        for j in 0..n {
            adjacency[[i, j]] = graph.adjacency[i * n + j];
        }
    }

    adjacency
}

/// Count coloring conflicts
fn count_conflicts(coloring: &[usize], adjacency: &Array2<bool>) -> usize {
    let n = coloring.len();
    let mut conflicts = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            if adjacency[[i, j]] && coloring[i] == coloring[j] {
                conflicts += 1;
            }
        }
    }

    conflicts
}

/// Simple greedy coloring with custom vertex ordering
///
/// Colors vertices in the given order using a greedy first-fit algorithm
pub fn greedy_coloring_with_ordering(
    graph: &Graph,
    vertex_order: &[usize],
) -> Result<ColoringSolution> {
    let start = std::time::Instant::now();
    let n = graph.num_vertices;

    if n == 0 {
        return Err(PRCTError::InvalidGraph("Empty graph".into()));
    }

    // Build adjacency matrix
    let adjacency = build_adjacency_matrix(graph);

    // Initialize coloring
    let mut coloring = vec![usize::MAX; n];

    // Color vertices in the given order
    for &vertex in vertex_order {
        if vertex >= n {
            return Err(PRCTError::InvalidGraph(format!(
                "Vertex {} out of range",
                vertex
            )));
        }

        // Find forbidden colors (used by neighbors)
        let mut forbidden = vec![false; n];
        for neighbor in 0..n {
            if adjacency[[vertex, neighbor]] && coloring[neighbor] != usize::MAX {
                forbidden[coloring[neighbor]] = true;
            }
        }

        // Find first available color
        let mut color = 0;
        while color < n && forbidden[color] {
            color += 1;
        }

        coloring[vertex] = color;
    }

    // Count conflicts (should be 0 for greedy)
    let conflicts = count_conflicts(&coloring, &adjacency);

    // Compute chromatic number
    let colors_used = coloring
        .iter()
        .filter(|&&c| c != usize::MAX)
        .map(|&c| c + 1)
        .max()
        .unwrap_or(0);

    let computation_time = start.elapsed().as_secs_f64() * 1000.0;

    Ok(ColoringSolution {
        colors: coloring,
        chromatic_number: colors_used,
        conflicts,
        quality_score: 1.0 - (conflicts as f64 / graph.num_edges as f64).min(1.0),
        computation_time_ms: computation_time,
    })
}

#[cfg(test)]
mod tests {
    use crate::coupling::PhysicsCouplingService;

    #[test]
    fn test_order_parameter_synchronized() {
        let phases = vec![0.0; 10];
        let order = PhysicsCouplingService::compute_order_parameter(&phases);
        assert!((order - 1.0).abs() < 0.01); // Perfect sync
    }

    #[test]
    fn test_order_parameter_random() {
        let phases: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.137) % (2.0 * core::f64::consts::PI))
            .collect();
        let order = PhysicsCouplingService::compute_order_parameter(&phases);
        assert!(order < 0.3); // Low sync
    }

    #[test]
    fn test_kuramoto_evolution() {
        let mut service = PhysicsCouplingService::new(1.0);
        let mut phases = vec![0.0, 1.0, 2.0, 3.0];
        let frequencies = vec![1.0; 4];

        let initial_order = PhysicsCouplingService::compute_order_parameter(&phases);

        // Evolve
        for _ in 0..100 {
            service
                .kuramoto_step(&mut phases, &frequencies, 0.1)
                .unwrap();
        }

        let final_order = PhysicsCouplingService::compute_order_parameter(&phases);
        assert!(final_order > initial_order); // Should synchronize
    }
}
