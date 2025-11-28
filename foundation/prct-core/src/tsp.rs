//! Phase-Guided TSP Tour Construction
//!
//! Builds TSP tours within each color class using quantum phase field guidance.

use crate::errors::*;
use shared_types::*;
use std::collections::HashSet;

/// Build TSP tours for each color class using phase guidance
pub fn phase_guided_tsp(
    graph: &Graph,
    coloring: &ColoringSolution,
    phase_field: &PhaseField,
) -> Result<Vec<TSPSolution>> {
    let num_colors = coloring.chromatic_number;
    let mut tours = Vec::with_capacity(num_colors);

    for color in 0..num_colors {
        // Get vertices in this color class
        let vertices: Vec<usize> = coloring
            .colors
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == color)
            .map(|(i, _)| i)
            .collect();

        if vertices.is_empty() {
            continue;
        }

        // Build TSP tour for this color class
        let tour = build_phase_guided_tour(graph, &vertices, phase_field)?;
        tours.push(tour);
    }

    Ok(tours)
}

/// Build TSP tour using phase-guided nearest neighbor
fn build_phase_guided_tour(
    graph: &Graph,
    vertices: &[usize],
    phase_field: &PhaseField,
) -> Result<TSPSolution> {
    let start_time = std::time::Instant::now();

    if vertices.is_empty() {
        return Err(PRCTError::TSPFailed("Empty vertex set".into()));
    }

    if vertices.len() == 1 {
        return Ok(TSPSolution {
            tour: vertices.to_vec(),
            tour_length: 0.0,
            num_vertices: 1,
            quality_score: 1.0,
            computation_time_ms: 0.0,
        });
    }

    let mut tour = Vec::with_capacity(vertices.len());
    let mut unvisited: HashSet<usize> = vertices.iter().copied().collect();

    // Start with vertex having smallest phase (most synchronized)
    let start_vertex = *vertices
        .iter()
        .min_by(|&&a, &&b| {
            let phase_a = get_vertex_phase(phase_field, a);
            let phase_b = get_vertex_phase(phase_field, b);
            phase_a
                .partial_cmp(&phase_b)
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .unwrap();

    tour.push(start_vertex);
    unvisited.remove(&start_vertex);

    // Phase-guided nearest neighbor
    while !unvisited.is_empty() {
        let current = *tour.last().unwrap();

        // Find next vertex: maximize (phase coherence × edge weight)
        let next = *unvisited
            .iter()
            .max_by(|&&a, &&b| {
                let score_a = compute_edge_score(graph, phase_field, current, a);
                let score_b = compute_edge_score(graph, phase_field, current, b);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .unwrap();

        tour.push(next);
        unvisited.remove(&next);
    }

    // Compute tour length
    let tour_length = compute_tour_length(graph, &tour);

    // Quality score (lower length = higher quality)
    // Normalize by number of vertices
    let quality_score = 1.0 / (1.0 + tour_length / vertices.len() as f64);

    let computation_time = start_time.elapsed().as_secs_f64() * 1000.0;

    Ok(TSPSolution {
        tour,
        tour_length,
        num_vertices: vertices.len(),
        quality_score,
        computation_time_ms: computation_time,
    })
}

/// Compute edge score for TSP construction
///
/// Score = phase_alignment × edge_weight
fn compute_edge_score(graph: &Graph, phase_field: &PhaseField, from: usize, to: usize) -> f64 {
    // Phase alignment: vertices with similar phases should be close in tour
    let phase_from = get_vertex_phase(phase_field, from);
    let phase_to = get_vertex_phase(phase_field, to);
    let phase_diff = (phase_from - phase_to).abs();
    let phase_alignment = (-phase_diff).exp(); // Gaussian decay

    // Edge weight from graph
    let edge_weight = get_edge_weight(graph, from, to);

    phase_alignment * edge_weight
}

/// Get vertex phase from phase field
fn get_vertex_phase(phase_field: &PhaseField, vertex: usize) -> f64 {
    if vertex < phase_field.phases.len() {
        phase_field.phases[vertex]
    } else {
        0.0
    }
}

/// Get edge weight between vertices
fn get_edge_weight(graph: &Graph, from: usize, to: usize) -> f64 {
    graph
        .edges
        .iter()
        .find(|(u, v, _)| (*u == from && *v == to) || (*u == to && *v == from))
        .map(|(_, _, w)| *w)
        .unwrap_or(1.0) // Default weight if edge not found
}

/// Compute total tour length
fn compute_tour_length(graph: &Graph, tour: &[usize]) -> f64 {
    if tour.len() < 2 {
        return 0.0;
    }

    let mut length = 0.0;

    for i in 0..tour.len() {
        let from = tour[i];
        let to = tour[(i + 1) % tour.len()];
        length += get_edge_distance(graph, from, to);
    }

    length
}

/// Get distance between two vertices
fn get_edge_distance(graph: &Graph, from: usize, to: usize) -> f64 {
    // Try edge weight first
    if let Some((_, _, weight)) = graph
        .edges
        .iter()
        .find(|(u, v, _)| (*u == from && *v == to) || (*u == to && *v == from))
    {
        return *weight;
    }

    // Use Euclidean distance if coordinates available
    if let Some(ref coords) = graph.coordinates {
        if from < coords.len() && to < coords.len() {
            let (x1, y1) = coords[from];
            let (x2, y2) = coords[to];
            return ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
        }
    }

    // Default: unit distance
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coloring::phase_guided_coloring;

    #[test]
    fn test_simple_coloring() {
        // Triangle graph
        let graph = Graph {
            num_vertices: 3,
            num_edges: 3,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
            adjacency: vec![false, true, true, true, false, true, true, true, false],
            coordinates: None,
        };

        let phase_field = PhaseField {
            phases: vec![0.0, 2.0, 4.0],
            coherence_matrix: vec![1.0; 9],
            order_parameter: 0.9,
            resonance_frequency: 50.0,
        };

        let kuramoto = KuramotoState {
            phases: vec![0.0, 2.0, 4.0],
            natural_frequencies: vec![1.0; 3],
            coupling_matrix: vec![0.5; 9],
            order_parameter: 0.95,
            mean_phase: 2.0,
        };

        let solution = phase_guided_coloring(&graph, &phase_field, &kuramoto, 3).unwrap();

        // Triangle needs 3 colors
        assert_eq!(solution.chromatic_number, 3);
        assert_eq!(solution.conflicts, 0);
    }

    #[test]
    fn test_tsp_tour() {
        let graph = Graph {
            num_vertices: 4,
            num_edges: 4,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
            adjacency: vec![
                false, true, false, true, true, false, true, false, false, true, false, true, true,
                false, true, false,
            ],
            coordinates: Some(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
        };

        let phase_field = PhaseField {
            phases: vec![0.0, 1.57, 3.14, 4.71],
            coherence_matrix: vec![0.8; 16],
            order_parameter: 0.9,
            resonance_frequency: 50.0,
        };

        let vertices = vec![0, 1, 2, 3];
        let tour = build_phase_guided_tour(&graph, &vertices, &phase_field).unwrap();

        assert_eq!(tour.num_vertices, 4);
        assert!(tour.tour_length > 0.0);
        assert_eq!(tour.tour.len(), 4);
    }
}
