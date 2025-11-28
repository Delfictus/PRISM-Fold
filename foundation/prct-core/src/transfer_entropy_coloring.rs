//! Transfer Entropy-Guided Graph Coloring
//!
//! Uses information flow analysis to guide vertex ordering in graph coloring.
//! Vertices with high information centrality are colored first to minimize conflicts.

use crate::errors::*;
use crate::geodesic::GeodesicFeatures;
use ndarray::Array2;
#[cfg(feature = "cuda")]
use neuromorphic_engine::{TransferEntropyConfig, TransferEntropyEngine};
use shared_types::*;
use std::collections::HashMap;

/// Compute transfer entropy-based vertex ordering
///
/// Returns vertices sorted by information centrality (highest first).
/// High-centrality vertices are "information hubs" that should be colored first.
///
/// # Arguments
/// * `graph` - Input graph
/// * `kuramoto_state` - Kuramoto phase synchronization state
/// * `geodesic_features` - Optional geodesic features for tie-breaking
/// * `geodesic_weight` - Weight for geodesic features (0.0 = TE only, 1.0 = geodesic only)
pub fn compute_transfer_entropy_ordering(
    graph: &Graph,
    kuramoto_state: &KuramotoState,
    geodesic_features: Option<&GeodesicFeatures>,
    geodesic_weight: f64,
) -> Result<Vec<usize>> {
    let n = graph.num_vertices;

    println!(
        "[TE-COLORING] Computing transfer entropy ordering for {} vertices",
        n
    );

    // Use adjacency-based TE computation (faster and more robust for graph coloring)
    let te_matrix = compute_te_from_adjacency(graph)?;

    println!("[TE-COLORING] Transfer entropy matrix computed");

    // Compute information centrality for each vertex
    let mut centrality: Vec<(usize, f64)> = (0..n)
        .map(|v| {
            // Total information flow (outgoing + incoming)
            let outgoing: f64 = (0..n).map(|u| te_matrix[[v, u]]).sum();
            let incoming: f64 = (0..n).map(|u| te_matrix[[u, v]]).sum();
            let te_score = outgoing + incoming;

            // Blend with geodesic features if provided
            let blended_score = if let Some(geo) = geodesic_features {
                let geo_score = geo.compute_score(v, 0.5, 0.5);
                (1.0 - geodesic_weight) * te_score + geodesic_weight * geo_score
            } else {
                te_score
            };

            (v, blended_score)
        })
        .collect();

    // Sort by centrality (descending)
    centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let ordering: Vec<usize> = centrality.iter().map(|(v, _)| *v).collect();

    // Log statistics
    let max_centrality = centrality.first().map(|(_, c)| *c).unwrap_or(0.0);
    let min_centrality = centrality.last().map(|(_, c)| *c).unwrap_or(0.0);
    let avg_centrality: f64 = centrality.iter().map(|(_, c)| c).sum::<f64>() / n as f64;

    println!(
        "[TE-COLORING] Centrality range: [{:.4}, {:.4}], avg: {:.4}",
        min_centrality, max_centrality, avg_centrality
    );
    println!(
        "[TE-COLORING] Top 5 hub vertices: {:?}",
        &ordering[..5.min(n)]
    );

    Ok(ordering)
}

/// Generate time series for each vertex based on graph dynamics
///
/// Uses Kuramoto phases + neighbor interactions to create dynamic time series
fn generate_vertex_time_series(
    graph: &Graph,
    kuramoto_state: &KuramotoState,
    n: usize,
) -> Result<Vec<Vec<f64>>> {
    const TIME_STEPS: usize = 100; // Length of time series

    let mut time_series = vec![Vec::with_capacity(TIME_STEPS); n];

    // Initialize with Kuramoto phases
    let phases = if kuramoto_state.phases.len() >= n {
        kuramoto_state.phases[..n].to_vec()
    } else {
        // Fallback: use vertex indices normalized
        (0..n)
            .map(|i| (i as f64 / n as f64) * 2.0 * std::f64::consts::PI)
            .collect()
    };

    // Simulate dynamics to generate time series
    let mut current_phases = phases.clone();

    for t in 0..TIME_STEPS {
        // Record current state
        for v in 0..n {
            time_series[v].push(current_phases[v]);
        }

        // Update phases based on neighbor coupling
        let mut next_phases = current_phases.clone();

        for v in 0..n {
            // Neighbor coupling (simplified Kuramoto update)
            let mut coupling_sum = 0.0;
            let mut neighbor_count = 0;

            for u in 0..n {
                if graph.adjacency[v * n + u] {
                    coupling_sum += (current_phases[u] - current_phases[v]).sin();
                    neighbor_count += 1;
                }
            }

            if neighbor_count > 0 {
                let coupling = 0.5 * coupling_sum / neighbor_count as f64;
                let natural_freq = phases[v]; // Use initial phase as natural frequency

                // Kuramoto equation: dθ/dt = ω + K Σ sin(θ_j - θ_i)
                next_phases[v] = current_phases[v] + 0.1 * (natural_freq + coupling);

                // Keep in [0, 2π]
                next_phases[v] = next_phases[v] % (2.0 * std::f64::consts::PI);
            }
        }

        current_phases = next_phases;
    }

    Ok(time_series)
}

/// Compute transfer entropy matrix from graph structure directly
///
/// Alternative approach: Use adjacency matrix patterns as "time series"
pub fn compute_te_from_adjacency(graph: &Graph) -> Result<Array2<f64>> {
    let n = graph.num_vertices;
    let mut te_matrix = Array2::zeros((n, n));

    // Compute vertex degrees
    let degrees: Vec<usize> = (0..n)
        .map(|v| (0..n).filter(|&u| graph.adjacency[v * n + u]).count())
        .collect();

    // For each pair of vertices, compute pseudo-transfer-entropy
    // based on neighborhood overlap and degree centrality
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }

            // Count common neighbors
            let mut common_neighbors = 0;
            let mut i_unique = 0;
            let mut j_unique = 0;

            for k in 0..n {
                let i_connected = graph.adjacency[i * n + k];
                let j_connected = graph.adjacency[j * n + k];

                if i_connected && j_connected {
                    common_neighbors += 1;
                } else if i_connected {
                    i_unique += 1;
                } else if j_connected {
                    j_unique += 1;
                }
            }

            // TE(i→j): Information flow from i to j
            // High if:
            // 1. i has many neighbors (high degree)
            // 2. i and j share neighbors (coupling)
            // 3. i is connected to j (direct influence)

            let degree_factor = degrees[i] as f64 / (n as f64);
            let overlap_factor = if degrees[i] > 0 {
                common_neighbors as f64 / degrees[i] as f64
            } else {
                0.0
            };
            let direct_influence = if graph.adjacency[i * n + j] { 1.0 } else { 0.0 };

            // Weighted combination: degree (50%) + overlap (30%) + direct (20%)
            te_matrix[[i, j]] = 0.5 * degree_factor + 0.3 * overlap_factor + 0.2 * direct_influence;
        }
    }

    Ok(te_matrix)
}

/// Hybrid: Combine Kuramoto ordering with Transfer Entropy
///
/// Uses both phase synchronization and information flow for best results
///
/// # Arguments
/// * `graph` - Input graph
/// * `kuramoto_state` - Kuramoto phase synchronization state
/// * `geodesic_features` - Optional geodesic features for tie-breaking
/// * `geodesic_weight` - Weight for geodesic features (default: 0.0 for backward compatibility)
pub fn hybrid_te_kuramoto_ordering(
    graph: &Graph,
    kuramoto_state: &KuramotoState,
    geodesic_features: Option<&GeodesicFeatures>,
    geodesic_weight: f64,
) -> Result<Vec<usize>> {
    let n = graph.num_vertices;

    // Get both orderings
    let te_ordering = compute_transfer_entropy_ordering(
        graph,
        kuramoto_state,
        geodesic_features,
        geodesic_weight,
    )?;

    // Compute Kuramoto ordering
    let mut kuramoto_ordering: Vec<(usize, f64)> = kuramoto_state
        .phases
        .iter()
        .take(n)
        .enumerate()
        .map(|(i, &phase)| (i, phase))
        .collect();
    kuramoto_ordering.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Hybrid score: 70% TE centrality + 30% Kuramoto phase
    let te_rank_map: HashMap<usize, usize> = te_ordering
        .iter()
        .enumerate()
        .map(|(rank, &v)| (v, rank))
        .collect();

    let kuramoto_rank_map: HashMap<usize, usize> = kuramoto_ordering
        .iter()
        .enumerate()
        .map(|(rank, &(v, _))| (v, rank))
        .collect();

    let mut hybrid_scores: Vec<(usize, f64)> = (0..n)
        .map(|v| {
            let te_rank = te_rank_map.get(&v).copied().unwrap_or(n);
            let kuramoto_rank = kuramoto_rank_map.get(&v).copied().unwrap_or(n);
            let te_score = 1.0 - (te_rank as f64 / n as f64); // Normalize to [0,1]
            let kuramoto_score = 1.0 - (kuramoto_rank as f64 / n as f64);
            let hybrid = 0.7 * te_score + 0.3 * kuramoto_score;
            (v, hybrid)
        })
        .collect();

    hybrid_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(hybrid_scores.into_iter().map(|(v, _)| v).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_te_ordering_triangle() {
        // Triangle: all vertices have equal centrality
        let graph = Graph {
            num_vertices: 3,
            num_edges: 3,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
            adjacency: vec![false, true, true, true, false, true, true, true, false],
            coordinates: None,
        };

        let kuramoto = KuramotoState {
            phases: vec![0.0, 1.0, 2.0],
            natural_frequencies: vec![1.0, 1.0, 1.0],
            coupling_matrix: vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            order_parameter: 0.5,
            mean_phase: 1.0,
        };

        let ordering = compute_transfer_entropy_ordering(&graph, &kuramoto).unwrap();
        assert_eq!(ordering.len(), 3);
    }
}
