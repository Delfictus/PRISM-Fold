//! Alternative Initial Coloring Strategies
//!
//! Provides multiple approaches to generating initial vertex orderings
//! before applying greedy coloring:
//! - Greedy: Standard degree-based ordering
//! - Spectral: Laplacian eigenvector ordering
//! - Community: Louvain/label propagation clustering
//! - Randomized: Multiple greedy runs with random tie-breaks

use crate::coloring::greedy_coloring_with_ordering;
use crate::errors::*;
use ndarray::{Array1, Array2};
use rand::Rng;
use shared_types::{ColoringSolution, Graph};
use std::collections::HashMap;

/// Compute vertex degree from adjacency matrix
fn vertex_degree(graph: &Graph, v: usize) -> usize {
    let n = graph.num_vertices;
    (0..n).filter(|&u| graph.adjacency[v * n + u]).count()
}

/// Collect neighbors of vertex v into provided vector
fn collect_neighbors(graph: &Graph, v: usize, neighbors: &mut Vec<usize>) {
    neighbors.clear();
    let n = graph.num_vertices;
    for u in 0..n {
        if graph.adjacency[v * n + u] {
            neighbors.push(u);
        }
    }
}

/// Initial coloring strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InitialColoringStrategy {
    /// Standard greedy with degree ordering
    Greedy,

    /// Laplacian eigenvector ordering
    Spectral,

    /// Community detection (label propagation)
    Community,

    /// Multiple randomized runs (best of N)
    Randomized,
}

/// Compute initial coloring using specified strategy
pub fn compute_initial_coloring(
    graph: &Graph,
    strategy: InitialColoringStrategy,
) -> Result<ColoringSolution> {
    match strategy {
        InitialColoringStrategy::Greedy => greedy_ordering(graph),
        InitialColoringStrategy::Spectral => spectral_ordering(graph),
        InitialColoringStrategy::Community => community_ordering(graph),
        InitialColoringStrategy::Randomized => randomized_ordering(graph, 10),
    }
}

/// Standard greedy coloring with degree ordering
fn greedy_ordering(graph: &Graph) -> Result<ColoringSolution> {
    let n = graph.num_vertices;
    let mut ordering: Vec<usize> = (0..n).collect();

    // Sort by degree (descending)
    ordering.sort_by_key(|&v| std::cmp::Reverse(vertex_degree(graph, v)));

    greedy_coloring_with_ordering(graph, &ordering)
}

/// Spectral ordering using Laplacian eigenvector
fn spectral_ordering(graph: &Graph) -> Result<ColoringSolution> {
    let n = graph.num_vertices;

    // Build Laplacian matrix: L = D - A
    let mut laplacian = Array2::<f64>::zeros((n, n));
    let mut scratch = Vec::new();

    for v in 0..n {
        let degree = vertex_degree(graph, v) as f64;
        laplacian[[v, v]] = degree;

        collect_neighbors(graph, v, &mut scratch);
        for &u in &scratch {
            laplacian[[v, u]] = -1.0;
        }
    }

    // Compute second smallest eigenvector (Fiedler vector)
    // For now, use simple power iteration approximation
    let fiedler = compute_fiedler_vector(&laplacian, n);

    // Sort vertices by Fiedler vector values
    let mut ordering: Vec<usize> = (0..n).collect();
    ordering.sort_by(|&a, &b| {
        fiedler[a]
            .partial_cmp(&fiedler[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    greedy_coloring_with_ordering(graph, &ordering)
}

/// Approximate Fiedler vector using power iteration
fn compute_fiedler_vector(laplacian: &Array2<f64>, n: usize) -> Vec<f64> {
    let mut v = Array1::<f64>::from_vec(vec![1.0; n]);

    // Remove constant eigenvector component
    let mean = v.mean().unwrap_or(0.0);
    v.mapv_inplace(|x| x - mean);

    // Power iteration (simplified)
    for _ in 0..100 {
        let mut v_new = Array1::<f64>::zeros(n);

        for i in 0..n {
            for j in 0..n {
                v_new[i] += laplacian[[i, j]] * v[j];
            }
        }

        // Normalize
        let norm = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            v_new /= norm;
        }

        // Remove constant component
        let mean = v_new.mean().unwrap_or(0.0);
        v_new.mapv_inplace(|x| x - mean);

        v = v_new;
    }

    v.to_vec()
}

/// Community-based ordering using label propagation
fn community_ordering(graph: &Graph) -> Result<ColoringSolution> {
    let n = graph.num_vertices;
    let mut labels: Vec<usize> = (0..n).collect();

    // Label propagation
    let mut scratch = Vec::new();
    for _ in 0..20 {
        let mut new_labels = labels.clone();

        for v in 0..n {
            let mut label_counts: HashMap<usize, usize> = HashMap::new();

            collect_neighbors(graph, v, &mut scratch);
            for &u in &scratch {
                *label_counts.entry(labels[u]).or_insert(0) += 1;
            }

            if let Some(&most_common) = label_counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(label, _)| label)
            {
                new_labels[v] = most_common;
            }
        }

        labels = new_labels;
    }

    // Group vertices by community, sort by community size
    let mut community_sizes: HashMap<usize, Vec<usize>> = HashMap::new();
    for (v, &label) in labels.iter().enumerate() {
        community_sizes
            .entry(label)
            .or_insert_with(Vec::new)
            .push(v);
    }

    let mut ordering = Vec::with_capacity(n);
    let mut sorted_communities: Vec<_> = community_sizes.into_iter().collect();
    sorted_communities.sort_by_key(|(_, vertices)| std::cmp::Reverse(vertices.len()));

    for (_, vertices) in sorted_communities {
        ordering.extend(vertices);
    }

    greedy_coloring_with_ordering(graph, &ordering)
}

/// Randomized ordering (best of N runs)
fn randomized_ordering(graph: &Graph, num_runs: usize) -> Result<ColoringSolution> {
    let n = graph.num_vertices;
    let mut best_solution: Option<ColoringSolution> = None;
    let mut rng = rand::thread_rng();

    for _ in 0..num_runs {
        let mut ordering: Vec<usize> = (0..n).collect();

        // Sort by degree with random tie-breaking
        ordering.sort_by_key(|&v| {
            let degree = vertex_degree(graph, v);
            let noise: u32 = rng.gen_range(0..100);
            (std::cmp::Reverse(degree), noise)
        });

        let solution = greedy_coloring_with_ordering(graph, &ordering)?;

        if let Some(ref best) = best_solution {
            if solution.chromatic_number < best.chromatic_number {
                best_solution = Some(solution);
            }
        } else {
            best_solution = Some(solution);
        }
    }

    best_solution.ok_or_else(|| PRCTError::ColoringFailed("No solution found".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_strategy() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(3, 4, 1.0);

        let solution = compute_initial_coloring(&graph, InitialColoringStrategy::Greedy)
            .expect("Failed to color");

        assert_eq!(solution.conflicts, 0);
        assert!(solution.chromatic_number <= 5);
    }

    #[test]
    fn test_randomized_strategy() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);

        let solution = compute_initial_coloring(&graph, InitialColoringStrategy::Randomized)
            .expect("Failed to color");

        assert_eq!(solution.conflicts, 0);
    }
}
