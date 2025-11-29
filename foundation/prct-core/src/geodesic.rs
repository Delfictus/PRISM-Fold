//! Geodesic Features for Graph Coloring
//!
//! Computes landmark-based geodesic distances for tie-breaking in DSATUR.
//! Provides structural priors that complement transfer entropy and Kuramoto phases.

use crate::errors::*;
use shared_types::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Geodesic features for vertices
#[derive(Debug, Clone)]
pub struct GeodesicFeatures {
    /// Distance to each landmark for each vertex: [num_vertices x num_landmarks]
    pub landmark_distances: Vec<Vec<f64>>,

    /// Landmark vertices (indices)
    pub landmarks: Vec<usize>,

    /// Geodesic centrality (sum of distances to all landmarks, normalized)
    pub centrality: Vec<f64>,

    /// Geodesic eccentricity (max distance to any landmark)
    pub eccentricity: Vec<f64>,
}

impl GeodesicFeatures {
    /// Compute combined geodesic score for tie-breaking
    ///
    /// Higher score = higher priority for coloring
    pub fn compute_score(&self, v: usize, centrality_weight: f64, eccentricity_weight: f64) -> f64 {
        if v >= self.centrality.len() {
            return 0.0;
        }

        // Invert centrality (lower distance sum = higher priority)
        let centrality_score = 1.0 / (1.0 + self.centrality[v]);

        // Invert eccentricity (lower max distance = higher priority)
        let eccentricity_score = 1.0 / (1.0 + self.eccentricity[v]);

        centrality_weight * centrality_score + eccentricity_weight * eccentricity_score
    }
}

/// Compute landmark-based geodesic distances
///
/// # Arguments
/// * `graph` - Input graph
/// * `num_landmarks` - Number of landmarks to select
/// * `metric` - Distance metric: "hop" (unweighted) or "weighted"
///
/// # Returns
/// Geodesic features for all vertices
pub fn compute_landmark_distances(
    graph: &Graph,
    num_landmarks: usize,
    metric: &str,
) -> Result<GeodesicFeatures> {
    let n = graph.num_vertices;

    if num_landmarks == 0 {
        return Err(PRCTError::ColoringFailed(
            "num_landmarks must be > 0".to_string(),
        ));
    }

    let num_landmarks = num_landmarks.min(n);

    println!(
        "[GEODESIC] Computing geodesic features: {} landmarks, metric={}",
        num_landmarks, metric
    );

    // Select landmarks (degree-based selection for diversity)
    let landmarks = select_landmarks(graph, num_landmarks)?;

    println!(
        "[GEODESIC] Selected landmarks: {:?}",
        &landmarks[..5.min(landmarks.len())]
    );

    // Compute distances from each landmark to all vertices
    let mut landmark_distances = vec![vec![0.0; num_landmarks]; n];

    for (lm_idx, &landmark) in landmarks.iter().enumerate() {
        let distances = if metric == "weighted" {
            compute_dijkstra_distances(graph, landmark)?
        } else {
            compute_hop_distances(graph, landmark)?
        };

        for v in 0..n {
            landmark_distances[v][lm_idx] = distances[v];
        }
    }

    // Compute centrality (sum of distances to all landmarks)
    let mut centrality = vec![0.0; n];
    let mut max_centrality: f64 = 0.0;

    for v in 0..n {
        let sum: f64 = landmark_distances[v].iter().sum();
        centrality[v] = sum;
        max_centrality = max_centrality.max(sum);
    }

    // Normalize centrality to [0, 1]
    if max_centrality > 0.0 {
        for v in 0..n {
            centrality[v] /= max_centrality;
        }
    }

    // Compute eccentricity (max distance to any landmark)
    let mut eccentricity = vec![0.0; n];
    let mut max_eccentricity: f64 = 0.0;

    for v in 0..n {
        let max_dist = landmark_distances[v]
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0);
        eccentricity[v] = max_dist;
        max_eccentricity = max_eccentricity.max(max_dist);
    }

    // Normalize eccentricity to [0, 1]
    if max_eccentricity > 0.0 {
        for v in 0..n {
            eccentricity[v] /= max_eccentricity;
        }
    }

    println!("[GEODESIC] ✅ Computed features: centrality range [{:.3}, {:.3}], eccentricity range [{:.3}, {:.3}]",
             centrality.iter().copied().min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal)).unwrap_or(0.0),
             centrality.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal)).unwrap_or(0.0),
             eccentricity.iter().copied().min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal)).unwrap_or(0.0),
             eccentricity.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal)).unwrap_or(0.0));

    Ok(GeodesicFeatures {
        landmark_distances,
        landmarks,
        centrality,
        eccentricity,
    })
}

/// Select landmarks using degree-based heuristic
///
/// Selects vertices with diverse degrees to maximize coverage
fn select_landmarks(graph: &Graph, num_landmarks: usize) -> Result<Vec<usize>> {
    let n = graph.num_vertices;

    // Compute degrees
    let mut degrees: Vec<(usize, usize)> = (0..n)
        .map(|v| {
            let degree = (0..n).filter(|&u| graph.adjacency[v * n + u]).count();
            (v, degree)
        })
        .collect();

    // Sort by degree (descending)
    degrees.sort_by(|a, b| b.1.cmp(&a.1));

    // Select landmarks: stratified sampling across degree distribution
    let mut landmarks = Vec::with_capacity(num_landmarks);
    let step = n / num_landmarks.max(1);

    for i in 0..num_landmarks {
        let idx = (i * step).min(n - 1);
        landmarks.push(degrees[idx].0);
    }

    Ok(landmarks)
}

/// Compute hop distances (unweighted shortest paths) using BFS
fn compute_hop_distances(graph: &Graph, source: usize) -> Result<Vec<f64>> {
    use std::collections::VecDeque;

    let n = graph.num_vertices;
    let mut distances = vec![f64::INFINITY; n];
    let mut queue = VecDeque::new();

    distances[source] = 0.0;
    queue.push_back(source);

    while let Some(v) = queue.pop_front() {
        let current_dist = distances[v];

        for u in 0..n {
            if graph.adjacency[v * n + u] && distances[u] == f64::INFINITY {
                distances[u] = current_dist + 1.0;
                queue.push_back(u);
            }
        }
    }

    Ok(distances)
}

/// Compute weighted shortest paths using Dijkstra's algorithm
fn compute_dijkstra_distances(graph: &Graph, source: usize) -> Result<Vec<f64>> {
    let n = graph.num_vertices;
    let mut distances = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();

    distances[source] = 0.0;
    heap.push(DijkstraState {
        vertex: source,
        distance: 0.0,
    });

    // Build edge weight map
    let mut edge_weights: HashMap<(usize, usize), f64> = HashMap::new();
    for &(u, v, weight) in &graph.edges {
        edge_weights.insert((u, v), weight);
        edge_weights.insert((v, u), weight);
    }

    while let Some(DijkstraState {
        vertex: v,
        distance: dist,
    }) = heap.pop()
    {
        // Skip if we've already found a better path
        if dist > distances[v] {
            continue;
        }

        // Check all neighbors
        for u in 0..n {
            if graph.adjacency[v * n + u] {
                let weight = edge_weights.get(&(v, u)).copied().unwrap_or(1.0);
                let new_dist = dist + weight;

                if new_dist < distances[u] {
                    distances[u] = new_dist;
                    heap.push(DijkstraState {
                        vertex: u,
                        distance: new_dist,
                    });
                }
            }
        }
    }

    Ok(distances)
}

/// State for Dijkstra's algorithm priority queue
#[derive(Debug, Clone, Copy)]
struct DijkstraState {
    vertex: usize,
    distance: f64,
}

impl Eq for DijkstraState {}

impl PartialEq for DijkstraState {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.vertex == other.vertex
    }
}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse ordering
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.vertex.cmp(&other.vertex))
    }
}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geodesic_triangle() {
        // Triangle graph: all vertices equidistant
        let graph = Graph {
            num_vertices: 3,
            num_edges: 3,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
            adjacency: vec![false, true, true, true, false, true, true, true, false],
            coordinates: None,
        };

        let features = compute_landmark_distances(&graph, 2, "hop").unwrap();

        assert_eq!(features.landmarks.len(), 2);
        assert_eq!(features.centrality.len(), 3);
        assert_eq!(features.eccentricity.len(), 3);

        // All vertices should have similar centrality in triangle
        let avg_centrality: f64 = features.centrality.iter().sum::<f64>() / 3.0;
        for &c in &features.centrality {
            assert!((c - avg_centrality).abs() < 0.5);
        }
    }

    #[test]
    fn test_geodesic_path() {
        // Path graph: 0-1-2-3
        let graph = Graph {
            num_vertices: 4,
            num_edges: 3,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)],
            adjacency: vec![
                false, true, false, false, true, false, true, false, false, true, false, true,
                false, false, true, false,
            ],
            coordinates: None,
        };

        let features = compute_landmark_distances(&graph, 2, "hop").unwrap();

        // Endpoints should have higher eccentricity than middle vertices
        let max_ecc = features
            .eccentricity
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0);

        assert!(
            features.eccentricity[0] >= max_ecc * 0.9 || features.eccentricity[3] >= max_ecc * 0.9
        );
    }
}
