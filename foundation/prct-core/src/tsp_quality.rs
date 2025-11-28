//! TSP Quality Analysis for Graph Coloring
//!
//! Analyzes color class structure using TSP tour quality metrics.
//! Compact color classes (short TSP tours) indicate good coloring structure.

use crate::errors::*;
use crate::tsp::*;
use crate::geodesic::GeodesicFeatures;
use shared_types::*;
use ndarray::Array2;

/// Quality metrics for a color class
#[derive(Debug, Clone)]
pub struct ColorClassQuality {
    /// Color class index
    pub color: usize,

    /// Number of vertices in this class
    pub size: usize,

    /// TSP tour length (normalized by class size)
    pub tour_length: f64,

    /// Compactness score (0.0 = scattered, 1.0 = compact)
    pub compactness: f64,

    /// Average edge weight in TSP tour
    pub avg_edge_weight: f64,

    /// Number of "long jumps" (edges > 2 * avg)
    pub long_jumps: usize,
}

/// Analyzer for color class quality using TSP
pub struct TSPQualityAnalyzer {
    tsp_solver: TSPSolver,
}

impl TSPQualityAnalyzer {
    pub fn new() -> Self {
        Self {
            tsp_solver: TSPSolver::new(TSPConfig::default()),
        }
    }

    /// Analyze all color classes in a solution
    pub fn analyze_solution(
        &self,
        solution: &ColoringSolution,
        graph: &Graph,
    ) -> Result<Vec<ColorClassQuality>> {
        let n = graph.num_vertices;
        let mut qualities = Vec::new();

        // Group vertices by color
        let color_classes = self.extract_color_classes(solution);

        for (color, vertices) in color_classes.iter().enumerate() {
            if vertices.is_empty() {
                continue;
            }

            let quality = self.analyze_color_class(color, vertices, graph)?;
            qualities.push(quality);
        }

        Ok(qualities)
    }

    /// Compute average compactness across all color classes
    pub fn compute_avg_compactness(
        &self,
        solution: &ColoringSolution,
        graph: &Graph,
    ) -> Result<f64> {
        let qualities = self.analyze_solution(solution, graph)?;

        if qualities.is_empty() {
            return Ok(0.0);
        }

        let total_compactness: f64 = qualities.iter()
            .map(|q| q.compactness)
            .sum();

        Ok(total_compactness / qualities.len() as f64)
    }

    /// Extract color classes from solution
    fn extract_color_classes(&self, solution: &ColoringSolution) -> Vec<Vec<usize>> {
        let max_color = solution.colors.iter()
            .filter(|&&c| c != usize::MAX)
            .max()
            .copied()
            .unwrap_or(0);

        let mut classes = vec![Vec::new(); max_color + 1];

        for (vertex, &color) in solution.colors.iter().enumerate() {
            if color != usize::MAX {
                classes[color].push(vertex);
            }
        }

        classes
    }

    /// Analyze quality of a single color class
    fn analyze_color_class(
        &self,
        color: usize,
        vertices: &[usize],
        graph: &Graph,
    ) -> Result<ColorClassQuality> {
        let size = vertices.len();

        // For single vertex, perfect compactness
        if size == 1 {
            return Ok(ColorClassQuality {
                color,
                size,
                tour_length: 0.0,
                compactness: 1.0,
                avg_edge_weight: 0.0,
                long_jumps: 0,
            });
        }

        // For two vertices, trivial tour
        if size == 2 {
            let distance = self.compute_distance(vertices[0], vertices[1], graph);
            return Ok(ColorClassQuality {
                color,
                size,
                tour_length: 2.0 * distance,
                compactness: if distance < 1.0 { 1.0 } else { 1.0 / distance },
                avg_edge_weight: distance,
                long_jumps: 0,
            });
        }

        // Build subgraph for this color class
        let subgraph = self.build_subgraph(vertices, graph);

        // Solve TSP on subgraph
        let tour = self.tsp_solver.solve_tsp(&subgraph)?;

        // Compute tour metrics
        let tour_length = self.compute_tour_length(&tour, &subgraph);
        let avg_edge_weight = tour_length / tour.len() as f64;

        // Count long jumps (edges significantly longer than average)
        let threshold = 2.0 * avg_edge_weight;
        let long_jumps = tour.windows(2)
            .filter(|window| {
                let dist = subgraph.distance_matrix[window[0]][window[1]];
                dist > threshold
            })
            .count();

        // Compactness: ideal tour length vs actual
        // Ideal = perimeter of minimal bounding box (approximation)
        let ideal_length = (size as f64).sqrt() * 2.0;
        let compactness = (ideal_length / (tour_length + 1.0)).min(1.0);

        Ok(ColorClassQuality {
            color,
            size,
            tour_length,
            compactness,
            avg_edge_weight,
            long_jumps,
        })
    }

    /// Build subgraph for a set of vertices
    fn build_subgraph(&self, vertices: &[usize], graph: &Graph) -> SubGraph {
        let n = vertices.len();
        let mut distance_matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    distance_matrix[i][j] = 0.0;
                } else {
                    let dist = self.compute_distance(vertices[i], vertices[j], graph);
                    distance_matrix[i][j] = dist;
                }
            }
        }

        SubGraph {
            vertices: vertices.to_vec(),
            distance_matrix,
        }
    }

    /// Compute distance between two vertices
    fn compute_distance(&self, v1: usize, v2: usize, graph: &Graph) -> f64 {
        self.compute_distance_with_geodesic(v1, v2, graph, None, false)
    }

    /// Compute distance between two vertices with optional geodesic blending
    ///
    /// # Arguments
    /// * `v1`, `v2` - Vertex pair
    /// * `graph` - Graph structure
    /// * `geodesic_features` - Optional geodesic features for enhanced distance
    /// * `use_geodesic_distance` - Blend geodesic distance into metric
    fn compute_distance_with_geodesic(
        &self,
        v1: usize,
        v2: usize,
        graph: &Graph,
        geodesic_features: Option<&GeodesicFeatures>,
        use_geodesic_distance: bool,
    ) -> f64 {
        // Base distance: Euclidean or hop
        let base_distance = if !graph.coordinates.is_empty() && v1 < graph.coordinates.len() && v2 < graph.coordinates.len() {
            let (x1, y1) = graph.coordinates[v1];
            let (x2, y2) = graph.coordinates[v2];
            let dx = x2 - x1;
            let dy = y2 - y1;
            (dx * dx + dy * dy).sqrt()
        } else {
            self.compute_hop_distance(v1, v2, graph)
        };

        // Blend with geodesic distance if requested
        if use_geodesic_distance {
            if let Some(geo) = geodesic_features {
                // Compute geodesic distance as L1 norm of landmark distance vectors
                let mut geo_dist = 0.0;
                for lm_idx in 0..geo.landmarks.len() {
                    if v1 < geo.landmark_distances.len() && v2 < geo.landmark_distances.len() {
                        let d1 = geo.landmark_distances[v1][lm_idx];
                        let d2 = geo.landmark_distances[v2][lm_idx];
                        geo_dist += (d1 - d2).abs();
                    }
                }
                // Blend 70% base + 30% geodesic
                return 0.7 * base_distance + 0.3 * geo_dist;
            }
        }

        base_distance
    }

    /// Compute hop distance (shortest path length)
    fn compute_hop_distance(&self, v1: usize, v2: usize, graph: &Graph) -> f64 {
        use std::collections::VecDeque;

        let n = graph.num_vertices;
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();

        queue.push_back((v1, 0));
        visited[v1] = true;

        while let Some((v, dist)) = queue.pop_front() {
            if v == v2 {
                return dist as f64;
            }

            for u in 0..n {
                if graph.adjacency[v * n + u] && !visited[u] {
                    visited[u] = true;
                    queue.push_back((u, dist + 1));
                }
            }
        }

        // No path found, return large distance
        n as f64
    }

    /// Compute total tour length
    fn compute_tour_length(&self, tour: &[usize], subgraph: &SubGraph) -> f64 {
        if tour.len() < 2 {
            return 0.0;
        }

        let mut length = 0.0;
        for i in 0..tour.len() {
            let next = (i + 1) % tour.len();
            length += subgraph.distance_matrix[tour[i]][tour[next]];
        }

        length
    }

    /// Find worst (least compact) color classes
    pub fn find_worst_classes(
        &self,
        qualities: &[ColorClassQuality],
        k: usize,
    ) -> Vec<usize> {
        let mut sorted = qualities.to_vec();
        sorted.sort_by(|a, b| a.compactness.partial_cmp(&b.compactness).unwrap_or(std::cmp::Ordering::Equal));

        sorted.iter().take(k).map(|q| q.color).collect()
    }

    /// Find best (most compact) color classes
    pub fn find_best_classes(
        &self,
        qualities: &[ColorClassQuality],
        k: usize,
    ) -> Vec<usize> {
        let mut sorted = qualities.to_vec();
        sorted.sort_by(|a, b| b.compactness.partial_cmp(&a.compactness).unwrap_or(std::cmp::Ordering::Equal));

        sorted.iter().take(k).map(|q| q.color).collect()
    }
}

/// Subgraph for TSP computation
struct SubGraph {
    vertices: Vec<usize>,
    distance_matrix: Vec<Vec<f64>>,
}
