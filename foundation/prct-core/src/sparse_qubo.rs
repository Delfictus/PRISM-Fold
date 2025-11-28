//! Sparse QUBO Formulation for Graph Coloring
//!
//! Reduces memory from O(n²k²) to O(nk + |E|k) using sparse matrix storage.
//! This makes DSJC1000 feasible: 2.4 TB → ~10 GB (240x reduction)

use crate::errors::*;
use ndarray::Array2;
use shared_types::*;
use std::collections::HashMap;

/// Sparse QUBO matrix in Coordinate (COO) format
/// Stores only non-zero entries
#[derive(Clone)]
pub struct SparseQUBO {
    /// Non-zero entries: (row, col, value)
    entries: Vec<(usize, usize, f64)>,
    /// Number of variables
    num_variables: usize,
    /// Number of vertices
    num_vertices: usize,
    /// Number of colors
    num_colors: usize,
}

impl SparseQUBO {
    /// Convert graph coloring to sparse QUBO formulation
    ///
    /// Memory: O(nk + |E|k) instead of O(n²k²)
    /// For DSJC1000: ~10 GB instead of 2.4 TB
    pub fn from_graph_coloring(graph: &Graph, num_colors: usize) -> Result<Self> {
        Self::from_graph_coloring_with_color_penalty(graph, num_colors, 0.0)
    }

    /// Convert graph coloring to sparse QUBO with color penalty
    /// color_penalty_weight: penalty for using higher-numbered colors (0.0 = no penalty)
    pub fn from_graph_coloring_with_color_penalty(
        graph: &Graph,
        num_colors: usize,
        color_penalty_weight: f64,
    ) -> Result<Self> {
        let n = graph.num_vertices;
        let num_vars = n * num_colors;
        let mut entries = Vec::new();

        println!("[SPARSE-QUBO] Creating sparse formulation:");
        println!("[SPARSE-QUBO]   Vertices: {}", n);
        println!("[SPARSE-QUBO]   Colors: {}", num_colors);
        println!("[SPARSE-QUBO]   Variables: {}", num_vars);
        if color_penalty_weight > 0.0 {
            println!(
                "[SPARSE-QUBO]   Color penalty: {:.3} (prefers lower colors)",
                color_penalty_weight
            );
        }

        // Helper: variable index for (vertex, color)
        let var_idx = |v: usize, c: usize| -> usize { v * num_colors + c };

        // Constraint 1: Each vertex must have exactly one color
        // Penalty: (Σ_c x_{v,c} - 1)²
        // Expands to: Σ_c x_{v,c}² + Σ_{c1<c2} 2*x_{v,c1}*x_{v,c2} - 2*Σ_c x_{v,c}
        let one_color_penalty = 10.0;

        for v in 0..n {
            // Diagonal terms: x_{v,c}²
            for c in 0..num_colors {
                let i = var_idx(v, c);
                // Base coefficient: +1 (from expansion) - 2 (linear term) = -1
                let mut coeff = -one_color_penalty;

                // Add color penalty: prefer lower-numbered colors
                // Higher colors get higher penalty
                if color_penalty_weight > 0.0 {
                    coeff += color_penalty_weight * c as f64;
                }

                entries.push((i, i, coeff));
            }

            // Off-diagonal terms: x_{v,c1} * x_{v,c2}
            for c1 in 0..num_colors {
                for c2 in (c1 + 1)..num_colors {
                    let i = var_idx(v, c1);
                    let j = var_idx(v, c2);
                    // Coefficient: 2 (from expansion)
                    entries.push((i, j, 2.0 * one_color_penalty));
                }
            }
        }

        println!(
            "[SPARSE-QUBO]   One-color constraint entries: {}",
            entries.len()
        );

        // Constraint 2: Adjacent vertices must have different colors
        // Penalty: x_{u,c} * x_{v,c} for edge (u,v) and color c
        let conflict_penalty = 100.0;
        let mut edge_entries = 0;

        for u in 0..n {
            for v in (u + 1)..n {
                if graph.adjacency[u * n + v] {
                    // Edge (u, v) exists
                    for c in 0..num_colors {
                        let i = var_idx(u, c);
                        let j = var_idx(v, c);
                        entries.push((i, j, conflict_penalty));
                        edge_entries += 1;
                    }
                }
            }
        }

        println!(
            "[SPARSE-QUBO]   Conflict constraint entries: {}",
            edge_entries
        );
        println!("[SPARSE-QUBO]   Total non-zero entries: {}", entries.len());

        let density = entries.len() as f64 / (num_vars * num_vars) as f64;
        println!("[SPARSE-QUBO]   Matrix density: {:.6}%", density * 100.0);

        let memory_mb =
            (entries.len() * std::mem::size_of::<(usize, usize, f64)>()) as f64 / (1024.0 * 1024.0);
        println!("[SPARSE-QUBO]   Memory usage: {:.2} MB", memory_mb);

        Ok(Self {
            entries,
            num_variables: num_vars,
            num_vertices: n,
            num_colors,
        })
    }

    /// Evaluate QUBO objective: x^T Q x (sparse)
    pub fn evaluate(&self, solution: &[f64]) -> f64 {
        let mut energy = 0.0;

        for &(i, j, q_ij) in &self.entries {
            if i == j {
                // Diagonal term
                energy += q_ij * solution[i] * solution[i];
            } else {
                // Off-diagonal term (symmetric)
                energy += q_ij * solution[i] * solution[j];
                // Since Q is symmetric, we store upper triangle only
                // But evaluation needs Q[i,j] + Q[j,i] = 2*Q[i,j]
                // Actually for QUBO, we typically store the full coefficient
                // So this is correct as-is
            }
        }

        energy
    }

    /// Apply a single-variable flip and compute energy delta (fast)
    pub fn delta_energy(&self, solution: &[f64], flip_idx: usize) -> f64 {
        let old_val = solution[flip_idx];
        let new_val = 1.0 - old_val;
        let diff = new_val - old_val;

        let mut delta = 0.0;

        // Only need to consider entries involving flip_idx
        for &(i, j, q_ij) in &self.entries {
            if i == flip_idx {
                // Q[flip_idx, j] term
                if i == j {
                    // Diagonal: Q[i,i] * (new² - old²)
                    delta += q_ij * (new_val * new_val - old_val * old_val);
                } else {
                    // Off-diagonal: Q[i,j] * x[j] * (new - old)
                    delta += q_ij * solution[j] * diff;
                }
            } else if j == flip_idx {
                // Q[i, flip_idx] term (symmetric part)
                delta += q_ij * solution[i] * diff;
            }
        }

        delta
    }

    /// Get number of variables
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Get number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.entries.len() * std::mem::size_of::<(usize, usize, f64)>()
    }

    /// Get direct access to QUBO entries (for GPU upload)
    pub fn entries(&self) -> &[(usize, usize, f64)] {
        &self.entries
    }

    /// Get number of vertices
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Get number of colors
    pub fn num_colors(&self) -> usize {
        self.num_colors
    }
}

/// Chromatic bounds from TDA
pub struct ChromaticBounds {
    /// Lower bound (from max clique size)
    pub lower: usize,
    /// Upper bound (from degree + 1)
    pub upper: usize,
    /// Maximal clique sizes found
    pub max_clique_size: usize,
    /// Number of connected components (Betti-0)
    pub num_components: usize,
}

impl ChromaticBounds {
    /// Compute tight chromatic bounds using TDA
    pub fn from_graph_tda(graph: &Graph) -> Result<Self> {
        use ndarray::Array2;

        let n = graph.num_vertices;

        println!("[TDA-BOUNDS] Computing chromatic bounds for {} vertices", n);

        // Build adjacency matrix
        let mut adj_matrix = Array2::from_elem((n, n), false);
        for i in 0..n {
            for j in 0..n {
                adj_matrix[[i, j]] = graph.adjacency[i * n + j];
            }
        }

        // Use CPU implementation for Phase 2
        // (GPU TDA will be integrated in Phase 3)
        Self::compute_bounds_cpu(graph, &adj_matrix)
    }

    fn compute_bounds_cpu(graph: &Graph, adj_matrix: &Array2<bool>) -> Result<Self> {
        let n = graph.num_vertices;

        // Compute connected components (CPU union-find)
        let num_components = Self::count_components_cpu(adj_matrix);
        println!("[TDA-BOUNDS] Connected components: {}", num_components);

        // Find max clique (greedy approximation)
        let max_clique_size = Self::find_max_clique_cpu(adj_matrix);
        println!("[TDA-BOUNDS] Max clique size (approx): {}", max_clique_size);

        // Compute max degree
        let mut max_degree = 0;
        for i in 0..n {
            let degree = (0..n).filter(|&j| adj_matrix[[i, j]]).count();
            max_degree = max_degree.max(degree);
        }

        let lower = max_clique_size;
        let upper = (max_degree + 1).max(3);

        println!("[TDA-BOUNDS] Chromatic bounds: [{}, {}]", lower, upper);

        Ok(Self {
            lower,
            upper,
            max_clique_size,
            num_components,
        })
    }

    fn count_components_cpu(adj_matrix: &Array2<bool>) -> usize {
        let n = adj_matrix.nrows();
        let mut visited = vec![false; n];
        let mut components = 0;

        for start in 0..n {
            if !visited[start] {
                components += 1;
                let mut stack = vec![start];
                while let Some(v) = stack.pop() {
                    if visited[v] {
                        continue;
                    }
                    visited[v] = true;
                    for u in 0..n {
                        if adj_matrix[[v, u]] && !visited[u] {
                            stack.push(u);
                        }
                    }
                }
            }
        }

        components
    }

    fn find_max_clique_cpu(adj_matrix: &Array2<bool>) -> usize {
        let n = adj_matrix.nrows();

        // Use degree-based heuristic for better initial bound
        let degrees: Vec<usize> = (0..n)
            .map(|i| (0..n).filter(|&j| adj_matrix[[i, j]]).count())
            .collect();

        // Sort vertices by degree (descending)
        let mut vertices: Vec<usize> = (0..n).collect();
        vertices.sort_by_key(|&v| std::cmp::Reverse(degrees[v]));

        let mut max_clique = 0;

        // Try to build cliques starting from high-degree vertices
        for &start in vertices.iter().take(n.min(100)) {
            let mut clique = vec![start];
            let mut candidates: Vec<usize> = (0..n)
                .filter(|&v| v != start && adj_matrix[[start, v]])
                .collect();

            // Sort candidates by degree within the candidate set
            let candidate_degrees: Vec<(usize, usize)> = candidates
                .iter()
                .map(|&v| {
                    let deg = candidates
                        .iter()
                        .filter(|&&u| u != v && adj_matrix[[v, u]])
                        .count();
                    (v, deg)
                })
                .collect();

            candidates.clear();
            let mut sorted_candidates = candidate_degrees;
            sorted_candidates.sort_by_key(|(_, deg)| std::cmp::Reverse(*deg));
            candidates = sorted_candidates.into_iter().map(|(v, _)| v).collect();

            for &v in &candidates {
                // Check if v is connected to all vertices in current clique
                if clique.iter().all(|&u| adj_matrix[[v, u]]) {
                    clique.push(v);
                }
            }

            max_clique = max_clique.max(clique.len());

            // Early termination if we found a very large clique
            if max_clique >= degrees[start] + 1 {
                break;
            }
        }

        max_clique
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_qubo_small_graph() {
        // Triangle graph: 3 vertices, 3 edges
        let graph = Graph {
            num_vertices: 3,
            num_edges: 3,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
            adjacency: vec![false, true, true, true, false, true, true, true, false],
            coordinates: None,
        };

        let qubo = SparseQUBO::from_graph_coloring(&graph, 3).unwrap();

        // For 3 vertices and 3 colors:
        // - One-color constraints: 3*3 diagonal + 3*C(3,2) off-diagonal = 9 + 9 = 18
        // - Conflict constraints: 3 edges * 3 colors = 9
        // Total: 27 entries

        assert_eq!(qubo.nnz(), 27);
        assert_eq!(qubo.num_variables(), 9);

        // Test evaluation
        let solution = vec![
            1.0, 0.0, 0.0, // v0 → color 0
            0.0, 1.0, 0.0, // v1 → color 1
            0.0, 0.0, 1.0, // v2 → color 2
        ];

        let energy = qubo.evaluate(&solution);
        println!("Valid 3-coloring energy: {}", energy);

        // Should have low energy (satisfies constraints)
        assert!(energy < 0.0); // One-color constraints are satisfied (-10 each)
    }

    #[test]
    fn test_chromatic_bounds_triangle() {
        let graph = Graph {
            num_vertices: 3,
            num_edges: 3,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
            adjacency: vec![false, true, true, true, false, true, true, true, false],
            coordinates: None,
        };

        let bounds = ChromaticBounds::from_graph_tda(&graph).unwrap();

        // Triangle has chromatic number = 3
        assert_eq!(bounds.lower, 3); // Max clique is K3
        assert_eq!(bounds.upper, 3); // Max degree is 2, so upper is 3
        assert_eq!(bounds.num_components, 1);
    }
}
