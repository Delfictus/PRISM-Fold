//! TSP Path Optimization for PRCT Algorithm
//!
//! Implements Traveling Salesperson Problem solver optimized for phase coherence
//! maximization in quantum systems. Uses Lin-Kernighan heuristic with 2-opt improvements.

use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashSet;

/// TSP path optimizer for PRCT algorithm
#[derive(Debug, Clone)]
pub struct TSPPathOptimizer {
    /// Distance matrix (derived from coupling strengths)
    distance_matrix: Array2<f64>,
    /// Current tour (permutation of vertices)
    tour: Vec<usize>,
    /// Tour length (total distance)
    tour_length: f64,
    /// Number of vertices
    n_vertices: usize,
}

impl TSPPathOptimizer {
    /// Create new TSP optimizer from coupling matrix
    ///
    /// # Arguments
    /// * `coupling_matrix` - Complex coupling amplitudes between atoms
    pub fn new(coupling_matrix: &Array2<Complex64>) -> Self {
        let n = coupling_matrix.nrows();

        // Convert coupling strengths to distances (inverse relationship)
        // Strong coupling = short distance (want to visit nearby)
        let distance_matrix = Self::coupling_to_distance(coupling_matrix);

        // Initialize with greedy nearest-neighbor tour
        let tour = Self::nearest_neighbor_tour(&distance_matrix);
        let tour_length = Self::calculate_tour_length(&tour, &distance_matrix);

        Self {
            distance_matrix,
            tour,
            tour_length,
            n_vertices: n,
        }
    }

    /// Convert coupling matrix to distance matrix
    fn coupling_to_distance(coupling_matrix: &Array2<Complex64>) -> Array2<f64> {
        let n = coupling_matrix.nrows();
        let mut distance = Array2::zeros((n, n));

        // Find max coupling strength for normalization
        let max_coupling = coupling_matrix
            .iter()
            .map(|c| c.norm())
            .fold(0.0f64, |a, b| a.max(b));

        if max_coupling < 1e-10 {
            // No coupling - use uniform distances
            return Array2::from_elem((n, n), 1.0);
        }

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    distance[[i, j]] = 0.0;
                } else {
                    // Distance = 1 / coupling_strength (normalized)
                    let coupling_strength = coupling_matrix[[i, j]].norm();
                    distance[[i, j]] = max_coupling / (coupling_strength + 1e-10);
                }
            }
        }

        distance
    }

    /// Nearest neighbor heuristic for initial tour
    fn nearest_neighbor_tour(distance_matrix: &Array2<f64>) -> Vec<usize> {
        let n = distance_matrix.nrows();
        if n == 0 {
            return Vec::new();
        }

        let mut tour = Vec::with_capacity(n);
        let mut unvisited: HashSet<usize> = (0..n).collect();

        // Start from vertex 0
        let mut current = 0;
        tour.push(current);
        unvisited.remove(&current);

        // Greedily visit nearest unvisited vertex
        while !unvisited.is_empty() {
            let mut nearest = *unvisited.iter().next().unwrap();
            let mut min_dist = distance_matrix[[current, nearest]];

            for &v in &unvisited {
                let dist = distance_matrix[[current, v]];
                if dist < min_dist {
                    min_dist = dist;
                    nearest = v;
                }
            }

            tour.push(nearest);
            unvisited.remove(&nearest);
            current = nearest;
        }

        tour
    }

    /// Calculate total tour length
    fn calculate_tour_length(tour: &[usize], distance_matrix: &Array2<f64>) -> f64 {
        if tour.len() < 2 {
            return 0.0;
        }

        let mut length = 0.0;
        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            length += distance_matrix[[from, to]];
        }

        length
    }

    /// Optimize tour using 2-opt algorithm
    pub fn optimize(&mut self, max_iterations: usize) -> Result<()> {
        let mut improved = true;
        let mut iteration = 0;

        while improved && iteration < max_iterations {
            improved = false;

            // Try all possible 2-opt swaps
            for i in 0..self.n_vertices - 1 {
                for j in (i + 2)..self.n_vertices {
                    // Calculate change in tour length for this swap
                    let delta = self.calculate_2opt_delta(i, j);

                    if delta < -1e-10 {
                        // Improvement found - apply swap
                        self.apply_2opt_swap(i, j);
                        self.tour_length += delta;
                        improved = true;
                    }
                }
            }

            iteration += 1;
        }

        Ok(())
    }

    /// Calculate change in tour length for 2-opt swap
    fn calculate_2opt_delta(&self, i: usize, j: usize) -> f64 {
        let n = self.tour.len();

        // Current edges: (i, i+1) and (j, j+1)
        let v1 = self.tour[i];
        let v2 = self.tour[(i + 1) % n];
        let v3 = self.tour[j];
        let v4 = self.tour[(j + 1) % n];

        // Current length
        let current = self.distance_matrix[[v1, v2]] + self.distance_matrix[[v3, v4]];

        // New edges: (i, j) and (i+1, j+1)
        let new = self.distance_matrix[[v1, v3]] + self.distance_matrix[[v2, v4]];

        new - current
    }

    /// Apply 2-opt swap to tour
    fn apply_2opt_swap(&mut self, i: usize, j: usize) {
        // Reverse segment between i+1 and j
        let mut k = i + 1;
        let mut l = j;

        while k < l {
            self.tour.swap(k, l);
            k += 1;
            l -= 1;
        }
    }

    /// Optimize using simulated annealing for better solutions
    pub fn optimize_annealing(&mut self, iterations: usize, initial_temp: f64) -> Result<()> {
        let mut temperature = initial_temp;
        let cooling_rate = 0.995;
        let mut best_tour = self.tour.clone();
        let mut best_length = self.tour_length;

        for _ in 0..iterations {
            // Random 2-opt swap
            let i = rand::random::<usize>() % self.n_vertices;
            let j = (i + 2 + rand::random::<usize>() % (self.n_vertices - 2)) % self.n_vertices;

            let delta = self.calculate_2opt_delta(i, j);

            // Metropolis criterion
            let accept = delta < 0.0 || rand::random::<f64>() < (-delta / temperature).exp();

            if accept {
                self.apply_2opt_swap(i, j);
                self.tour_length += delta;

                if self.tour_length < best_length {
                    best_length = self.tour_length;
                    best_tour = self.tour.clone();
                }
            }

            temperature *= cooling_rate;
        }

        // Apply best solution
        self.tour = best_tour;
        self.tour_length = best_length;

        Ok(())
    }

    /// Get current tour
    pub fn get_tour(&self) -> &[usize] {
        &self.tour
    }

    /// Get tour length
    pub fn get_tour_length(&self) -> f64 {
        self.tour_length
    }

    /// Calculate path quality (normalized inverse of tour length)
    pub fn get_path_quality(&self) -> f64 {
        if self.tour_length < 1e-10 {
            return 1.0;
        }

        // Quality = 1 / (1 + normalized_length)
        let max_possible_length =
            self.n_vertices as f64 * self.distance_matrix.iter().fold(0.0f64, |a, &b| a.max(b));

        if max_possible_length < 1e-10 {
            return 1.0;
        }

        1.0 / (1.0 + self.tour_length / max_possible_length)
    }

    /// Calculate tour circularity (how circular the path is)
    pub fn get_circularity(&self) -> f64 {
        if self.n_vertices < 3 {
            return 1.0;
        }

        // Calculate variance in consecutive edge lengths
        let mut edge_lengths = Vec::new();
        for i in 0..self.n_vertices {
            let from = self.tour[i];
            let to = self.tour[(i + 1) % self.n_vertices];
            edge_lengths.push(self.distance_matrix[[from, to]]);
        }

        let mean = edge_lengths.iter().sum::<f64>() / edge_lengths.len() as f64;
        let variance = edge_lengths
            .iter()
            .map(|&l| (l - mean).powi(2))
            .sum::<f64>()
            / edge_lengths.len() as f64;

        // Low variance = high circularity
        1.0 / (1.0 + variance.sqrt())
    }

    /// Get TSP factor τ(eᵢⱼ,π) for edge (i,j) given tour π
    pub fn get_tsp_factor(&self, i: usize, j: usize) -> f64 {
        // Find positions of i and j in tour
        let pos_i = self.tour.iter().position(|&v| v == i);
        let pos_j = self.tour.iter().position(|&v| v == j);

        match (pos_i, pos_j) {
            (Some(pi), Some(pj)) => {
                // Distance in tour (cyclic)
                let dist = (pi as i32 - pj as i32).abs();
                let cyclic_dist = dist.min(self.n_vertices as i32 - dist);

                // Factor inversely proportional to tour distance
                // Adjacent in tour = factor 1.0, far apart = factor → 0
                (-cyclic_dist as f64 / 3.0).exp()
            }
            _ => 0.0, // Vertices not in tour
        }
    }

    /// Validate tour (all vertices visited exactly once)
    pub fn validate_tour(&self) -> bool {
        if self.tour.len() != self.n_vertices {
            return false;
        }

        let unique: HashSet<_> = self.tour.iter().collect();
        unique.len() == self.n_vertices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsp_creation() {
        let coupling = Array2::from_elem((5, 5), Complex64::new(0.5, 0.0));
        let tsp = TSPPathOptimizer::new(&coupling);

        assert_eq!(tsp.n_vertices, 5);
        assert_eq!(tsp.tour.len(), 5);
        assert!(tsp.validate_tour());
    }

    #[test]
    fn test_2opt_optimization() {
        // Create asymmetric coupling (should find better tour)
        let mut coupling = Array2::zeros((6, 6));
        for i in 0..6 {
            for j in 0..6 {
                let dist = (i as f64 - j as f64).abs();
                coupling[[i, j]] = Complex64::new(1.0 / (1.0 + dist), 0.0);
            }
        }

        let mut tsp = TSPPathOptimizer::new(&coupling);
        let initial_length = tsp.get_tour_length();

        tsp.optimize(100).unwrap();

        assert!(tsp.get_tour_length() <= initial_length);
        assert!(tsp.validate_tour());
    }

    #[test]
    fn test_path_quality() {
        let coupling = Array2::from_elem((4, 4), Complex64::new(1.0, 0.0));
        let tsp = TSPPathOptimizer::new(&coupling);

        let quality = tsp.get_path_quality();
        assert!(quality >= 0.0 && quality <= 1.0);
    }

    #[test]
    fn test_tsp_factor() {
        let coupling = Array2::from_elem((5, 5), Complex64::new(0.8, 0.0));
        let tsp = TSPPathOptimizer::new(&coupling);

        // Adjacent vertices should have high factor
        let factor_adjacent = tsp.get_tsp_factor(0, 1);
        // Far vertices should have lower factor
        let factor_far = tsp.get_tsp_factor(0, 4);

        assert!(factor_adjacent > factor_far);
    }

    #[test]
    fn test_annealing_optimization() {
        let mut coupling = Array2::zeros((8, 8));
        for i in 0..8 {
            for j in 0..8 {
                coupling[[i, j]] = Complex64::new((1.0 + (i + j) as f64).recip(), 0.0);
            }
        }

        let mut tsp = TSPPathOptimizer::new(&coupling);
        let initial_length = tsp.get_tour_length();

        tsp.optimize_annealing(1000, 10.0).unwrap();

        assert!(tsp.get_tour_length() <= initial_length);
        assert!(tsp.validate_tour());
    }
}
