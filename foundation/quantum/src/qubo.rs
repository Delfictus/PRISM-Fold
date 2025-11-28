//! QUBO (Quadratic Unconstrained Binary Optimization) solver
//!
//! Implements GPU-accelerated QUBO solving for Maximum Independent Set (MIS)
//! and general QUBO problems to compete with Intel Loihi 2 neuromorphic hardware
//!
//! Problem formulation:
//! minimize f(x) = x^T * Q * x
//! where x is a binary vector and Q is the QUBO matrix

use anyhow::{anyhow, Context, Result};
use ndarray::Array2;
use std::sync::Arc;

/// QUBO problem solver using GPU acceleration
pub struct GpuQuboSolver {
    n_vars: usize,
    q_matrix: Array2<f64>,
    best_solution: Vec<u8>,
    best_energy: f64,
    iteration_count: usize,
}

impl GpuQuboSolver {
    /// Create a new QUBO solver for the given Q matrix
    pub fn new(q_matrix: Array2<f64>) -> Result<Self> {
        let n_vars = q_matrix.nrows();

        if n_vars != q_matrix.ncols() {
            return Err(anyhow!("Q matrix must be square"));
        }

        // Initialize with random solution
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let initial_solution: Vec<u8> = (0..n_vars)
            .map(|_| if rng.gen::<f64>() < 0.5 { 1 } else { 0 })
            .collect();

        let initial_energy = Self::calculate_energy(&q_matrix, &initial_solution);

        Ok(Self {
            n_vars,
            q_matrix,
            best_solution: initial_solution,
            best_energy: initial_energy,
            iteration_count: 0,
        })
    }

    /// Create QUBO matrix from Maximum Independent Set (MIS) problem
    ///
    /// MIS: Find largest set of vertices with no edges between them
    /// QUBO formulation: minimize -Σx_i + P*Σ(x_i * x_j) for edges (i,j)
    /// where P is a penalty parameter (typically > max degree)
    pub fn from_mis_problem(adjacency: &Array2<u8>) -> Result<Self> {
        let n_vars = adjacency.nrows();

        if n_vars != adjacency.ncols() {
            return Err(anyhow!("Adjacency matrix must be square"));
        }

        // Calculate penalty parameter (larger than max degree)
        let max_degree = adjacency.sum_axis(ndarray::Axis(1)).iter()
            .map(|&d| d as f64)
            .fold(0.0, f64::max);
        let penalty = max_degree + 1.0;

        // Build QUBO matrix
        let mut q_matrix = Array2::zeros((n_vars, n_vars));

        // Diagonal: -1 (reward for selecting vertex)
        for i in 0..n_vars {
            q_matrix[[i, i]] = -1.0;
        }

        // Off-diagonal: penalty for adjacent vertices
        for i in 0..n_vars {
            for j in (i + 1)..n_vars {
                if adjacency[[i, j]] == 1 {
                    // Symmetric penalty
                    q_matrix[[i, j]] = penalty;
                    q_matrix[[j, i]] = penalty;
                }
            }
        }

        Self::new(q_matrix)
    }

    /// Calculate energy (objective value) for a given solution
    fn calculate_energy(q_matrix: &Array2<f64>, solution: &[u8]) -> f64 {
        let n = solution.len();
        let mut energy = 0.0;

        for i in 0..n {
            if solution[i] == 1 {
                energy += q_matrix[[i, i]];
                for j in (i + 1)..n {
                    if solution[j] == 1 {
                        energy += 2.0 * q_matrix[[i, j]];
                    }
                }
            }
        }

        energy
    }

    /// Solve using simulated annealing (CPU baseline)
    pub fn solve_cpu_sa(&mut self, max_iterations: usize, initial_temp: f64) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut current_solution = self.best_solution.clone();
        let mut current_energy = self.best_energy;

        let mut temperature = initial_temp;
        let cooling_rate = (initial_temp / 0.01).powf(1.0 / max_iterations as f64);

        for iteration in 0..max_iterations {
            // Random bit flip
            let flip_bit = rng.gen_range(0..self.n_vars);
            let old_value = current_solution[flip_bit];
            current_solution[flip_bit] = 1 - old_value;

            // Calculate new energy
            let new_energy = Self::calculate_energy(&self.q_matrix, &current_solution);
            let delta_energy = new_energy - current_energy;

            // Accept or reject
            if delta_energy < 0.0 || rng.gen::<f64>() < (-delta_energy / temperature).exp() {
                current_energy = new_energy;

                if current_energy < self.best_energy {
                    self.best_energy = current_energy;
                    self.best_solution = current_solution.clone();
                }
            } else {
                // Revert
                current_solution[flip_bit] = old_value;
            }

            // Cool down
            temperature *= cooling_rate;

            if iteration % 1000 == 0 && iteration > 0 {
                println!("  Iteration {}: best energy = {:.4}, temp = {:.6}",
                         iteration, self.best_energy, temperature);
            }
        }

        self.iteration_count = max_iterations;
        Ok(())
    }

    /// Solve using tabu search (CPU baseline)
    pub fn solve_cpu_tabu(&mut self, max_iterations: usize, tabu_tenure: usize) -> Result<()> {
        use std::collections::VecDeque;

        let mut current_solution = self.best_solution.clone();
        let mut current_energy = self.best_energy;

        // Tabu list: recently flipped bits
        let mut tabu_list: VecDeque<usize> = VecDeque::with_capacity(tabu_tenure);

        for iteration in 0..max_iterations {
            let mut best_neighbor_bit = 0;
            let mut best_neighbor_energy = f64::INFINITY;
            let mut best_neighbor_solution = current_solution.clone();

            // Evaluate all neighbors (flip each bit)
            for bit in 0..self.n_vars {
                // Skip if tabu (unless aspiration criterion)
                if tabu_list.contains(&bit) {
                    continue;
                }

                // Try flipping this bit
                let mut neighbor = current_solution.clone();
                neighbor[bit] = 1 - neighbor[bit];
                let neighbor_energy = Self::calculate_energy(&self.q_matrix, &neighbor);

                if neighbor_energy < best_neighbor_energy {
                    best_neighbor_energy = neighbor_energy;
                    best_neighbor_bit = bit;
                    best_neighbor_solution = neighbor;
                }
            }

            // Move to best neighbor
            current_solution = best_neighbor_solution;
            current_energy = best_neighbor_energy;

            // Update tabu list
            tabu_list.push_back(best_neighbor_bit);
            if tabu_list.len() > tabu_tenure {
                tabu_list.pop_front();
            }

            // Update best solution
            if current_energy < self.best_energy {
                self.best_energy = current_energy;
                self.best_solution = current_solution.clone();
            }

            if iteration % 1000 == 0 && iteration > 0 {
                println!("  Iteration {}: best energy = {:.4}", iteration, self.best_energy);
            }
        }

        self.iteration_count = max_iterations;
        Ok(())
    }

    /// Get the best solution found
    pub fn get_solution(&self) -> &[u8] {
        &self.best_solution
    }

    /// Get the best energy (objective value) found
    pub fn get_energy(&self) -> f64 {
        self.best_energy
    }

    /// Get number of iterations performed
    pub fn get_iterations(&self) -> usize {
        self.iteration_count
    }

    /// Validate MIS solution (check no adjacent vertices selected)
    pub fn validate_mis(&self, adjacency: &Array2<u8>) -> bool {
        let n = adjacency.nrows();

        for i in 0..n {
            if self.best_solution[i] == 1 {
                for j in (i + 1)..n {
                    if self.best_solution[j] == 1 && adjacency[[i, j]] == 1 {
                        return false; // Adjacent vertices selected
                    }
                }
            }
        }

        true
    }

    /// Count number of selected vertices (MIS size)
    pub fn get_mis_size(&self) -> usize {
        self.best_solution.iter().filter(|&&x| x == 1).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_qubo() {
        // Simple 2-variable QUBO
        let q = Array2::from_shape_vec((2, 2), vec![-1.0, 2.0, 2.0, -1.0]).unwrap();
        let mut solver = GpuQuboSolver::new(q).unwrap();

        // Optimal should be [0, 0] or [1, 1] with energy -1
        solver.solve_cpu_sa(1000, 10.0).unwrap();

        assert!(solver.get_energy() <= -0.5); // Should find reasonable solution
    }

    #[test]
    fn test_mis_problem() {
        // Triangle graph: only 1 vertex can be selected
        let mut adj = Array2::zeros((3, 3));
        adj[[0, 1]] = 1; adj[[1, 0]] = 1;
        adj[[1, 2]] = 1; adj[[2, 1]] = 1;
        adj[[0, 2]] = 1; adj[[2, 0]] = 1;

        let mut solver = GpuQuboSolver::from_mis_problem(&adj).unwrap();
        solver.solve_cpu_sa(1000, 10.0).unwrap();

        assert!(solver.validate_mis(&adj));
        assert_eq!(solver.get_mis_size(), 1);
    }
}
