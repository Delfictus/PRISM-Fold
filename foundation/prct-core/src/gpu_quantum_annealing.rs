//! GPU-Accelerated QUBO Simulated Annealing for Graph Coloring
//!
//! Implements sparse-QUBO simulated annealing entirely on GPU using:
//! - CSR (Compressed Sparse Row) matrix format for memory efficiency
//! - cuRAND Philox4x32-10 for on-device RNG
//! - Warp-level parallel batch flip evaluation
//! - Metropolis-Hastings acceptance criterion
//!
//! Memory footprint for DSJC1000.5:
//! - Variables: n*k ≈ 100,000
//! - CSR entries: ~15M non-zeros → 12 MB
//! - State buffers: <1 MB
//! - Total: <30 MB VRAM

use crate::errors::*;
use crate::sparse_qubo::SparseQUBO;
use cudarc::driver::*;
use shared_types::*;
use std::sync::Arc;

/// GPU QUBO Simulated Annealing Configuration
#[derive(Debug, Clone)]
pub struct GpuQuboConfig {
    pub iterations: usize,
    pub batch_size: usize,
    pub t_initial: f64,
    pub t_final: f64,
    pub seed: u64,
}

impl Default for GpuQuboConfig {
    fn default() -> Self {
        Self {
            iterations: 10_000,
            batch_size: 256,
            t_initial: 1.0,
            t_final: 0.01,
            seed: 42,
        }
    }
}

/// CSR (Compressed Sparse Row) representation for GPU upload
#[derive(Debug)]
pub struct CsrMatrix {
    pub row_ptr: Vec<i32>,
    pub col_idx: Vec<i32>,
    pub values: Vec<f64>,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl CsrMatrix {
    /// Convert QUBO COO entries to CSR format (upper triangular)
    pub fn from_qubo_coo(entries: &[(usize, usize, f64)], num_vars: usize) -> Self {
        let mut row_ptr = vec![0i32; num_vars + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        // Count entries per row
        for &(row, col, _) in entries {
            if row <= col && row < num_vars && col < num_vars {
                row_ptr[row + 1] += 1;
            }
        }

        // Compute cumulative sum
        for i in 1..=num_vars {
            row_ptr[i] += row_ptr[i - 1];
        }

        // Allocate storage
        let nnz = row_ptr[num_vars] as usize;
        col_idx.resize(nnz, 0);
        values.resize(nnz, 0.0);

        // Fill entries
        let mut insert_pos = row_ptr.clone();
        for &(row, col, val) in entries {
            if row <= col && row < num_vars && col < num_vars {
                let pos = insert_pos[row] as usize;
                col_idx[pos] = col as i32;
                values[pos] = val;
                insert_pos[row] += 1;
            }
        }

        CsrMatrix {
            row_ptr,
            col_idx,
            values,
            num_rows: num_vars,
            num_cols: num_vars,
        }
    }
}

/// GPU QUBO Simulated Annealing Solver
pub struct GpuQuboSolver {
    device: Arc<CudaDevice>,
    energy_kernel: Arc<CudaFunction>,
    flip_kernel: Arc<CudaFunction>,
    metropolis_kernel: Arc<CudaFunction>,
    init_rng_kernel: Arc<CudaFunction>,
}

impl GpuQuboSolver {
    /// Initialize GPU QUBO solver with loaded kernels
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        // Load PTX module
        let ptx_path = std::env::var("PRISM_QUANTUM_PTX")
            .unwrap_or_else(|_| "foundation/kernels/quantum_evolution.ptx".to_string());

        let ptx = std::fs::read_to_string(&ptx_path).map_err(|e| {
            PRCTError::GpuError(format!("Failed to load PTX from {}: {}", ptx_path, e))
        })?;

        device
            .load_ptx(
                ptx.into(),
                "quantum_evolution",
                &[
                    "qubo_energy_kernel",
                    "qubo_flip_batch_kernel",
                    "qubo_metropolis_kernel",
                    "init_curand_states",
                ],
            )
            .map_err(|e| PRCTError::GpuError(format!("Failed to load PTX module: {}", e)))?;

        let energy_kernel = device
            .get_func("quantum_evolution", "qubo_energy_kernel")
            .ok_or_else(|| PRCTError::GpuError("Failed to get qubo_energy_kernel".to_string()))?;

        let flip_kernel = device
            .get_func("quantum_evolution", "qubo_flip_batch_kernel")
            .ok_or_else(|| {
                PRCTError::GpuError("Failed to get qubo_flip_batch_kernel".to_string())
            })?;

        let metropolis_kernel = device
            .get_func("quantum_evolution", "qubo_metropolis_kernel")
            .ok_or_else(|| {
                PRCTError::GpuError("Failed to get qubo_metropolis_kernel".to_string())
            })?;

        let init_rng_kernel = device
            .get_func("quantum_evolution", "init_curand_states")
            .ok_or_else(|| PRCTError::GpuError("Failed to get init_curand_states".to_string()))?;

        Ok(Self {
            device,
            energy_kernel: Arc::new(energy_kernel),
            flip_kernel: Arc::new(flip_kernel),
            metropolis_kernel: Arc::new(metropolis_kernel),
            init_rng_kernel: Arc::new(init_rng_kernel),
        })
    }

    /// Run QUBO simulated annealing on GPU
    pub fn solve(
        &self,
        qubo: &SparseQUBO,
        initial_state: &[bool],
        config: &GpuQuboConfig,
    ) -> Result<Vec<bool>> {
        let num_vars = qubo.num_variables();

        if initial_state.len() != num_vars {
            return Err(PRCTError::InvalidInput(format!(
                "Initial state size {} != num_vars {}",
                initial_state.len(),
                num_vars
            )));
        }

        println!("[GPU-QUBO] Starting GPU QUBO SA");
        println!("[GPU-QUBO]   Variables: {}", num_vars);
        println!("[GPU-QUBO]   Iterations: {}", config.iterations);
        println!(
            "[GPU-QUBO]   Temperature: {:.3} → {:.3}",
            config.t_initial, config.t_final
        );
        println!("[GPU-QUBO]   Batch size: {}", config.batch_size);

        // Convert QUBO to CSR
        let csr = CsrMatrix::from_qubo_coo(qubo.entries(), num_vars);
        println!("[GPU-QUBO]   CSR nnz: {}", csr.values.len());

        // Upload CSR to device
        let d_row_ptr = self
            .device
            .htod_copy(csr.row_ptr.clone())
            .map_err(|e| PRCTError::GpuError(format!("Failed to upload row_ptr: {}", e)))?;
        let d_col_idx = self
            .device
            .htod_copy(csr.col_idx.clone())
            .map_err(|e| PRCTError::GpuError(format!("Failed to upload col_idx: {}", e)))?;
        let d_values = self
            .device
            .htod_copy(csr.values.clone())
            .map_err(|e| PRCTError::GpuError(format!("Failed to upload values: {}", e)))?;

        // Convert initial state to u8
        let state_u8: Vec<u8> = initial_state.iter().map(|&b| b as u8).collect();
        let mut d_state_current = self
            .device
            .htod_copy(state_u8.clone())
            .map_err(|e| PRCTError::GpuError(format!("Failed to upload state: {}", e)))?;
        let mut d_state_best = self
            .device
            .htod_copy(state_u8)
            .map_err(|e| PRCTError::GpuError(format!("Failed to upload best state: {}", e)))?;

        // Allocate working buffers
        let d_delta_energy = self
            .device
            .alloc_zeros::<f64>(config.batch_size)
            .map_err(|e| PRCTError::GpuError(format!("Failed to allocate delta_energy: {}", e)))?;
        let d_flip_candidates = self
            .device
            .alloc_zeros::<i32>(config.batch_size)
            .map_err(|e| {
                PRCTError::GpuError(format!("Failed to allocate flip_candidates: {}", e))
            })?;
        let mut d_best_energy = self
            .device
            .htod_copy(vec![f64::INFINITY])
            .map_err(|e| PRCTError::GpuError(format!("Failed to allocate best_energy: {}", e)))?;

        // Initialize RNG states
        let num_threads = 256;
        let num_blocks = config.batch_size.div_ceil(num_threads);
        let rng_state_size = config.batch_size * 64; // curandStatePhilox4_32_10_t is ~64 bytes
        let d_rng_states = self
            .device
            .alloc_zeros::<u8>(rng_state_size)
            .map_err(|e| PRCTError::GpuError(format!("Failed to allocate RNG states: {}", e)))?;

        unsafe {
            (*self.init_rng_kernel)
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (num_blocks as u32, 1, 1),
                        block_dim: (num_threads as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&d_rng_states, config.seed, config.batch_size as i32),
                )
                .map_err(|e| PRCTError::GpuError(format!("RNG init launch failed: {}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| PRCTError::GpuError(format!("RNG init sync failed: {}", e)))?;

        println!("[GPU-QUBO] GPU buffers allocated and RNG initialized");

        // Temperature schedule (geometric cooling)
        let temp_ratio = (config.t_final / config.t_initial).ln();

        let start = std::time::Instant::now();

        // Main annealing loop
        for iter in 0..config.iterations {
            let progress = iter as f64 / config.iterations as f64;
            let temperature = config.t_initial * (temp_ratio * progress).exp();

            // Batch flip evaluation
            unsafe {
                (*self.flip_kernel)
                    .clone()
                    .launch(
                        LaunchConfig {
                            grid_dim: (num_blocks as u32, 1, 1),
                            block_dim: (num_threads as u32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &d_row_ptr,
                            &d_col_idx,
                            &d_values,
                            &d_state_current,
                            &d_rng_states,
                            &d_delta_energy,
                            &d_flip_candidates,
                            config.batch_size as i32,
                            num_vars as i32,
                        ),
                    )
                    .map_err(|e| {
                        PRCTError::GpuError(format!(
                            "Flip kernel launch failed at iter {}: {}",
                            iter, e
                        ))
                    })?;
            }

            // Metropolis acceptance
            unsafe {
                (*self.metropolis_kernel)
                    .clone()
                    .launch(
                        LaunchConfig {
                            grid_dim: (num_blocks as u32, 1, 1),
                            block_dim: (num_threads as u32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &mut d_state_current,
                            &mut d_state_best,
                            &d_delta_energy,
                            &d_flip_candidates,
                            &mut d_best_energy,
                            temperature,
                            &d_rng_states,
                            config.batch_size as i32,
                            num_vars as i32,
                        ),
                    )
                    .map_err(|e| {
                        PRCTError::GpuError(format!(
                            "Metropolis kernel launch failed at iter {}: {}",
                            iter, e
                        ))
                    })?;
            }

            // Progress logging
            if (iter + 1) % 1000 == 0 {
                self.device.synchronize().map_err(|e| {
                    PRCTError::GpuError(format!("Sync failed at iter {}: {}", iter, e))
                })?;

                let elapsed = start.elapsed().as_secs_f64();
                let iter_per_sec = (iter + 1) as f64 / elapsed;
                println!(
                    "[GPU-QUBO]   Iter {}/{} | T={:.4} | {:.1} iter/s",
                    iter + 1,
                    config.iterations,
                    temperature,
                    iter_per_sec
                );
            }
        }

        // Final sync
        self.device
            .synchronize()
            .map_err(|e| PRCTError::GpuError(format!("Final sync failed: {}", e)))?;

        let elapsed = start.elapsed().as_secs_f64();
        println!("[GPU-QUBO] Annealing completed in {:.2}s", elapsed);

        // Download best solution
        let best_state_u8 = self
            .device
            .dtoh_sync_copy(&d_state_best)
            .map_err(|e| PRCTError::GpuError(format!("Failed to download result: {}", e)))?;

        let best_energy_vec = self
            .device
            .dtoh_sync_copy(&d_best_energy)
            .map_err(|e| PRCTError::GpuError(format!("Failed to download energy: {}", e)))?;

        println!("[GPU-QUBO] Best energy: {:.6}", best_energy_vec[0]);

        // Convert back to bool
        let result: Vec<bool> = best_state_u8.iter().map(|&x| x != 0).collect();

        Ok(result)
    }
}

/// Decode QUBO binary solution to graph coloring
///
/// Variables are indexed as x[v*num_colors + c] for vertex v, color c
/// Each vertex should have exactly one color bit set
pub fn qubo_solution_to_coloring(
    qubo_solution: &[bool],
    num_vertices: usize,
    num_colors: usize,
) -> Result<Vec<usize>> {
    if qubo_solution.len() != num_vertices * num_colors {
        return Err(PRCTError::InvalidInput(format!(
            "QUBO solution size {} != n*k {}",
            qubo_solution.len(),
            num_vertices * num_colors
        )));
    }

    let mut coloring = vec![0; num_vertices];
    let mut num_conflicts = 0;

    for v in 0..num_vertices {
        let mut assigned_color = None;
        let mut count = 0;

        for c in 0..num_colors {
            let idx = v * num_colors + c;
            if qubo_solution[idx] {
                assigned_color = Some(c);
                count += 1;
            }
        }

        match (count, assigned_color) {
            (1, Some(c)) => {
                coloring[v] = c;
            }
            (0, _) => {
                // No color assigned - use first available
                coloring[v] = 0;
                num_conflicts += 1;
            }
            (_, Some(c)) => {
                // Multiple colors - use first one
                coloring[v] = c;
                num_conflicts += 1;
            }
            _ => unreachable!(),
        }
    }

    if num_conflicts > 0 {
        println!(
            "[GPU-QUBO][WARN] {} vertices with constraint violations",
            num_conflicts
        );
    }

    Ok(coloring)
}

/// Main entry point for GPU QUBO simulated annealing
pub fn gpu_qubo_simulated_annealing(
    cuda_device: &Arc<CudaDevice>,
    qubo: &SparseQUBO,
    initial_state: &[bool],
    iterations: usize,
    t_initial: f64,
    t_final: f64,
    seed: u64,
) -> Result<Vec<bool>> {
    let config = GpuQuboConfig {
        iterations,
        batch_size: 256,
        t_initial,
        t_final,
        seed,
    };

    let solver = GpuQuboSolver::new(cuda_device.clone())?;
    solver.solve(qubo, initial_state, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_conversion() {
        // Simple 3x3 QUBO matrix
        let entries = vec![
            (0, 0, 1.0),
            (0, 1, 2.0),
            (1, 1, 3.0),
            (1, 2, 4.0),
            (2, 2, 5.0),
        ];

        let csr = CsrMatrix::from_qubo_coo(&entries, 3);

        assert_eq!(csr.num_rows, 3);
        assert_eq!(csr.num_cols, 3);
        assert_eq!(csr.row_ptr, vec![0, 2, 4, 5]);
        assert_eq!(csr.col_idx, vec![0, 1, 1, 2, 2]);
        assert_eq!(csr.values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_qubo_to_coloring() {
        // 3 vertices, 2 colors
        let solution = vec![
            true, false, // v0 -> c0
            false, true, // v1 -> c1
            true, false, // v2 -> c0
        ];

        let coloring = qubo_solution_to_coloring(&solution, 3, 2).unwrap();
        assert_eq!(coloring, vec![0, 1, 0]);
    }
}
