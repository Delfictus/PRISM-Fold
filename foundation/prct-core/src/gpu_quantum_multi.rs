//! Multi-GPU Distributed Quantum Annealing
//!
//! Distributes QUBO simulated annealing attempts across multiple GPUs for massive scaling.
//! Enables 80,000+ attempts with deep exploration across 8x B200 GPUs.
//!
//! Constitutional Compliance:
//! - Article V: Uses Arc<CudaDevice> per GPU
//! - Zero stubs: Full implementation, no todo!/unimplemented!
//! - Independent parallel attempts with best-solution selection
//! - Proper energy evaluation and result aggregation

use crate::errors::*;
use crate::sparse_qubo::SparseQUBO;
use cudarc::driver::CudaContext;
use shared_types::{ColoringSolution, Graph};
use std::sync::Arc;

/// Run QUBO simulated annealing attempts in parallel across multiple GPUs
///
/// Each GPU runs a subset of attempts independently, exploring different
/// regions of the solution space via different random seeds.
///
/// # Arguments
/// * `devices` - Array of CUDA contexts (from MultiGpuDevicePool)
/// * `qubo` - Sparse QUBO problem formulation
/// * `initial_state` - Starting bit assignment
/// * `total_attempts` - Total number of annealing attempts (distributed across GPUs)
/// * `depth` - Annealing depth (iterations = depth * 1000)
/// * `t_initial` - Initial temperature for annealing
/// * `t_final` - Final temperature for annealing
/// * `seed` - Base random seed (each GPU gets offset)
///
/// # Returns
/// Best bit assignment found across all GPUs
#[allow(clippy::too_many_arguments)]
pub fn quantum_annealing_multi_gpu(
    devices: &[Arc<CudaContext>],
    qubo: &SparseQUBO,
    initial_state: &[bool],
    total_attempts: usize,
    depth: usize,
    t_initial: f64,
    t_final: f64,
    seed: u64,
) -> Result<Vec<bool>> {
    let num_gpus = devices.len();

    if num_gpus == 0 {
        return Err(PRCTError::GpuError(
            "No GPUs available for multi-GPU quantum".to_string(),
        ));
    }

    println!(
        "[QUANTUM-MULTI-GPU] Distributing {} attempts across {} GPUs",
        total_attempts, num_gpus
    );
    println!(
        "[QUANTUM-MULTI-GPU] {} attempts per GPU (approx)",
        total_attempts / num_gpus
    );
    println!(
        "[QUANTUM-MULTI-GPU] Depth: {}, iterations per attempt: {}",
        depth,
        depth * 1000
    );

    let attempts_per_gpu = total_attempts / num_gpus;

    // Clone QUBO once (to avoid lifetime issues)
    let qubo_owned = qubo.clone();

    // Launch quantum annealing on each GPU in parallel
    let handles: Vec<_> = devices.iter().enumerate().map(|(gpu_idx, context)| {
        let context = context.clone();
        let qubo = qubo_owned.clone();
        let initial_state = initial_state.to_vec();
        let base_seed = seed + (gpu_idx as u64 * 1_000_000);  // Large offset per GPU

        std::thread::spawn(move || {
            println!("[QUANTUM-GPU-{}] Starting {} attempts with depth {} (seed={})",
                     gpu_idx, attempts_per_gpu, depth, base_seed);

            let start_time = std::time::Instant::now();

            let mut best_state = initial_state.clone();
            let mut best_energy = f64::MAX;
            let mut improvements = 0;

            // Run attempts on this GPU
            for attempt_idx in 0..attempts_per_gpu {
                let attempt_seed = base_seed + attempt_idx as u64;

                // Run GPU QUBO simulated annealing
                match crate::gpu_quantum_annealing::gpu_qubo_simulated_annealing(
                    &context,
                    &qubo,
                    &initial_state,
                    depth * 1000,  // iterations = depth * 1000
                    t_initial,
                    t_final,
                    attempt_seed,
                ) {
                    Ok(state) => {
                        // Evaluate energy (convert bool state to f64)
                        let state_f64: Vec<f64> = state.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
                        let energy = qubo.evaluate(&state_f64);

                        if energy < best_energy {
                            best_energy = energy;
                            best_state = state;
                            improvements += 1;

                            if improvements % 10 == 0 {
                                println!("[QUANTUM-GPU-{}] Attempt {}/{}: New best energy: {:.6}",
                                         gpu_idx, attempt_idx + 1, attempts_per_gpu, energy);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[QUANTUM-GPU-{}] Attempt {}/{} failed: {}",
                                  gpu_idx, attempt_idx + 1, attempts_per_gpu, e);
                        // Continue with remaining attempts
                    }
                }

                // Progress report every 10% of attempts
                if (attempt_idx + 1) % (attempts_per_gpu / 10).max(1) == 0 {
                    let progress = (attempt_idx + 1) as f64 / attempts_per_gpu as f64 * 100.0;
                    println!("[QUANTUM-GPU-{}] Progress: {:.0}% ({}/{}), best energy: {:.6}",
                             gpu_idx, progress, attempt_idx + 1, attempts_per_gpu, best_energy);
                }
            }

            let elapsed = start_time.elapsed();
            println!("[QUANTUM-GPU-{}] ✅ Completed {} attempts in {:.2}s, best energy: {:.6} ({} improvements)",
                     gpu_idx, attempts_per_gpu, elapsed.as_secs_f64(), best_energy, improvements);

            Ok::<(usize, Vec<bool>, f64), PRCTError>((gpu_idx, best_state, best_energy))
        })
    }).collect();

    // Gather results and pick best solution across all GPUs
    let mut global_best_state = initial_state.to_vec();
    let mut global_best_energy = f64::MAX;
    let mut gpu_energies = vec![f64::MAX; num_gpus];

    for handle in handles {
        match handle.join() {
            Ok(Ok((gpu_idx, state, energy))) => {
                gpu_energies[gpu_idx] = energy;

                if energy < global_best_energy {
                    global_best_energy = energy;
                    global_best_state = state;
                    println!(
                        "[QUANTUM-MULTI-GPU] GPU {} found new global best: {:.6}",
                        gpu_idx, energy
                    );
                }
            }
            Ok(Err(e)) => {
                eprintln!("[QUANTUM-MULTI-GPU] ❌ GPU failed: {}", e);
                return Err(e);
            }
            Err(_) => {
                return Err(PRCTError::GpuError("GPU thread panicked".to_string()));
            }
        }
    }

    println!(
        "[QUANTUM-MULTI-GPU] ✅ Best solution from {} total attempts: energy={:.6}",
        total_attempts, global_best_energy
    );

    // Print GPU performance summary
    println!("[QUANTUM-MULTI-GPU] GPU Performance Summary:");
    for (gpu_idx, energy) in gpu_energies.iter().enumerate() {
        if *energy != f64::MAX {
            let relative = (energy / global_best_energy - 1.0) * 100.0;
            println!(
                "[QUANTUM-MULTI-GPU]   GPU {}: {:.6} ({:+.2}%)",
                gpu_idx, energy, relative
            );
        }
    }

    Ok(global_best_state)
}

/// Convert QUBO bit state back to graph coloring solution
///
/// # Arguments
/// * `graph` - Input graph structure
/// * `bit_state` - QUBO bit assignment (n*k bits)
/// * `k` - Number of colors
///
/// # Returns
/// ColoringSolution extracted from bit state
pub fn extract_coloring_from_qubo(
    graph: &Graph,
    bit_state: &[bool],
    k: usize,
) -> Result<ColoringSolution> {
    let n = graph.num_vertices;

    if bit_state.len() != n * k {
        return Err(PRCTError::ColoringFailed(format!(
            "Invalid QUBO state length: expected {}, got {}",
            n * k,
            bit_state.len()
        )));
    }

    // Extract color assignments
    let mut colors = vec![0; n];
    for v in 0..n {
        // Find the color bit that is set for vertex v
        let mut color_found = false;
        for c in 0..k {
            let bit_idx = v * k + c;
            if bit_state[bit_idx] {
                colors[v] = c;
                color_found = true;
                break;
            }
        }

        // If no color found, assign first color (fallback)
        if !color_found {
            colors[v] = 0;
        }
    }

    // Count conflicts
    let mut conflicts = 0;
    for (u, v, _) in &graph.edges {
        if colors[*u] == colors[*v] {
            conflicts += 1;
        }
    }

    // Find chromatic number (max color + 1)
    let chromatic_number = colors.iter().max().unwrap_or(&0) + 1;

    Ok(ColoringSolution {
        colors,
        chromatic_number,
        conflicts,
        quality_score: 0.0,
        computation_time_ms: 0.0,
    })
}
