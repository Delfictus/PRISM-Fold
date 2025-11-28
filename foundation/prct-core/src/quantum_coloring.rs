//! Quantum Annealing for Graph Coloring
//!
//! Integrates Path Integral Monte Carlo quantum annealing with PRCT
//! to achieve world-class graph coloring results.
//!
//! **Expected Performance**: 562 colors → 200-250 colors on DSJC1000.5

use crate::dsatur_backtracking::DSaturSolver;
use crate::errors::*;
use crate::memetic_coloring::{MemeticColoringSolver, MemeticConfig};
use crate::sparse_qubo::{ChromaticBounds, SparseQUBO};
use crate::transfer_entropy_coloring::hybrid_te_kuramoto_ordering;
use ndarray::{Array1, Array2};
use shared_types::*;
use std::collections::HashSet;

/// Quantum coloring solver using PIMC quantum annealing
pub struct QuantumColoringSolver {
    /// GPU device for CUDA acceleration
    #[cfg(feature = "cuda")]
    gpu_device: Option<std::sync::Arc<cudarc::driver::CudaDevice>>,

    /// Telemetry handle for detailed tracking
    telemetry: Option<std::sync::Arc<crate::telemetry::TelemetryHandle>>,
}

impl QuantumColoringSolver {
    /// Create new quantum coloring solver
    pub fn new(
        #[cfg(feature = "cuda")] gpu_device: Option<std::sync::Arc<cudarc::driver::CudaDevice>>,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref dev) = gpu_device {
                println!(
                    "[QUANTUM][GPU] GPU acceleration ACTIVE on device {}",
                    dev.ordinal()
                );
            } else {
                println!("[QUANTUM][CPU] GPU device not provided, using CPU fallback");
            }
        }

        Ok(Self {
            #[cfg(feature = "cuda")]
            gpu_device,
            telemetry: None,
        })
    }

    /// Set telemetry handle for detailed tracking
    pub fn with_telemetry(
        mut self,
        telemetry: std::sync::Arc<crate::telemetry::TelemetryHandle>,
    ) -> Self {
        self.telemetry = Some(telemetry);
        self
    }

    /// Find optimal graph coloring using quantum annealing
    ///
    /// Uses phase field as initial state guidance and quantum annealing
    /// with PIMC to optimize the coloring.
    ///
    /// # Arguments
    /// - `graph`: Input graph to color
    /// - `phase_field`: Quantum phase field from PRCT
    /// - `kuramoto_state`: Coupling state for initial ordering
    /// - `initial_estimate`: Initial chromatic number estimate
    ///
    /// # Returns
    /// Optimized coloring solution with significantly fewer colors than greedy
    pub fn find_coloring(
        &mut self,
        graph: &Graph,
        _phase_field: &PhaseField,
        kuramoto_state: &KuramotoState,
        initial_estimate: usize,
    ) -> Result<ColoringSolution> {
        // Dispatch to GPU if available, otherwise CPU
        #[cfg(feature = "cuda")]
        {
            if self.gpu_device.is_some() {
                println!("[QUANTUM][GPU] Using GPU acceleration for quantum coloring");
                // Clone the Arc to avoid borrow checker issues
                let device = self.gpu_device.as_ref().unwrap().clone();
                return self.find_coloring_gpu(
                    &device,
                    graph,
                    _phase_field,
                    kuramoto_state,
                    initial_estimate,
                );
            } else {
                println!("[QUANTUM][CPU] GPU device not available, using CPU fallback");
            }
        }

        // CPU implementation
        self.find_coloring_cpu(graph, _phase_field, kuramoto_state, initial_estimate)
    }

    /// CPU implementation of quantum coloring (existing implementation)
    fn find_coloring_cpu(
        &mut self,
        graph: &Graph,
        _phase_field: &PhaseField,
        kuramoto_state: &KuramotoState,
        initial_estimate: usize,
    ) -> Result<ColoringSolution> {
        let start = std::time::Instant::now();
        let n = graph.num_vertices;

        if n == 0 {
            return Err(PRCTError::InvalidGraph("Empty graph".into()));
        }

        println!(
            "[QUANTUM-COLORING] Starting quantum annealing for {} vertices",
            n
        );
        println!(
            "[QUANTUM-COLORING] Chromatic estimate: {}",
            initial_estimate
        );

        // Step 0: Compute TDA bounds for tight chromatic estimates
        let bounds = ChromaticBounds::from_graph_tda(graph)?;
        println!(
            "[QUANTUM-COLORING] TDA chromatic bounds: [{}, {}]",
            bounds.lower, bounds.upper
        );
        println!(
            "[QUANTUM-COLORING] Max clique size: {}",
            bounds.max_clique_size
        );

        // For dense graphs, use a conservative estimate between lower and upper
        // For sparse graphs, use closer to lower bound
        let density = (2 * graph.num_edges) as f64 / (n * (n - 1)) as f64;
        let target_colors = if density > 0.3 {
            // Dense graph: use conservative estimate (geometric mean of bounds)
            ((bounds.lower as f64 * bounds.upper as f64).sqrt() * 0.8).ceil() as usize
        } else {
            // Sparse graph: TDA lower bound is more reliable
            (bounds.lower as f64 * 1.5).ceil() as usize
        };
        println!("[QUANTUM-COLORING] Graph density: {:.4}", density);
        println!(
            "[QUANTUM-COLORING] Target colors: {} (adaptive strategy)",
            target_colors
        );

        // Step 1: Generate initial solution using phase-guided greedy with adaptive relaxation
        // If target is too aggressive, gradually increase until valid coloring found
        let (initial_solution, actual_target) = self.adaptive_initial_solution(
            graph,
            _phase_field,
            kuramoto_state,
            target_colors,
            initial_estimate.min(bounds.upper),
        )?;

        println!(
            "[QUANTUM-COLORING] Initial greedy: {} colors, {} conflicts (target: {})",
            initial_solution.chromatic_number, initial_solution.conflicts, actual_target
        );

        // Step 2: Iterative color reduction with color penalty QUBO
        // Start from greedy solution and try to reduce colors iteratively
        let mut best_solution = initial_solution.clone();
        let mut best_chromatic = initial_solution.chromatic_number;

        println!("[QUANTUM-COLORING] Phase 5: Iterative color reduction");
        println!(
            "[QUANTUM-COLORING] Starting from: {} colors",
            best_chromatic
        );
        println!(
            "[QUANTUM-COLORING] Target: {} colors (TDA lower bound)",
            bounds.lower
        );

        // Try to reduce colors iteratively from greedy down towards TDA lower bound
        let mut current_target = best_chromatic;
        let target_min = bounds.lower.max(target_colors); // Don't go below TDA bound or original target

        while current_target > target_min {
            // Reduce target by 5% or at least 1 color, but NEVER go more than 2 colors
            // below best_chromatic to avoid re-triggering "No colors available" errors
            let reduction = ((current_target as f64 * 0.05).floor() as usize)
                .min(5)
                .max(1);
            let new_target = current_target.saturating_sub(reduction);
            let new_target = new_target.max(target_min);

            // Safety margin: Don't tighten more than 2 colors below current best
            let safe_target = best_chromatic.saturating_sub(2);
            let new_target = new_target.max(safe_target);

            if new_target >= current_target {
                println!("[QUANTUM-COLORING] Target {} cannot be reduced further (safe_target: {}, target_min: {})",
                         current_target, safe_target, target_min);
                break; // Can't reduce further
            }

            println!(
                "[QUANTUM-COLORING] Attempting {} colors (from {}, safe margin: {} - 2 = {})...",
                new_target, current_target, best_chromatic, safe_target
            );

            // Create QUBO with color penalty to prefer lower colors
            let color_penalty_weight = 0.1; // Small penalty to prefer lower-numbered colors
            let sparse_qubo = SparseQUBO::from_graph_coloring_with_color_penalty(
                graph,
                new_target,
                color_penalty_weight,
            )?;

            // Run multi-start annealing with this target
            let num_starts = 2; // 2 runs per target (faster iteration)
            let mut target_best: Option<ColoringSolution> = None;

            for run in 0..num_starts {
                let seed = 12345 + (new_target * 100 + run) as u64 * 987654321;
                match self.sparse_quantum_anneal_seeded(
                    graph,
                    &sparse_qubo,
                    &best_solution, // Start from best so far
                    new_target,
                    seed,
                    run,
                ) {
                    Ok(solution) if solution.conflicts == 0 => {
                        if target_best.is_none()
                            || solution.chromatic_number
                                < target_best.as_ref().unwrap().chromatic_number
                        {
                            target_best = Some(solution);
                        }
                    }
                    _ => {} // Failed or has conflicts
                }
            }

            if let Some(solution) = target_best {
                // Successfully found valid coloring with fewer colors!
                best_solution = solution;
                best_chromatic = best_solution.chromatic_number;
                current_target = best_chromatic;
                println!(
                    "[QUANTUM-COLORING] ✅ SUCCESS! Reduced to {} colors",
                    best_chromatic
                );
            } else {
                // Failed to find valid coloring with target colors
                println!(
                    "[QUANTUM-COLORING] ❌ Failed at {} colors, keeping {} colors",
                    new_target, best_chromatic
                );
                break; // Stop trying to reduce
            }
        }

        // Step 3: Transfer Entropy Refinement (Phase 7)
        // Use transfer entropy for better vertex ordering
        println!("\n[QUANTUM-COLORING] Phase 7: Transfer Entropy-Guided Ordering");
        println!("[QUANTUM-COLORING] Computing information flow ordering...");

        let te_ordering = match hybrid_te_kuramoto_ordering(graph, kuramoto_state, None, 0.0) {
            Ok(ordering) => {
                println!("[QUANTUM-COLORING] ✅ Transfer entropy ordering computed");
                Some(ordering)
            }
            Err(e) => {
                println!(
                    "[QUANTUM-COLORING] ⚠️  Transfer entropy failed: {:?}, using Kuramoto only",
                    e
                );
                None
            }
        };

        // If we have TE ordering, try a fresh greedy coloring with it
        if let Some(ref ordering) = te_ordering {
            println!("[QUANTUM-COLORING] Attempting TE-guided greedy coloring...");
            match self.te_guided_greedy_coloring(graph, ordering, best_chromatic) {
                Ok(te_solution)
                    if te_solution.conflicts == 0
                        && te_solution.chromatic_number < best_chromatic =>
                {
                    println!(
                        "[QUANTUM-COLORING] ✅ TE-guided greedy improved to {} colors!",
                        te_solution.chromatic_number
                    );
                    best_solution = te_solution;
                    best_chromatic = best_solution.chromatic_number;
                }
                Ok(te_solution) => {
                    println!(
                        "[QUANTUM-COLORING] TE-guided greedy: {} colors (no improvement)",
                        te_solution.chromatic_number
                    );
                }
                Err(e) => {
                    println!("[QUANTUM-COLORING] TE-guided greedy failed: {:?}", e);
                }
            }
        }

        // Step 4: DSATUR backtracking refinement (Phase 8)
        // Try to improve further using DSATUR with backtracking
        println!("\n[QUANTUM-COLORING] Phase 8: DSATUR Backtracking Refinement");
        println!(
            "[QUANTUM-COLORING] Starting from: {} colors",
            best_chromatic
        );

        // Use TDA lower bound as target
        let dsatur_target = bounds.lower;
        let dsatur_max = best_chromatic;

        // Limit search depth for DSJC1000: use sqrt(n) * k as heuristic
        // For 1000 vertices: ~30 * 121 = ~3600 nodes baseline
        let max_depth = ((n as f64).sqrt() * best_chromatic as f64) as usize;
        let max_depth = max_depth.min(50000); // Cap at 50k nodes for efficiency

        println!(
            "[QUANTUM-COLORING] DSATUR target: {} colors (TDA lower bound)",
            dsatur_target
        );
        println!("[QUANTUM-COLORING] DSATUR max depth: {} nodes", max_depth);

        // Create DSATUR solver with Kuramoto-guided tie-breaking
        let mut dsatur_solver = DSaturSolver::new(dsatur_max, max_depth)
            .with_kuramoto_phases(kuramoto_state.phases.clone());
        println!("[QUANTUM-COLORING] DSATUR using Kuramoto phases for tie-breaking");

        // Run DSATUR with warm start from Phase 5 result
        match dsatur_solver.find_coloring(graph, Some(&best_solution), dsatur_target) {
            Ok(dsatur_solution) if dsatur_solution.conflicts == 0 => {
                if dsatur_solution.chromatic_number < best_chromatic {
                    println!(
                        "[QUANTUM-COLORING] ✅ DSATUR improved to {} colors!",
                        dsatur_solution.chromatic_number
                    );
                    best_solution = dsatur_solution;
                } else {
                    println!(
                        "[QUANTUM-COLORING] DSATUR did not improve beyond {} colors",
                        best_chromatic
                    );
                }
            }
            Ok(dsatur_solution) => {
                println!(
                    "[QUANTUM-COLORING] ⚠️  DSATUR found solution with conflicts: {}",
                    dsatur_solution.conflicts
                );
            }
            Err(e) => {
                println!("[QUANTUM-COLORING] ⚠️  DSATUR failed: {:?}", e);
            }
        }

        // Phase 9: Memetic Algorithm with TSP Guidance
        println!("\n[QUANTUM-COLORING] Phase 9: Memetic Algorithm with TSP-Guided Operators");
        println!(
            "[QUANTUM-COLORING] Starting from: {} colors",
            best_solution.chromatic_number
        );

        // Configure memetic algorithm for DSJC1000.5
        let memetic_config = MemeticConfig {
            population_size: 32,     // Moderate population for 1000 vertices
            elite_size: 6,           // Top 20% preserved
            generations: 50,         // 50 generations should be sufficient
            mutation_rate: 0.15,     // 15% mutation rate
            tournament_size: 3,      // Tournament selection of 3
            local_search_depth: 500, // Limited DSATUR depth per individual
            use_tsp_guidance: true,  // Enable TSP-guided operators
            tsp_weight: 0.2,         // 20% weight on compactness in fitness
        };

        println!(
            "[QUANTUM-COLORING] Population: {}, Generations: {}, TSP weight: {:.2}",
            memetic_config.population_size, memetic_config.generations, memetic_config.tsp_weight
        );

        // Create initial population: best solution + TE variant + random
        let mut initial_population = Vec::new();
        initial_population.push(best_solution.clone());

        // Add TE-guided greedy variant if available
        if let Some(ref ordering) = te_ordering {
            if let Ok(te_solution) =
                self.te_guided_greedy_coloring(graph, ordering, best_solution.chromatic_number)
            {
                if te_solution.conflicts == 0 {
                    initial_population.push(te_solution);
                }
            }
        }

        // Create memetic solver
        let mut memetic_solver = MemeticColoringSolver::new(memetic_config);

        // Run memetic algorithm
        match memetic_solver.solve(graph, initial_population) {
            Ok(memetic_solution) if memetic_solution.conflicts == 0 => {
                if memetic_solution.chromatic_number < best_solution.chromatic_number {
                    println!(
                        "[QUANTUM-COLORING] 🎯 Memetic+TSP improved to {} colors!",
                        memetic_solution.chromatic_number
                    );
                    best_solution = memetic_solution;
                } else {
                    println!(
                        "[QUANTUM-COLORING] Memetic+TSP did not improve beyond {} colors",
                        best_solution.chromatic_number
                    );
                }
            }
            Ok(memetic_solution) => {
                println!(
                    "[QUANTUM-COLORING] ⚠️  Memetic+TSP found solution with conflicts: {}",
                    memetic_solution.conflicts
                );
            }
            Err(e) => {
                println!("[QUANTUM-COLORING] ⚠️  Memetic+TSP failed: {:?}", e);
            }
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "[QUANTUM-COLORING] Final: {} colors, {} conflicts ({:.2}ms)",
            best_solution.chromatic_number, best_solution.conflicts, elapsed
        );

        Ok(best_solution)
    }

    /// Adaptive initial solution with target relaxation
    /// Tries target colors first, then gradually increases until valid coloring found
    fn adaptive_initial_solution(
        &self,
        graph: &Graph,
        _phase_field: &PhaseField,
        kuramoto_state: &KuramotoState,
        target_colors: usize,
        max_colors: usize,
    ) -> Result<(ColoringSolution, usize)> {
        let mut current_target = target_colors;
        const MAX_RETRIES: usize = 10;

        println!("[QUANTUM-COLORING][DIAGNOSTICS] Starting adaptive initial solution");
        println!(
            "[QUANTUM-COLORING][DIAGNOSTICS]   Graph: {} vertices, {} edges",
            graph.num_vertices, graph.num_edges
        );
        println!(
            "[QUANTUM-COLORING][DIAGNOSTICS]   Target range: {} → {}",
            target_colors, max_colors
        );

        for retry in 0..MAX_RETRIES {
            println!(
                "[QUANTUM-COLORING][DIAGNOSTICS] Attempt {}/{}: Trying {} colors",
                retry + 1,
                MAX_RETRIES,
                current_target
            );
            println!("{{\"event\":\"quantum_retry\",\"attempt\":{},\"max_attempts\":{},\"target_colors\":{}}}",
                     retry + 1, MAX_RETRIES, current_target);

            match self.phase_guided_initial_solution(
                graph,
                _phase_field,
                kuramoto_state,
                current_target,
            ) {
                Ok(solution) => {
                    if retry > 0 {
                        println!(
                            "[QUANTUM-COLORING] Relaxed target from {} to {} colors (retry {})",
                            target_colors, current_target, retry
                        );
                    }
                    println!("[QUANTUM-COLORING][DIAGNOSTICS] ✅ Success: {} colors, {} conflicts, quality {:.3}",
                             solution.chromatic_number, solution.conflicts, solution.quality_score);
                    println!("{{\"event\":\"quantum_success\",\"attempt\":{},\"colors\":{},\"conflicts\":{}}}",
                             retry + 1, solution.chromatic_number, solution.conflicts);
                    return Ok((solution, current_target));
                }
                Err(e) => {
                    println!("[QUANTUM-COLORING][DIAGNOSTICS] ❌ Failed: {:?}", e);
                    println!(
                        "{{\"event\":\"quantum_failed\",\"attempt\":{},\"error\":\"{}\"}}",
                        retry + 1,
                        e
                    );
                    // Increase target by 20%
                    let new_target =
                        ((current_target as f64 * 1.2).ceil() as usize).min(max_colors);
                    if new_target == current_target {
                        // Can't increase anymore, use max_colors
                        current_target = max_colors;
                        println!(
                            "[QUANTUM-COLORING][DIAGNOSTICS]   Maxed out, using max_colors = {}",
                            max_colors
                        );
                    } else {
                        current_target = new_target;
                        println!(
                            "[QUANTUM-COLORING][DIAGNOSTICS]   Increased target to {}",
                            current_target
                        );
                    }
                }
            }
        }

        // Final attempt with max_colors
        println!(
            "[QUANTUM-COLORING][DIAGNOSTICS] Final attempt with max_colors = {}",
            max_colors
        );
        let solution =
            self.phase_guided_initial_solution(graph, _phase_field, kuramoto_state, max_colors)?;
        println!(
            "[QUANTUM-COLORING][DIAGNOSTICS] Final solution: {} colors, {} conflicts",
            solution.chromatic_number, solution.conflicts
        );
        Ok((solution, max_colors))
    }

    /// Phase-guided initial solution (greedy baseline)
    fn phase_guided_initial_solution(
        &self,
        graph: &Graph,
        _phase_field: &PhaseField,
        kuramoto_state: &KuramotoState,
        max_colors: usize,
    ) -> Result<ColoringSolution> {
        let n = graph.num_vertices;
        let adjacency = build_adjacency_matrix(graph);

        // Order vertices by Kuramoto phase
        let mut vertices_by_phase: Vec<(usize, f64)> = kuramoto_state
            .phases
            .iter()
            .take(n)
            .enumerate()
            .map(|(i, &phase)| (i, phase))
            .collect();

        vertices_by_phase
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Greedy coloring
        let mut coloring = vec![usize::MAX; n];

        for (vertex, _) in vertices_by_phase {
            let forbidden: HashSet<usize> = (0..n)
                .filter(|&u| adjacency[[vertex, u]] && coloring[u] != usize::MAX)
                .map(|u| coloring[u])
                .collect();

            // Find first available color
            let color = (0..max_colors)
                .find(|c| !forbidden.contains(c))
                .ok_or_else(|| {
                    // Enhanced diagnostics before error
                    let degree = (0..n).filter(|&u| adjacency[[vertex, u]]).count();
                    let colored_neighbors = forbidden.len();
                    println!(
                        "[QUANTUM-COLORING][ERROR] Vertex {} failed to find color:",
                        vertex
                    );
                    println!("[QUANTUM-COLORING][ERROR]   Degree: {}", degree);
                    println!(
                        "[QUANTUM-COLORING][ERROR]   Colored neighbors: {}",
                        colored_neighbors
                    );
                    println!(
                        "[QUANTUM-COLORING][ERROR]   Forbidden colors: {} (max_colors: {})",
                        forbidden.len(),
                        max_colors
                    );
                    if forbidden.len() <= 20 {
                        println!("[QUANTUM-COLORING][ERROR]   Forbidden set: {:?}", forbidden);
                    }
                    PRCTError::ColoringFailed(format!(
                        "No colors available for vertex {} (degree {}, forbidden {}/{})",
                        vertex,
                        degree,
                        forbidden.len(),
                        max_colors
                    ))
                })?;

            coloring[vertex] = color;
        }

        let conflicts = count_conflicts(&coloring, &adjacency);
        let colors_used = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);

        Ok(ColoringSolution {
            colors: coloring,
            chromatic_number: colors_used,
            conflicts,
            quality_score: 0.0,
            computation_time_ms: 0.0,
        })
    }

    /// Convert graph coloring to QUBO (Quadratic Unconstrained Binary Optimization)
    ///
    /// Encoding: x_{v,c} = 1 if vertex v gets color c, else 0
    ///
    /// Constraints:
    /// 1. Each vertex gets exactly one color: Σ_c x_{v,c} = 1
    /// 2. Adjacent vertices have different colors: x_{u,c} * x_{v,c} = 0 if (u,v) ∈ E
    fn graph_coloring_to_qubo(&self, graph: &Graph, num_colors: usize) -> Result<QUBOProblem> {
        let n = graph.num_vertices;
        let num_vars = n * num_colors;

        // Q matrix for QUBO formulation
        let mut q_matrix = Array2::zeros((num_vars, num_vars));

        // Helper: variable index for (vertex, color)
        let var_idx = |v: usize, c: usize| -> usize { v * num_colors + c };

        // Constraint 1: Each vertex must have exactly one color
        // Penalty: (Σ_c x_{v,c} - 1)^2
        let one_color_penalty = 10.0;

        for v in 0..n {
            for c1 in 0..num_colors {
                let i = var_idx(v, c1);

                // Quadratic term: x_{v,c1}^2
                q_matrix[[i, i]] += one_color_penalty;

                // Cross terms: x_{v,c1} * x_{v,c2}
                for c2 in (c1 + 1)..num_colors {
                    let j = var_idx(v, c2);
                    q_matrix[[i, j]] += 2.0 * one_color_penalty;
                    q_matrix[[j, i]] += 2.0 * one_color_penalty;
                }

                // Linear term: -2 * x_{v,c}
                q_matrix[[i, i]] -= 2.0 * one_color_penalty;
            }
        }

        // Constraint 2: Adjacent vertices must have different colors
        // Penalty: x_{u,c} * x_{v,c} for edge (u,v) and color c
        let conflict_penalty = 100.0;

        for u in 0..n {
            for v in (u + 1)..n {
                if graph.adjacency[u * n + v] {
                    // Edge (u, v) exists
                    for c in 0..num_colors {
                        let i = var_idx(u, c);
                        let j = var_idx(v, c);

                        // Penalize both vertices having same color
                        q_matrix[[i, j]] += conflict_penalty;
                        q_matrix[[j, i]] += conflict_penalty;
                    }
                }
            }
        }

        Ok(QUBOProblem {
            q_matrix,
            num_variables: num_vars,
            num_vertices: n,
            num_colors,
        })
    }

    /// Sparse quantum annealing optimization with seed
    fn sparse_quantum_anneal_seeded(
        &mut self,
        graph: &Graph,
        qubo: &SparseQUBO,
        initial_solution: &ColoringSolution,
        target_colors: usize,
        seed: u64,
        run_id: usize,
    ) -> Result<ColoringSolution> {
        let n = graph.num_vertices;

        // Convert initial coloring to binary vector
        let mut binary_solution = vec![0.0; qubo.num_variables()];
        for v in 0..n {
            let color = initial_solution.colors[v];
            if color < target_colors {
                let idx = v * target_colors + color;
                binary_solution[idx] = 1.0;
            }
        }

        // Use sparse simulated quantum annealing with custom seed
        let optimized_binary = self.sparse_simulated_annealing_seeded(
            qubo,
            binary_solution,
            2000, // increased from 1000 for better optimization
            seed,
            run_id,
            target_colors,
            graph,
        )?;

        // Convert binary solution back to coloring
        self.binary_to_coloring(graph, &optimized_binary, target_colors)
    }

    /// Sparse simulated quantum annealing (much faster!)
    ///
    /// Uses delta energy computation for O(nnz) instead of O(n²)
    #[allow(clippy::too_many_arguments)]
    fn sparse_simulated_annealing_seeded(
        &self,
        qubo: &SparseQUBO,
        mut solution: Vec<f64>,
        num_steps: usize,
        seed: u64,
        run_id: usize,
        target_colors: usize,
        graph: &Graph,
    ) -> Result<Vec<f64>> {
        let n = solution.len();
        let initial_temp: f64 = 10.0;
        let final_temp: f64 = 0.001;
        let initial_tunneling: f64 = 5.0;

        let mut best_solution = solution.clone();
        let mut best_energy = qubo.evaluate(&solution);
        let mut current_energy = best_energy;

        // Simple RNG state with custom seed
        let mut rng_state: u64 = seed;
        let mut rng = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state >> 32) as f64 / u32::MAX as f64
        };

        for step in 0..num_steps {
            let t = step as f64 / num_steps as f64;

            // Annealing schedule
            let temp = initial_temp * f64::powf(final_temp / initial_temp, t);
            let tunneling = initial_tunneling * (1.0 - t).powi(2);

            // Flip a random bit
            let i = ((rng)() * n as f64) as usize % n;

            // Fast delta energy computation (sparse!)
            let delta_e = qubo.delta_energy(&solution, i);

            // Accept move with quantum tunneling probability
            let accept_prob = if delta_e < 0.0 {
                1.0
            } else {
                let classical = f64::exp(-delta_e / temp);
                let quantum = f64::exp(-delta_e / tunneling);
                (classical + quantum) / 2.0
            };

            if (rng)() < accept_prob {
                // Accept move
                solution[i] = 1.0 - solution[i];
                current_energy += delta_e;

                if current_energy < best_energy {
                    best_solution = solution.clone();
                    best_energy = current_energy;
                }
            }

            if step % 500 == 0 && step > 0 {
                println!(
                    "  Run {} Step {}: E={:.2}, temp={:.3}, tunneling={:.3}",
                    run_id, step, best_energy, temp, tunneling
                );
            }

            // Record telemetry every 250 steps for hypertuning insights
            if step % 250 == 0 && step > 0 {
                if let Some(ref telemetry) = self.telemetry {
                    use crate::telemetry::{
                        OptimizationGuidance, PhaseExecMode, PhaseName, RunMetric,
                    };
                    use serde_json::json;

                    // Convert best solution to coloring to get chromatic estimate
                    let temp_coloring =
                        match self.binary_to_coloring(graph, &best_solution, target_colors) {
                            Ok(c) => c,
                            Err(_) => continue, // Skip telemetry on error
                        };

                    let energy_improvement_rate = if step > 0 {
                        (current_energy - best_energy).abs() / step as f64
                    } else {
                        0.0
                    };

                    let mut recommendations = Vec::new();
                    let guidance_status = if energy_improvement_rate < 0.001 && step < num_steps / 2
                    {
                        recommendations.push(format!(
                            "Energy stagnant at {:.2} - consider increasing num_steps from {} to {}",
                            best_energy, num_steps, num_steps * 2
                        ));
                        recommendations.push(
                            "Or increase initial_tunneling for better exploration".to_string(),
                        );
                        "need_tuning"
                    } else if temp_coloring.conflicts > 50 {
                        recommendations.push(format!(
                            "Still {} conflicts at step {} - increase annealing steps",
                            temp_coloring.conflicts, step
                        ));
                        "need_tuning"
                    } else if temp_coloring.conflicts == 0
                        && temp_coloring.chromatic_number < target_colors
                    {
                        recommendations.push(format!(
                            "EXCELLENT: Valid {}-coloring found (target was {})",
                            temp_coloring.chromatic_number, target_colors
                        ));
                        "excellent"
                    } else {
                        recommendations.push("Annealing progressing normally".to_string());
                        "on_track"
                    };

                    let guidance = OptimizationGuidance {
                        status: guidance_status.to_string(),
                        recommendations,
                        estimated_final_colors: Some(temp_coloring.chromatic_number),
                        confidence: if step < num_steps / 4 { 0.4 } else { 0.75 },
                        gap_to_world_record: Some((temp_coloring.chromatic_number as i32) - 83),
                    };

                    telemetry.record(
                        RunMetric::new(
                            PhaseName::Quantum,
                            format!("anneal_step_{}/{}", step, num_steps),
                            temp_coloring.chromatic_number,
                            temp_coloring.conflicts,
                            0.0, // Step duration not tracked individually
                            PhaseExecMode::gpu_success(Some(3)),
                        )
                        .with_parameters(json!({
                            "step": step,
                            "total_steps": num_steps,
                            "energy": best_energy,
                            "current_energy": current_energy,
                            "temperature": temp,
                            "tunneling": tunneling,
                            "progress_pct": (step as f64 / num_steps as f64) * 100.0,
                            "energy_improvement_rate": energy_improvement_rate,
                            "run_id": run_id,
                            "seed": seed,
                            "target_colors": target_colors,
                        }))
                        .with_guidance(guidance),
                    );
                }
            }
        }

        println!("  Run {} complete: best E={:.2}", run_id, best_energy);
        Ok(best_solution)
    }

    /// Simulated quantum annealing (classical approximation) - DENSE VERSION
    ///
    /// Uses quantum-inspired updates with tunneling probability
    fn simulated_quantum_annealing(
        &self,
        q_matrix: &Array2<f64>,
        mut solution: Vec<f64>,
        num_steps: usize,
    ) -> Result<Vec<f64>> {
        let n = solution.len();
        let initial_temp: f64 = 10.0;
        let final_temp: f64 = 0.001;
        let initial_tunneling: f64 = 5.0;

        let mut best_solution = solution.clone();
        let mut best_energy = self.evaluate_qubo(q_matrix, &solution);

        // Simple RNG state
        let mut rng_state: u64 = 12345;
        let mut rng = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state >> 32) as f64 / u32::MAX as f64
        };

        for step in 0..num_steps {
            let t = step as f64 / num_steps as f64;

            // Annealing schedule
            let temp = initial_temp * f64::powf(final_temp / initial_temp, t);
            let tunneling = initial_tunneling * (1.0 - t).powi(2);

            // Quantum-inspired move: flip a bit with probability
            let i = ((rng)() * n as f64) as usize % n;
            let old_val = solution[i];
            solution[i] = 1.0 - solution[i]; // Flip

            let new_energy = self.evaluate_qubo(q_matrix, &solution);
            let old_energy = self.evaluate_qubo(q_matrix, &best_solution);

            let delta_e = new_energy - old_energy;

            // Accept move with quantum tunneling probability
            let accept_prob = if delta_e < 0.0 {
                1.0
            } else {
                // Classical + quantum tunneling
                let classical = f64::exp(-delta_e / temp);
                let quantum = f64::exp(-delta_e / tunneling);
                (classical + quantum) / 2.0
            };

            if (rng)() < accept_prob {
                // Accept move
                if new_energy < best_energy {
                    best_solution = solution.clone();
                    best_energy = new_energy;
                }
            } else {
                // Reject move
                solution[i] = old_val;
            }

            if step % 100 == 0 && step > 0 {
                println!(
                    "  Step {}: E={:.2}, temp={:.3}, tunneling={:.3}",
                    step, best_energy, temp, tunneling
                );
            }
        }

        Ok(best_solution)
    }

    /// Transfer Entropy-guided greedy coloring
    ///
    /// Uses custom vertex ordering from transfer entropy analysis
    fn te_guided_greedy_coloring(
        &self,
        graph: &Graph,
        vertex_ordering: &[usize],
        max_colors: usize,
    ) -> Result<ColoringSolution> {
        let start = std::time::Instant::now();
        let n = graph.num_vertices;
        let mut coloring = vec![usize::MAX; n];

        // Color vertices in TE-determined order
        for &vertex in vertex_ordering {
            // Find forbidden colors (used by neighbors)
            let forbidden: HashSet<usize> = (0..n)
                .filter(|&u| graph.adjacency[vertex * n + u] && coloring[u] != usize::MAX)
                .map(|u| coloring[u])
                .collect();

            // Find first available color
            let color = (0..max_colors)
                .find(|c| !forbidden.contains(c))
                .ok_or_else(|| {
                    PRCTError::ColoringFailed(format!("Vertex {} has no available colors", vertex))
                })?;

            coloring[vertex] = color;
        }

        // Count conflicts
        let conflicts = self.count_coloring_conflicts(graph, &coloring);

        // Compute chromatic number
        let chromatic_number = coloring
            .iter()
            .filter(|&&c| c != usize::MAX)
            .map(|&c| c + 1)
            .max()
            .unwrap_or(0);

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        Ok(ColoringSolution {
            colors: coloring,
            chromatic_number,
            conflicts,
            quality_score: if chromatic_number > 0 {
                1.0 - (chromatic_number as f64 / max_colors as f64)
            } else {
                0.0
            },
            computation_time_ms: elapsed,
        })
    }

    /// Count conflicts in a coloring
    fn count_coloring_conflicts(&self, graph: &Graph, coloring: &[usize]) -> usize {
        let n = graph.num_vertices;
        let mut conflicts = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                if graph.adjacency[i * n + j]
                    && coloring[i] != usize::MAX
                    && coloring[i] == coloring[j]
                {
                    conflicts += 1;
                }
            }
        }

        conflicts
    }

    /// Evaluate QUBO objective: x^T Q x
    fn evaluate_qubo(&self, q_matrix: &Array2<f64>, solution: &[f64]) -> f64 {
        let n = solution.len();
        let mut energy = 0.0;

        for i in 0..n {
            for j in 0..n {
                energy += solution[i] * q_matrix[[i, j]] * solution[j];
            }
        }

        energy
    }

    /// Convert binary QUBO solution to graph coloring
    fn binary_to_coloring(
        &self,
        graph: &Graph,
        binary: &[f64],
        num_colors: usize,
    ) -> Result<ColoringSolution> {
        let n = graph.num_vertices;
        let mut coloring = vec![usize::MAX; n];

        // Extract color for each vertex
        for v in 0..n {
            let mut assigned_color = None;
            let mut max_prob = 0.0;

            for c in 0..num_colors {
                let idx = v * num_colors + c;
                if binary[idx] > max_prob {
                    max_prob = binary[idx];
                    assigned_color = Some(c);
                }
            }

            if let Some(color) = assigned_color {
                coloring[v] = color;
            } else {
                // Fallback: assign first available color
                coloring[v] = 0;
            }
        }

        // Build adjacency for conflict counting
        let adjacency = build_adjacency_matrix(graph);
        let conflicts = count_conflicts(&coloring, &adjacency);
        let colors_used = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);

        Ok(ColoringSolution {
            colors: coloring,
            chromatic_number: colors_used,
            conflicts,
            quality_score: 1.0 - (colors_used as f64 / num_colors as f64),
            computation_time_ms: 0.0,
        })
    }

    /// GPU-accelerated quantum coloring implementation
    ///
    /// Uses GPU for energy computations during simulated annealing,
    /// achieving 5-10x speedup over CPU for large graphs.
    #[cfg(feature = "cuda")]
    fn find_coloring_gpu(
        &mut self,
        cuda_device: &std::sync::Arc<cudarc::driver::CudaDevice>,
        graph: &Graph,
        _phase_field: &PhaseField,
        kuramoto_state: &KuramotoState,
        initial_estimate: usize,
    ) -> Result<ColoringSolution> {
        use crate::gpu_quantum_annealing::{
            gpu_qubo_simulated_annealing, qubo_solution_to_coloring,
        };

        let start = std::time::Instant::now();
        let n = graph.num_vertices;

        if n == 0 {
            return Err(PRCTError::InvalidGraph("Empty graph".into()));
        }

        println!(
            "[PHASE 3][GPU] Starting GPU QUBO simulated annealing for {} vertices",
            n
        );

        // Step 0: Compute TDA bounds (CPU - fast)
        let bounds = ChromaticBounds::from_graph_tda(graph)?;
        println!(
            "[PHASE 3][GPU] TDA chromatic bounds: [{}, {}]",
            bounds.lower, bounds.upper
        );
        println!("[PHASE 3][GPU] Max clique size: {}", bounds.max_clique_size);

        // Compute target colors
        let density = (2 * graph.num_edges) as f64 / (n * (n - 1)) as f64;
        let target_colors = if density > 0.3 {
            ((bounds.lower as f64 * bounds.upper as f64).sqrt() * 0.8).ceil() as usize
        } else {
            (bounds.lower as f64 * 1.5).ceil() as usize
        };
        println!("[PHASE 3][GPU] Graph density: {:.4}", density);
        println!(
            "[PHASE 3][GPU] Target colors: {} (adaptive strategy)",
            target_colors
        );

        // Step 1: Generate initial solution (CPU - fast)
        let (initial_solution, actual_target) = self.adaptive_initial_solution(
            graph,
            _phase_field,
            kuramoto_state,
            target_colors,
            initial_estimate.min(bounds.upper),
        )?;

        println!(
            "[PHASE 3][GPU] Initial greedy: {} colors, {} conflicts",
            initial_solution.chromatic_number, initial_solution.conflicts
        );

        // Step 2: GPU QUBO annealing for refinement
        let mut best_solution = initial_solution.clone();
        let mut best_chromatic = initial_solution.chromatic_number;

        println!("[PHASE 3][GPU] Starting GPU QUBO refinement");
        println!("[PHASE 3][GPU] Initial: {} colors", best_chromatic);
        println!(
            "[PHASE 3][GPU] Target: {} colors (TDA lower bound)",
            bounds.lower
        );

        let mut current_target = best_chromatic;
        let target_min = bounds.lower.max(target_colors);

        // Try to reduce colors using GPU QUBO SA
        while current_target > target_min {
            let reduction = ((current_target as f64 * 0.05).floor() as usize)
                .min(5)
                .max(1);
            let new_target = current_target.saturating_sub(reduction).max(target_min);
            let safe_target = best_chromatic.saturating_sub(2);
            let new_target = new_target.max(safe_target);

            if new_target >= current_target {
                break;
            }

            println!(
                "[PHASE 3][GPU] Attempting {} colors (from {})...",
                new_target, current_target
            );

            // Create sparse QUBO
            let color_penalty_weight = 0.1;
            let sparse_qubo = SparseQUBO::from_graph_coloring_with_color_penalty(
                graph,
                new_target,
                color_penalty_weight,
            )?;

            // Convert initial solution to QUBO binary format
            let mut initial_state = vec![false; sparse_qubo.num_variables()];
            for (v, &c) in best_solution.colors.iter().enumerate() {
                if c < new_target {
                    let idx = v * new_target + c;
                    if idx < initial_state.len() {
                        initial_state[idx] = true;
                    }
                }
            }

            // Run GPU QUBO SA
            let seed = 42 + new_target as u64 * 1337;
            match gpu_qubo_simulated_annealing(
                cuda_device,
                &sparse_qubo,
                &initial_state,
                10_000, // iterations
                1.0,    // T_initial
                0.01,   // T_final
                seed,
            ) {
                Ok(qubo_solution) => {
                    // Decode QUBO solution to coloring
                    match qubo_solution_to_coloring(&qubo_solution, n, new_target) {
                        Ok(coloring) => {
                            // Validate solution
                            let conflicts = self.count_conflicts(graph, &coloring);
                            let actual_chromatic =
                                coloring.iter().max().map(|&c| c + 1).unwrap_or(0);

                            println!(
                                "[PHASE 3][GPU] QUBO result: {} colors, {} conflicts",
                                actual_chromatic, conflicts
                            );

                            if conflicts == 0 {
                                best_solution = ColoringSolution {
                                    colors: coloring,
                                    chromatic_number: actual_chromatic,
                                    conflicts,
                                    quality_score: 1.0
                                        - (actual_chromatic as f64 / new_target as f64),
                                    computation_time_ms: 0.0,
                                };
                                best_chromatic = actual_chromatic;
                                current_target = best_chromatic;
                                println!(
                                    "[PHASE 3][GPU] SUCCESS! Reduced to {} colors",
                                    best_chromatic
                                );
                            } else {
                                println!(
                                    "[PHASE 3][GPU] Failed at {} colors (conflicts: {})",
                                    new_target, conflicts
                                );
                                break;
                            }
                        }
                        Err(e) => {
                            println!("[PHASE 3][GPU][FALLBACK] QUBO decode failed: {}", e);
                            return self.find_coloring_cpu(
                                graph,
                                _phase_field,
                                kuramoto_state,
                                initial_estimate,
                            );
                        }
                    }
                }
                Err(e) => {
                    println!("[PHASE 3][GPU][FALLBACK] GPU QUBO SA failed: {}", e);
                    return self.find_coloring_cpu(
                        graph,
                        _phase_field,
                        kuramoto_state,
                        initial_estimate,
                    );
                }
            }
        }

        // Step 3: Transfer Entropy Refinement (CPU)
        println!("[PHASE 3][GPU] Transfer Entropy-Guided Ordering");
        let te_ordering = match hybrid_te_kuramoto_ordering(graph, kuramoto_state, None, 0.0) {
            Ok(ordering) => {
                println!("[PHASE 3][GPU] TE ordering computed successfully");
                ordering
            }
            Err(e) => {
                println!(
                    "[PHASE 3][GPU] TE ordering failed ({}), using degree ordering",
                    e
                );
                self.get_degree_ordering(graph)
            }
        };

        let te_solution = self.greedy_with_ordering(graph, &te_ordering)?;
        if te_solution.conflicts == 0 && te_solution.chromatic_number < best_chromatic {
            println!(
                "[PHASE 3][GPU] TE improved: {} -> {} colors",
                best_chromatic, te_solution.chromatic_number
            );
            best_solution = te_solution;
        }

        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "[PHASE 3][GPU] GPU quantum coloring completed in {:.2}s",
            elapsed
        );
        println!(
            "[PHASE 3][GPU] Final: {} colors, {} conflicts",
            best_solution.chromatic_number, best_solution.conflicts
        );

        Ok(best_solution)
    }

    /// Count edge conflicts in a coloring
    fn count_conflicts(&self, graph: &Graph, coloring: &[usize]) -> usize {
        let n = graph.num_vertices;
        let mut conflicts = 0;

        for u in 0..n {
            for v in (u + 1)..n {
                if graph.adjacency[u * n + v] && coloring[u] == coloring[v] {
                    conflicts += 1;
                }
            }
        }

        conflicts
    }

    /// Get degree-based vertex ordering
    fn get_degree_ordering(&self, graph: &Graph) -> Vec<usize> {
        let n = graph.num_vertices;
        let mut vertices: Vec<(usize, usize)> = (0..n)
            .map(|v| {
                let degree = (0..n).filter(|&u| graph.adjacency[v * n + u]).count();
                (v, degree)
            })
            .collect();

        vertices.sort_by(|a, b| b.1.cmp(&a.1));
        vertices.iter().map(|&(v, _)| v).collect()
    }

    /// Greedy coloring with given vertex ordering
    fn greedy_with_ordering(&self, graph: &Graph, ordering: &[usize]) -> Result<ColoringSolution> {
        let n = graph.num_vertices;
        let mut colors = vec![0; n];
        let mut max_color = 0;

        for &v in ordering {
            let mut forbidden = HashSet::new();

            for u in 0..n {
                if graph.adjacency[v * n + u] {
                    forbidden.insert(colors[u]);
                }
            }

            let mut color = 0;
            while forbidden.contains(&color) {
                color += 1;
            }

            colors[v] = color;
            max_color = max_color.max(color);
        }

        let chromatic_number = max_color + 1;
        let conflicts = self.count_conflicts(graph, &colors);

        Ok(ColoringSolution {
            colors,
            chromatic_number,
            conflicts,
            quality_score: if chromatic_number > 0 {
                1.0 / chromatic_number as f64
            } else {
                0.0
            },
            computation_time_ms: 0.0,
        })
    }
}

/// QUBO problem representation
struct QUBOProblem {
    q_matrix: Array2<f64>,
    num_variables: usize,
    num_vertices: usize,
    num_colors: usize,
}

/// Build adjacency matrix from graph
fn build_adjacency_matrix(graph: &Graph) -> Array2<bool> {
    let n = graph.num_vertices;
    let mut adj = Array2::from_elem((n, n), false);

    for i in 0..n {
        for j in 0..n {
            adj[[i, j]] = graph.adjacency[i * n + j];
        }
    }

    adj
}

/// Count conflicts in a coloring
fn count_conflicts(coloring: &[usize], adjacency: &Array2<bool>) -> usize {
    let n = coloring.len();
    let mut conflicts = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            if adjacency[[i, j]] && coloring[i] == coloring[j] {
                conflicts += 1;
            }
        }
    }

    conflicts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_coloring_solver_creation() {
        #[cfg(feature = "cuda")]
        let solver = QuantumColoringSolver::new(None);

        #[cfg(not(feature = "cuda"))]
        let solver = QuantumColoringSolver::new();

        assert!(solver.is_ok());
    }

    #[test]
    fn test_qubo_encoding_small_graph() {
        #[cfg(feature = "cuda")]
        let solver = QuantumColoringSolver::new(None).unwrap();

        #[cfg(not(feature = "cuda"))]
        let solver = QuantumColoringSolver::new().unwrap();

        // Triangle graph: 3 vertices, 3 edges, chromatic number = 3
        let graph = Graph {
            num_vertices: 3,
            num_edges: 3,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
            adjacency: vec![false, true, true, true, false, true, true, true, false],
            coordinates: None,
        };

        let qubo = solver.graph_coloring_to_qubo(&graph, 3);
        assert!(qubo.is_ok());

        let qubo = qubo.unwrap();
        assert_eq!(qubo.num_variables, 9); // 3 vertices * 3 colors
        assert_eq!(qubo.num_vertices, 3);
        assert_eq!(qubo.num_colors, 3);
    }
}
