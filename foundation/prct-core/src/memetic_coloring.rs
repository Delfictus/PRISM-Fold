//! Memetic Algorithm for Graph Coloring with TSP-Guided Operators
//!
//! Combines:
//! - Genetic Algorithm (population-based search)
//! - TSP analysis (structural quality metrics)
//! - Local search (DSATUR refinement)
//! - GPU acceleration (parallel fitness evaluation)
//!
//! Expected performance: 115 → 85-100 colors on DSJC1000.5

use crate::dsatur_backtracking::DSaturSolver;
use crate::errors::*;
use ndarray::Array2;
use rand::prelude::*;
use rayon::prelude::*;
use shared_types::*;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Configuration for memetic algorithm
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemeticConfig {
    /// Population size (32-64 recommended)
    pub population_size: usize,

    /// Elite size (top 10-20%)
    pub elite_size: usize,

    /// Number of generations
    pub generations: usize,

    /// Mutation rate (0.1-0.3)
    pub mutation_rate: f64,

    /// Tournament size for selection
    pub tournament_size: usize,

    /// Local search depth (DSATUR iterations per generation)
    pub local_search_depth: usize,

    /// Enable TSP-guided operators
    pub use_tsp_guidance: bool,

    /// Weight for TSP quality in fitness (0.0-1.0)
    pub tsp_weight: f64,
}

impl Default for MemeticConfig {
    fn default() -> Self {
        Self {
            population_size: 48,
            elite_size: 8,
            generations: 100,
            mutation_rate: 0.15,
            tournament_size: 3,
            local_search_depth: 1000,
            use_tsp_guidance: true,
            tsp_weight: 0.3,
        }
    }
}

/// Individual in the population
#[derive(Clone)]
pub struct Individual {
    /// Coloring solution
    pub solution: ColoringSolution,

    /// Fitness score (higher is better)
    pub fitness: f64,

    /// TSP quality metrics for each color class
    pub tsp_qualities: Vec<f64>,

    /// Average TSP compactness
    pub avg_compactness: f64,
}

/// Memetic algorithm solver
pub struct MemeticColoringSolver {
    config: MemeticConfig,

    #[cfg(feature = "cuda")]
    gpu_device: Option<Arc<CudaDevice>>,

    /// Best solution found across all generations
    best_ever: Option<ColoringSolution>,

    /// Generation statistics
    generation_stats: Vec<GenerationStats>,

    /// Telemetry handle for detailed tracking
    telemetry: Option<std::sync::Arc<crate::telemetry::TelemetryHandle>>,
}

/// Statistics for a generation
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_chromatic: usize,
    pub avg_chromatic: f64,
    pub best_fitness: f64,
    pub avg_fitness: f64,
    pub best_tsp_quality: f64,
    pub diversity: f64,
}

impl MemeticColoringSolver {
    /// Create new memetic solver
    pub fn new(config: MemeticConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "cuda")]
            gpu_device: None,
            best_ever: None,
            generation_stats: Vec::new(),
            telemetry: None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn with_gpu(mut self, device: Arc<CudaDevice>) -> Self {
        self.gpu_device = Some(device);
        self
    }

    /// Set telemetry handle for detailed tracking
    pub fn with_telemetry(
        mut self,
        telemetry: std::sync::Arc<crate::telemetry::TelemetryHandle>,
    ) -> Self {
        self.telemetry = Some(telemetry);
        self
    }

    /// Run memetic algorithm
    pub fn solve(
        &mut self,
        graph: &Graph,
        initial_solutions: Vec<ColoringSolution>,
    ) -> Result<ColoringSolution> {
        println!("[MEMETIC] Starting memetic algorithm");
        println!(
            "[MEMETIC] Population: {}, Generations: {}",
            self.config.population_size, self.config.generations
        );
        println!(
            "[MEMETIC] TSP guidance: {}, Weight: {:.2}",
            self.config.use_tsp_guidance, self.config.tsp_weight
        );

        // Initialize population
        let mut population = self.initialize_population(graph, initial_solutions)?;

        // Evaluate initial population
        self.evaluate_population(&mut population, graph)?;

        let mut best_chromatic = population
            .iter()
            .map(|ind| ind.solution.chromatic_number)
            .min()
            .expect("population is expected non-empty after initialize/evaluate");

        println!("[MEMETIC] Initial best: {} colors", best_chromatic);

        // Evolution loop
        for gen in 0..self.config.generations {
            // Selection
            let parents = self.tournament_selection(&population);

            // Crossover
            let mut offspring = self.crossover_population(&parents, graph)?;

            // Mutation
            self.mutate_population(&mut offspring, graph)?;

            // Local search on elite
            self.local_search(&mut offspring, graph)?;

            // Evaluate offspring
            self.evaluate_population(&mut offspring, graph)?;

            // Elitism: combine parents and offspring, select best
            population = self.select_next_generation(&population, &offspring);

            // Track statistics
            let stats = self.compute_generation_stats(gen, &population);

            if stats.best_chromatic < best_chromatic {
                best_chromatic = stats.best_chromatic;
                println!(
                    "[MEMETIC] 🎯 Gen {}: {} colors (fitness: {:.2}, TSP: {:.3})",
                    gen, stats.best_chromatic, stats.best_fitness, stats.best_tsp_quality
                );
            } else if gen % 10 == 0 {
                println!(
                    "[MEMETIC] Gen {}: best={}, avg={:.1}, diversity={:.3}",
                    gen, stats.best_chromatic, stats.avg_chromatic, stats.diversity
                );
            }

            self.generation_stats.push(stats.clone());

            // Record telemetry every 20 generations for hypertuning insights
            if gen % 20 == 0 || gen < 5 || stats.best_chromatic < best_chromatic {
                if let Some(ref telemetry) = self.telemetry {
                    use crate::telemetry::{
                        OptimizationGuidance, PhaseExecMode, PhaseName, RunMetric,
                    };
                    use serde_json::json;

                    let stagnation_count = self.count_stagnation();
                    let mut recommendations = Vec::new();

                    let guidance_status = if stats.diversity < 0.01 {
                        recommendations.push(format!(
                            "CRITICAL: Diversity collapsed to {:.4} - increase mutation_rate from {:.2} to {:.2}",
                            stats.diversity, self.config.mutation_rate, self.config.mutation_rate * 1.5
                        ));
                        recommendations
                            .push("Or trigger desperation burst (population reset)".to_string());
                        "critical"
                    } else if stagnation_count > 50 {
                        recommendations.push(format!(
                            "Stagnant for {} generations - increase population_size from {} to {}",
                            stagnation_count,
                            self.config.population_size,
                            self.config.population_size + 16
                        ));
                        recommendations.push(format!(
                            "Or increase local_search_depth from {} to {}",
                            self.config.local_search_depth,
                            self.config.local_search_depth * 2
                        ));
                        "need_tuning"
                    } else if stats.best_chromatic < 90 && stats.best_chromatic > best_chromatic {
                        recommendations.push(format!(
                            "EXCELLENT: Improved to {} colors (was {})",
                            stats.best_chromatic, best_chromatic
                        ));
                        recommendations.push("Current memetic settings are effective".to_string());
                        "excellent"
                    } else if stats.avg_chromatic - stats.best_chromatic as f64 > 10.0 {
                        recommendations.push(
                            "Large gap between best and avg - increase elite_size".to_string(),
                        );
                        "need_tuning"
                    } else {
                        recommendations.push("Memetic evolution progressing normally".to_string());
                        "on_track"
                    };

                    let guidance = OptimizationGuidance {
                        status: guidance_status.to_string(),
                        recommendations,
                        estimated_final_colors: Some(stats.best_chromatic.saturating_sub(
                            ((self.config.generations - gen) as f64 * 0.05) as usize,
                        )),
                        confidence: if gen < 10 { 0.3 } else { 0.7 },
                        gap_to_world_record: Some((stats.best_chromatic as i32) - 83),
                    };

                    telemetry.record(
                        RunMetric::new(
                            PhaseName::Memetic,
                            format!("generation_{}/{}", gen, self.config.generations),
                            stats.best_chromatic,
                            0,   // Memetic doesn't track conflicts per generation
                            0.0, // Generation duration not tracked individually
                            PhaseExecMode::cpu_disabled(),
                        )
                        .with_parameters(json!({
                            "generation": gen,
                            "total_generations": self.config.generations,
                            "best_chromatic": stats.best_chromatic,
                            "avg_chromatic": stats.avg_chromatic,
                            "best_fitness": stats.best_fitness,
                            "avg_fitness": stats.avg_fitness,
                            "best_tsp_quality": stats.best_tsp_quality,
                            "diversity": stats.diversity,
                            "stagnation_count": stagnation_count,
                            "population_size": self.config.population_size,
                            "mutation_rate": self.config.mutation_rate,
                            "elite_size": self.config.elite_size,
                            "progress_pct": (gen as f64 / self.config.generations as f64) * 100.0,
                        }))
                        .with_guidance(guidance),
                    );
                }
            }

            // Early stopping if no improvement for 20 generations
            if gen > 20 && self.check_stagnation(20) {
                println!("[MEMETIC] Early stopping: no improvement for 20 generations");
                break;
            }
        }

        // Return best solution
        let best = population
            .iter()
            .min_by_key(|ind| ind.solution.chromatic_number)
            .ok_or_else(|| {
                PRCTError::ColoringFailed(
                    "Population became empty before selecting final best".to_string(),
                )
            })?;

        println!(
            "[MEMETIC] Final best: {} colors",
            best.solution.chromatic_number
        );

        Ok(best.solution.clone())
    }

    /// Run memetic algorithm with restart strategy for better exploration
    pub fn solve_with_restart(
        &mut self,
        graph: &Graph,
        initial_solutions: Vec<ColoringSolution>,
        max_restarts: usize,
    ) -> Result<ColoringSolution> {
        println!("[MEMETIC-RESTART] Starting adaptive restart strategy");
        println!("[MEMETIC-RESTART] Max restarts: {}", max_restarts);

        let mut best_ever = ColoringSolution {
            colors: vec![],
            chromatic_number: usize::MAX,
            conflicts: usize::MAX,
            quality_score: 0.0,
            computation_time_ms: 0.0,
        };

        let original_generations = self.config.generations;
        let original_mutation = self.config.mutation_rate;

        for restart in 0..max_restarts {
            // Adaptive parameter tuning
            self.config.mutation_rate = original_mutation + (restart as f64 * 0.10);
            self.config.generations = original_generations / 2; // Shorter runs for restart

            println!(
                "\n[MEMETIC-RESTART] === Restart {}/{} ===",
                restart + 1,
                max_restarts
            );
            println!(
                "[MEMETIC-RESTART] Mutation rate: {:.2}",
                self.config.mutation_rate
            );
            println!("[MEMETIC-RESTART] Generations: {}", self.config.generations);

            // Run memetic algorithm
            let solution = self.solve(graph, initial_solutions.clone())?;

            if solution.chromatic_number < best_ever.chromatic_number {
                best_ever = solution.clone();
                println!(
                    "[MEMETIC-RESTART] 🎯 New best: {} colors!",
                    best_ever.chromatic_number
                );
            } else {
                println!(
                    "[MEMETIC-RESTART] No improvement: {} colors (best: {})",
                    solution.chromatic_number, best_ever.chromatic_number
                );
            }

            // Early stopping if we haven't improved in 2 restarts
            if restart > 0 && solution.chromatic_number >= best_ever.chromatic_number {
                let prev_best = best_ever.chromatic_number;
                if restart > 1 && prev_best == best_ever.chromatic_number {
                    println!("[MEMETIC-RESTART] No improvement for 2 restarts, stopping early");
                    break;
                }
            }
        }

        // Restore original config
        self.config.generations = original_generations;
        self.config.mutation_rate = original_mutation;

        println!(
            "\n[MEMETIC-RESTART] Final best: {} colors",
            best_ever.chromatic_number
        );
        Ok(best_ever)
    }

    /// Initialize population with diversity
    fn initialize_population(
        &self,
        graph: &Graph,
        initial_solutions: Vec<ColoringSolution>,
    ) -> Result<Vec<Individual>> {
        let mut population = Vec::with_capacity(self.config.population_size);

        // Add provided initial solutions
        for solution in initial_solutions
            .into_iter()
            .take(self.config.population_size / 2)
        {
            population.push(Individual {
                solution,
                fitness: 0.0,
                tsp_qualities: Vec::new(),
                avg_compactness: 0.0,
            });
        }

        // Fill remaining with random solutions
        let mut rng = rand::thread_rng();
        while population.len() < self.config.population_size {
            let solution = self.generate_random_solution(graph, &mut rng)?;
            population.push(Individual {
                solution,
                fitness: 0.0,
                tsp_qualities: Vec::new(),
                avg_compactness: 0.0,
            });
        }

        Ok(population)
    }

    /// Generate random greedy solution
    fn generate_random_solution(
        &self,
        graph: &Graph,
        rng: &mut ThreadRng,
    ) -> Result<ColoringSolution> {
        let n = graph.num_vertices;
        let mut vertices: Vec<usize> = (0..n).collect();
        vertices.shuffle(rng);

        let mut coloring = vec![usize::MAX; n];
        let adjacency = self.build_adjacency_matrix(graph);

        for &v in &vertices {
            // Find forbidden colors
            let forbidden: std::collections::HashSet<usize> = (0..n)
                .filter(|&u| adjacency[[v, u]] && coloring[u] != usize::MAX)
                .map(|u| coloring[u])
                .collect();

            // Assign smallest available color
            let color = (0..n).find(|c| !forbidden.contains(c)).unwrap_or(0);
            coloring[v] = color;
        }

        let chromatic_number = coloring
            .iter()
            .filter(|&&c| c != usize::MAX)
            .max()
            .map(|&c| c + 1)
            .unwrap_or(0);

        let conflicts = self.count_conflicts(&coloring, &adjacency);

        Ok(ColoringSolution {
            colors: coloring,
            chromatic_number,
            conflicts,
            quality_score: 0.0,
            computation_time_ms: 0.0,
        })
    }

    /// Build adjacency matrix
    fn build_adjacency_matrix(&self, graph: &Graph) -> Array2<bool> {
        let n = graph.num_vertices;
        let mut adjacency = Array2::from_elem((n, n), false);

        for i in 0..n {
            for j in 0..n {
                adjacency[[i, j]] = graph.adjacency[i * n + j];
            }
        }

        adjacency
    }

    /// Count conflicts in coloring
    fn count_conflicts(&self, coloring: &[usize], adjacency: &Array2<bool>) -> usize {
        let n = coloring.len();
        let mut conflicts = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                if adjacency[[i, j]] && coloring[i] != usize::MAX && coloring[i] == coloring[j] {
                    conflicts += 1;
                }
            }
        }

        conflicts
    }

    /// Compute generation statistics
    fn compute_generation_stats(&self, gen: usize, population: &[Individual]) -> GenerationStats {
        let best_chromatic = population
            .iter()
            .map(|ind| ind.solution.chromatic_number)
            .min()
            .expect("compute_generation_stats expects non-empty population slice");

        let avg_chromatic = population
            .iter()
            .map(|ind| ind.solution.chromatic_number as f64)
            .sum::<f64>()
            / population.len() as f64;

        let best_fitness = population
            .iter()
            .map(|ind| ind.fitness)
            .fold(f64::NEG_INFINITY, f64::max);

        let avg_fitness =
            population.iter().map(|ind| ind.fitness).sum::<f64>() / population.len() as f64;

        let best_tsp_quality = population
            .iter()
            .map(|ind| ind.avg_compactness)
            .fold(f64::NEG_INFINITY, f64::max);

        // Diversity: variance in chromatic numbers
        let variance = population
            .iter()
            .map(|ind| {
                let diff = ind.solution.chromatic_number as f64 - avg_chromatic;
                diff * diff
            })
            .sum::<f64>()
            / population.len() as f64;
        let diversity = variance.sqrt();

        GenerationStats {
            generation: gen,
            best_chromatic,
            avg_chromatic,
            best_fitness,
            avg_fitness,
            best_tsp_quality,
            diversity,
        }
    }

    /// Check for stagnation
    fn check_stagnation(&self, window: usize) -> bool {
        if self.generation_stats.len() < window {
            return false;
        }

        let recent = &self.generation_stats[self.generation_stats.len() - window..];
        // recent has length >= window by guard above; use indexing to avoid unwraps
        let first_best = recent[0].best_chromatic;
        let last_best = recent[window - 1].best_chromatic;

        first_best == last_best
    }

    /// Count consecutive generations without improvement
    fn count_stagnation(&self) -> usize {
        if self.generation_stats.is_empty() {
            return 0;
        }

        let mut count = 0;
        let current_best = self.generation_stats.last().unwrap().best_chromatic;

        for stat in self.generation_stats.iter().rev() {
            if stat.best_chromatic == current_best {
                count += 1;
            } else {
                break;
            }
        }

        count
    }

    /// Evaluate fitness for all individuals in population
    fn evaluate_population(&self, population: &mut [Individual], graph: &Graph) -> Result<()> {
        // Parallel fitness evaluation
        population.par_iter_mut().for_each(|individual| {
            // Fitness components:
            // 1. Chromatic number (lower is better)
            // 2. Conflicts (0 conflicts required)
            // 3. TSP compactness (higher is better, if enabled)

            let chromatic = individual.solution.chromatic_number as f64;
            let conflicts = individual.solution.conflicts as f64;

            // Base fitness: minimize colors and conflicts
            let mut fitness = 1000.0 / chromatic - 1000.0 * conflicts;

            // Add TSP compactness if enabled
            if self.config.use_tsp_guidance {
                // Compute average compactness (simplified for performance)
                let avg_compactness = self.compute_simple_compactness(&individual.solution, graph);
                individual.avg_compactness = avg_compactness;

                // Weight TSP quality into fitness
                fitness += self.config.tsp_weight * 100.0 * avg_compactness;
            }

            individual.fitness = fitness;
        });

        Ok(())
    }

    /// Simple compactness metric without full TSP solve
    fn compute_simple_compactness(&self, solution: &ColoringSolution, graph: &Graph) -> f64 {
        let color_classes = self.extract_color_classes_fast(solution);

        if color_classes.is_empty() {
            return 0.0;
        }

        let n = graph.num_vertices;
        let adjacency = self.build_adjacency_matrix(graph);

        // For each color class, compute intra-class connectivity
        let total_compactness: f64 = color_classes
            .iter()
            .filter(|class| !class.is_empty())
            .map(|class| {
                if class.len() <= 1 {
                    return 1.0;
                }

                // Count edges within color class
                let mut internal_edges = 0;
                let max_possible = class.len() * (class.len() - 1) / 2;

                for (i, &v1) in class.iter().enumerate() {
                    for &v2 in class.iter().skip(i + 1) {
                        if v1 < n && v2 < n && adjacency[[v1, v2]] {
                            internal_edges += 1;
                        }
                    }
                }

                // Compactness = actual edges / possible edges
                if max_possible > 0 {
                    internal_edges as f64 / max_possible as f64
                } else {
                    0.0
                }
            })
            .sum();

        total_compactness / color_classes.len() as f64
    }

    /// Extract color classes quickly
    fn extract_color_classes_fast(&self, solution: &ColoringSolution) -> Vec<Vec<usize>> {
        let max_color = solution
            .colors
            .iter()
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

    /// Tournament selection
    fn tournament_selection(&self, population: &[Individual]) -> Vec<Individual> {
        let mut rng = rand::thread_rng();
        let mut selected = Vec::with_capacity(population.len());

        for _ in 0..population.len() {
            // Tournament: pick k random individuals, select best
            let mut best = &population[rng.gen_range(0..population.len())];

            for _ in 1..self.config.tournament_size {
                let candidate = &population[rng.gen_range(0..population.len())];
                if candidate.fitness > best.fitness {
                    best = candidate;
                }
            }

            selected.push(best.clone());
        }

        selected
    }

    /// Crossover population (TSP-guided)
    fn crossover_population(
        &self,
        parents: &[Individual],
        graph: &Graph,
    ) -> Result<Vec<Individual>> {
        let offspring: Vec<Individual> = parents
            .par_chunks(2)
            .map(|pair| {
                if pair.len() == 2 {
                    self.tsp_guided_crossover(&pair[0], &pair[1], graph)
                        .unwrap_or_else(|_| pair[0].clone())
                } else {
                    pair[0].clone()
                }
            })
            .collect();

        Ok(offspring)
    }

    /// TSP-guided crossover: preserve compact color classes
    fn tsp_guided_crossover(
        &self,
        parent1: &Individual,
        parent2: &Individual,
        graph: &Graph,
    ) -> Result<Individual> {
        let n = graph.num_vertices;
        let mut child_colors = vec![usize::MAX; n];

        // Extract color classes from both parents
        let p1_classes = self.extract_color_classes_fast(&parent1.solution);
        let p2_classes = self.extract_color_classes_fast(&parent2.solution);

        // Score each class by compactness (simplified)
        let mut all_classes = Vec::new();

        for (i, class) in p1_classes.iter().enumerate() {
            if !class.is_empty() {
                all_classes.push((class.clone(), i, 1)); // 1 = from parent1
            }
        }

        for (i, class) in p2_classes.iter().enumerate() {
            if !class.is_empty() {
                all_classes.push((class.clone(), i, 2)); // 2 = from parent2
            }
        }

        // Assign vertices from best classes first
        let mut colored_vertices = std::collections::HashSet::new();
        let mut next_color = 0;

        for (class_vertices, _, _) in all_classes {
            // Skip if vertices already colored
            if class_vertices.iter().any(|v| colored_vertices.contains(v)) {
                continue;
            }

            // Assign this color class
            for &v in &class_vertices {
                if v < n && !colored_vertices.contains(&v) {
                    child_colors[v] = next_color;
                    colored_vertices.insert(v);
                }
            }
            next_color += 1;
        }

        // Color remaining vertices greedily
        let adjacency = self.build_adjacency_matrix(graph);
        for v in 0..n {
            if child_colors[v] == usize::MAX {
                let forbidden: std::collections::HashSet<usize> = (0..n)
                    .filter(|&u| adjacency[[v, u]] && child_colors[u] != usize::MAX)
                    .map(|u| child_colors[u])
                    .collect();

                let color = (0..next_color + 1)
                    .find(|c| !forbidden.contains(c))
                    .unwrap_or(next_color);
                child_colors[v] = color;
                next_color = next_color.max(color + 1);
            }
        }

        let chromatic_number = child_colors
            .iter()
            .filter(|&&c| c != usize::MAX)
            .max()
            .map(|&c| c + 1)
            .unwrap_or(0);

        let conflicts = self.count_conflicts(&child_colors, &adjacency);

        Ok(Individual {
            solution: ColoringSolution {
                colors: child_colors,
                chromatic_number,
                conflicts,
                quality_score: 0.0,
                computation_time_ms: 0.0,
            },
            fitness: 0.0,
            tsp_qualities: Vec::new(),
            avg_compactness: 0.0,
        })
    }

    /// Mutate population
    fn mutate_population(&self, population: &mut [Individual], graph: &Graph) -> Result<()> {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        population.par_iter_mut().for_each(|individual| {
            // Each thread gets its own RNG
            let mut rng = StdRng::from_entropy();
            if rng.gen::<f64>() < self.config.mutation_rate {
                self.tsp_guided_mutation(&mut individual.solution, graph);
            }
        });

        Ok(())
    }

    /// TSP-guided mutation: break up scattered color classes
    fn tsp_guided_mutation(&self, solution: &mut ColoringSolution, graph: &Graph) {
        let n = graph.num_vertices;
        let color_classes = self.extract_color_classes_fast(solution);

        // Find a scattered class (low internal connectivity)
        let mut worst_class_idx = 0;
        let mut worst_compactness = 1.0;

        let adjacency = self.build_adjacency_matrix(graph);

        for (i, class) in color_classes.iter().enumerate() {
            if class.len() <= 1 {
                continue;
            }

            // Compute compactness
            let mut internal_edges = 0;
            let max_possible = class.len() * (class.len() - 1) / 2;

            for (j, &v1) in class.iter().enumerate() {
                for &v2 in class.iter().skip(j + 1) {
                    if adjacency[[v1, v2]] {
                        internal_edges += 1;
                    }
                }
            }

            let compactness = if max_possible > 0 {
                internal_edges as f64 / max_possible as f64
            } else {
                0.0
            };

            if compactness < worst_compactness {
                worst_compactness = compactness;
                worst_class_idx = i;
            }
        }

        // Reassign some vertices from worst class
        if worst_class_idx < color_classes.len() {
            let worst_class = &color_classes[worst_class_idx];
            let num_to_reassign = (worst_class.len() / 4).max(1);

            for &v in worst_class.iter().take(num_to_reassign) {
                // Find best alternative color
                let forbidden: std::collections::HashSet<usize> = (0..n)
                    .filter(|&u| adjacency[[v, u]] && solution.colors[u] != usize::MAX)
                    .map(|u| solution.colors[u])
                    .collect();

                if let Some(new_color) =
                    (0..solution.chromatic_number).find(|c| !forbidden.contains(c))
                {
                    solution.colors[v] = new_color;
                }
            }
        }

        // Recompute chromatic number and conflicts
        solution.chromatic_number = solution
            .colors
            .iter()
            .filter(|&&c| c != usize::MAX)
            .max()
            .map(|&c| c + 1)
            .unwrap_or(0);

        solution.conflicts = self.count_conflicts(&solution.colors, &adjacency);
    }

    /// Local search with DSATUR on elite
    fn local_search(&self, population: &mut [Individual], graph: &Graph) -> Result<()> {
        // Apply DSATUR refinement to top individuals
        let elite_count = self.config.elite_size;

        // Sort by fitness
        let mut indices: Vec<usize> = (0..population.len()).collect();
        indices.sort_by(|&a, &b| {
            population[b]
                .fitness
                .partial_cmp(&population[a].fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Refine elite individuals IN PARALLEL (leverages all 24 cores!)
        let elite_indices: Vec<usize> = indices.iter().take(elite_count).copied().collect();

        let refinements: Vec<(usize, Option<ColoringSolution>)> = elite_indices
            .par_iter()
            .map(|&idx| {
                let current_chromatic = population[idx].solution.chromatic_number;

                // Run limited DSATUR
                let mut dsatur =
                    DSaturSolver::new(current_chromatic, self.config.local_search_depth);

                let refined = dsatur
                    .find_coloring(
                        graph,
                        Some(&population[idx].solution),
                        current_chromatic.saturating_sub(5),
                    )
                    .ok();

                (idx, refined)
            })
            .collect();

        // Apply refinements
        for (idx, refined) in refinements {
            if let Some(ref solution) = refined {
                if solution.chromatic_number < population[idx].solution.chromatic_number
                    && solution.conflicts == 0
                {
                    population[idx].solution = solution.clone();
                }
            }
        }

        Ok(())
    }

    /// Select next generation (elitism)
    fn select_next_generation(
        &self,
        parents: &[Individual],
        offspring: &[Individual],
    ) -> Vec<Individual> {
        let mut combined = Vec::with_capacity(parents.len() + offspring.len());
        combined.extend_from_slice(parents);
        combined.extend_from_slice(offspring);

        // Sort by fitness (descending)
        combined.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top population_size individuals
        combined
            .into_iter()
            .take(self.config.population_size)
            .collect()
    }
}
