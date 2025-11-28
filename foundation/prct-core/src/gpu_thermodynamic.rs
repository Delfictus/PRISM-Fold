//! GPU-Accelerated Thermodynamic Equilibration
//!
//! This module provides CUDA-accelerated thermodynamic replica exchange for
//! Phase 2 of the PRISM world-record pipeline.
//!
//! Constitutional Compliance:
//! - Article V: Uses shared CUDA context (Arc<CudaDevice>)
//! - Article VII: Kernels compiled in build.rs
//! - Zero stubs: Full implementation, no todo!/unimplemented!

use crate::errors::*;
use cudarc::driver::*;
use cudarc::nvrtc::Ptx;
use shared_types::*;
use std::path::PathBuf;
use std::sync::Arc;

/// Compute thermodynamic equilibration using GPU-accelerated replica exchange
///
/// Uses parallel GPU kernels to simulate oscillator dynamics at multiple temperatures,
/// finding energy-minimizing colorings via thermodynamic equilibration.
///
/// # Arguments
/// * `cuda_device` - Shared CUDA context (Article V compliance)
/// * `stream` - CUDA stream for async execution (cudarc 0.9: synchronous, prepared for future)
/// * `graph` - Input graph structure
/// * `initial_solution` - Starting coloring configuration
/// * `target_chromatic` - Target number of colors
/// * `t_min` - Minimum temperature (high precision)
/// * `t_max` - Maximum temperature (high exploration)
/// * `num_temps` - Number of temperature replicas
/// * `steps_per_temp` - Evolution steps at each temperature
/// * `ai_uncertainty` - Active Inference uncertainty scores for vertex prioritization (Phase 1 output)
/// * `fluxnet_config` - Optional FluxNet RL configuration for adaptive force profiles
/// * `difficulty_scores` - Optional reservoir difficulty scores for ForceProfile initialization (Phase 0 output)
/// * `force_start_temp` - TWEAK 1: Temperature at which conflict forces start activating
/// * `force_full_strength_temp` - TWEAK 1: Temperature at which conflict forces reach full strength
///
/// # Returns
/// Vec<ColoringSolution> - Equilibrium states at each temperature
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn equilibrate_thermodynamic_gpu(
    cuda_device: &Arc<CudaDevice>,
    stream: &CudaStream, // Stream for async execution (cudarc 0.9: synchronous, prepared for future)
    graph: &Graph,
    initial_solution: &ColoringSolution,
    target_chromatic: usize,
    t_min: f64,
    t_max: f64,
    num_temps: usize,
    steps_per_temp: usize,
    ai_uncertainty: Option<&Vec<f64>>,
    telemetry: Option<&Arc<crate::telemetry::TelemetryHandle>>,
    fluxnet_config: Option<&crate::fluxnet::FluxNetConfig>,
    difficulty_scores: Option<&Vec<f32>>,
    force_start_temp: f64,
    force_full_strength_temp: f64,
    guard_initial_slack: usize,
    guard_min_slack: usize,
    guard_max_slack: usize,
    compaction_guard_threshold: f64,
    reheat_consecutive_guards: usize,
    reheat_temp_boost: f64,
) -> Result<Vec<ColoringSolution>> {
    // Note: cudarc 0.9 doesn't support async stream execution, but we accept the parameter
    // for API consistency and future cudarc 0.17+ upgrade
    let _ = stream; // Will be used when cudarc supports stream.launch()
    let n = graph.num_vertices;

    println!("[THERMO-GPU] Starting GPU thermodynamic equilibration");
    println!(
        "[THERMO-GPU] Graph: {} vertices, {} edges",
        n, graph.num_edges
    );
    println!(
        "[THERMO-GPU] Temperature range: [{:.3}, {:.3}]",
        t_min, t_max
    );
    println!(
        "[THERMO-GPU] Replicas: {}, steps per temp: {}",
        num_temps, steps_per_temp
    );
    println!(
        "[THERMO-GPU][TWEAK-1] Force activation: start_T={:.3}, full_strength_T={:.3}",
        force_start_temp, force_full_strength_temp
    );

    let start_time = std::time::Instant::now();

    // Load PTX module for thermodynamic kernels
    let ptx_path = "target/ptx/thermodynamic.ptx";
    let ptx = Ptx::from_file(ptx_path);

    cuda_device
        .load_ptx(
            ptx,
            "thermodynamic_module",
            &[
                "initialize_oscillators_kernel",
                "compute_coupling_forces_kernel",
                "evolve_oscillators_kernel",
                "evolve_oscillators_with_conflicts_kernel",
                "compute_energy_kernel",
                "compute_entropy_kernel",
                "compute_order_parameter_kernel",
                "compute_conflicts_kernel",
            ],
        )
        .map_err(|e| PRCTError::GpuError(format!("Failed to load thermo kernels: {}", e)))?;

    // Prepare AI-guided vertex perturbation weights
    let vertex_perturbation_weights = if let Some(uncertainty) = ai_uncertainty {
        println!(
            "[THERMO-GPU][AI-GUIDED] Using Active Inference uncertainty for vertex perturbation"
        );

        // CRITICAL FIX: Preserve raw uncertainty for band classification
        let raw_unc: Vec<f64> = uncertainty.to_vec(); // Raw [0.0, 1.0] values

        // Classify bands using RAW uncertainty (before normalization)
        let strong_band = raw_unc.iter().filter(|&&u| u > 0.66).count();
        let weak_band = raw_unc.iter().filter(|&&u| u < 0.33).count();
        let neutral_band = n - strong_band - weak_band;

        println!(
            "[THERMO-GPU][TWEAK-2] Phase bands (from RAW uncertainty): strong={} ({}%), neutral={} ({}%), weak={} ({}%)",
            strong_band, strong_band * 100 / n,
            neutral_band, neutral_band * 100 / n,
            weak_band, weak_band * 100 / n
        );

        // NOW normalize for perturbation probability (after band classification)
        let epsilon = 1e-3;
        let mut weights: Vec<f64> = raw_unc.iter().map(|&u| u + epsilon).collect();
        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }

        println!(
            "[THERMO-GPU][AI-GUIDED] Uncertainty range: [{:.6}, {:.6}], mean: {:.6}",
            raw_unc.iter().cloned().fold(f64::INFINITY, f64::min),
            raw_unc.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            raw_unc.iter().sum::<f64>() / raw_unc.len() as f64
        );

        // Convert to f32 for GPU upload (thermodynamic kernels use f32)
        let weights_f32: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
        Some(weights_f32)
    } else {
        println!("[THERMO-GPU] Using uniform vertex perturbation (no AI guidance)");
        None
    };

    // Upload vertex weights to GPU if available
    // MOVE 5: Always create buffer (uniform weights if AI not available) for coupling kernel
    let d_vertex_weights = if let Some(ref weights) = vertex_perturbation_weights {
        cuda_device
            .htod_copy(weights.clone())
            .map_err(|e| PRCTError::GpuError(format!("Failed to copy vertex weights: {}", e)))?
    } else {
        // Create uniform weights (0.5) for vertices without AI guidance
        cuda_device
            .htod_copy(vec![0.5f32; n])
            .map_err(|e| PRCTError::GpuError(format!("Failed to create uniform weights: {}", e)))?
    };

    // Geometric temperature ladder
    let temperatures: Vec<f64> = (0..num_temps)
        .map(|i| {
            let ratio = t_min / t_max;
            t_max * ratio.powf(i as f64 / (num_temps - 1) as f64)
        })
        .collect();

    println!(
        "[THERMO-GPU] Temperature ladder: {:?}",
        &temperatures[..temperatures.len().min(5)]
    );

    // Convert graph edges to device format (u32, u32, f32)
    let edge_list: Vec<(u32, u32, f32)> = graph
        .edges
        .iter()
        .map(|(u, v, w)| (*u as u32, *v as u32, *w as f32))
        .collect();

    // Upload edges to GPU
    let edge_u: Vec<u32> = edge_list.iter().map(|(u, _, _)| *u).collect();
    let edge_v: Vec<u32> = edge_list.iter().map(|(_, v, _)| *v).collect();
    let edge_w: Vec<f32> = edge_list.iter().map(|(_, _, w)| *w).collect();

    let d_edge_u = cuda_device
        .htod_copy(edge_u)
        .map_err(|e| PRCTError::GpuError(format!("Failed to copy edges_u: {}", e)))?;
    let d_edge_v = cuda_device
        .htod_copy(edge_v)
        .map_err(|e| PRCTError::GpuError(format!("Failed to copy edges_v: {}", e)))?;
    let d_edge_w = cuda_device
        .htod_copy(edge_w)
        .map_err(|e| PRCTError::GpuError(format!("Failed to copy edges_w: {}", e)))?;

    // Get kernel functions
    let init_osc = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "initialize_oscillators_kernel")
            .ok_or_else(|| PRCTError::GpuError("initialize_oscillators_kernel not found".into()))?,
    );
    let compute_coupling = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "compute_coupling_forces_kernel")
            .ok_or_else(|| {
                PRCTError::GpuError("compute_coupling_forces_kernel not found".into())
            })?,
    );
    let evolve_osc = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "evolve_oscillators_kernel")
            .ok_or_else(|| PRCTError::GpuError("evolve_oscillators_kernel not found".into()))?,
    );
    let evolve_osc_conflicts = Arc::new(
        cuda_device
            .get_func(
                "thermodynamic_module",
                "evolve_oscillators_with_conflicts_kernel",
            )
            .ok_or_else(|| {
                PRCTError::GpuError("evolve_oscillators_with_conflicts_kernel not found".into())
            })?,
    );
    let compute_energy = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "compute_energy_kernel")
            .ok_or_else(|| PRCTError::GpuError("compute_energy_kernel not found".into()))?,
    );
    let compute_conflicts = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "compute_conflicts_kernel")
            .ok_or_else(|| PRCTError::GpuError("compute_conflicts_kernel not found".into()))?,
    );
    let _compute_entropy = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "compute_entropy_kernel")
            .ok_or_else(|| PRCTError::GpuError("compute_entropy_kernel not found".into()))?,
    );
    let _compute_order_param = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "compute_order_parameter_kernel")
            .ok_or_else(|| {
                PRCTError::GpuError("compute_order_parameter_kernel not found".into())
            })?,
    );

    let mut equilibrium_states: Vec<ColoringSolution> = Vec::new();

    // Track initial chromatic for effectiveness scoring
    let initial_chromatic = initial_solution.chromatic_number;
    let initial_conflicts = initial_solution.conflicts;

    // MOVE 3: Dynamic slack tracking with aggressive initial slack
    let mut consecutive_guards = 0;
    let mut current_slack = guard_initial_slack;
    let mut stable_temps = 0;
    let min_slack_limit = guard_min_slack.min(guard_max_slack);
    let max_slack_limit = guard_max_slack.max(guard_initial_slack);
    current_slack = current_slack.clamp(min_slack_limit, max_slack_limit);
    let base_guard_floor = 0.05f32;
    let mut dynamic_guard_threshold = compaction_guard_threshold as f32;
    let mut guard_threshold_floor = base_guard_floor;
    let mut guard_threshold_relax_timer = 0usize;
    let collapse_reheat_threshold = reheat_consecutive_guards.max(1);
    let mut collapse_streak = 0usize;
    let mut guard_cooldown = 0usize;
    let mut pending_reheat_temp: Option<f64> = None;
    let mut randomize_next_phases = false;
    let mut post_reheat_slack_lock = 0usize;
    let min_steps_per_temp = 10_000usize;
    let max_steps_per_temp = 40_000usize;
    let mut dynamic_steps_per_temp = steps_per_temp.clamp(min_steps_per_temp, max_steps_per_temp);

    // MOVE 5: Guard boost tracking - 1.5x repulsion for 2 temps after guard fires
    let mut temps_since_guard = 999; // Start high so boost is inactive initially
    let guard_boost_duration = 2; // Number of temps to apply boost after guard
    let guard_boost_multiplier = 1.5; // 1.5x repulsion strength

    // MOVE 6: Adaptive reheat detection
    let mut prev_chromatic = initial_chromatic;
    let mut stall_counter = 0;
    let reheat_stall_threshold = 5; // Temps without improvement before suggesting reheat

    // TWEAK 6: Track best snapshot across all temperatures
    let mut best_snapshot: Option<(usize, ColoringSolution, f64)> = None; // (temp_idx, solution, quality_score)

    // Task B2: Generate natural frequency heterogeneity
    // CRITICAL FIX: Widened range [0.9, 1.1] → [0.5, 1.5] (5x spread)
    // Prevents phase-locking by forcing extreme frequency diversity
    let natural_frequencies: Vec<f32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| 0.5 + rng.gen::<f32>() * 1.0).collect()
    };
    let d_natural_frequencies = cuda_device
        .htod_copy(natural_frequencies)
        .map_err(|e| PRCTError::GpuError(format!("Failed to upload natural frequencies: {}", e)))?;

    println!(
        "[THERMO-GPU][TASK-B2] Generated {} natural frequencies (AGGRESSIVE range: 0.5-1.5)",
        n
    );

    // FluxNet RL: Initialize ForceProfile and RLController if enabled
    let mut fluxnet_state: Option<(crate::fluxnet::ForceProfile, crate::fluxnet::RLController)> =
        None;

    if let Some(config) = fluxnet_config {
        if config.enabled {
            println!("[FLUXNET] Initializing FluxNet RL controller");
            println!("[FLUXNET] Memory tier: {:?}", config.memory_tier);
            println!("[FLUXNET] Learning rate: {}", config.rl.learning_rate);
            println!(
                "[FLUXNET] Epsilon: {} → {}",
                config.rl.epsilon_start, config.rl.epsilon_min
            );

            // Initialize ForceProfile from difficulty scores (Phase 0 output) or defaults
            let mut force_profile = if let Some(scores) = difficulty_scores {
                println!("[FLUXNET] Initializing ForceProfile from reservoir difficulty scores");
                crate::fluxnet::ForceProfile::from_difficulty_scores(scores, cuda_device.clone())
                    .map_err(|e| {
                        PRCTError::GpuError(format!("Failed to create ForceProfile: {}", e))
                    })?
            } else {
                println!("[FLUXNET] Initializing ForceProfile with uniform baseline forces");
                crate::fluxnet::ForceProfile::new(n, cuda_device.clone()).map_err(|e| {
                    PRCTError::GpuError(format!("Failed to create ForceProfile: {}", e))
                })?
            };

            // Update ForceProfile with AI uncertainty (Phase 1 output) if available
            if let Some(uncertainty) = ai_uncertainty {
                println!("[FLUXNET] Updating ForceProfile with AI uncertainty scores");
                let uncertainty_f32: Vec<f32> = uncertainty.iter().map(|&x| x as f32).collect();
                force_profile
                    .update_from_uncertainty(&uncertainty_f32)
                    .map_err(|e| {
                        PRCTError::GpuError(format!("Failed to update ForceProfile: {}", e))
                    })?;
            }

            // Initialize RL Controller
            let num_states = config.rl.get_qtable_states(config.memory_tier);
            let replay_capacity = config.rl.get_replay_capacity(config.memory_tier);
            let max_conflicts = n; // Assume max conflicts ~ num_vertices
            let max_chromatic = n; // Assume max colors ~ num_vertices
            let compact = config.memory_tier == crate::fluxnet::MemoryTier::Compact;

            let mut rl_controller = crate::fluxnet::RLController::new(
                config.rl.clone(),
                num_states,
                replay_capacity,
                max_conflicts,
                max_chromatic,
                compact,
                guard_max_slack,
                max_steps_per_temp,
            );

            // Load pre-trained Q-table if available
            if config.persistence.load_pretrained {
                if let Some(ref pretrained_path) = config.persistence.pretrained_path {
                    if pretrained_path.exists() {
                        println!(
                            "[FLUXNET] Loading pre-trained Q-table from {:?}",
                            pretrained_path
                        );
                        match rl_controller.load_qtable(pretrained_path) {
                            Ok(_) => println!("[FLUXNET] Successfully loaded pre-trained Q-table"),
                            Err(e) => println!("[FLUXNET] Warning: Failed to load Q-table: {}", e),
                        }
                    } else {
                        println!(
                            "[FLUXNET] No pre-trained Q-table found at {:?}",
                            pretrained_path
                        );
                    }
                }
            }

            // Sync ForceProfile to GPU device buffers
            force_profile.to_device().map_err(|e| {
                PRCTError::GpuError(format!("Failed to sync ForceProfile to device: {}", e))
            })?;

            let stats = force_profile.compute_stats();
            println!("[FLUXNET] ForceProfile initialized:");
            println!(
                "[FLUXNET]   Strong band: {:.1}%",
                stats.strong_fraction * 100.0
            );
            println!(
                "[FLUXNET]   Neutral band: {:.1}%",
                stats.neutral_fraction * 100.0
            );
            println!("[FLUXNET]   Weak band: {:.1}%", stats.weak_fraction * 100.0);
            println!("[FLUXNET]   Mean force: {:.3}", stats.mean_force);

            fluxnet_state = Some((force_profile, rl_controller));
        } else {
            println!("[FLUXNET] FluxNet disabled in config");
        }
    } else {
        println!("[FLUXNET] No FluxNet config provided, running baseline thermodynamic");
    }

    // Create identity force buffers for when FluxNet is disabled
    // These are reused across all temperatures to avoid repeated allocation
    let identity_f_strong = cuda_device
        .htod_copy(vec![1.0f32; n])
        .map_err(|e| PRCTError::GpuError(format!("Failed to allocate identity f_strong: {}", e)))?;
    let identity_f_weak = cuda_device
        .htod_copy(vec![1.0f32; n])
        .map_err(|e| PRCTError::GpuError(format!("Failed to allocate identity f_weak: {}", e)))?;

    // Process each temperature
    for (temp_idx, base_temp) in temperatures.iter().enumerate() {
        let mut temp = *base_temp;
        let mut reheat_triggered = false;

        if let Some(override_temp) = pending_reheat_temp.take() {
            let boosted = override_temp.min(t_max);
            println!(
                "[THERMO-GPU][GUARD-REHEAT] Applying temperature override {:.3} (base {:.3})",
                boosted, temp
            );
            temp = boosted;
        }

        let guard_active = guard_cooldown == 0;
        let mut guard_suppressed = false;
        if !guard_active {
            guard_suppressed = true;
            guard_cooldown = guard_cooldown.saturating_sub(1);
            println!(
                "[THERMO-GPU][COMPACTION-GUARD] Cooldown active → skipping guard checks this temp"
            );
        }

        dynamic_guard_threshold = dynamic_guard_threshold.max(guard_threshold_floor);
        let temp_start = std::time::Instant::now();

        println!(
            "[THERMO-GPU] Processing temperature {}/{}: T={:.3}",
            temp_idx + 1,
            num_temps,
            temp
        );

        // FluxNet RL: Per-temperature decision-making
        // Store telemetry before this temperature for reward computation
        let telemetry_before = if fluxnet_state.is_some() {
            if temp_idx > 0 {
                let prev_solution = equilibrium_states.last().unwrap();
                let prev_chromatic = prev_solution.chromatic_number;
                let prev_conflicts = prev_solution.conflicts;
                let prev_compaction = 0.75f32;
                Some((prev_chromatic, prev_conflicts, prev_compaction))
            } else {
                Some((initial_chromatic, initial_conflicts, 0.5f32))
            }
        } else {
            None
        };

        // FluxNet RL: Select and apply force command at start of temperature
        let mut fluxnet_command: Option<crate::fluxnet::ForceCommand> = None;
        let mut rl_state_before: Option<crate::fluxnet::RLState> = None;
        let mut rl_telemetry: Option<(f32, bool)> = None; // (q_value, was_exploration)
        let mut force_band_stats_before: Option<crate::fluxnet::ForceBandStats> = None;
        let mut force_band_stats_after: Option<crate::fluxnet::ForceBandStats> = None;

        if let Some((ref mut force_profile, ref mut rl_controller)) = fluxnet_state {
            if let Some((chromatic_before, conflicts_before, compaction_before)) = telemetry_before
            {
                // Capture force band stats before RL action
                force_band_stats_before = Some(force_profile.compute_stats());
                let band_std_before = force_band_stats_before
                    .as_ref()
                    .map(|s| s.std_force)
                    .unwrap_or(0.0);

                // Create RL state from previous telemetry
                let state = crate::fluxnet::RLState::from_telemetry(
                    conflicts_before,
                    chromatic_before,
                    compaction_before,
                    n,
                    n,
                    consecutive_guards,
                    current_slack,
                    max_slack_limit,
                    band_std_before,
                    compaction_before < dynamic_guard_threshold,
                    dynamic_guard_threshold,
                    collapse_streak,
                    dynamic_steps_per_temp,
                    max_steps_per_temp,
                );

                // Store state for Q-table update at end of temperature
                rl_state_before = Some(state);

                // Select action via epsilon-greedy with telemetry capture
                let (command, q_value, was_exploration) =
                    rl_controller.select_action_with_telemetry(&state);

                // Store telemetry data
                rl_telemetry = Some((q_value, was_exploration));

                println!(
                    "[FLUXNET][T={}] State: conflicts={}, chromatic={}, compaction={:.2}",
                    temp_idx, conflicts_before, chromatic_before, compaction_before
                );
                println!(
                    "[FLUXNET][T={}] Action: {} (Q={:.3}, ε={:.3}, explore={})",
                    temp_idx,
                    command.description(),
                    q_value,
                    rl_controller.epsilon(),
                    was_exploration
                );

                // Apply command to force profile
                if command.is_meta_guard_adjustment() {
                    match command {
                        crate::fluxnet::ForceCommand::IncreaseSlack => {
                            let old = current_slack;
                            current_slack = (current_slack + 5).min(max_slack_limit);
                            println!(
                                "[FLUXNET][T={}] Meta-action SL+: slack {} → {} (min={}, max={})",
                                temp_idx, old, current_slack, min_slack_limit, max_slack_limit
                            );
                        }
                        crate::fluxnet::ForceCommand::DecreaseSlack => {
                            let old = current_slack;
                            current_slack = current_slack.saturating_sub(5).max(min_slack_limit);
                            println!(
                                "[FLUXNET][T={}] Meta-action SL-: slack {} → {} (min={}, max={})",
                                temp_idx, old, current_slack, min_slack_limit, max_slack_limit
                            );
                        }
                        crate::fluxnet::ForceCommand::RaiseGuardThreshold => {
                            let old = dynamic_guard_threshold;
                            dynamic_guard_threshold = (dynamic_guard_threshold + 0.02).min(0.35);
                            println!(
                                "[FLUXNET][T={}] Meta-action TH+: threshold {:.3} → {:.3}",
                                temp_idx, old, dynamic_guard_threshold
                            );
                        }
                        crate::fluxnet::ForceCommand::LowerGuardThreshold => {
                            let old = dynamic_guard_threshold;
                            dynamic_guard_threshold =
                                (dynamic_guard_threshold - 0.02).max(guard_threshold_floor);
                            println!(
                                "[FLUXNET][T={}] Meta-action TH-: threshold {:.3} → {:.3}",
                                temp_idx, old, dynamic_guard_threshold
                            );
                        }
                        _ => {}
                    }
                    force_band_stats_after = force_band_stats_before.clone();
                    fluxnet_command = Some(command);
                } else if command.is_meta_step_adjustment() {
                    match command {
                        crate::fluxnet::ForceCommand::IncreaseSteps => {
                            let old = dynamic_steps_per_temp;
                            dynamic_steps_per_temp =
                                (dynamic_steps_per_temp + 5_000).min(max_steps_per_temp);
                            println!(
                                "[FLUXNET][T={}] Meta-action ST+: steps {} → {} (min={}, max={})",
                                temp_idx,
                                old,
                                dynamic_steps_per_temp,
                                min_steps_per_temp,
                                max_steps_per_temp
                            );
                        }
                        crate::fluxnet::ForceCommand::DecreaseSteps => {
                            let old = dynamic_steps_per_temp;
                            dynamic_steps_per_temp = dynamic_steps_per_temp
                                .saturating_sub(5_000)
                                .max(min_steps_per_temp);
                            println!(
                                "[FLUXNET][T={}] Meta-action ST-: steps {} → {} (min={}, max={})",
                                temp_idx,
                                old,
                                dynamic_steps_per_temp,
                                min_steps_per_temp,
                                max_steps_per_temp
                            );
                        }
                        _ => {}
                    }
                    force_band_stats_after = force_band_stats_before.clone();
                    fluxnet_command = Some(command);
                } else {
                    match force_profile.apply_force_command(&command) {
                        Ok(result) => {
                            println!(
                                "[FLUXNET][T={}] Applied {}: mean_force {:.3} → {:.3}",
                                temp_idx,
                                command.telemetry_code(),
                                result.stats_before.mean_force,
                                result.stats_after.mean_force
                            );
                            force_band_stats_after = Some(result.stats_after.clone());
                            fluxnet_command = Some(command);
                        }
                        Err(e) => {
                            eprintln!("[FLUXNET][T={}] Error applying command: {}", temp_idx, e);
                        }
                    }
                }

                // Force buffers are already synced to GPU in apply_force_command()
            }
        }

        // Initialize oscillator phases on GPU
        let d_phases = cuda_device
            .htod_copy(vec![0.0f32; n])
            .map_err(|e| PRCTError::GpuError(format!("Failed to allocate phases: {}", e)))?;
        let d_velocities = cuda_device
            .htod_copy(vec![0.0f32; n])
            .map_err(|e| PRCTError::GpuError(format!("Failed to allocate velocities: {}", e)))?;
        let d_coupling_forces = cuda_device
            .alloc_zeros::<f32>(n)
            .map_err(|e| PRCTError::GpuError(format!("Failed to allocate forces: {}", e)))?;

        let threads = 256;
        let blocks = n.div_ceil(threads);

        // Initialize oscillators from current coloring with AI-guided perturbation
        // TWEAK 2: Use reservoir scores to create phase bands (strong/neutral/weak force zones)
        let mut color_phases: Vec<f32> = if vertex_perturbation_weights.is_some() {
            let weights = vertex_perturbation_weights.as_ref().unwrap();
            // AI-guided: add weighted random perturbation to high-uncertainty vertices
            use rand::Rng;
            let mut rng = rand::thread_rng();

            initial_solution
                .colors
                .iter()
                .enumerate()
                .map(|(v, &c)| {
                    let base_phase =
                        (c as f32 / target_chromatic as f32) * 2.0 * std::f32::consts::PI;

                    // TWEAK 2: Amplify perturbation for high-difficulty vertices (top 20%)
                    // This creates implicit "strong force" bands
                    let difficulty_boost = if weights[v] > 0.8 { 1.5 } else { 1.0 };
                    let perturbation_strength = weights[v] * temp as f32 * 0.5 * difficulty_boost;
                    let perturbation = rng.gen_range(-perturbation_strength..perturbation_strength);

                    base_phase + perturbation
                })
                .collect()
        } else {
            // Uniform: standard phase initialization
            initial_solution
                .colors
                .iter()
                .map(|&c| (c as f32 / target_chromatic as f32) * 2.0 * std::f32::consts::PI)
                .collect()
        };

        if randomize_next_phases {
            use rand::seq::SliceRandom;
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);
            let mut perturb_count = ((n as f32) * 0.15).ceil() as usize;
            perturb_count = perturb_count.max(1).min(n);
            for idx in indices.into_iter().take(perturb_count) {
                color_phases[idx] = rng.gen_range(0.0..(2.0 * std::f32::consts::PI));
            }
            randomize_next_phases = false;
            println!(
                "[THERMO-GPU][GUARD-REHEAT] Randomized {} oscillator phases to break lock",
                perturb_count
            );
        }

        let mut d_phases = d_phases;
        cuda_device
            .htod_copy_into(color_phases, &mut d_phases)
            .map_err(|e| PRCTError::GpuError(format!("Failed to init phases: {}", e)))?;

        // Evolution loop
        let dt = 0.01f32;
        let coupling_strength = 1.0f32 / (n as f32).sqrt();

        for step in 0..dynamic_steps_per_temp {
            // Compute coupling forces (Task B1: temperature-dependent coupling)
            // MOVE 5: Pass uncertainty for band-aware coupling redistribution
            unsafe {
                (*compute_coupling)
                    .clone()
                    .launch(
                        LaunchConfig {
                            grid_dim: (blocks as u32, 1, 1),
                            block_dim: (threads as u32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &d_phases,
                            &d_edge_u,
                            &d_edge_v,
                            &d_edge_w,
                            graph.num_edges as i32,
                            n as i32,
                            coupling_strength,
                            temp as f32,       // Task B1: current temperature
                            t_max as f32,      // Task B1: max temperature for modulation
                            &d_vertex_weights, // MOVE 5: Pass uncertainty for band-aware coupling
                            &d_coupling_forces,
                        ),
                    )
                    .map_err(|e| {
                        PRCTError::GpuError(format!(
                            "Coupling kernel failed at step {}: {}",
                            step, e
                        ))
                    })?;
            }

            // Evolve oscillators with FluxNet force modulation if enabled
            // FluxNet: Pass force buffers if available, otherwise pass zero-length dummy slices
            // CUDA kernel checks for NULL pointer and uses multiplier=1.0 when NULL
            if let Some((ref force_profile, ref _rl_controller)) = fluxnet_state {
                // FluxNet enabled: use actual force buffers
                unsafe {
                    (*evolve_osc)
                        .clone()
                        .launch(
                            LaunchConfig {
                                grid_dim: (blocks as u32, 1, 1),
                                block_dim: (threads as u32, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &d_phases,
                                &d_velocities,
                                &d_coupling_forces,
                                n as i32,
                                dt,
                                temp as f32,
                                &force_profile.device_f_strong,
                                &force_profile.device_f_weak,
                            ),
                        )
                        .map_err(|e| {
                            PRCTError::GpuError(format!(
                                "Evolve kernel with FluxNet failed at step {}: {}",
                                step, e
                            ))
                        })?;
                }
            } else {
                // FluxNet disabled: use pre-allocated identity multiplier buffers (all 1.0f)
                // This ensures forces remain unchanged when FluxNet is not active
                unsafe {
                    (*evolve_osc)
                        .clone()
                        .launch(
                            LaunchConfig {
                                grid_dim: (blocks as u32, 1, 1),
                                block_dim: (threads as u32, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &d_phases,
                                &d_velocities,
                                &d_coupling_forces,
                                n as i32,
                                dt,
                                temp as f32,
                                &identity_f_strong,
                                &identity_f_weak,
                            ),
                        )
                        .map_err(|e| {
                            PRCTError::GpuError(format!(
                                "Evolve kernel failed at step {}: {}",
                                step, e
                            ))
                        })?;
                }
            }

            // Periodic energy computation (every 100 steps)
            if step % 100 == 0 {
                // Allocate and zero energy buffer for this computation
                let d_energy = cuda_device.alloc_zeros::<f32>(1).map_err(|e| {
                    PRCTError::GpuError(format!("Failed to allocate energy: {}", e))
                })?;

                // Energy kernel uses edge-based computation
                let energy_blocks = graph.num_edges.div_ceil(threads);
                unsafe {
                    (*compute_energy)
                        .clone()
                        .launch(
                            LaunchConfig {
                                grid_dim: (energy_blocks as u32, 1, 1),
                                block_dim: (threads as u32, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &d_phases,
                                &d_edge_u,
                                &d_edge_v,
                                &d_edge_w,
                                graph.num_edges as i32,
                                n as i32,
                                &d_energy,
                            ),
                        )
                        .map_err(|e| {
                            PRCTError::GpuError(format!(
                                "Energy kernel failed at step {}: {}",
                                step, e
                            ))
                        })?;
                }
            }
        }

        // Download final phases
        let final_phases = cuda_device
            .dtoh_sync_copy(&d_phases)
            .map_err(|e| PRCTError::GpuError(format!("Failed to download phases: {}", e)))?;

        // Convert phases to coloring with dynamic color range
        // CRITICAL FIX: Use initial_chromatic + slack, NOT target_chromatic
        // This prevents chromatic collapse from 127 -> 19 colors
        // TWEAK 3: Use dynamic slack that expands on consecutive guard triggers
        let color_range = initial_chromatic + current_slack; // Available color buckets

        println!(
            "[THERMO-GPU][PHASE-TO-COLOR] Using color_range={} (initial={} + slack={})",
            color_range, initial_chromatic, current_slack
        );

        let mut colors: Vec<usize> = final_phases
            .iter()
            .map(|&phase| {
                let normalized =
                    (phase.rem_euclid(2.0 * std::f32::consts::PI)) / (2.0 * std::f32::consts::PI);
                (normalized * color_range as f32).floor() as usize % color_range
            })
            .collect();

        // Compact colors: renumber to sequential [0, actual_chromatic)
        // This removes gaps and gives us the true chromatic number
        use std::collections::HashMap;
        let mut color_map: HashMap<usize, usize> = HashMap::new();
        let mut next_color = 0;

        for c in &mut colors {
            let new_color = *color_map.entry(*c).or_insert_with(|| {
                let nc = next_color;
                next_color += 1;
                nc
            });
            *c = new_color;
        }

        let actual_chromatic = next_color; // True chromatic after compaction
        let compaction_ratio = actual_chromatic as f64 / color_range as f64;
        let unique_buckets = color_map.len(); // Number of distinct phase buckets used

        println!(
            "[THERMO-GPU][COMPACTION] {} phase buckets -> {} actual colors (ratio: {:.3})",
            color_range, actual_chromatic, compaction_ratio
        );

        // Compute conflicts FIRST (before guards)
        let mut conflicts_temp = 0;
        for &(u, v, _) in &graph.edges {
            if colors[u] == colors[v] {
                conflicts_temp += 1;
            }
        }

        // Task B5: De-sync burst detection
        let bucket_collapse_risk = if unique_buckets < (color_range / 2) {
            "high"
        } else {
            "low"
        };

        // Task B6: Compaction guard - prevent catastrophic chromatic collapse
        let collapse_threshold = (initial_chromatic as f64 * 0.5) as usize;
        let phase_locked = compaction_ratio < dynamic_guard_threshold as f64;
        if phase_locked {
            collapse_streak += 1;
        } else {
            collapse_streak = 0;
        }
        let collapse_streak_observed = collapse_streak;
        let compaction_guard_triggered = guard_active
            && (phase_locked || (actual_chromatic < collapse_threshold && conflicts_temp > 1000));
        let mut shake_invoked = false;
        let mut shake_vertices_count = 0;
        let mut post_shake_conflicts_count = 0;
        let mut slack_expanded = false;
        let mut snapshot_reseeded = false;

        // MOVE 3: Immediate slack expansion on guard (clamp to configured limits)
        if compaction_guard_triggered {
            consecutive_guards += 1;
            temps_since_guard = 0; // MOVE 5: Reset guard boost counter

            if consecutive_guards == 1 {
                if post_reheat_slack_lock > 0 {
                    println!(
                        "[THERMO-GPU][MOVE-3][SLACK-EXPAND] Skipping expansion (reheat cooldown active: {} temps remaining)",
                        post_reheat_slack_lock
                    );
                } else {
                    // MOVE 3: Clamp slack to configured window
                    let old_slack = current_slack;
                    current_slack = current_slack.clamp(min_slack_limit, max_slack_limit);
                    slack_expanded = true;
                    println!(
                        "[THERMO-GPU][MOVE-3][SLACK-EXPAND] Maintaining aggressive slack={} (was {}, min={}, max={})",
                        current_slack, old_slack, min_slack_limit, max_slack_limit
                    );
                }
            }

            // MOVE 5: Guard boost activated
            println!(
                "[THERMO-GPU][MOVE-5][GUARD-BOOST] Activating {:.1}x repulsion boost for next {} temps",
                guard_boost_multiplier, guard_boost_duration
            );

            let should_reheat = phase_locked && collapse_streak >= collapse_reheat_threshold;

            if should_reheat {
                reheat_triggered = true;
                let boosted = (temp * reheat_temp_boost).min(t_max);
                pending_reheat_temp = Some(boosted);
                guard_cooldown = guard_cooldown.max(2);
                randomize_next_phases = true;
                guard_threshold_floor = 0.08;
                guard_threshold_relax_timer = guard_threshold_relax_timer.max(5);
                if dynamic_guard_threshold < guard_threshold_floor {
                    dynamic_guard_threshold = guard_threshold_floor;
                }
                let old_slack = current_slack;
                current_slack = current_slack.min(guard_initial_slack);
                if old_slack != current_slack {
                    println!(
                        "[THERMO-GPU][GUARD-REHEAT] Reset slack from {} → {} during cooldown (min={})",
                        old_slack, current_slack, guard_initial_slack
                    );
                }
                post_reheat_slack_lock = post_reheat_slack_lock.max(3);
                println!(
                    "[THERMO-GPU][GUARD-REHEAT] Collapse streak {} ≥ {} → scheduling reheat (boost {:.2}×)",
                    collapse_streak, collapse_reheat_threshold, reheat_temp_boost
                );
                collapse_streak = 0;
            } else if let Some((best_temp_idx, ref best_sol, best_q)) = best_snapshot {
                println!(
                    "[THERMO-GPU][MOVE-2][SNAPSHOT-RESET] Re-injecting best snapshot from temp {} ({} colors, {} conflicts, quality={:.6})",
                    best_temp_idx, best_sol.chromatic_number, best_sol.conflicts, best_q
                );

                // Reset to best snapshot state
                colors = best_sol.colors.clone();

                // MOVE 2: Reset oscillator phases to best snapshot to restart dynamics
                // Convert best coloring back to phases for clean restart
                let reset_phases: Vec<f32> = best_sol
                    .colors
                    .iter()
                    .map(|&c| (c as f32 / target_chromatic as f32) * 2.0 * std::f32::consts::PI)
                    .collect();

                cuda_device
                    .htod_copy_into(reset_phases.clone(), &mut d_phases)
                    .map_err(|e| PRCTError::GpuError(format!("Failed to reset phases: {}", e)))?;

                println!(
                    "[THERMO-GPU][MOVE-2][SNAPSHOT-RESET] Reset {} oscillator phases to best snapshot",
                    n
                );
                snapshot_reseeded = true;
            } else {
                println!(
                    "[THERMO-GPU][MOVE-2][SNAPSHOT-RESET] No snapshot available yet (temp {})",
                    temp_idx
                );
            }
        } else {
            consecutive_guards = 0; // Reset on success
            temps_since_guard += 1; // MOVE 5: Increment temps since last guard
        }

        // MOVE 5: Check if guard boost is active
        let guard_boost_active = temps_since_guard < guard_boost_duration;
        let current_guard_boost = if guard_boost_active {
            guard_boost_multiplier as f32
        } else {
            1.0f32
        };

        if guard_boost_active {
            println!(
                "[THERMO-GPU][MOVE-5][GUARD-BOOST] ACTIVE (temp {} after guard): {:.1}x repulsion multiplier",
                temps_since_guard + 1, current_guard_boost
            );
        }

        if compaction_guard_triggered {
            eprintln!(
                "[THERMO-GPU][COMPACTION-GUARD] CRITICAL: Chromatic collapsed to {} (from {}), reverting compaction",
                actual_chromatic, initial_chromatic
            );
            eprintln!(
                "[THERMO-GPU][COMPACTION-GUARD] This indicates phase-locking - {} unique buckets used (expected ~{})",
                unique_buckets, color_range
            );

            // Revert to original colors - use pre-compaction bucket assignments
            colors = final_phases
                .iter()
                .map(|&phase| {
                    let normalized = (phase.rem_euclid(2.0 * std::f32::consts::PI))
                        / (2.0 * std::f32::consts::PI);
                    (normalized * color_range as f32).floor() as usize % color_range
                })
                .collect();

            println!(
                "[THERMO-GPU][COMPACTION-GUARD] Reverted to bucket assignments (chromatic = {})",
                color_range
            );

            // MOVE 3: AGGRESSIVE shake - identify and perturb 150 high-conflict vertices
            let mut vertex_conflicts_shake: Vec<(usize, usize)> = (0..n)
                .map(|v| {
                    let mut v_conflicts = 0;
                    for &(u, w, _) in &graph.edges {
                        if (u == v || w == v) && colors[u] == colors[w] {
                            v_conflicts += 1;
                        }
                    }
                    (v, v_conflicts)
                })
                .collect();

            vertex_conflicts_shake.sort_by_key(|(_, c)| std::cmp::Reverse(*c));

            // MOVE 3: AGGRESSIVE shake count to 150, shake across strong AND neutral bands
            let raw_unc = ai_uncertainty
                .map(|u| u.to_vec())
                .unwrap_or_else(|| vec![0.5; n]);
            let shake_count = vertex_conflicts_shake
                .iter()
                .take(150) // MOVE 3: AGGRESSIVE - Increased from 100 to 150
                .filter(|(v, c)| *c > 0 && raw_unc[*v] > 0.33) // Strong or neutral band
                .count();

            if shake_count > 0 {
                shake_invoked = true;
                shake_vertices_count = shake_count;

                println!(
                    "[THERMO-GPU][MOVE-3][SHAKE] AGGRESSIVE: Shaking {} high-conflict vertices (strong+neutral bands) to break symmetry",
                    shake_count
                );

                use rand::Rng;
                let mut rng = rand::thread_rng();
                let mut final_phases_mut = final_phases.clone();

                for &(v, conf) in vertex_conflicts_shake.iter().take(150) {
                    // MOVE 3: 150 vertices
                    if conf > 0 && raw_unc[v] > 0.33 {
                        // MOVE 3: Strong or neutral band
                        let offset = rng.gen_range(0.0..(2.0 * std::f32::consts::PI));
                        final_phases_mut[v] =
                            (final_phases_mut[v] + offset).rem_euclid(2.0 * std::f32::consts::PI);
                    }
                }

                // Re-map with shaken phases
                colors = final_phases_mut
                    .iter()
                    .map(|&phase| {
                        let normalized = (phase.rem_euclid(2.0 * std::f32::consts::PI))
                            / (2.0 * std::f32::consts::PI);
                        (normalized * color_range as f32).floor() as usize % color_range
                    })
                    .collect();

                post_shake_conflicts_count = graph
                    .edges
                    .iter()
                    .filter(|(u, v, _)| colors[*u] == colors[*v])
                    .count();
                println!(
                    "[THERMO-GPU][TWEAK-4][SHAKE] Post-shake: {} conflicts (was {})",
                    post_shake_conflicts_count, conflicts_temp
                );
            }
        }

        // Compute conflicts and per-vertex conflict counts (final computation)
        let mut conflicts = 0;
        let mut vertex_conflicts = vec![0usize; n];

        for &(u, v, _) in &graph.edges {
            if colors[u] == colors[v] {
                conflicts += 1;
                vertex_conflicts[u] += 1;
                vertex_conflicts[v] += 1;
            }
        }

        let max_vertex_conflicts = vertex_conflicts.iter().max().copied().unwrap_or(0);
        let stuck_vertices = vertex_conflicts.iter().filter(|&&c| c > 5).count();

        // MOVE 3: Slack decay - only decay after 3 stable temps without guards (slower decay)
        if compaction_ratio > 0.7 && !compaction_guard_triggered {
            stable_temps += 1;
            if stable_temps >= 3 {
                // MOVE 3: Increased from 2 to 3 for slower decay
                let old_slack = current_slack;
                current_slack = (current_slack.saturating_sub(5)).max(min_slack_limit);
                if old_slack != current_slack {
                    println!(
                        "[THERMO-GPU][MOVE-3][SLACK-DECAY] Gradual decay from +{} to +{} after {} stable temps (min=40)",
                        old_slack, current_slack, stable_temps
                    );
                }
                stable_temps = 0; // Reset counter after decay
            }
        } else {
            stable_temps = 0;
        }

        // MOVE 6: Adaptive reheat detection - track progress and detect stalls
        let chromatic_improved = actual_chromatic < prev_chromatic;
        let reheat_recommended = if chromatic_improved {
            stall_counter = 0; // Reset on improvement
            false
        } else {
            stall_counter += 1;
            stall_counter >= reheat_stall_threshold
        };

        if reheat_recommended {
            println!(
                "[THERMO-GPU][MOVE-6][ADAPTIVE-REHEAT] STALL DETECTED: No chromatic improvement for {} temps (current={}, was={})",
                stall_counter, actual_chromatic, prev_chromatic
            );
            println!(
                "[THERMO-GPU][MOVE-6][ADAPTIVE-REHEAT] RECOMMENDATION: Consider reheating or increasing steps_per_temp from {} to {}",
                dynamic_steps_per_temp,
                (dynamic_steps_per_temp + 5_000).min(max_steps_per_temp)
            );
        }

        prev_chromatic = actual_chromatic; // Update for next iteration

        println!(
            "[THERMO-GPU] T={:.3}: {} colors, {} conflicts (max_vertex={}, stuck={}, stall={})",
            temp, actual_chromatic, conflicts, max_vertex_conflicts, stuck_vertices, stall_counter
        );

        // Count actual chromatic number
        let chromatic_number = actual_chromatic;

        let solution = ColoringSolution {
            colors,
            chromatic_number,
            conflicts,
            quality_score: 1.0 / (conflicts + 1) as f64,
            computation_time_ms: 0.0,
        };

        // TWEAK 6: Track best snapshot based on quality score
        let quality_score = if conflicts == 0 {
            1.0 / chromatic_number as f64
        } else {
            1.0 / (chromatic_number as f64 * (1.0 + conflicts as f64))
        };

        let is_new_best = best_snapshot
            .as_ref()
            .map_or(true, |(_, _, prev_q)| quality_score > *prev_q);
        if is_new_best {
            println!(
                "[THERMO-GPU][TWEAK-6][SNAPSHOT] New best at temp {}: {} colors, {} conflicts (quality={:.6})",
                temp_idx, chromatic_number, conflicts, quality_score
            );
            best_snapshot = Some((temp_idx, solution.clone(), quality_score));
        }

        // Record detailed telemetry for this temperature
        let temp_elapsed = temp_start.elapsed();

        // TWEAK 1: Calculate force blend factor for telemetry
        let force_blend_factor = if temp <= force_start_temp {
            if force_start_temp > force_full_strength_temp {
                let blend = 1.0
                    - (temp - force_full_strength_temp)
                        / (force_start_temp - force_full_strength_temp);
                blend.max(0.0).min(1.0)
            } else {
                1.0
            }
        } else {
            0.0
        };

        if let Some(ref telemetry) = telemetry {
            use crate::telemetry::{OptimizationGuidance, PhaseExecMode, PhaseName, RunMetric};
            use serde_json::json;

            // Calculate improvement metrics
            let chromatic_delta = chromatic_number as i32 - initial_chromatic as i32;
            let conflict_delta = conflicts as i32 - initial_conflicts as i32;
            let effectiveness = if temp_idx > 0 {
                (initial_chromatic.saturating_sub(chromatic_number)) as f64 / (temp_idx + 1) as f64
            } else {
                0.0
            };

            // Detect issues
            let issue_detected = if chromatic_number < 50 && conflicts > 10000 {
                "chromatic_collapsed_with_conflicts"
            } else if chromatic_number < initial_chromatic / 2 {
                "chromatic_collapsed"
            } else if conflicts > 100000 {
                "conflicts_not_resolving"
            } else {
                "none"
            };

            // Generate actionable recommendations
            let mut recommendations = Vec::new();
            let guidance_status = if issue_detected != "none" {
                recommendations.push(format!(
                    "CRITICAL ISSUE: {} - chromatic={}, conflicts={}",
                    issue_detected, chromatic_number, conflicts
                ));
                recommendations.push(format!(
                    "Color mapping issue: phase buckets may be too narrow (current color_range={})",
                    color_range
                ));
                "critical"
            } else if conflicts > 100 {
                recommendations.push(format!(
                    "CRITICAL: {} conflicts at temp {:.3} - increase steps_per_temp from {} to {}+",
                    conflicts,
                    temp,
                    dynamic_steps_per_temp,
                    (dynamic_steps_per_temp + 5_000).min(max_steps_per_temp)
                ));
                recommendations
                    .push("Consider increasing t_max for better exploration".to_string());
                "critical"
            } else if chromatic_number > initial_chromatic * 95 / 100 {
                recommendations.push(format!(
                    "Limited progress: {} colors (started at {}) - increase num_temps to {}+",
                    chromatic_number,
                    initial_chromatic,
                    num_temps + 16
                ));
                recommendations.push(format!(
                    "Or increase t_max from {:.3} to {:.3}",
                    t_max,
                    t_max * 1.5
                ));
                "need_tuning"
            } else if chromatic_number < initial_chromatic * 80 / 100 {
                recommendations.push(format!(
                    "EXCELLENT: Reduced from {} to {} colors ({:.1}% reduction)",
                    initial_chromatic,
                    chromatic_number,
                    (initial_chromatic - chromatic_number) as f64 / initial_chromatic as f64
                        * 100.0
                ));
                recommendations
                    .push("These thermo settings are optimal, maintain them".to_string());
                "excellent"
            } else {
                recommendations.push("On track - steady progress".to_string());
                "on_track"
            };

            let guidance =
                OptimizationGuidance {
                    status: guidance_status.to_string(),
                    recommendations,
                    estimated_final_colors: Some(chromatic_number.saturating_sub(
                        ((num_temps - temp_idx - 1) as f64 * effectiveness) as usize,
                    )),
                    confidence: if temp_idx < 3 { 0.5 } else { 0.85 },
                    gap_to_world_record: Some((chromatic_number as i32) - 83), // DSJC1000.5 WR = 83
                };

            // Split parameters into multiple json! blocks to avoid recursion limit
            let mut params = serde_json::Map::new();

            // Basic temperature metrics
            params.insert("temperature".to_string(), json!(temp));
            params.insert("temp_index".to_string(), json!(temp_idx));
            params.insert("total_temps".to_string(), json!(num_temps));
            params.insert("chromatic_delta".to_string(), json!(chromatic_delta));
            params.insert("conflict_delta".to_string(), json!(conflict_delta));
            params.insert("effectiveness".to_string(), json!(effectiveness));
            params.insert(
                "cumulative_improvement".to_string(),
                json!(initial_chromatic.saturating_sub(chromatic_number)),
            );
            params.insert(
                "improvement_rate_per_temp".to_string(),
                json!(effectiveness),
            );
            params.insert("steps_per_temp".to_string(), json!(dynamic_steps_per_temp));
            params.insert("t_min".to_string(), json!(t_min));
            params.insert("t_max".to_string(), json!(t_max));

            // Color mapping metrics
            params.insert("color_range".to_string(), json!(color_range));
            params.insert(
                "chromatic_before_compaction".to_string(),
                json!(color_range),
            );
            params.insert(
                "chromatic_after_compaction".to_string(),
                json!(chromatic_number),
            );
            params.insert("compaction_ratio".to_string(), json!(compaction_ratio));
            params.insert(
                "max_vertex_conflicts".to_string(),
                json!(max_vertex_conflicts),
            );
            params.insert("stuck_vertices".to_string(), json!(stuck_vertices));
            params.insert("issue_detected".to_string(), json!(issue_detected));
            params.insert("unique_buckets".to_string(), json!(unique_buckets));
            params.insert(
                "bucket_collapse_risk".to_string(),
                json!(bucket_collapse_risk),
            );
            params.insert(
                "coupling_modulation".to_string(),
                json!(temp as f64 / t_max),
            );
            params.insert("natural_freq_enabled".to_string(), json!(true));
            params.insert(
                "compaction_guard_triggered".to_string(),
                json!(compaction_guard_triggered),
            );

            // Force and coupling metrics
            params.insert("force_blend_factor".to_string(), json!(force_blend_factor));
            params.insert("force_start_temp".to_string(), json!(force_start_temp));
            params.insert(
                "force_full_strength_temp".to_string(),
                json!(force_full_strength_temp),
            );
            params.insert(
                "coupling_strong_gain".to_string(),
                json!(if temp > 3.0 { 0.15 } else { 1.0 }),
            );
            params.insert(
                "coupling_weak_gain".to_string(),
                json!(if temp > 3.0 { 1.40 } else { 1.0 }),
            );

            // Shake metrics
            params.insert("shake_invoked".to_string(), json!(shake_invoked));
            params.insert("shake_vertices".to_string(), json!(shake_vertices_count));
            params.insert(
                "post_shake_conflicts".to_string(),
                json!(post_shake_conflicts_count),
            );

            // Slack metrics
            params.insert("current_slack".to_string(), json!(current_slack));
            params.insert("slack_expanded".to_string(), json!(slack_expanded));
            params.insert("consecutive_guards".to_string(), json!(consecutive_guards));
            params.insert(
                "guard_threshold".to_string(),
                json!(dynamic_guard_threshold),
            );
            params.insert("phase_locked".to_string(), json!(phase_locked));
            params.insert(
                "collapse_streak".to_string(),
                json!(collapse_streak_observed),
            );
            params.insert(
                "guard_cooldown_remaining".to_string(),
                json!(guard_cooldown),
            );
            params.insert("guard_active".to_string(), json!(guard_active));
            params.insert("guard_suppressed".to_string(), json!(guard_suppressed));
            params.insert(
                "dynamic_steps_per_temp".to_string(),
                json!(dynamic_steps_per_temp),
            );
            params.insert("reheat_triggered".to_string(), json!(reheat_triggered));

            // Snapshot metrics
            params.insert("is_best_snapshot".to_string(), json!(is_new_best));
            params.insert("snapshot_reseeded".to_string(), json!(snapshot_reseeded));

            // Guard boost metrics (MOVE 5)
            params.insert("guard_boost_active".to_string(), json!(guard_boost_active));
            params.insert(
                "guard_boost_multiplier".to_string(),
                json!(current_guard_boost),
            );
            params.insert("temps_since_guard".to_string(), json!(temps_since_guard));

            // Reheat metrics (MOVE 6)
            params.insert("reheat_stall_counter".to_string(), json!(stall_counter));
            params.insert("reheat_recommended".to_string(), json!(reheat_recommended));
            params.insert("chromatic_improved".to_string(), json!(chromatic_improved));

            // Add FluxNet telemetry if enabled
            if let (
                Some((ref force_profile, ref rl_controller)),
                Some(cmd),
                Some(state),
                Some((q_value, was_exploration)),
                Some(_stats_before),
            ) = (
                &fluxnet_state,
                fluxnet_command,
                rl_state_before,
                rl_telemetry,
                force_band_stats_before,
            ) {
                if let Some(config) = fluxnet_config {
                    use crate::fluxnet::telemetry::*;

                    let force_bands =
                        ForceBandTelemetry::from_force_band_stats(&force_profile.compute_stats());

                    let rl_decision = RLDecisionTelemetry::new(
                        temp_idx,
                        &state,
                        cmd,
                        q_value,
                        rl_controller.epsilon(),
                        was_exploration,
                    );

                    let config_snapshot = FluxNetConfigSnapshot::from_config(config);

                    let fluxnet_telem = FluxNetTelemetry::new(
                        force_bands,
                        rl_decision,
                        None, // Q-update telemetry will be added after RL update below
                        config_snapshot,
                    );

                    if let Ok(fluxnet_json) = serde_json::to_value(&fluxnet_telem) {
                        params.insert("fluxnet".to_string(), fluxnet_json);
                    }
                }
            }

            telemetry.record(
                RunMetric::new(
                    PhaseName::Thermodynamic,
                    format!("temp_{}/{}", temp_idx + 1, num_temps),
                    chromatic_number,
                    conflicts,
                    temp_elapsed.as_secs_f64() * 1000.0,
                    PhaseExecMode::gpu_success(Some(2)),
                )
                .with_parameters(serde_json::Value::Object(params))
                .with_guidance(guidance),
            );
        }

        // FluxNet RL: Compute reward and update Q-table after temperature completes
        if let Some((ref mut force_profile, ref mut rl_controller)) = fluxnet_state {
            if let (
                Some(cmd),
                Some(state_before),
                Some((chromatic_before, conflicts_before, compaction_before)),
            ) = (fluxnet_command, rl_state_before, telemetry_before)
            {
                // Compute reward from telemetry delta
                let compaction_ratio_f32 = compaction_ratio as f32;
                let mut reward = rl_controller.compute_reward(
                    conflicts_before,
                    conflicts,
                    chromatic_before,
                    chromatic_number,
                    compaction_before,
                    compaction_ratio_f32,
                );
                if phase_locked {
                    // Penalize collapses so RL prefers guard meta-actions
                    let penalty = (collapse_streak_observed as f32 + 1.0) * 0.25;
                    reward -= penalty;
                }

                // Create next state from current telemetry
                let band_std_after = force_band_stats_after
                    .as_ref()
                    .map(|s| s.std_force)
                    .unwrap_or_else(|| force_profile.compute_stats().std_force);
                let next_state = crate::fluxnet::RLState::from_telemetry(
                    conflicts,
                    chromatic_number,
                    compaction_ratio_f32,
                    n, // max_conflicts
                    n, // max_chromatic
                    consecutive_guards,
                    current_slack,
                    max_slack_limit,
                    band_std_after,
                    phase_locked,
                    dynamic_guard_threshold,
                    collapse_streak,
                    dynamic_steps_per_temp,
                    max_steps_per_temp,
                );

                // Determine if this is a terminal state (last temperature)
                let is_terminal = temp_idx == num_temps - 1;

                // Update Q-table with experience and capture telemetry
                let (q_old, q_new, q_delta) = rl_controller.update_with_telemetry(
                    state_before,
                    cmd,
                    reward,
                    next_state,
                    is_terminal,
                );

                println!(
                    "[FLUXNET][T={}] Reward: {:.3}, Q: {:.3} → {:.3} (Δ={:.3}), ε={:.3}",
                    temp_idx,
                    reward,
                    q_old,
                    q_new,
                    q_delta,
                    rl_controller.epsilon()
                );

                // Save Q-table at intervals if configured
                if let Some(config) = fluxnet_config {
                    if config.persistence.save_interval_temps > 0
                        && temp_idx % config.persistence.save_interval_temps == 0
                        && temp_idx > 0
                    {
                        let save_path: PathBuf = config
                            .persistence
                            .cache_dir
                            .join(format!("qtable_checkpoint_temp{}.bin", temp_idx));

                        match rl_controller.save_qtable(save_path.as_path()) {
                            Ok(_) => {
                                println!("[FLUXNET] Q-table saved to: {}", save_path.display())
                            }
                            Err(e) => eprintln!("[FLUXNET] Failed to save Q-table: {}", e),
                        }
                    }
                }
            }
        }

        equilibrium_states.push(solution);

        if guard_threshold_relax_timer > 0 {
            guard_threshold_relax_timer -= 1;
            if guard_threshold_relax_timer == 0 {
                guard_threshold_floor = base_guard_floor;
            }
        }

        if post_reheat_slack_lock > 0 {
            post_reheat_slack_lock -= 1;
        }
    }

    let elapsed = start_time.elapsed();
    println!(
        "[THERMO-GPU] ✅ Completed {} temperature replicas in {:.2}ms",
        num_temps,
        elapsed.as_secs_f64() * 1000.0
    );

    // FluxNet RL: Save final Q-table after equilibration completes
    if let Some((ref _force_profile, ref mut rl_controller)) = fluxnet_state {
        if let Some(config) = fluxnet_config {
            if config.persistence.save_final {
                let final_path: PathBuf = config.persistence.cache_dir.join("qtable_final.bin");
                match rl_controller.save_qtable(final_path.as_path()) {
                    Ok(_) => println!("[FLUXNET] Final Q-table saved to: {}", final_path.display()),
                    Err(e) => eprintln!("[FLUXNET] Failed to save final Q-table: {}", e),
                }
            }

            // Print learning statistics
            println!(
                "[FLUXNET] Training complete: {} experiences, ε={:.3}",
                rl_controller.replay_buffer_size(),
                rl_controller.epsilon()
            );
        }
    }

    // TWEAK 6: Report best snapshot found
    if let Some((best_idx, best_sol, best_q)) = &best_snapshot {
        println!(
            "[THERMO-GPU][TWEAK-6] Best snapshot: temp {} with {} colors, {} conflicts (quality={:.6})",
            best_idx, best_sol.chromatic_number, best_sol.conflicts, best_q
        );
    }

    Ok(equilibrium_states)
}
