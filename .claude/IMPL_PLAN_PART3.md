# PRISM Ultra Implementation Plan - Part 3: MBRL, FluxNet Ultra, Async Pipeline, Multi-GPU

## Document Metadata
- **Version**: 1.0.0
- **Created**: 2025-11-29
- **Purpose**: MBRL, FluxNet, Async, Multi-GPU specifications
- **Scope**: Phase 1D, 1E, Phase 2, Phase 3

---

# SECTION F: PHASE 1D - MBRL WORLD MODEL

## F.1 MBRL World Model

**File**: `crates/prism-fluxnet/src/mbrl.rs`

**AGENT**: prism-architect

**Target**: ~600 LOC

```rust
//! MBRL: Model-Based Reinforcement Learning World Model
//!
//! GNN-based world model that predicts kernel outcomes from state+action,
//! enabling synthetic experience generation for FluxNet training.

use anyhow::Result;
use ndarray::{Array1, Array2};
use ort::{Session, SessionBuilder, GraphOptimizationLevel};
use std::path::Path;
use std::sync::Arc;

use prism_core::{RuntimeConfig, KernelTelemetry};

/// Predicted outcome from world model
#[derive(Debug, Clone)]
pub struct PredictedOutcome {
    /// Predicted conflict count after action
    pub predicted_conflicts: f32,
    /// Predicted colors used
    pub predicted_colors: f32,
    /// Predicted moves applied
    pub predicted_moves: f32,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Predicted reward
    pub predicted_reward: f32,
}

/// Kernel state representation for world model
#[derive(Debug, Clone)]
pub struct KernelState {
    /// Current conflict count
    pub conflicts: i32,
    /// Current colors used
    pub colors_used: i32,
    /// Current iteration
    pub iteration: i32,
    /// Graph density
    pub density: f32,
    /// Betti numbers
    pub betti: [f32; 3],
    /// Reservoir activity
    pub reservoir_activity: f32,
    /// Recent conflict trajectory (last 10 iterations)
    pub conflict_history: [f32; 10],
    /// Current temperature
    pub temperature: f32,
    /// Phase ID
    pub phase_id: i32,
}

impl KernelState {
    /// Convert to feature vector for ONNX model
    pub fn to_features(&self) -> Array1<f32> {
        let mut features = Vec::with_capacity(32);

        // Basic state
        features.push(self.conflicts as f32);
        features.push(self.colors_used as f32);
        features.push(self.iteration as f32);
        features.push(self.density);

        // Topology
        features.extend_from_slice(&self.betti);

        // Reservoir
        features.push(self.reservoir_activity);

        // History
        features.extend_from_slice(&self.conflict_history);

        // Control
        features.push(self.temperature);
        features.push(self.phase_id as f32);

        Array1::from_vec(features)
    }
}

/// Delta to RuntimeConfig (action representation)
#[derive(Debug, Clone, Default)]
pub struct RuntimeConfigDelta {
    pub d_chemical_potential: f32,
    pub d_tunneling_prob: f32,
    pub d_temperature: f32,
    pub d_belief_weight: f32,
    pub d_reservoir_leak: f32,
    // ... other deltas
}

impl RuntimeConfigDelta {
    /// Convert to feature vector
    pub fn to_features(&self) -> Array1<f32> {
        Array1::from_vec(vec![
            self.d_chemical_potential,
            self.d_tunneling_prob,
            self.d_temperature,
            self.d_belief_weight,
            self.d_reservoir_leak,
        ])
    }

    /// Apply delta to config
    pub fn apply(&self, config: &mut RuntimeConfig) {
        config.chemical_potential += self.d_chemical_potential;
        config.tunneling_prob_base += self.d_tunneling_prob;
        config.global_temperature += self.d_temperature;
        config.belief_weight += self.d_belief_weight;
        config.reservoir_leak_rate += self.d_reservoir_leak;

        // Clamp values to valid ranges
        config.chemical_potential = config.chemical_potential.clamp(0.0, 10.0);
        config.tunneling_prob_base = config.tunneling_prob_base.clamp(0.0, 1.0);
        config.global_temperature = config.global_temperature.clamp(0.01, 100.0);
        config.belief_weight = config.belief_weight.clamp(0.0, 1.0);
        config.reservoir_leak_rate = config.reservoir_leak_rate.clamp(0.01, 0.99);
    }
}

/// MBRL World Model using GNN
pub struct MBRLWorldModel {
    /// ONNX session for GNN inference
    world_model: Session,

    /// Number of rollouts for MCTS
    num_rollouts: usize,

    /// Rollout horizon (steps to simulate)
    horizon: usize,

    /// Discount factor
    gamma: f64,

    /// Experience buffer for training
    experience_buffer: Vec<Experience>,

    /// Buffer capacity
    buffer_capacity: usize,
}

/// Single experience tuple
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: KernelState,
    pub action: RuntimeConfigDelta,
    pub next_state: KernelState,
    pub reward: f32,
    pub done: bool,
}

impl MBRLWorldModel {
    /// Load world model from ONNX file
    pub fn new(model_path: &Path) -> Result<Self> {
        let world_model = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self {
            world_model,
            num_rollouts: 100,
            horizon: 10,
            gamma: 0.99,
            experience_buffer: Vec::with_capacity(10000),
            buffer_capacity: 10000,
        })
    }

    /// Predict outcome of action from state
    pub fn predict_outcome(
        &self,
        state: &KernelState,
        action: &RuntimeConfigDelta,
    ) -> Result<PredictedOutcome> {
        // Concatenate state and action features
        let state_features = state.to_features();
        let action_features = action.to_features();

        let input_dim = state_features.len() + action_features.len();
        let mut input = Array2::zeros((1, input_dim));

        for (i, &v) in state_features.iter().enumerate() {
            input[[0, i]] = v;
        }
        for (i, &v) in action_features.iter().enumerate() {
            input[[0, state_features.len() + i]] = v;
        }

        // Run inference
        let outputs = self.world_model.run(ort::inputs![input]?)?;

        // Parse output
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let output_view = output_tensor.view();

        Ok(PredictedOutcome {
            predicted_conflicts: output_view[[0, 0]],
            predicted_colors: output_view[[0, 1]],
            predicted_moves: output_view[[0, 2]],
            confidence: output_view[[0, 3]].clamp(0.0, 1.0),
            predicted_reward: output_view[[0, 4]],
        })
    }

    /// Generate synthetic experience via rollouts
    pub fn generate_synthetic_experience(
        &self,
        initial_state: &KernelState,
        num_experiences: usize,
    ) -> Result<Vec<Experience>> {
        let mut experiences = Vec::with_capacity(num_experiences);

        for _ in 0..num_experiences {
            let mut state = initial_state.clone();

            for _ in 0..self.horizon {
                // Generate random action
                let action = self.sample_action();

                // Predict next state
                let outcome = self.predict_outcome(&state, &action)?;

                // Compute reward
                let reward = self.compute_reward(&state, &outcome);

                // Create next state from prediction
                let next_state = KernelState {
                    conflicts: outcome.predicted_conflicts as i32,
                    colors_used: outcome.predicted_colors as i32,
                    iteration: state.iteration + 1,
                    density: state.density,
                    betti: state.betti, // Would update from TPTP prediction
                    reservoir_activity: state.reservoir_activity * 0.9,
                    conflict_history: {
                        let mut h = state.conflict_history;
                        h.rotate_left(1);
                        h[9] = outcome.predicted_conflicts;
                        h
                    },
                    temperature: (state.temperature * 0.99).max(0.01),
                    phase_id: state.phase_id,
                };

                let done = outcome.predicted_conflicts == 0;

                experiences.push(Experience {
                    state: state.clone(),
                    action,
                    next_state: next_state.clone(),
                    reward,
                    done,
                });

                if done {
                    break;
                }

                state = next_state;
            }
        }

        Ok(experiences)
    }

    /// MCTS-style action selection
    pub fn mcts_action_selection(
        &self,
        state: &KernelState,
        num_candidates: usize,
    ) -> Result<Vec<(RuntimeConfigDelta, f32)>> {
        let mut candidates = Vec::with_capacity(num_candidates);

        for _ in 0..num_candidates {
            let action = self.sample_action();
            let value = self.rollout_value(state, &action)?;
            candidates.push((action, value));
        }

        // Sort by value (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(candidates)
    }

    /// Compute rollout value for action
    fn rollout_value(&self, state: &KernelState, action: &RuntimeConfigDelta) -> Result<f32> {
        let mut total_value = 0.0;
        let mut current_state = state.clone();
        let mut current_action = action.clone();
        let mut discount = 1.0;

        for _ in 0..self.horizon {
            let outcome = self.predict_outcome(&current_state, &current_action)?;
            let reward = self.compute_reward(&current_state, &outcome);

            total_value += discount * reward;
            discount *= self.gamma as f32;

            if outcome.predicted_conflicts == 0.0 {
                break;
            }

            // Update state
            current_state.conflicts = outcome.predicted_conflicts as i32;
            current_state.colors_used = outcome.predicted_colors as i32;
            current_state.iteration += 1;

            // Sample new action for next step
            current_action = self.sample_action();
        }

        Ok(total_value)
    }

    /// Sample random action delta
    fn sample_action(&self) -> RuntimeConfigDelta {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        RuntimeConfigDelta {
            d_chemical_potential: rng.gen_range(-0.5..0.5),
            d_tunneling_prob: rng.gen_range(-0.1..0.1),
            d_temperature: rng.gen_range(-0.2..0.2),
            d_belief_weight: rng.gen_range(-0.1..0.1),
            d_reservoir_leak: rng.gen_range(-0.05..0.05),
        }
    }

    /// Compute reward from state and outcome
    fn compute_reward(&self, state: &KernelState, outcome: &PredictedOutcome) -> f32 {
        let conflict_reduction = state.conflicts as f32 - outcome.predicted_conflicts;
        let color_reduction = state.colors_used as f32 - outcome.predicted_colors;

        // Reward = conflict reduction + small bonus for color reduction
        let reward = conflict_reduction + 0.1 * color_reduction;

        // Scale by confidence
        reward * outcome.confidence
    }

    /// Add experience to buffer
    pub fn add_experience(&mut self, exp: Experience) {
        if self.experience_buffer.len() >= self.buffer_capacity {
            self.experience_buffer.remove(0);
        }
        self.experience_buffer.push(exp);
    }

    /// Sample batch from experience buffer
    pub fn sample_batch(&self, batch_size: usize) -> Vec<&Experience> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        self.experience_buffer
            .choose_multiple(&mut rng, batch_size.min(self.experience_buffer.len()))
            .collect()
    }
}

/// Dyna-style integration of MBRL with FluxNet
pub struct DynaFluxNet {
    /// Real Q-table/network
    fluxnet: Arc<crate::FluxNetController>,

    /// World model for synthetic experience
    world_model: MBRLWorldModel,

    /// Ratio of synthetic to real experience
    synthetic_ratio: usize,
}

impl DynaFluxNet {
    pub fn new(
        fluxnet: Arc<crate::FluxNetController>,
        model_path: &Path,
    ) -> Result<Self> {
        Ok(Self {
            fluxnet,
            world_model: MBRLWorldModel::new(model_path)?,
            synthetic_ratio: 5, // 5 synthetic updates per real update
        })
    }

    /// Update with real experience + synthetic rollouts
    pub fn update(
        &mut self,
        state: &KernelState,
        action: &RuntimeConfigDelta,
        next_state: &KernelState,
        reward: f32,
    ) -> Result<()> {
        // 1. Update with real experience
        self.world_model.add_experience(Experience {
            state: state.clone(),
            action: action.clone(),
            next_state: next_state.clone(),
            reward,
            done: next_state.conflicts == 0,
        });

        // 2. Generate and update with synthetic experience
        let synthetic = self.world_model.generate_synthetic_experience(
            state,
            self.synthetic_ratio,
        )?;

        for exp in synthetic {
            // Update FluxNet Q-values with synthetic experience
            // self.fluxnet.update_from_experience(&exp)?;
        }

        Ok(())
    }

    /// Select best action using MCTS
    pub fn select_action(&self, state: &KernelState) -> Result<RuntimeConfigDelta> {
        let candidates = self.world_model.mcts_action_selection(state, 20)?;

        // Return best action
        Ok(candidates.into_iter().next().map(|(a, _)| a).unwrap_or_default())
    }
}
```

---

# SECTION G: PHASE 1E - ULTRA FLUXNET CONTROLLER

## G.1 Ultra FluxNet Controller

**File**: `crates/prism-fluxnet/src/ultra_controller.rs`

**AGENT**: prism-architect

**Target**: ~500 LOC

```rust
//! Ultra FluxNet Controller
//!
//! Unified RL controller that manages all PRISM phases using the RuntimeConfig
//! struct for action space and KernelTelemetry for state observation.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

use prism_core::{RuntimeConfig, KernelTelemetry};
use crate::mbrl::{MBRLWorldModel, KernelState, RuntimeConfigDelta, DynaFluxNet};

/// State discretization for Q-table
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct DiscreteState {
    /// Conflict bucket (0-10 = low, 11-50 = medium, 51+ = high)
    pub conflict_bucket: u8,
    /// Color bucket
    pub color_bucket: u8,
    /// Temperature bucket
    pub temp_bucket: u8,
    /// Phase transition flag
    pub transition_active: bool,
    /// Stagnation counter bucket
    pub stagnation_bucket: u8,
}

impl DiscreteState {
    pub fn from_telemetry(telemetry: &KernelTelemetry, config: &RuntimeConfig) -> Self {
        Self {
            conflict_bucket: match telemetry.conflicts {
                0..=10 => 0,
                11..=50 => 1,
                51..=200 => 2,
                _ => 3,
            },
            color_bucket: match telemetry.colors_used {
                0..=20 => 0,
                21..=40 => 1,
                41..=60 => 2,
                _ => 3,
            },
            temp_bucket: if config.global_temperature < 0.1 { 0 }
                        else if config.global_temperature < 1.0 { 1 }
                        else if config.global_temperature < 10.0 { 2 }
                        else { 3 },
            transition_active: telemetry.phase_transitions > 0,
            stagnation_bucket: 0, // Updated from history
        }
    }
}

/// Discrete action for Q-table
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum DiscreteAction {
    /// Increase chemical potential
    IncreaseChemicalPotential,
    /// Decrease chemical potential
    DecreaseChemicalPotential,
    /// Increase tunneling probability
    IncreaseTunneling,
    /// Decrease tunneling probability
    DecreaseTunneling,
    /// Increase temperature
    IncreaseTemperature,
    /// Decrease temperature
    DecreaseTemperature,
    /// Boost reservoir influence
    BoostReservoir,
    /// Reduce reservoir influence
    ReduceReservoir,
    /// Enable phase transition response
    EnableTransitionResponse,
    /// Disable phase transition response
    DisableTransitionResponse,
    /// No change (exploit current)
    NoOp,
}

impl DiscreteAction {
    pub const ALL: [DiscreteAction; 11] = [
        DiscreteAction::IncreaseChemicalPotential,
        DiscreteAction::DecreaseChemicalPotential,
        DiscreteAction::IncreaseTunneling,
        DiscreteAction::DecreaseTunneling,
        DiscreteAction::IncreaseTemperature,
        DiscreteAction::DecreaseTemperature,
        DiscreteAction::BoostReservoir,
        DiscreteAction::ReduceReservoir,
        DiscreteAction::EnableTransitionResponse,
        DiscreteAction::DisableTransitionResponse,
        DiscreteAction::NoOp,
    ];

    /// Apply action to config
    pub fn apply(&self, config: &mut RuntimeConfig) {
        match self {
            DiscreteAction::IncreaseChemicalPotential => {
                config.chemical_potential = (config.chemical_potential * 1.2).min(10.0);
            }
            DiscreteAction::DecreaseChemicalPotential => {
                config.chemical_potential = (config.chemical_potential * 0.8).max(0.1);
            }
            DiscreteAction::IncreaseTunneling => {
                config.tunneling_prob_base = (config.tunneling_prob_base * 1.5).min(0.5);
            }
            DiscreteAction::DecreaseTunneling => {
                config.tunneling_prob_base = (config.tunneling_prob_base * 0.7).max(0.01);
            }
            DiscreteAction::IncreaseTemperature => {
                config.global_temperature *= 1.5;
            }
            DiscreteAction::DecreaseTemperature => {
                config.global_temperature = (config.global_temperature * 0.8).max(0.01);
            }
            DiscreteAction::BoostReservoir => {
                config.reservoir_leak_rate = (config.reservoir_leak_rate * 1.2).min(0.9);
            }
            DiscreteAction::ReduceReservoir => {
                config.reservoir_leak_rate = (config.reservoir_leak_rate * 0.8).max(0.1);
            }
            DiscreteAction::EnableTransitionResponse => {
                config.tunneling_prob_boost = 3.0;
            }
            DiscreteAction::DisableTransitionResponse => {
                config.tunneling_prob_boost = 1.0;
            }
            DiscreteAction::NoOp => {}
        }
    }
}

/// Ultra FluxNet Controller
pub struct UltraFluxNetController {
    /// Q-table: state → action → value
    q_table: HashMap<(DiscreteState, DiscreteAction), f64>,

    /// Learning rate
    alpha: f64,

    /// Discount factor
    gamma: f64,

    /// Exploration rate
    epsilon: f64,

    /// Epsilon decay
    epsilon_decay: f64,

    /// Minimum epsilon
    epsilon_min: f64,

    /// Previous state for TD update
    prev_state: Option<DiscreteState>,

    /// Previous action
    prev_action: Option<DiscreteAction>,

    /// Best configuration seen
    best_config: Option<RuntimeConfig>,

    /// Best conflicts achieved
    best_conflicts: i32,

    /// Stagnation counter
    stagnation_counter: usize,

    /// MBRL world model (optional)
    world_model: Option<DynaFluxNet>,
}

impl UltraFluxNetController {
    pub fn new() -> Self {
        Self {
            q_table: HashMap::new(),
            alpha: 0.1,
            gamma: 0.95,
            epsilon: 0.3,
            epsilon_decay: 0.999,
            epsilon_min: 0.05,
            prev_state: None,
            prev_action: None,
            best_config: None,
            best_conflicts: i32::MAX,
            stagnation_counter: 0,
            world_model: None,
        }
    }

    /// Attach MBRL world model for Dyna-style learning
    pub fn with_world_model(mut self, model: DynaFluxNet) -> Self {
        self.world_model = Some(model);
        self
    }

    /// Select action given current telemetry and config
    pub fn select_action(
        &mut self,
        telemetry: &KernelTelemetry,
        config: &RuntimeConfig,
    ) -> DiscreteAction {
        let state = DiscreteState::from_telemetry(telemetry, config);

        // Epsilon-greedy action selection
        let action = if rand::random::<f64>() < self.epsilon {
            // Explore: random action
            DiscreteAction::ALL[rand::random::<usize>() % DiscreteAction::ALL.len()]
        } else {
            // Exploit: best Q-value action
            self.best_action(&state)
        };

        // Store for next update
        self.prev_state = Some(state);
        self.prev_action = Some(action);

        action
    }

    /// Update Q-values after observing reward
    pub fn update(
        &mut self,
        telemetry: &KernelTelemetry,
        config: &RuntimeConfig,
    ) {
        // Compute reward
        let reward = self.compute_reward(telemetry);

        // Track best
        if telemetry.conflicts < self.best_conflicts {
            self.best_conflicts = telemetry.conflicts;
            self.best_config = Some(config.clone());
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }

        // TD update
        if let (Some(prev_state), Some(prev_action)) = (&self.prev_state, &self.prev_action) {
            let next_state = DiscreteState::from_telemetry(telemetry, config);
            let next_best_q = self.best_q_value(&next_state);

            let key = (prev_state.clone(), *prev_action);
            let current_q = *self.q_table.get(&key).unwrap_or(&0.0);

            // Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
            let new_q = current_q + self.alpha * (reward + self.gamma * next_best_q - current_q);
            self.q_table.insert(key, new_q);
        }

        // Decay epsilon
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);

        // MBRL: Generate synthetic experience
        if let Some(ref mut world_model) = self.world_model {
            // Would generate and learn from synthetic experience
        }
    }

    /// Compute reward from telemetry
    fn compute_reward(&self, telemetry: &KernelTelemetry) -> f64 {
        let mut reward = 0.0;

        // Primary: conflict reduction
        if telemetry.conflicts == 0 {
            reward += 100.0; // Big bonus for zero conflicts
        } else {
            reward -= telemetry.conflicts as f64 * 0.1;
        }

        // Secondary: color efficiency
        reward -= telemetry.colors_used as f64 * 0.01;

        // Bonus for moves applied (active optimization)
        reward += telemetry.moves_applied as f64 * 0.001;

        // Penalty for stagnation
        if self.stagnation_counter > 100 {
            reward -= 10.0;
        }

        reward
    }

    /// Get best action for state
    fn best_action(&self, state: &DiscreteState) -> DiscreteAction {
        let mut best_action = DiscreteAction::NoOp;
        let mut best_value = f64::NEG_INFINITY;

        for action in DiscreteAction::ALL.iter() {
            let key = (state.clone(), *action);
            let value = *self.q_table.get(&key).unwrap_or(&0.0);
            if value > best_value {
                best_value = value;
                best_action = *action;
            }
        }

        best_action
    }

    /// Get best Q-value for state
    fn best_q_value(&self, state: &DiscreteState) -> f64 {
        DiscreteAction::ALL.iter()
            .map(|a| *self.q_table.get(&(state.clone(), *a)).unwrap_or(&0.0))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get current best configuration
    pub fn best_config(&self) -> Option<&RuntimeConfig> {
        self.best_config.as_ref()
    }

    /// Reset for new graph
    pub fn reset_episode(&mut self) {
        self.prev_state = None;
        self.prev_action = None;
        self.best_conflicts = i32::MAX;
        self.stagnation_counter = 0;
        // Keep Q-table for transfer learning
    }

    /// Save Q-table to file
    pub fn save(&self, path: &str) -> Result<()> {
        let data = bincode::serialize(&self.q_table)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load Q-table from file
    pub fn load(&mut self, path: &str) -> Result<()> {
        let data = std::fs::read(path)?;
        self.q_table = bincode::deserialize(&data)?;
        Ok(())
    }
}

impl Default for UltraFluxNetController {
    fn default() -> Self {
        Self::new()
    }
}
```

---

# SECTION H: PHASE 2 - ASYNC PIPELINE

## H.1 Stream Manager

**File**: `crates/prism-gpu/src/stream_manager.rs`

**AGENT**: prism-gpu-specialist

**Target**: ~400 LOC

```rust
//! Stream Manager for Async GPU Operations
//!
//! Centralized stream management with triple-buffering for maximum overlap.

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaStream};
use std::sync::Arc;

/// Stream purposes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPurpose {
    /// Config upload stream
    ConfigUpload,
    /// Main kernel execution stream
    KernelExecution,
    /// Telemetry download stream
    TelemetryDownload,
    /// P2P transfer stream (multi-GPU)
    P2PTransfer,
    /// Auxiliary compute stream
    AuxCompute,
}

/// Managed CUDA stream with purpose tagging
pub struct ManagedStream {
    stream: CudaStream,
    purpose: StreamPurpose,
    in_use: bool,
}

/// Stream pool for a single GPU
pub struct StreamPool {
    ctx: Arc<CudaContext>,
    streams: Vec<ManagedStream>,
}

impl StreamPool {
    /// Create pool with default streams
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self> {
        let mut streams = Vec::new();

        // Create one stream for each purpose
        for purpose in [
            StreamPurpose::ConfigUpload,
            StreamPurpose::KernelExecution,
            StreamPurpose::TelemetryDownload,
            StreamPurpose::P2PTransfer,
            StreamPurpose::AuxCompute,
        ] {
            streams.push(ManagedStream {
                stream: ctx.new_stream()?,
                purpose,
                in_use: false,
            });
        }

        Ok(Self { ctx, streams })
    }

    /// Get stream for specific purpose
    pub fn get_stream(&mut self, purpose: StreamPurpose) -> &CudaStream {
        let stream = self.streams.iter_mut()
            .find(|s| s.purpose == purpose)
            .expect("Stream purpose not found");
        stream.in_use = true;
        &stream.stream
    }

    /// Get config upload stream
    pub fn config_stream(&mut self) -> &CudaStream {
        self.get_stream(StreamPurpose::ConfigUpload)
    }

    /// Get kernel execution stream
    pub fn kernel_stream(&mut self) -> &CudaStream {
        self.get_stream(StreamPurpose::KernelExecution)
    }

    /// Get telemetry download stream
    pub fn telemetry_stream(&mut self) -> &CudaStream {
        self.get_stream(StreamPurpose::TelemetryDownload)
    }

    /// Synchronize all streams
    pub fn synchronize_all(&self) -> Result<()> {
        for stream in &self.streams {
            stream.stream.synchronize()?;
        }
        Ok(())
    }

    /// Synchronize specific purpose
    pub fn synchronize(&self, purpose: StreamPurpose) -> Result<()> {
        if let Some(stream) = self.streams.iter().find(|s| s.purpose == purpose) {
            stream.stream.synchronize()?;
        }
        Ok(())
    }
}

/// Triple-buffer state for async pipeline
pub struct TripleBuffer<T> {
    /// Buffer being written by producer
    write_buffer: usize,
    /// Buffer being read by consumer
    read_buffer: usize,
    /// Buffer ready for swap
    ready_buffer: usize,
    /// The actual buffers
    buffers: [T; 3],
}

impl<T: Default + Clone> TripleBuffer<T> {
    pub fn new() -> Self {
        Self {
            write_buffer: 0,
            read_buffer: 1,
            ready_buffer: 2,
            buffers: [T::default(), T::default(), T::default()],
        }
    }

    /// Get mutable reference to write buffer
    pub fn write_buf(&mut self) -> &mut T {
        &mut self.buffers[self.write_buffer]
    }

    /// Get reference to read buffer
    pub fn read_buf(&self) -> &T {
        &self.buffers[self.read_buffer]
    }

    /// Publish write buffer (make it ready)
    pub fn publish(&mut self) {
        std::mem::swap(&mut self.write_buffer, &mut self.ready_buffer);
    }

    /// Consume ready buffer (make it the read buffer)
    pub fn consume(&mut self) {
        std::mem::swap(&mut self.read_buffer, &mut self.ready_buffer);
    }
}

/// Async pipeline coordinator
pub struct AsyncPipelineCoordinator {
    /// Stream pool
    streams: StreamPool,

    /// Config triple buffer
    config_buffer: TripleBuffer<prism_core::RuntimeConfig>,

    /// Telemetry triple buffer
    telemetry_buffer: TripleBuffer<prism_core::KernelTelemetry>,

    /// Pipeline stage
    stage: PipelineStage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    Idle,
    ConfigUploading,
    KernelRunning,
    TelemetryDownloading,
}

impl AsyncPipelineCoordinator {
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            streams: StreamPool::new(ctx)?,
            config_buffer: TripleBuffer::new(),
            telemetry_buffer: TripleBuffer::new(),
            stage: PipelineStage::Idle,
        })
    }

    /// Start async config upload
    pub fn begin_config_upload(&mut self, config: prism_core::RuntimeConfig) -> Result<()> {
        *self.config_buffer.write_buf() = config;
        // Async copy would happen here
        self.stage = PipelineStage::ConfigUploading;
        Ok(())
    }

    /// Start async kernel execution (after config ready)
    pub fn begin_kernel_execution(&mut self) -> Result<()> {
        self.config_buffer.publish();
        // Kernel launch would happen here
        self.stage = PipelineStage::KernelRunning;
        Ok(())
    }

    /// Start async telemetry download (after kernel done)
    pub fn begin_telemetry_download(&mut self) -> Result<()> {
        // Async copy would happen here
        self.stage = PipelineStage::TelemetryDownloading;
        Ok(())
    }

    /// Complete pipeline iteration, get telemetry
    pub fn complete_iteration(&mut self) -> Result<prism_core::KernelTelemetry> {
        self.telemetry_buffer.consume();
        self.stage = PipelineStage::Idle;
        Ok(self.telemetry_buffer.read_buf().clone())
    }

    /// Check if config upload is complete
    pub fn is_config_ready(&self) -> bool {
        // Would query stream
        true
    }

    /// Check if kernel is complete
    pub fn is_kernel_done(&self) -> bool {
        // Would query stream
        true
    }

    /// Check if telemetry is ready
    pub fn is_telemetry_ready(&self) -> bool {
        // Would query stream
        true
    }
}
```

---

# SECTION I: PHASE 3 - MULTI-GPU

## I.1 Multi-Device Pool

**File**: `crates/prism-gpu/src/multi_device_pool.rs`

**AGENT**: prism-gpu-specialist

**Target**: ~600 LOC (modified)

```rust
//! Multi-GPU Device Pool with P2P Support
//!
//! Manages multiple CUDA devices with peer-to-peer memory access for
//! efficient cross-GPU communication.

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::stream_manager::StreamPool;

/// P2P capability between two devices
#[derive(Debug, Clone, Copy)]
pub struct P2PCapability {
    pub can_access: bool,
    pub atomic_supported: bool,
    pub bandwidth_gbps: f32,
}

/// Multi-GPU device pool
pub struct MultiGpuDevicePool {
    /// Contexts for each GPU
    contexts: Vec<Arc<CudaContext>>,

    /// Stream pools for each GPU
    stream_pools: Vec<StreamPool>,

    /// P2P capability matrix [src][dst]
    p2p_matrix: Vec<Vec<P2PCapability>>,

    /// Primary device index
    primary_device: usize,
}

impl MultiGpuDevicePool {
    /// Create pool from device IDs
    pub fn new(device_ids: &[usize]) -> Result<Self> {
        let mut contexts = Vec::with_capacity(device_ids.len());
        let mut stream_pools = Vec::with_capacity(device_ids.len());

        // Create contexts
        for &device_id in device_ids {
            let ctx = Arc::new(CudaContext::new(device_id)?);
            stream_pools.push(StreamPool::new(ctx.clone())?);
            contexts.push(ctx);
        }

        // Initialize P2P matrix
        let num_devices = device_ids.len();
        let mut p2p_matrix = vec![vec![P2PCapability {
            can_access: false,
            atomic_supported: false,
            bandwidth_gbps: 0.0,
        }; num_devices]; num_devices];

        // Enable P2P access where possible
        for i in 0..num_devices {
            for j in 0..num_devices {
                if i == j {
                    p2p_matrix[i][j] = P2PCapability {
                        can_access: true,
                        atomic_supported: true,
                        bandwidth_gbps: f32::MAX, // Same device
                    };
                    continue;
                }

                // Check and enable P2P
                // In cudarc 0.18+, would use:
                // contexts[i].enable_peer_access(&contexts[j])?;

                p2p_matrix[i][j] = P2PCapability {
                    can_access: true, // Assume success
                    atomic_supported: false, // Conservative
                    bandwidth_gbps: 25.0, // Assume PCIe 4.0 x16
                };
            }
        }

        Ok(Self {
            contexts,
            stream_pools,
            p2p_matrix,
            primary_device: 0,
        })
    }

    /// Get number of devices
    pub fn num_devices(&self) -> usize {
        self.contexts.len()
    }

    /// Get context for device
    pub fn context(&self, device_idx: usize) -> &Arc<CudaContext> {
        &self.contexts[device_idx]
    }

    /// Get stream pool for device
    pub fn stream_pool(&mut self, device_idx: usize) -> &mut StreamPool {
        &mut self.stream_pools[device_idx]
    }

    /// Check P2P capability
    pub fn can_p2p(&self, src: usize, dst: usize) -> bool {
        self.p2p_matrix[src][dst].can_access
    }

    /// Get estimated P2P bandwidth
    pub fn p2p_bandwidth(&self, src: usize, dst: usize) -> f32 {
        self.p2p_matrix[src][dst].bandwidth_gbps
    }

    /// Set primary device
    pub fn set_primary(&mut self, device_idx: usize) {
        self.primary_device = device_idx;
    }

    /// Get primary context
    pub fn primary_context(&self) -> &Arc<CudaContext> {
        &self.contexts[self.primary_device]
    }

    /// Synchronize all devices
    pub fn synchronize_all(&self) -> Result<()> {
        for pool in &self.stream_pools {
            pool.synchronize_all()?;
        }
        Ok(())
    }

    /// Distribute work across GPUs (round-robin)
    pub fn distribute_work<T>(&self, items: &[T]) -> Vec<(usize, Vec<T>)>
    where
        T: Clone,
    {
        let num_devices = self.num_devices();
        let mut distributions: Vec<Vec<T>> = vec![Vec::new(); num_devices];

        for (i, item) in items.iter().enumerate() {
            let device = i % num_devices;
            distributions[device].push(item.clone());
        }

        distributions.into_iter().enumerate().collect()
    }
}

/// Cross-GPU replica exchange coordinator
pub struct ReplicaExchangeCoordinator {
    /// Device pool
    pool: MultiGpuDevicePool,

    /// Replica to device mapping
    replica_to_device: Vec<usize>,

    /// Device to replicas mapping
    device_to_replicas: Vec<Vec<usize>>,
}

impl ReplicaExchangeCoordinator {
    pub fn new(pool: MultiGpuDevicePool, num_replicas: usize) -> Self {
        let num_devices = pool.num_devices();

        // Distribute replicas across devices
        let mut replica_to_device = vec![0; num_replicas];
        let mut device_to_replicas: Vec<Vec<usize>> = vec![Vec::new(); num_devices];

        for replica in 0..num_replicas {
            let device = replica % num_devices;
            replica_to_device[replica] = device;
            device_to_replicas[device].push(replica);
        }

        Self {
            pool,
            replica_to_device,
            device_to_replicas,
        }
    }

    /// Execute parallel tempering across GPUs
    pub fn parallel_tempering_step(&mut self) -> Result<()> {
        // 1. Execute local tempering on each GPU in parallel
        // (Each GPU runs its replicas independently)

        // 2. Synchronize all GPUs
        self.pool.synchronize_all()?;

        // 3. Exchange replicas between GPUs (P2P)
        self.cross_gpu_exchange()?;

        Ok(())
    }

    /// Cross-GPU replica exchange
    fn cross_gpu_exchange(&mut self) -> Result<()> {
        let num_replicas = self.replica_to_device.len();

        // Exchange adjacent replicas (even pairs, then odd pairs)
        for phase in 0..2 {
            for i in (phase..num_replicas - 1).step_by(2) {
                let r1 = i;
                let r2 = i + 1;

                let d1 = self.replica_to_device[r1];
                let d2 = self.replica_to_device[r2];

                if d1 != d2 {
                    // Cross-GPU exchange required
                    self.exchange_replicas_p2p(r1, d1, r2, d2)?;
                }
                // Same-GPU exchanges handled in local step
            }
        }

        Ok(())
    }

    /// P2P replica exchange between two GPUs
    fn exchange_replicas_p2p(
        &mut self,
        r1: usize,
        d1: usize,
        r2: usize,
        d2: usize,
    ) -> Result<()> {
        // Check P2P capability
        if !self.pool.can_p2p(d1, d2) {
            // Fall back to CPU staging
            return self.exchange_replicas_staged(r1, d1, r2, d2);
        }

        // P2P copy:
        // 1. Copy r1's coloring from d1 to d2
        // 2. Copy r2's coloring from d2 to d1
        // Using P2P streams

        // Would use:
        // stream_d1.memcpy_peer_async(dst_d2, src_d1, size)?;
        // stream_d2.memcpy_peer_async(dst_d1, src_d2, size)?;

        Ok(())
    }

    /// CPU-staged replica exchange (fallback)
    fn exchange_replicas_staged(
        &mut self,
        r1: usize,
        d1: usize,
        r2: usize,
        d2: usize,
    ) -> Result<()> {
        // Copy to CPU, then to other GPU
        // (Much slower, avoid if possible)
        Ok(())
    }

    /// Get device for replica
    pub fn replica_device(&self, replica: usize) -> usize {
        self.replica_to_device[replica]
    }

    /// Get replicas for device
    pub fn device_replicas(&self, device: usize) -> &[usize] {
        &self.device_to_replicas[device]
    }
}
```

---

# END OF PART 3

**Next**: Part 4 covers Agent-Specific Instructions, Verification, and Benchmarks
