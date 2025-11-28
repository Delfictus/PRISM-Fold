//! FluxNet RL Controller - Q-Learning Agent
//!
//! Implements tabular Q-learning for adaptive force profile control during
//! Phase 2 thermodynamic equilibration.
//!
//! # Architecture
//!
//! ```text
//! Per Temperature Step:
//! ┌─────────────────┐
//! │  Telemetry      │ (conflicts, colors, compaction)
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  RLState        │ Discretize to state index
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  QTable         │ Q(s, a) lookup
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Epsilon-Greedy  │ Explore vs Exploit
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ ForceCommand    │ Selected action
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ ForceProfile    │ Apply command
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ GPU Kernel      │ Thermodynamic evolution
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Reward          │ Δconflicts, Δcolors
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Q-Update        │ Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
//! └─────────────────┘
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;

use super::approximator::QApproximator;
use super::command::ForceCommand;
use super::config::RLConfig;

/// RL state observation (discretized telemetry)
///
/// Captures key metrics from thermodynamic equilibration:
/// - conflicts: Number of constraint violations
/// - chromatic: Number of colors used
/// - compaction_ratio: Convergence health metric
/// - guard/slack context: collapse history and guard tuning knobs
/// - band variance: stability from force profile
///
/// State is discretized into bins for tabular Q-learning while keeping raw values
/// for telemetry and the hybrid approximator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RLState {
    /// Discretized conflict count (0-255)
    pub conflict_bin: u8,

    /// Discretized chromatic number (0-255)
    pub chromatic_bin: u8,

    /// Discretized compaction ratio (0-255)
    pub compaction_bin: u8,

    /// Discretized guard pressure (0-255)
    pub guard_bin: u8,

    /// Discretized slack level (0-255)
    pub slack_bin: u8,

    /// Discretized band variance (0-255)
    pub band_std_bin: u8,

    /// Discretized steps per temperature (0-255)
    pub steps_bin: u8,

    /// Discretized guard threshold (0-255)
    pub guard_threshold_bin: u8,

    /// Discretized collapse streak (0-255)
    pub collapse_bin: u8,

    /// Raw conflict count (for telemetry)
    pub conflicts: usize,

    /// Raw chromatic number (for telemetry)
    pub chromatic_number: usize,

    /// Raw compaction ratio (for telemetry)
    pub compaction_ratio: f32,

    /// Number of recent guard triggers
    pub guard_count: usize,

    /// Current slack (colors)
    pub current_slack: usize,

    /// Current steps per temperature
    pub current_steps: usize,

    /// Dynamic guard threshold used this step
    pub dynamic_guard_threshold: f32,

    /// Number of consecutive guard collapses
    pub collapse_streak: usize,

    /// Standard deviation for force bands
    pub band_std: f32,

    /// Whether system is currently phase-locked (chromatic collapsed)
    pub phase_locked: bool,
}

// Manual PartialEq and Eq implementations that only compare bins (for hashing/Q-table lookup)
impl PartialEq for RLState {
    fn eq(&self, other: &Self) -> bool {
        self.conflict_bin == other.conflict_bin
            && self.chromatic_bin == other.chromatic_bin
            && self.compaction_bin == other.compaction_bin
            && self.guard_bin == other.guard_bin
            && self.slack_bin == other.slack_bin
            && self.band_std_bin == other.band_std_bin
            && self.steps_bin == other.steps_bin
            && self.guard_threshold_bin == other.guard_threshold_bin
            && self.collapse_bin == other.collapse_bin
            && self.phase_locked == other.phase_locked
    }
}

impl Eq for RLState {}

impl std::hash::Hash for RLState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.conflict_bin.hash(state);
        self.chromatic_bin.hash(state);
        self.compaction_bin.hash(state);
        self.guard_bin.hash(state);
        self.slack_bin.hash(state);
        self.band_std_bin.hash(state);
        self.steps_bin.hash(state);
        self.guard_threshold_bin.hash(state);
        self.collapse_bin.hash(state);
        self.phase_locked.hash(state);
    }
}

impl RLState {
    pub const FEATURE_DIM: usize = 10;

    /// Create state from raw telemetry values
    ///
    /// # Arguments
    /// - `conflicts`: Raw conflict count
    /// - `chromatic`: Raw chromatic number
    /// - `compaction_ratio`: Raw compaction ratio [0.0, 1.0]
    /// - `max_conflicts`: Maximum expected conflicts (for normalization)
    /// - `max_chromatic`: Maximum expected colors (for normalization)
    #[allow(clippy::too_many_arguments)]
    pub fn from_telemetry(
        conflicts: usize,
        chromatic: usize,
        compaction_ratio: f32,
        max_conflicts: usize,
        max_chromatic: usize,
        guard_count: usize,
        current_slack: usize,
        max_slack: usize,
        band_std: f32,
        phase_locked: bool,
        dynamic_guard_threshold: f32,
        collapse_streak: usize,
        current_steps: usize,
        max_steps: usize,
    ) -> Self {
        let conflict_bin =
            ((conflicts.min(max_conflicts) as f32 / max_conflicts as f32) * 255.0).min(255.0) as u8;

        let chromatic_bin =
            ((chromatic.min(max_chromatic) as f32 / max_chromatic as f32) * 255.0).min(255.0) as u8;

        let compaction_bin = (compaction_ratio.clamp(0.0, 1.0) * 255.0).min(255.0) as u8;
        let guard_bin = ((guard_count.min(32) as f32 / 32.0) * 255.0).min(255.0) as u8;
        let slack_bin = if max_slack == 0 {
            0
        } else {
            ((current_slack.min(max_slack) as f32 / max_slack as f32) * 255.0).min(255.0) as u8
        };
        let band_std_bin = (band_std.clamp(0.0, 2.0) / 2.0 * 255.0).min(255.0) as u8;
        let steps_bin = if max_steps == 0 {
            0
        } else {
            ((current_steps.min(max_steps) as f32 / max_steps as f32) * 255.0).min(255.0) as u8
        };
        let guard_threshold_clamped = dynamic_guard_threshold.clamp(0.05, 0.35);
        let guard_threshold_bin = ((guard_threshold_clamped / 0.35) * 255.0).min(255.0) as u8;
        let collapse_clamped = collapse_streak.min(16);
        let collapse_bin = ((collapse_clamped as f32 / 16.0) * 255.0).min(255.0) as u8;

        Self {
            conflict_bin,
            chromatic_bin,
            compaction_bin,
            guard_bin,
            slack_bin,
            band_std_bin,
            steps_bin,
            guard_threshold_bin,
            collapse_bin,
            conflicts,
            chromatic_number: chromatic,
            compaction_ratio,
            guard_count,
            current_slack,
            current_steps,
            dynamic_guard_threshold: guard_threshold_clamped,
            collapse_streak,
            band_std,
            phase_locked,
        }
    }

    pub fn feature_vector(&self, max_slack: usize, max_steps: usize) -> [f32; Self::FEATURE_DIM] {
        let slack_norm = if max_slack == 0 {
            0.0
        } else {
            self.current_slack as f32 / max_slack as f32
        };
        let steps_norm = if max_steps == 0 {
            0.0
        } else {
            self.current_steps as f32 / max_steps as f32
        };
        let band_std_norm = (self.band_std / 2.0).clamp(0.0, 1.0);
        let guard_threshold_norm =
            ((self.dynamic_guard_threshold - 0.05).max(0.0) / 0.3).clamp(0.0, 1.0);
        let collapse_norm = (self.collapse_streak.min(16) as f32) / 16.0;

        [
            self.conflicts as f32 / 1000.0,
            self.chromatic_number as f32 / 1000.0,
            self.compaction_ratio,
            self.guard_count as f32 / 32.0,
            slack_norm,
            band_std_norm,
            guard_threshold_norm,
            collapse_norm,
            if self.phase_locked { 1.0 } else { 0.0 },
            steps_norm,
        ]
    }

    /// Convert state to index for Q-table lookup
    ///
    /// Uses bit-packing: [conflict_bin | chromatic_bin | compaction_bin]
    /// State space size: 256 × 256 × 256 = 16,777,216 (full)
    ///
    /// For compact mode, we reduce to top bits only
    pub fn to_index(&self, compact: bool) -> usize {
        let phase_bit = if self.phase_locked { 1 } else { 0 };

        let base = if compact {
            let c1 = (self.conflict_bin >> 4) as usize;
            let c2 = (self.chromatic_bin >> 4) as usize;
            let c3 = (self.compaction_bin >> 4) as usize;
            let g = (self.guard_bin >> 5) as usize;
            let s = (self.slack_bin >> 5) as usize;
            let b = (self.band_std_bin >> 5) as usize;
            let st = (self.steps_bin >> 5) as usize;
            let th = (self.guard_threshold_bin >> 6) as usize;
            let collapse = (self.collapse_bin >> 6) as usize;
            ((c1 << 22)
                ^ (c2 << 18)
                ^ (c3 << 14)
                ^ (g << 11)
                ^ (s << 8)
                ^ (b << 5)
                ^ (st << 3)
                ^ (th << 1)
                ^ collapse)
                << 1
                ^ phase_bit
        } else {
            let c1 = self.conflict_bin as usize;
            let c2 = self.chromatic_bin as usize;
            let c3 = self.compaction_bin as usize;
            let g = self.guard_bin as usize;
            let s = self.slack_bin as usize;
            let b = self.band_std_bin as usize;
            let st = self.steps_bin as usize;
            let th = self.guard_threshold_bin as usize;
            let collapse = self.collapse_bin as usize;
            ((c1 << 32)
                ^ (c2 << 24)
                ^ (c3 << 16)
                ^ (g << 12)
                ^ (s << 8)
                ^ (b << 5)
                ^ (st << 3)
                ^ (th << 1)
                ^ collapse)
                << 1
                ^ phase_bit
        };

        if compact {
            base % 4096
        } else {
            base % 16384
        }
    }
}

impl Default for RLState {
    fn default() -> Self {
        Self {
            conflict_bin: 128,
            chromatic_bin: 128,
            compaction_bin: 128,
            guard_bin: 0,
            slack_bin: 128,
            band_std_bin: 0,
            steps_bin: 0,
            guard_threshold_bin: 0,
            collapse_bin: 0,
            conflicts: 0,
            chromatic_number: 0,
            compaction_ratio: 0.5,
            guard_count: 0,
            current_slack: 0,
            current_steps: 0,
            dynamic_guard_threshold: 0.15,
            collapse_streak: 0,
            band_std: 0.0,
            phase_locked: false,
        }
    }
}

/// Q-Table for tabular Q-learning
///
/// Stores Q-values Q(s, a) for state-action pairs.
/// Uses simple 2D array: states × actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QTable {
    /// Q-values: [num_states × num_actions]
    q_values: Vec<Vec<f32>>,

    /// Number of states
    num_states: usize,

    /// Number of actions (always 7 for ForceCommand)
    num_actions: usize,
}

impl QTable {
    /// Create a new Q-table with zeros
    pub fn new(num_states: usize) -> Self {
        let num_actions = ForceCommand::ACTION_SPACE_SIZE;
        let q_values = vec![vec![0.0; num_actions]; num_states];

        Self {
            q_values,
            num_states,
            num_actions,
        }
    }

    /// Get Q-value for state-action pair
    pub fn get(&self, state_idx: usize, action_idx: usize) -> f32 {
        if state_idx < self.num_states && action_idx < self.num_actions {
            self.q_values[state_idx][action_idx]
        } else {
            0.0 // Out of bounds → default to 0
        }
    }

    /// Get Q-value for state-action pair (alias for get, used by telemetry)
    pub fn get_q_value(&self, state_idx: usize, action_idx: usize) -> f32 {
        self.get(state_idx, action_idx)
    }

    /// Set Q-value for state-action pair
    pub fn set(&mut self, state_idx: usize, action_idx: usize, value: f32) {
        if state_idx < self.num_states && action_idx < self.num_actions {
            self.q_values[state_idx][action_idx] = value;
        }
    }

    /// Get best action for state (argmax Q(s, a))
    pub fn best_action(&self, state_idx: usize) -> usize {
        if state_idx >= self.num_states {
            return ForceCommand::NoOp.to_action_index();
        }

        self.q_values[state_idx]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(ForceCommand::NoOp.to_action_index())
    }

    /// Get max Q-value for state (max_a Q(s, a))
    pub fn max_q_value(&self, state_idx: usize) -> f32 {
        if state_idx >= self.num_states {
            return 0.0;
        }

        self.q_values[state_idx]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Update Q-value using Q-learning rule
    ///
    /// Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    pub fn update(
        &mut self,
        state_idx: usize,
        action_idx: usize,
        reward: f32,
        next_state_idx: usize,
        alpha: f32,
        gamma: f32,
    ) {
        let current_q = self.get(state_idx, action_idx);
        let max_next_q = self.max_q_value(next_state_idx);

        let td_target = reward + gamma * max_next_q;
        let td_error = td_target - current_q;
        let new_q = current_q + alpha * td_error;

        self.set(state_idx, action_idx, new_q);
    }

    /// Save Q-table to binary file
    pub fn save(&self, path: &Path) -> Result<()> {
        let bytes = bincode::serialize(self).context("Failed to serialize Q-table")?;
        std::fs::write(path, bytes).context("Failed to write Q-table to file")?;
        Ok(())
    }

    /// Load Q-table from binary file
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).context("Failed to read Q-table file")?;
        let qtable: QTable =
            bincode::deserialize(&bytes).context("Failed to deserialize Q-table")?;
        Ok(qtable)
    }
}

/// Experience tuple for replay buffer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub state: RLState,
    pub action: ForceCommand,
    pub reward: f32,
    pub next_state: RLState,
    pub done: bool,
    pub priority: bool,
}

/// Experience replay buffer for off-policy learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    /// Create new replay buffer with capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add experience to buffer
    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    /// Sample random batch from buffer with optional priority weighting
    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let buffer_vec: Vec<_> = self.buffer.iter().cloned().collect();

        buffer_vec
            .choose_multiple(&mut rng, batch_size.min(self.buffer.len()))
            .cloned()
            .collect()
    }

    pub fn sample_prioritized(&self, batch_size: usize) -> Vec<Experience> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let (priority, normal): (Vec<_>, Vec<_>) =
            self.buffer.iter().cloned().partition(|e| e.priority);

        let mut batch = Vec::new();
        let pri_take = batch_size.min(priority.len());
        batch.extend(priority.choose_multiple(&mut rng, pri_take).cloned());

        if batch.len() < batch_size {
            let remaining = batch_size - batch.len();
            batch.extend(
                normal
                    .choose_multiple(&mut rng, remaining.min(normal.len()))
                    .cloned(),
            );
        }

        batch
    }

    /// Get buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn last(&self) -> Option<&Experience> {
        self.buffer.back()
    }
}

/// FluxNet RL Controller
///
/// Q-learning agent for adaptive force profile control.
pub struct RLController {
    /// Q-table for state-action values
    qtable: QTable,

    /// Experience replay buffer
    replay_buffer: ReplayBuffer,

    /// RL hyperparameters
    config: RLConfig,

    /// Current epsilon (decays over time)
    epsilon: f32,

    /// Compact mode (reduces state space)
    compact: bool,

    /// Maximum conflicts for normalization
    max_conflicts: usize,

    /// Maximum chromatic for normalization
    max_chromatic: usize,

    /// Maximum slack for normalization
    max_slack: usize,

    /// Hybrid approximator for generalization
    approximator: QApproximator,

    /// Maximum steps per temperature for normalization
    max_steps_per_temp: usize,
}

impl RLController {
    /// Create new RL controller
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: RLConfig,
        num_states: usize,
        replay_capacity: usize,
        max_conflicts: usize,
        max_chromatic: usize,
        compact: bool,
        max_slack: usize,
        max_steps_per_temp: usize,
    ) -> Self {
        let epsilon_start = config.epsilon_start;
        let approx_lr = config.learning_rate * 0.5;
        Self {
            qtable: QTable::new(num_states),
            replay_buffer: ReplayBuffer::new(replay_capacity),
            config,
            epsilon: epsilon_start,
            compact,
            max_conflicts,
            max_chromatic,
            max_slack,
            approximator: QApproximator::new(
                ForceCommand::ACTION_SPACE_SIZE,
                RLState::FEATURE_DIM,
                approx_lr,
            ),
            max_steps_per_temp,
        }
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&mut self, state: &RLState) -> ForceCommand {
        use rand::Rng;

        let state_idx = state.to_index(self.compact);
        let features = state.feature_vector(self.max_slack, self.max_steps_per_temp);
        let approx = self.approximator.predict(&features);

        if rand::thread_rng().gen::<f32>() < self.epsilon {
            let action_idx = rand::thread_rng().gen_range(0..ForceCommand::ACTION_SPACE_SIZE);
            ForceCommand::from_action_index(action_idx)
        } else {
            let action_idx = self.best_action_with_approx(state_idx, &approx);
            ForceCommand::from_action_index(action_idx)
        }
    }

    /// Update Q-table from experience
    pub fn update(
        &mut self,
        state: RLState,
        action: ForceCommand,
        reward: f32,
        next_state: RLState,
        done: bool,
    ) {
        let state_idx = state.to_index(self.compact);
        let action_idx = action.to_action_index();
        let next_state_idx = next_state.to_index(self.compact);

        let td_target = if done {
            reward
        } else {
            reward + self.config.discount_factor * self.qtable.max_q_value(next_state_idx)
        };

        let priority = state.phase_locked || next_state.phase_locked || reward.abs() > 1.0;
        self.replay_buffer.push(Experience {
            state,
            action,
            reward,
            next_state,
            done,
            priority,
        });

        self.qtable.update(
            state_idx,
            action_idx,
            reward,
            next_state_idx,
            self.config.learning_rate,
            self.config.discount_factor,
        );

        if let Some(exp) = self.replay_buffer.last() {
            let feats = exp
                .state
                .feature_vector(self.max_slack, self.max_steps_per_temp);
            self.approximator.update(&feats, action_idx, td_target);
        }

        for exp in self.replay_buffer.sample_prioritized(4) {
            self.train_from_experience(&exp);
        }

        self.epsilon *= self.config.epsilon_decay;
        self.epsilon = self.epsilon.max(self.config.epsilon_min);
    }

    /// Compute reward from telemetry delta
    pub fn compute_reward(
        &self,
        conflicts_before: usize,
        conflicts_after: usize,
        chromatic_before: usize,
        chromatic_after: usize,
        compaction_before: f32,
        compaction_after: f32,
    ) -> f32 {
        let delta_conflicts =
            (conflicts_before as f32 - conflicts_after as f32) / self.max_conflicts as f32;
        let delta_chromatic =
            (chromatic_before as f32 - chromatic_after as f32) / self.max_chromatic as f32;
        let delta_compaction = compaction_after - compaction_before;

        // Weighted reward
        self.config.reward_conflict_weight * delta_conflicts
            + self.config.reward_color_weight * delta_chromatic
            + self.config.reward_compaction_weight * delta_compaction
    }

    /// Get current epsilon value (for telemetry)
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Current replay buffer occupancy
    pub fn replay_buffer_size(&self) -> usize {
        self.replay_buffer.len()
    }

    /// Get Q-value for a specific state-action pair (for telemetry)
    pub fn get_q_value(&self, state: &RLState, action: &ForceCommand) -> f32 {
        let state_idx = state.to_index(self.compact);
        let action_idx = action.to_action_index();
        self.qtable.get_q_value(state_idx, action_idx)
    }

    /// Select action with telemetry metadata
    ///
    /// Returns (action, q_value, was_exploration)
    pub fn select_action_with_telemetry(&mut self, state: &RLState) -> (ForceCommand, f32, bool) {
        use rand::Rng;

        let state_idx = state.to_index(self.compact);
        let mut rng = rand::thread_rng();
        let features = state.feature_vector(self.max_slack, self.max_steps_per_temp);
        let approx = self.approximator.predict(&features);

        let was_exploration = rng.gen::<f32>() < self.epsilon;

        let (action, q_value) = if was_exploration {
            // Explore: random action
            let action_idx = rng.gen_range(0..ForceCommand::ACTION_SPACE_SIZE);
            let action = ForceCommand::from_action_index(action_idx);
            let q_val = self.qtable.get_q_value(state_idx, action_idx) + approx[action_idx];
            (action, q_val)
        } else {
            // Exploit: best action from Q-table + approximator
            let action_idx = self.best_action_with_approx(state_idx, &approx);
            let action = ForceCommand::from_action_index(action_idx);
            let q_val = self.qtable.get_q_value(state_idx, action_idx) + approx[action_idx];
            (action, q_val)
        };

        (action, q_value, was_exploration)
    }

    /// Update with telemetry capture
    ///
    /// Returns (q_old, q_new, q_delta) for telemetry reporting
    pub fn update_with_telemetry(
        &mut self,
        state: RLState,
        action: ForceCommand,
        reward: f32,
        next_state: RLState,
        done: bool,
    ) -> (f32, f32, f32) {
        let state_idx = state.to_index(self.compact);
        let action_idx = action.to_action_index();
        let next_state_idx = next_state.to_index(self.compact);

        // Get Q-value before update
        let q_old = self.qtable.get_q_value(state_idx, action_idx);

        let td_target = if done {
            reward
        } else {
            reward + self.config.discount_factor * self.qtable.max_q_value(next_state_idx)
        };

        let priority = state.phase_locked || next_state.phase_locked || reward.abs() > 1.0;
        self.replay_buffer.push(Experience {
            state,
            action,
            reward,
            next_state,
            done,
            priority,
        });

        self.qtable.update(
            state_idx,
            action_idx,
            reward,
            next_state_idx,
            self.config.learning_rate,
            self.config.discount_factor,
        );

        // Get Q-value after update
        let q_new = self.qtable.get_q_value(state_idx, action_idx);
        let q_delta = q_new - q_old;

        if let Some(exp) = self.replay_buffer.last() {
            let features = exp
                .state
                .feature_vector(self.max_slack, self.max_steps_per_temp);
            self.approximator.update(&features, action_idx, td_target);
        }

        for exp in self.replay_buffer.sample_prioritized(4) {
            self.train_from_experience(&exp);
        }

        // Decay epsilon
        self.epsilon *= self.config.epsilon_decay;
        self.epsilon = self.epsilon.max(self.config.epsilon_min);

        (q_old, q_new, q_delta)
    }

    /// Load pre-trained Q-table
    pub fn load_qtable(&mut self, path: &Path) -> Result<()> {
        self.qtable = QTable::load(path)?;
        Ok(())
    }

    /// Save Q-table
    pub fn save_qtable(&self, path: &Path) -> Result<()> {
        self.qtable.save(path)?;
        Ok(())
    }

    fn best_action_with_approx(&self, state_idx: usize, approx: &[f32]) -> usize {
        let mut best_idx = ForceCommand::NoOp.to_action_index();
        let mut best_val = f32::MIN;
        for action_idx in 0..ForceCommand::ACTION_SPACE_SIZE {
            let value = self.qtable.get(state_idx, action_idx) + approx[action_idx];
            if value > best_val {
                best_val = value;
                best_idx = action_idx;
            }
        }
        best_idx
    }

    fn train_from_experience(&mut self, exp: &Experience) {
        let next_idx = exp.next_state.to_index(self.compact);
        let target = if exp.done {
            exp.reward
        } else {
            exp.reward + self.config.discount_factor * self.qtable.max_q_value(next_idx)
        };
        let features = exp
            .state
            .feature_vector(self.max_slack, self.max_steps_per_temp);
        self.approximator
            .update(&features, exp.action.to_action_index(), target);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rl_state_discretization() {
        let state = RLState::from_telemetry(
            50, 100, 0.75, 100, 200, 2, 50, 100, 0.1, false, 0.15, 1, 20_000, 40_000,
        );
        assert_eq!(state.conflict_bin, 127); // 50/100 * 255 ≈ 127
        assert_eq!(state.chromatic_bin, 127); // 100/200 * 255 ≈ 127
        assert_eq!(state.compaction_bin, 191); // 0.75 * 255 ≈ 191
    }

    #[test]
    fn test_qtable_operations() {
        let mut qtable = QTable::new(256);

        qtable.set(0, 0, 1.5);
        assert_eq!(qtable.get(0, 0), 1.5);

        qtable.set(0, 1, 2.5);
        assert_eq!(qtable.best_action(0), 1);
        assert_eq!(qtable.max_q_value(0), 2.5);
    }

    #[test]
    fn test_q_learning_update() {
        let mut qtable = QTable::new(256);

        // Initial Q(s,a) = 0
        qtable.update(0, 0, 1.0, 1, 0.1, 0.9);

        // Q(s,a) should increase toward reward
        assert!(qtable.get(0, 0) > 0.0);
        assert!(qtable.get(0, 0) < 1.0); // Not fully converged yet
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(10);

        for i in 0..15 {
            buffer.push(Experience {
                state: RLState::default(),
                action: ForceCommand::NoOp,
                reward: i as f32,
                next_state: RLState::default(),
                done: false,
                priority: false,
            });
        }

        assert_eq!(buffer.len(), 10); // Capped at capacity
        let batch = buffer.sample(5);
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn test_epsilon_decay() {
        let config = RLConfig {
            epsilon_start: 0.3,
            epsilon_decay: 0.99,
            epsilon_min: 0.05,
            ..Default::default()
        };

        let mut controller = RLController::new(config, 256, 1024, 100, 200, true, 100, 20_000);

        assert_eq!(controller.epsilon(), 0.3);

        // Simulate 100 updates
        for _ in 0..100 {
            controller.update(
                RLState::default(),
                ForceCommand::NoOp,
                0.0,
                RLState::default(),
                false,
            );
        }

        // Epsilon should decay but not below min
        assert!(controller.epsilon() < 0.3);
        assert!(controller.epsilon() >= 0.05);
    }
}
