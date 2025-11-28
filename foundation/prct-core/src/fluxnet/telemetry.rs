//! FluxNet RL Telemetry Extension
//!
//! Provides telemetry data structures for tracking FluxNet RL decisions,
//! force band statistics, and Q-learning updates during Phase 2 execution.

use crate::fluxnet::{ForceCommand, RLState};
use serde::{Deserialize, Serialize};

/// FluxNet-specific telemetry data
///
/// This structure is serialized to JSON and embedded in the `parameters` field
/// of `RunMetric` during Phase 2 (Thermodynamic) execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxNetTelemetry {
    /// Force band statistics (from ForceProfile)
    pub force_bands: ForceBandTelemetry,

    /// RL action and decision info
    pub rl_decision: RLDecisionTelemetry,

    /// Q-learning update details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_update: Option<QUpdateTelemetry>,

    /// FluxNet configuration snapshot
    pub config: FluxNetConfigSnapshot,
}

impl FluxNetTelemetry {
    /// Create new FluxNet telemetry snapshot
    pub fn new(
        force_bands: ForceBandTelemetry,
        rl_decision: RLDecisionTelemetry,
        q_update: Option<QUpdateTelemetry>,
        config: FluxNetConfigSnapshot,
    ) -> Self {
        Self {
            force_bands,
            rl_decision,
            q_update,
            config,
        }
    }
}

/// Force band statistics from ForceProfile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceBandTelemetry {
    /// Fraction of vertices in Strong band [0.0, 1.0]
    pub strong_fraction: f32,

    /// Fraction of vertices in Weak band [0.0, 1.0]
    pub weak_fraction: f32,

    /// Fraction of vertices in Neutral band [0.0, 1.0]
    pub neutral_fraction: f32,

    /// Mean force multiplier across all vertices
    pub mean_force: f32,

    /// Min force multiplier
    pub min_force: f32,

    /// Max force multiplier
    pub max_force: f32,

    /// Standard deviation of force multipliers
    pub force_stddev: f32,
}

impl ForceBandTelemetry {
    /// Create from force band counts and statistics
    #[allow(clippy::too_many_arguments)]
    pub fn from_stats(
        strong_count: usize,
        neutral_count: usize,
        weak_count: usize,
        total_vertices: usize,
        mean_force: f32,
        min_force: f32,
        max_force: f32,
        force_stddev: f32,
    ) -> Self {
        let total = total_vertices as f32;
        Self {
            strong_fraction: (strong_count as f32) / total,
            neutral_fraction: (neutral_count as f32) / total,
            weak_fraction: (weak_count as f32) / total,
            mean_force,
            min_force,
            max_force,
            force_stddev,
        }
    }

    /// Create from ForceBandStats (from ForceProfile)
    pub fn from_force_band_stats(stats: &crate::fluxnet::ForceBandStats) -> Self {
        Self {
            strong_fraction: stats.strong_fraction,
            neutral_fraction: stats.neutral_fraction,
            weak_fraction: stats.weak_fraction,
            mean_force: stats.mean_force,
            min_force: stats.min_force,
            max_force: stats.max_force,
            force_stddev: stats.std_force,
        }
    }
}

/// RL decision and action telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLDecisionTelemetry {
    /// Temperature index when decision was made
    pub temp_index: usize,

    /// RL state observation (discretized)
    pub state: RLStateTelemetry,

    /// Action taken by RL controller
    pub action: ForceCommand,

    /// Q-value for selected action (before update)
    pub q_value: f32,

    /// Exploration epsilon at decision time
    pub epsilon: f32,

    /// Whether action was exploratory (random) or exploitative (greedy)
    pub was_exploration: bool,
}

impl RLDecisionTelemetry {
    /// Create from RL controller state and action
    pub fn new(
        temp_index: usize,
        state: &RLState,
        action: ForceCommand,
        q_value: f32,
        epsilon: f32,
        was_exploration: bool,
    ) -> Self {
        Self {
            temp_index,
            state: RLStateTelemetry::from_rl_state(state),
            action,
            q_value,
            epsilon,
            was_exploration,
        }
    }
}

/// RL state observation for telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLStateTelemetry {
    /// Conflict count
    pub conflicts: usize,

    /// Chromatic number
    pub chromatic_number: usize,

    /// Compaction ratio [0.0, 1.0]
    pub compaction_ratio: f32,

    /// Guard streak
    pub guard_count: usize,

    /// Current slack
    pub current_slack: usize,

    /// Whether compaction guard detected collapse
    pub phase_locked: bool,

    /// Steps per temperature
    pub steps_per_temp: usize,

    /// Guard threshold in effect
    pub guard_threshold: f32,

    /// Collapse streak count
    pub collapse_streak: usize,

    /// Force band variance
    pub band_std: f32,

    /// Discretized state index
    pub state_index: usize,
}

impl RLStateTelemetry {
    /// Create from RLState
    pub fn from_rl_state(state: &RLState) -> Self {
        Self {
            conflicts: state.conflicts,
            chromatic_number: state.chromatic_number,
            compaction_ratio: state.compaction_ratio,
            guard_count: state.guard_count,
            current_slack: state.current_slack,
            phase_locked: state.phase_locked,
            steps_per_temp: state.current_steps,
            guard_threshold: state.dynamic_guard_threshold,
            collapse_streak: state.collapse_streak,
            band_std: state.band_std,
            state_index: state.to_index(true),
        }
    }
}

/// Q-learning update telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QUpdateTelemetry {
    /// Reward computed for this transition
    pub reward: f32,

    /// Previous Q-value (before update)
    pub q_old: f32,

    /// New Q-value (after update)
    pub q_new: f32,

    /// Q-value delta (q_new - q_old)
    pub q_delta: f32,

    /// Learning rate used for update
    pub learning_rate: f32,

    /// Whether this was a terminal state
    pub is_terminal: bool,

    /// Next state index (for debugging)
    pub next_state_index: usize,
}

impl QUpdateTelemetry {
    /// Create from Q-learning update parameters
    pub fn new(
        reward: f32,
        q_old: f32,
        q_new: f32,
        learning_rate: f32,
        is_terminal: bool,
        next_state_index: usize,
    ) -> Self {
        Self {
            reward,
            q_old,
            q_new,
            q_delta: q_new - q_old,
            learning_rate,
            is_terminal,
            next_state_index,
        }
    }
}

/// FluxNet configuration snapshot for telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxNetConfigSnapshot {
    /// Memory tier: "compact" or "extended"
    pub memory_tier: String,

    /// Q-table state space size
    pub qtable_states: usize,

    /// Replay buffer capacity
    pub replay_capacity: usize,

    /// Learning rate
    pub learning_rate: f32,

    /// Discount factor (gamma)
    pub discount_factor: f32,

    /// Epsilon start value
    pub epsilon_start: f32,

    /// Epsilon decay rate
    pub epsilon_decay: f32,

    /// Epsilon minimum value
    pub epsilon_min: f32,
}

impl FluxNetConfigSnapshot {
    /// Create from FluxNetConfig
    pub fn from_config(config: &crate::fluxnet::FluxNetConfig) -> Self {
        Self {
            memory_tier: format!("{:?}", config.memory_tier),
            qtable_states: config.rl.get_qtable_states(config.memory_tier),
            replay_capacity: config.rl.get_replay_capacity(config.memory_tier),
            learning_rate: config.rl.learning_rate,
            discount_factor: config.rl.discount_factor,
            epsilon_start: config.rl.epsilon_start,
            epsilon_decay: config.rl.epsilon_decay,
            epsilon_min: config.rl.epsilon_min,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_force_band_telemetry_fractions() {
        let telem = ForceBandTelemetry::from_stats(
            100,  // strong
            300,  // neutral
            600,  // weak
            1000, // total
            1.0, 0.5, 1.5, 0.25,
        );

        assert_eq!(telem.strong_fraction, 0.1);
        assert_eq!(telem.neutral_fraction, 0.3);
        assert_eq!(telem.weak_fraction, 0.6);
    }

    #[test]
    fn test_telemetry_serialization() {
        let force_bands = ForceBandTelemetry {
            strong_fraction: 0.2,
            neutral_fraction: 0.5,
            weak_fraction: 0.3,
            mean_force: 1.0,
            min_force: 0.5,
            max_force: 1.5,
            force_stddev: 0.2,
        };

        let rl_state = RLStateTelemetry {
            conflicts: 10,
            chromatic_number: 95,
            compaction_ratio: 0.75,
            guard_count: 2,
            current_slack: 55,
            phase_locked: false,
            state_index: 42,
        };

        let rl_decision = RLDecisionTelemetry {
            temp_index: 5,
            state: rl_state,
            action: ForceCommand::IncreaseStrong,
            q_value: 0.8,
            epsilon: 0.1,
            was_exploration: false,
        };

        let config = FluxNetConfigSnapshot {
            memory_tier: "Compact".to_string(),
            qtable_states: 256,
            replay_capacity: 1024,
            learning_rate: 0.001,
            discount_factor: 0.95,
            epsilon_start: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
        };

        let telemetry = FluxNetTelemetry::new(force_bands, rl_decision, None, config);

        let json = serde_json::to_string(&telemetry).expect("Failed to serialize");
        let _deserialized: FluxNetTelemetry =
            serde_json::from_str(&json).expect("Failed to deserialize");
    }
}
