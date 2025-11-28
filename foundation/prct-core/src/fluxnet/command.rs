//! FluxNet Force Commands - RL Action Representation
//!
//! Defines the action space for the RL controller to manipulate force profiles
//! during Phase 2 thermodynamic equilibration.
//!
//! # Command Architecture
//!
//! Commands are discrete actions that modify ForceProfile band multipliers:
//! - **Target**: Which force band to modify (Strong/Neutral/Weak)
//! - **Direction**: Increase or Decrease
//! - **Magnitude**: Fixed adjustment factor (typically 1.1 or 0.9)
//!
//! # RL Integration
//!
//! The RL controller:
//! 1. Observes telemetry state (conflicts, colors, compaction_ratio)
//! 2. Selects a ForceCommand via Q-learning policy
//! 3. Applies command to ForceProfile
//! 4. Observes reward (improvement in telemetry)
//! 5. Updates Q-table based on reward

use super::ForceBand;
use serde::{Deserialize, Serialize};

/// Force command issued by RL controller
///
/// Represents discrete actions in the RL action space.
/// Each command specifies which force band to modify and by how much.
///
/// # Action Space
///
/// Total of 11 discrete actions:
/// - 3 increase actions (Strong, Neutral, Weak)
/// - 3 decrease actions (Strong, Neutral, Weak)
/// - 4 meta-actions for guard tuning (Slack ±, Threshold ±)
/// - 2 meta-actions for step tuning (Steps ±)
/// - 1 no-op action (do nothing)
///
/// # Magnitudes
///
/// - Increase: multiply by 1.1 (10% boost)
/// - Decrease: multiply by 0.9 (10% reduction)
/// - NoOp: no change
///
/// # Example Usage
///
/// ```rust,ignore
/// use fluxnet::{ForceCommand, ForceBand};
///
/// // RL controller selects action
/// let action = q_table.select_action(state);
/// let command = ForceCommand::from_action_index(action);
///
/// // Apply to force profile
/// force_profile.apply_force_command(&command)?;
///
/// // Sync to GPU before kernel launch
/// force_profile.to_device()?;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForceCommand {
    /// Increase strong force band by 10% (action index 0)
    IncreaseStrong,

    /// Increase neutral force band by 10% (action index 1)
    IncreaseNeutral,

    /// Increase weak force band by 10% (action index 2)
    IncreaseWeak,

    /// Decrease strong force band by 10% (action index 3)
    DecreaseStrong,

    /// Decrease neutral force band by 10% (action index 4)
    DecreaseNeutral,

    /// Decrease weak force band by 10% (action index 5)
    DecreaseWeak,

    /// Increase guard slack window (+5 colors) (action index 6)
    IncreaseSlack,

    /// Decrease guard slack window (-5 colors) (action index 7)
    DecreaseSlack,

    /// Raise compaction guard threshold (+0.01) (action index 8)
    RaiseGuardThreshold,

    /// Lower compaction guard threshold (-0.01) (action index 9)
    LowerGuardThreshold,

    /// Increase steps per temp (+5k) (action index 10)
    IncreaseSteps,

    /// Decrease steps per temp (-5k) (action index 11)
    DecreaseSteps,

    /// No operation - maintain current forces (action index 6)
    NoOp,
}

impl ForceCommand {
    /// Total number of discrete actions in action space
    pub const ACTION_SPACE_SIZE: usize = 13;

    /// Convert from action index (0-6) to ForceCommand
    ///
    /// Used by RL controller to map Q-table actions to commands.
    ///
    /// # Arguments
    /// - `action_idx`: Action index from Q-table (0-6)
    ///
    /// # Returns
    /// Corresponding ForceCommand, or NoOp if index out of range
    pub fn from_action_index(action_idx: usize) -> Self {
        match action_idx {
            0 => ForceCommand::IncreaseStrong,
            1 => ForceCommand::IncreaseNeutral,
            2 => ForceCommand::IncreaseWeak,
            3 => ForceCommand::DecreaseStrong,
            4 => ForceCommand::DecreaseNeutral,
            5 => ForceCommand::DecreaseWeak,
            6 => ForceCommand::IncreaseSlack,
            7 => ForceCommand::DecreaseSlack,
            8 => ForceCommand::RaiseGuardThreshold,
            9 => ForceCommand::LowerGuardThreshold,
            10 => ForceCommand::IncreaseSteps,
            11 => ForceCommand::DecreaseSteps,
            12 => ForceCommand::NoOp,
            _ => ForceCommand::NoOp, // Default to no-op for invalid indices
        }
    }

    /// Convert ForceCommand to action index (0-6)
    ///
    /// Used by RL controller to map commands back to Q-table indices.
    pub fn to_action_index(&self) -> usize {
        match self {
            ForceCommand::IncreaseStrong => 0,
            ForceCommand::IncreaseNeutral => 1,
            ForceCommand::IncreaseWeak => 2,
            ForceCommand::DecreaseStrong => 3,
            ForceCommand::DecreaseNeutral => 4,
            ForceCommand::DecreaseWeak => 5,
            ForceCommand::IncreaseSlack => 6,
            ForceCommand::DecreaseSlack => 7,
            ForceCommand::RaiseGuardThreshold => 8,
            ForceCommand::LowerGuardThreshold => 9,
            ForceCommand::IncreaseSteps => 10,
            ForceCommand::DecreaseSteps => 11,
            ForceCommand::NoOp => 12,
        }
    }

    /// Get the target force band for this command
    ///
    /// Returns None for NoOp commands.
    pub fn target_band(&self) -> Option<ForceBand> {
        match self {
            ForceCommand::IncreaseStrong | ForceCommand::DecreaseStrong => Some(ForceBand::Strong),
            ForceCommand::IncreaseNeutral | ForceCommand::DecreaseNeutral => {
                Some(ForceBand::Neutral)
            }
            ForceCommand::IncreaseWeak | ForceCommand::DecreaseWeak => Some(ForceBand::Weak),
            _ => None,
        }
    }

    /// Get the adjustment multiplier for this command
    ///
    /// - Increase commands: 1.1 (10% boost)
    /// - Decrease commands: 0.9 (10% reduction)
    /// - NoOp: 1.0 (no change)
    pub fn multiplier(&self) -> f32 {
        match self {
            ForceCommand::IncreaseStrong
            | ForceCommand::IncreaseNeutral
            | ForceCommand::IncreaseWeak => 1.1,

            ForceCommand::DecreaseStrong
            | ForceCommand::DecreaseNeutral
            | ForceCommand::DecreaseWeak => 0.9,

            ForceCommand::NoOp => 1.0,

            ForceCommand::IncreaseSlack
            | ForceCommand::DecreaseSlack
            | ForceCommand::RaiseGuardThreshold
            | ForceCommand::LowerGuardThreshold
            | ForceCommand::IncreaseSteps
            | ForceCommand::DecreaseSteps => 1.0,
        }
    }

    /// Check if this is a no-op command
    pub fn is_noop(&self) -> bool {
        matches!(self, ForceCommand::NoOp)
    }

    /// Get human-readable description of command
    ///
    /// Used for telemetry logging and debugging.
    pub fn description(&self) -> &'static str {
        match self {
            ForceCommand::IncreaseStrong => "Increase Strong Force",
            ForceCommand::IncreaseNeutral => "Increase Neutral Force",
            ForceCommand::IncreaseWeak => "Increase Weak Force",
            ForceCommand::DecreaseStrong => "Decrease Strong Force",
            ForceCommand::DecreaseNeutral => "Decrease Neutral Force",
            ForceCommand::DecreaseWeak => "Decrease Weak Force",
            ForceCommand::IncreaseSlack => "Increase Guard Slack",
            ForceCommand::DecreaseSlack => "Decrease Guard Slack",
            ForceCommand::RaiseGuardThreshold => "Raise Guard Threshold",
            ForceCommand::LowerGuardThreshold => "Lower Guard Threshold",
            ForceCommand::IncreaseSteps => "Increase Steps/Temp",
            ForceCommand::DecreaseSteps => "Decrease Steps/Temp",
            ForceCommand::NoOp => "No Operation",
        }
    }

    /// Get short code for telemetry (2-3 chars)
    ///
    /// Used in compact telemetry logs:
    /// - `S+`: Increase Strong
    /// - `N-`: Decrease Neutral
    /// - `W+`: Increase Weak
    /// - `--`: No-op
    pub fn telemetry_code(&self) -> &'static str {
        match self {
            ForceCommand::IncreaseStrong => "S+",
            ForceCommand::IncreaseNeutral => "N+",
            ForceCommand::IncreaseWeak => "W+",
            ForceCommand::DecreaseStrong => "S-",
            ForceCommand::DecreaseNeutral => "N-",
            ForceCommand::DecreaseWeak => "W-",
            ForceCommand::IncreaseSlack => "SL+",
            ForceCommand::DecreaseSlack => "SL-",
            ForceCommand::RaiseGuardThreshold => "TH+",
            ForceCommand::LowerGuardThreshold => "TH-",
            ForceCommand::IncreaseSteps => "ST+",
            ForceCommand::DecreaseSteps => "ST-",
            ForceCommand::NoOp => "--",
        }
    }

    /// Get all possible commands (for RL exploration)
    pub fn all_actions() -> Vec<ForceCommand> {
        (0..Self::ACTION_SPACE_SIZE)
            .map(Self::from_action_index)
            .collect()
    }

    /// Check whether the command adjusts guard parameters instead of force bands
    pub fn is_meta_guard_adjustment(&self) -> bool {
        matches!(
            self,
            ForceCommand::IncreaseSlack
                | ForceCommand::DecreaseSlack
                | ForceCommand::RaiseGuardThreshold
                | ForceCommand::LowerGuardThreshold
        )
    }

    /// Check whether command adjusts thermodynamic steps per temp
    pub fn is_meta_step_adjustment(&self) -> bool {
        matches!(
            self,
            ForceCommand::IncreaseSteps | ForceCommand::DecreaseSteps
        )
    }
}

impl Default for ForceCommand {
    /// Default command is NoOp
    fn default() -> Self {
        ForceCommand::NoOp
    }
}

impl std::fmt::Display for ForceCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Command execution result with telemetry
///
/// Returned when applying a ForceCommand to a ForceProfile.
/// Captures before/after state for reward computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    /// Command that was executed
    pub command: ForceCommand,

    /// Force band statistics before command
    pub stats_before: super::ForceBandStats,

    /// Force band statistics after command
    pub stats_after: super::ForceBandStats,

    /// Whether command modified the profile (false for NoOp)
    pub modified: bool,
}

impl CommandResult {
    /// Create a new command result
    pub fn new(
        command: ForceCommand,
        stats_before: super::ForceBandStats,
        stats_after: super::ForceBandStats,
        modified: bool,
    ) -> Self {
        Self {
            command,
            stats_before,
            stats_after,
            modified,
        }
    }

    /// Compute change in mean force
    ///
    /// Positive = force increased, Negative = force decreased
    pub fn mean_force_delta(&self) -> f32 {
        self.stats_after.mean_force - self.stats_before.mean_force
    }

    /// Compute change in strong fraction
    ///
    /// Positive = more vertices in strong band
    pub fn strong_fraction_delta(&self) -> f32 {
        self.stats_after.strong_fraction - self.stats_before.strong_fraction
    }

    /// Compute change in weak fraction
    ///
    /// Positive = more vertices in weak band
    pub fn weak_fraction_delta(&self) -> f32 {
        self.stats_after.weak_fraction - self.stats_before.weak_fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_index_conversion() {
        for i in 0..ForceCommand::ACTION_SPACE_SIZE {
            let command = ForceCommand::from_action_index(i);
            assert_eq!(command.to_action_index(), i);
        }
    }

    #[test]
    fn test_action_space_size() {
        let all = ForceCommand::all_actions();
        assert_eq!(all.len(), ForceCommand::ACTION_SPACE_SIZE);
    }

    #[test]
    fn test_multipliers() {
        assert_eq!(ForceCommand::IncreaseStrong.multiplier(), 1.1);
        assert_eq!(ForceCommand::DecreaseWeak.multiplier(), 0.9);
        assert_eq!(ForceCommand::NoOp.multiplier(), 1.0);
    }

    #[test]
    fn test_target_bands() {
        assert_eq!(
            ForceCommand::IncreaseStrong.target_band(),
            Some(ForceBand::Strong)
        );
        assert_eq!(
            ForceCommand::DecreaseNeutral.target_band(),
            Some(ForceBand::Neutral)
        );
        assert_eq!(ForceCommand::NoOp.target_band(), None);
    }

    #[test]
    fn test_telemetry_codes() {
        assert_eq!(ForceCommand::IncreaseStrong.telemetry_code(), "S+");
        assert_eq!(ForceCommand::DecreaseNeutral.telemetry_code(), "N-");
        assert_eq!(ForceCommand::NoOp.telemetry_code(), "--");
    }

    #[test]
    fn test_noop_detection() {
        assert!(ForceCommand::NoOp.is_noop());
        assert!(!ForceCommand::IncreaseStrong.is_noop());
    }

    #[test]
    fn test_display() {
        let cmd = ForceCommand::IncreaseWeak;
        assert_eq!(format!("{}", cmd), "Increase Weak Force");
    }
}
