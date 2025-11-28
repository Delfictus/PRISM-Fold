//! Hypertuning Actions and Events
//!
//! Defines telemetry-triggered events and corresponding control actions.

use crate::telemetry::PhaseName;
use serde::{Deserialize, Serialize};

/// Telemetry event that triggers hypertuning action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TelemetryEvent {
    /// Phase is making no progress (stuck)
    PhaseStalled {
        phase: PhaseName,
        duration_sec: f64,
        iterations_without_improvement: usize,
    },

    /// Phase efficiency is below threshold
    LowEfficiency {
        phase: PhaseName,
        metric: String,
        current_value: f64,
        threshold: f64,
    },

    /// No improvement across multiple iterations
    NoImprovement {
        iterations: usize,
        chromatic_stuck_at: usize,
    },

    /// High conflict rate detected
    HighConflicts {
        phase: PhaseName,
        conflicts: usize,
        threshold: usize,
    },
}

/// Control action to adjust pipeline parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdpControl {
    /// Adjust thermodynamic temperature ladder
    AdjustThermoTemps { delta_percent: i32 },

    /// Set transfer entropy weight
    SetTeWeight { weight: f64 },

    /// Increase quantum annealing iterations
    IncreaseQuantumIterations { additional_iters: usize },

    /// Adjust memetic population size
    AdjustMemeticPopulation { delta: i32 },

    /// Enable/disable specific phase
    TogglePhase { phase: PhaseName, enabled: bool },

    /// Adjust ADP learning rate
    AdjustAdpAlpha { new_alpha: f64 },

    /// Reset pipeline to baseline config
    ResetToBaseline,
}

impl AdpControl {
    /// Get human-readable description
    pub fn description(&self) -> String {
        match self {
            AdpControl::AdjustThermoTemps { delta_percent } => {
                format!("Adjust thermodynamic temps by {}%", delta_percent)
            }
            AdpControl::SetTeWeight { weight } => {
                format!("Set TE weight to {:.2}", weight)
            }
            AdpControl::IncreaseQuantumIterations { additional_iters } => {
                format!("Add {} quantum iterations", additional_iters)
            }
            AdpControl::AdjustMemeticPopulation { delta } => {
                format!("Adjust memetic population by {}", delta)
            }
            AdpControl::TogglePhase { phase, enabled } => {
                format!(
                    "{} phase {}",
                    if *enabled { "Enable" } else { "Disable" },
                    phase
                )
            }
            AdpControl::AdjustAdpAlpha { new_alpha } => {
                format!("Set ADP alpha to {:.3}", new_alpha)
            }
            AdpControl::ResetToBaseline => "Reset to baseline config".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_serialization() {
        let event = TelemetryEvent::PhaseStalled {
            phase: PhaseName::Thermodynamic,
            duration_sec: 30.0,
            iterations_without_improvement: 100,
        };

        let json = serde_json::to_string(&event).expect("Failed to serialize");
        let _deserialized: TelemetryEvent =
            serde_json::from_str(&json).expect("Failed to deserialize");
    }

    #[test]
    fn test_action_description() {
        let action = AdpControl::AdjustThermoTemps { delta_percent: 10 };
        assert!(action.description().contains("10%"));
    }
}
