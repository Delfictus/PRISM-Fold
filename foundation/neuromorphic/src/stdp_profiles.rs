//! STDP Configuration Profiles for Different Use Cases
//!
//! Spike-Timing-Dependent Plasticity (STDP) profiles optimized for various application domains.
//! Each profile provides pre-configured learning parameters tailored to specific requirements.

use serde::{Deserialize, Serialize};

/// STDP learning profile presets for different application domains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum STDPProfile {
    /// Slow learning, high stability - for production systems requiring reliability
    Conservative,
    /// Balanced learning - recommended default for most applications
    #[default]
    Balanced,
    /// Fast adaptation - for research and rapid prototyping
    Aggressive,
    /// Optimized for financial pattern recognition and trading
    Financial,
    /// Optimized for optical systems (DARPA Narcissus)
    Optical,
    /// Custom profile - user-defined parameters
    Custom,
}

/// STDP learning configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STDPConfig {
    /// Learning rate (η) - controls adaptation speed
    pub learning_rate: f64,
    /// Positive time constant (τ+) - LTP window in milliseconds
    pub time_constant_pos: f64,
    /// Negative time constant (τ-) - LTD window in milliseconds
    pub time_constant_neg: f64,
    /// Maximum synaptic weight (upper saturation bound)
    pub max_weight: f64,
    /// Minimum synaptic weight (lower saturation bound)
    pub min_weight: f64,
    /// Enable heterosynaptic plasticity (competition between synapses)
    pub enable_heterosynaptic: bool,
    /// Weight decay rate (0.0 = no decay, 1.0 = full decay)
    pub weight_decay: f64,
    /// Enable homeostatic regulation (maintains network stability)
    pub enable_homeostasis: bool,
    /// Target mean activity level for homeostasis
    pub target_activity: f64,
}

impl Default for STDPConfig {
    fn default() -> Self {
        STDPProfile::Balanced.get_config()
    }
}

impl STDPProfile {
    /// Get STDP configuration parameters for this profile
    pub fn get_config(&self) -> STDPConfig {
        match self {
            STDPProfile::Conservative => STDPConfig {
                learning_rate: 0.001,
                time_constant_pos: 20.0,
                time_constant_neg: 20.0,
                max_weight: 2.0,
                min_weight: 0.1,
                enable_heterosynaptic: false,
                weight_decay: 0.0001,
                enable_homeostasis: true,
                target_activity: 0.1,
            },
            STDPProfile::Balanced => STDPConfig {
                learning_rate: 0.005,
                time_constant_pos: 15.0,
                time_constant_neg: 15.0,
                max_weight: 3.0,
                min_weight: 0.05,
                enable_heterosynaptic: true,
                weight_decay: 0.0005,
                enable_homeostasis: true,
                target_activity: 0.15,
            },
            STDPProfile::Aggressive => STDPConfig {
                learning_rate: 0.02,
                time_constant_pos: 10.0,
                time_constant_neg: 10.0,
                max_weight: 5.0,
                min_weight: 0.01,
                enable_heterosynaptic: true,
                weight_decay: 0.001,
                enable_homeostasis: false,
                target_activity: 0.2,
            },
            STDPProfile::Financial => STDPConfig {
                learning_rate: 0.008,
                time_constant_pos: 12.0,
                time_constant_neg: 18.0, // Asymmetric for momentum detection
                max_weight: 4.0,
                min_weight: 0.1,
                enable_heterosynaptic: true,
                weight_decay: 0.0008,
                enable_homeostasis: true,
                target_activity: 0.12,
            },
            STDPProfile::Optical => STDPConfig {
                learning_rate: 0.015,
                time_constant_pos: 8.0, // Fast adaptation for calibration
                time_constant_neg: 12.0,
                max_weight: 6.0,
                min_weight: 0.01,
                enable_heterosynaptic: true,
                weight_decay: 0.002,
                enable_homeostasis: false,
                target_activity: 0.25,
            },
            STDPProfile::Custom => {
                // Return balanced as base for customization
                Self::Balanced.get_config()
            }
        }
    }

    /// Get human-readable description of this profile
    pub fn description(&self) -> &'static str {
        match self {
            STDPProfile::Conservative => {
                "Conservative learning with high stability. Suitable for production systems requiring reliability."
            }
            STDPProfile::Balanced => {
                "Balanced learning rate with moderate adaptation. Recommended default for most applications."
            }
            STDPProfile::Aggressive => {
                "Fast adaptation with high learning rate. Best for research and rapid prototyping."
            }
            STDPProfile::Financial => {
                "Optimized for financial pattern recognition with asymmetric time windows for momentum detection."
            }
            STDPProfile::Optical => {
                "Fast adaptation for optical system calibration (DARPA Narcissus). High learning rate for real-time tuning."
            }
            STDPProfile::Custom => {
                "User-defined custom parameters. Start with Balanced and adjust as needed."
            }
        }
    }

    /// Get all available profiles
    pub fn all() -> Vec<Self> {
        vec![
            Self::Conservative,
            Self::Balanced,
            Self::Aggressive,
            Self::Financial,
            Self::Optical,
            Self::Custom,
        ]
    }
}

/// Learning statistics for monitoring STDP adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStats {
    /// Mean synaptic weight across all connections
    pub mean_weight: f64,
    /// Variance of synaptic weights
    pub weight_variance: f64,
    /// Maximum synaptic weight
    pub max_weight: f64,
    /// Minimum synaptic weight
    pub min_weight: f64,
    /// Percentage of weights at saturation bounds
    pub saturation_percentage: f64,
    /// Total number of weight updates performed
    pub total_updates: usize,
    /// Current learning rate
    pub learning_rate: f64,
    /// Mean network activity level
    pub mean_activity: f64,
    /// Weight distribution entropy (measure of diversity)
    pub weight_entropy: f64,
}

impl Default for LearningStats {
    fn default() -> Self {
        Self {
            mean_weight: 0.0,
            weight_variance: 0.0,
            max_weight: 0.0,
            min_weight: 0.0,
            saturation_percentage: 0.0,
            total_updates: 0,
            learning_rate: 0.0,
            mean_activity: 0.0,
            weight_entropy: 0.0,
        }
    }
}

impl LearningStats {
    /// Check if learning has converged (low variance in recent updates)
    pub fn is_converged(&self, variance_threshold: f64) -> bool {
        self.weight_variance < variance_threshold
    }

    /// Check if weights are saturated (stuck at bounds)
    pub fn is_saturated(&self, threshold_percentage: f64) -> bool {
        self.saturation_percentage > threshold_percentage
    }

    /// Get learning health score (0.0 = unhealthy, 1.0 = optimal)
    pub fn health_score(&self) -> f64 {
        let mut score = 1.0;

        // Penalize high saturation
        if self.saturation_percentage > 50.0 {
            score -= 0.3;
        } else if self.saturation_percentage > 25.0 {
            score -= 0.1;
        }

        // Penalize extremely low entropy (not diverse)
        if self.weight_entropy < 0.5 {
            score -= 0.2;
        }

        // Penalize extreme mean weights
        if self.mean_weight < 0.5 || self.mean_weight > 4.0 {
            score -= 0.2;
        }

        f64::max(score, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_configs() {
        for profile in STDPProfile::all() {
            let config = profile.get_config();
            assert!(config.learning_rate > 0.0);
            assert!(config.time_constant_pos > 0.0);
            assert!(config.time_constant_neg > 0.0);
            assert!(config.max_weight > config.min_weight);
            assert!(!profile.description().is_empty());
        }
    }

    #[test]
    fn test_conservative_profile() {
        let config = STDPProfile::Conservative.get_config();
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.enable_heterosynaptic, false);
        assert!(config.enable_homeostasis);
    }

    #[test]
    fn test_aggressive_profile() {
        let config = STDPProfile::Aggressive.get_config();
        assert!(config.learning_rate > 0.01);
        assert!(config.enable_heterosynaptic);
    }

    #[test]
    fn test_financial_profile() {
        let config = STDPProfile::Financial.get_config();
        // Financial profile has asymmetric time constants
        assert!(config.time_constant_neg > config.time_constant_pos);
    }

    #[test]
    fn test_learning_stats_health() {
        let mut stats = LearningStats::default();
        stats.mean_weight = 1.5;
        stats.saturation_percentage = 10.0;
        stats.weight_entropy = 0.8;

        let health = stats.health_score();
        assert!(health > 0.8); // Should be healthy

        stats.saturation_percentage = 60.0;
        let health = stats.health_score();
        assert!(health < 0.8); // Should be unhealthy
    }
}
