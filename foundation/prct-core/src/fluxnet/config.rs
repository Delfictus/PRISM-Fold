//! FluxNet Configuration - TOML Configuration Structure
//!
//! Defines configuration options for FluxNet RL system, loaded from TOML files.
//!
//! # Configuration Hierarchy
//!
//! ```toml
//! [fluxnet]
//! enabled = true
//! memory_tier = "compact"  # or "extended"
//!
//! [fluxnet.force_profile]
//! strong_multiplier = 1.5
//! weak_multiplier = 0.7
//! force_clamp_min = 0.5
//! force_clamp_max = 2.0
//!
//! [fluxnet.rl]
//! learning_rate = 0.1
//! discount_factor = 0.95
//! epsilon_start = 0.3
//! epsilon_decay = 0.995
//! epsilon_min = 0.05
//! replay_capacity = 1024
//! qtable_states = 256
//!
//! [fluxnet.persistence]
//! cache_dir = "target/fluxnet_cache"
//! save_interval_temps = 10
//! load_pretrained = true
//! pretrained_path = "target/fluxnet_cache/dsjc250_qtable.bin"
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Memory tier for FluxNet RL controller
///
/// Determines replay buffer size, Q-table dimensions, and GPU memory usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryTier {
    /// Compact mode: 8GB GPU, small replay buffer, reduced Q-table
    Compact,

    /// Extended mode: 24GB+ GPU, large replay buffer, full Q-table
    Extended,
}

impl MemoryTier {
    /// Get replay buffer capacity for this tier
    pub fn replay_capacity(&self) -> usize {
        match self {
            MemoryTier::Compact => 1024,   // 1K transitions
            MemoryTier::Extended => 16384, // 16K transitions
        }
    }

    /// Get Q-table state space size for this tier
    pub fn qtable_states(&self) -> usize {
        match self {
            MemoryTier::Compact => 256,   // 256 states × 7 actions = 1,792 Q-values
            MemoryTier::Extended => 1024, // 1K states × 7 actions = 7,168 Q-values
        }
    }

    /// Get batch size for experience replay
    pub fn batch_size(&self) -> usize {
        match self {
            MemoryTier::Compact => 32,
            MemoryTier::Extended => 64,
        }
    }
}

impl Default for MemoryTier {
    fn default() -> Self {
        MemoryTier::Compact
    }
}

/// Force profile initialization and update settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceProfileConfig {
    /// Default strong force multiplier [1.0, 2.0]
    #[serde(default = "default_strong_multiplier")]
    pub strong_multiplier: f32,

    /// Default weak force multiplier [0.5, 1.0]
    #[serde(default = "default_weak_multiplier")]
    pub weak_multiplier: f32,

    /// Minimum force clamp value
    #[serde(default = "default_force_clamp_min")]
    pub force_clamp_min: f32,

    /// Maximum force clamp value
    #[serde(default = "default_force_clamp_max")]
    pub force_clamp_max: f32,

    /// Strong band percentile cutoff (top X%)
    #[serde(default = "default_strong_percentile")]
    pub strong_percentile: f32,

    /// Weak band percentile cutoff (bottom X%)
    #[serde(default = "default_weak_percentile")]
    pub weak_percentile: f32,
}

fn default_strong_multiplier() -> f32 {
    1.5
}
fn default_weak_multiplier() -> f32 {
    0.7
}
fn default_force_clamp_min() -> f32 {
    0.5
}
fn default_force_clamp_max() -> f32 {
    2.0
}
fn default_strong_percentile() -> f32 {
    0.2
} // Top 20%
fn default_weak_percentile() -> f32 {
    0.2
} // Bottom 20%

impl Default for ForceProfileConfig {
    fn default() -> Self {
        Self {
            strong_multiplier: default_strong_multiplier(),
            weak_multiplier: default_weak_multiplier(),
            force_clamp_min: default_force_clamp_min(),
            force_clamp_max: default_force_clamp_max(),
            strong_percentile: default_strong_percentile(),
            weak_percentile: default_weak_percentile(),
        }
    }
}

/// RL controller hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLConfig {
    /// Q-learning learning rate (alpha) [0.0, 1.0]
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,

    /// Discount factor (gamma) for future rewards [0.0, 1.0]
    #[serde(default = "default_discount_factor")]
    pub discount_factor: f32,

    /// Initial epsilon for epsilon-greedy exploration [0.0, 1.0]
    #[serde(default = "default_epsilon_start")]
    pub epsilon_start: f32,

    /// Epsilon decay rate per temperature step
    #[serde(default = "default_epsilon_decay")]
    pub epsilon_decay: f32,

    /// Minimum epsilon (exploration floor)
    #[serde(default = "default_epsilon_min")]
    pub epsilon_min: f32,

    /// Replay buffer capacity (overrides memory tier if set)
    #[serde(default)]
    pub replay_capacity: Option<usize>,

    /// Q-table state space size (overrides memory tier if set)
    #[serde(default)]
    pub qtable_states: Option<usize>,

    /// Batch size for experience replay (overrides memory tier if set)
    #[serde(default)]
    pub batch_size: Option<usize>,

    /// Reward shaping: weight for conflict reduction
    #[serde(default = "default_reward_conflict_weight")]
    pub reward_conflict_weight: f32,

    /// Reward shaping: weight for color reduction
    #[serde(default = "default_reward_color_weight")]
    pub reward_color_weight: f32,

    /// Reward shaping: weight for compaction ratio improvement
    #[serde(default = "default_reward_compaction_weight")]
    pub reward_compaction_weight: f32,
}

fn default_learning_rate() -> f32 {
    0.1
}
fn default_discount_factor() -> f32 {
    0.95
}
fn default_epsilon_start() -> f32 {
    0.3
}
fn default_epsilon_decay() -> f32 {
    0.995
}
fn default_epsilon_min() -> f32 {
    0.05
}
fn default_reward_conflict_weight() -> f32 {
    1.0
}
fn default_reward_color_weight() -> f32 {
    0.5
}
fn default_reward_compaction_weight() -> f32 {
    0.3
}

impl Default for RLConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_learning_rate(),
            discount_factor: default_discount_factor(),
            epsilon_start: default_epsilon_start(),
            epsilon_decay: default_epsilon_decay(),
            epsilon_min: default_epsilon_min(),
            replay_capacity: None,
            qtable_states: None,
            batch_size: None,
            reward_conflict_weight: default_reward_conflict_weight(),
            reward_color_weight: default_reward_color_weight(),
            reward_compaction_weight: default_reward_compaction_weight(),
        }
    }
}

impl RLConfig {
    /// Get replay capacity, using override or memory tier default
    pub fn get_replay_capacity(&self, memory_tier: MemoryTier) -> usize {
        self.replay_capacity
            .unwrap_or_else(|| memory_tier.replay_capacity())
    }

    /// Get Q-table states, using override or memory tier default
    pub fn get_qtable_states(&self, memory_tier: MemoryTier) -> usize {
        self.qtable_states
            .unwrap_or_else(|| memory_tier.qtable_states())
    }

    /// Get batch size, using override or memory tier default
    pub fn get_batch_size(&self, memory_tier: MemoryTier) -> usize {
        self.batch_size.unwrap_or_else(|| memory_tier.batch_size())
    }
}

/// Persistence and caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Directory for FluxNet cache files (Q-tables, replay buffers)
    #[serde(default = "default_cache_dir")]
    pub cache_dir: PathBuf,

    /// Save Q-table every N temperature steps (0 = disabled)
    #[serde(default = "default_save_interval")]
    pub save_interval_temps: usize,

    /// Load pre-trained Q-table if available
    #[serde(default = "default_load_pretrained")]
    pub load_pretrained: bool,

    /// Path to pre-trained Q-table (from DSJC250 pre-training)
    #[serde(default)]
    pub pretrained_path: Option<PathBuf>,

    /// Save final Q-table after run completion
    #[serde(default = "default_save_final")]
    pub save_final: bool,
}

fn default_cache_dir() -> PathBuf {
    PathBuf::from("target/fluxnet_cache")
}

fn default_save_interval() -> usize {
    10 // Save every 10 temps
}

fn default_load_pretrained() -> bool {
    true
}

fn default_save_final() -> bool {
    true
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_cache_dir(),
            save_interval_temps: default_save_interval(),
            load_pretrained: default_load_pretrained(),
            pretrained_path: None,
            save_final: default_save_final(),
        }
    }
}

/// Complete FluxNet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxNetConfig {
    /// Enable FluxNet RL system (global toggle)
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Memory tier: determines buffer sizes and Q-table dimensions
    #[serde(default)]
    pub memory_tier: MemoryTier,

    /// Force profile initialization and update settings
    #[serde(default)]
    pub force_profile: ForceProfileConfig,

    /// RL controller hyperparameters
    #[serde(default)]
    pub rl: RLConfig,

    /// Persistence and caching settings
    #[serde(default)]
    pub persistence: PersistenceConfig,

    /// Verbose logging for FluxNet operations
    #[serde(default = "default_verbose")]
    pub verbose: bool,
}

fn default_enabled() -> bool {
    false // Disabled by default, opt-in
}

fn default_verbose() -> bool {
    false
}

impl Default for FluxNetConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            memory_tier: MemoryTier::default(),
            force_profile: ForceProfileConfig::default(),
            rl: RLConfig::default(),
            persistence: PersistenceConfig::default(),
            verbose: default_verbose(),
        }
    }
}

impl FluxNetConfig {
    /// Create a new FluxNet config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a compact tier config (8GB GPU)
    pub fn compact() -> Self {
        Self {
            enabled: true,
            memory_tier: MemoryTier::Compact,
            ..Default::default()
        }
    }

    /// Create an extended tier config (24GB+ GPU)
    pub fn extended() -> Self {
        Self {
            enabled: true,
            memory_tier: MemoryTier::Extended,
            ..Default::default()
        }
    }

    /// Load from TOML file
    pub fn from_toml_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: FluxNetConfig = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Load from TOML string
    pub fn from_toml_str(toml: &str) -> anyhow::Result<Self> {
        let config: FluxNetConfig = toml::from_str(toml)?;
        Ok(config)
    }

    /// Save to TOML file
    pub fn to_toml_file(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let toml = toml::to_string_pretty(self)?;
        std::fs::write(path, toml)?;
        Ok(())
    }

    /// Validate configuration values
    pub fn validate(&self) -> anyhow::Result<()> {
        // Validate force profile
        if self.force_profile.strong_multiplier < 1.0 || self.force_profile.strong_multiplier > 2.0
        {
            anyhow::bail!("strong_multiplier must be in range [1.0, 2.0]");
        }
        if self.force_profile.weak_multiplier < 0.5 || self.force_profile.weak_multiplier > 1.0 {
            anyhow::bail!("weak_multiplier must be in range [0.5, 1.0]");
        }
        if self.force_profile.force_clamp_min >= self.force_profile.force_clamp_max {
            anyhow::bail!("force_clamp_min must be < force_clamp_max");
        }

        // Validate RL hyperparameters
        if self.rl.learning_rate <= 0.0 || self.rl.learning_rate > 1.0 {
            anyhow::bail!("learning_rate must be in range (0.0, 1.0]");
        }
        if self.rl.discount_factor < 0.0 || self.rl.discount_factor > 1.0 {
            anyhow::bail!("discount_factor must be in range [0.0, 1.0]");
        }
        if self.rl.epsilon_start < 0.0 || self.rl.epsilon_start > 1.0 {
            anyhow::bail!("epsilon_start must be in range [0.0, 1.0]");
        }
        if self.rl.epsilon_min < 0.0 || self.rl.epsilon_min > self.rl.epsilon_start {
            anyhow::bail!("epsilon_min must be in range [0.0, epsilon_start]");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FluxNetConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.memory_tier, MemoryTier::Compact);
    }

    #[test]
    fn test_compact_config() {
        let config = FluxNetConfig::compact();
        assert!(config.enabled);
        assert_eq!(config.memory_tier, MemoryTier::Compact);
        assert_eq!(config.rl.get_replay_capacity(config.memory_tier), 1024);
        assert_eq!(config.rl.get_qtable_states(config.memory_tier), 256);
    }

    #[test]
    fn test_extended_config() {
        let config = FluxNetConfig::extended();
        assert!(config.enabled);
        assert_eq!(config.memory_tier, MemoryTier::Extended);
        assert_eq!(config.rl.get_replay_capacity(config.memory_tier), 16384);
        assert_eq!(config.rl.get_qtable_states(config.memory_tier), 1024);
    }

    #[test]
    fn test_toml_roundtrip() {
        let config = FluxNetConfig::compact();
        let toml = toml::to_string(&config).unwrap();
        let parsed: FluxNetConfig = toml::from_str(&toml).unwrap();

        assert_eq!(parsed.enabled, config.enabled);
        assert_eq!(parsed.memory_tier, config.memory_tier);
    }

    #[test]
    fn test_validation() {
        let mut config = FluxNetConfig::default();

        // Valid config
        assert!(config.validate().is_ok());

        // Invalid learning rate
        config.rl.learning_rate = 1.5;
        assert!(config.validate().is_err());
        config.rl.learning_rate = 0.1;

        // Invalid force multipliers
        config.force_profile.strong_multiplier = 3.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_memory_tier_overrides() {
        let mut config = FluxNetConfig::compact();

        // Default uses tier
        assert_eq!(config.rl.get_replay_capacity(config.memory_tier), 1024);

        // Override with custom value
        config.rl.replay_capacity = Some(2048);
        assert_eq!(config.rl.get_replay_capacity(config.memory_tier), 2048);
    }
}
