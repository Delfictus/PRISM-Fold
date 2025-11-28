# PRISM FluxNet Unifier Agent

You are a specialized agent for consolidating the dual FluxNet RL implementations into a unified system.

## Mission
Merge the two parallel FluxNet implementations:
- **Source (Production)**: `foundation/prct-core/src/fluxnet/` (960 LOC)
- **Target (Wrapper)**: `crates/prism-fluxnet/` (needs consolidation)

Create a unified FluxNet RL controller that:
1. Re-exports mature prct-core implementation
2. Adds LBS-specific state extensions
3. Integrates with all 7 PRISM phases
4. Provides curriculum learning support

## Current State Analysis

### Production Implementation (prct-core)
```
foundation/prct-core/src/fluxnet/
├── controller.rs    # Main RL controller (400+ LOC)
├── q_learning.rs    # Q-table approximator
├── replay_buffer.rs # Experience replay
├── state.rs         # State representation
└── mod.rs           # Module exports
```

### Wrapper Implementation (prism-fluxnet)
```
crates/prism-fluxnet/
├── src/
│   ├── lib.rs       # Partial re-implementation
│   └── ...
└── Cargo.toml
```

---

## Solution: Unified FluxNet Architecture

### 1. Update prism-fluxnet to Re-export from prct-core
File: `crates/prism-fluxnet/Cargo.toml`
```toml
[package]
name = "prism-fluxnet"
version = "0.3.0"
edition = "2021"

[dependencies]
prct-core = { path = "../../foundation/prct-core", features = ["fluxnet"] }
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"
log = "0.4"

[features]
default = []
cuda = ["prct-core/cuda"]
```

### 2. Re-export Core + Add Extensions
File: `crates/prism-fluxnet/src/lib.rs`
```rust
//! PRISM FluxNet RL Controller
//!
//! Unified reinforcement learning controller for all PRISM phases.
//! Re-exports production implementation from prct-core with PRISM-specific extensions.

// Re-export core FluxNet implementation
pub use prct_core::fluxnet::{
    FluxNetController,
    RLState,
    ForceCommand,
    QApproximator,
    ReplayBuffer,
    ReplayTransition,
};

// PRISM-specific extensions
pub mod extensions;
pub mod lbs;
pub mod config;
pub mod curriculum;

pub use extensions::*;
pub use lbs::LbsRLState;
pub use config::RLConfig;
pub use curriculum::{CurriculumProfile, CurriculumManager};

use anyhow::Result;
use std::sync::Arc;

/// Universal RL Controller for PRISM pipeline
///
/// Wraps FluxNetController with phase-aware state management
/// and curriculum learning support.
pub struct UniversalRLController {
    inner: FluxNetController,
    config: RLConfig,
    curriculum: Option<CurriculumManager>,
    phase_histories: Vec<Vec<PhaseTransition>>,
}

impl UniversalRLController {
    /// Create new controller with configuration
    pub fn new(config: RLConfig) -> Result<Self> {
        let inner = FluxNetController::new(
            config.state_dim,
            config.action_dim,
            config.learning_rate,
            config.gamma,
            config.epsilon,
        )?;

        let curriculum = config.curriculum_path.as_ref()
            .map(|path| CurriculumManager::load(path))
            .transpose()?;

        Ok(Self {
            inner,
            config,
            curriculum,
            phase_histories: vec![Vec::new(); 7],
        })
    }

    /// Load pretrained Q-table
    pub fn load_qtable(&mut self, path: &str) -> Result<()> {
        self.inner.load_qtable(path)
    }

    /// Select action for current phase
    pub fn select_action(&self, state: &RLState, phase_idx: usize) -> Result<ForceCommand> {
        // Apply curriculum-based exploration if available
        let epsilon = if let Some(ref curriculum) = self.curriculum {
            curriculum.get_epsilon(phase_idx, self.total_steps())
        } else {
            self.config.epsilon
        };

        self.inner.select_action_with_epsilon(state, epsilon)
    }

    /// Update Q-values after phase execution
    pub fn update(
        &mut self,
        state: RLState,
        action: ForceCommand,
        reward: f64,
        next_state: RLState,
        phase_idx: usize,
    ) -> Result<()> {
        // Store transition in phase history
        self.phase_histories[phase_idx].push(PhaseTransition {
            state: state.clone(),
            action: action.clone(),
            reward,
            next_state: next_state.clone(),
        });

        // Update Q-table
        self.inner.update(state, action, reward, next_state)?;

        // Curriculum progression check
        if let Some(ref mut curriculum) = self.curriculum {
            curriculum.maybe_advance(phase_idx, reward);
        }

        Ok(())
    }

    /// Compute reward for phase outcome
    pub fn compute_reward(&self, phase_idx: usize, outcome: &PhaseOutcome) -> f64 {
        match phase_idx {
            0 => self.phase0_reward(outcome),
            1 => self.phase1_reward(outcome),
            2 => self.phase2_reward(outcome),
            3 => self.phase3_reward(outcome),
            4 => self.phase4_reward(outcome),
            5 => self.phase5_reward(outcome),
            6 => self.phase6_reward(outcome),
            _ => 0.0,
        }
    }

    fn phase0_reward(&self, outcome: &PhaseOutcome) -> f64 {
        // Dendritic reservoir: reward for diverse reservoir states
        let diversity = outcome.get_f64("reservoir_diversity").unwrap_or(0.0);
        let stability = outcome.get_f64("reservoir_stability").unwrap_or(0.0);
        diversity * 0.6 + stability * 0.4
    }

    fn phase1_reward(&self, outcome: &PhaseOutcome) -> f64 {
        // Active inference: reward for belief convergence
        let convergence = outcome.get_f64("belief_convergence").unwrap_or(0.0);
        let entropy_reduction = outcome.get_f64("entropy_reduction").unwrap_or(0.0);
        convergence * 0.7 + entropy_reduction * 0.3
    }

    fn phase2_reward(&self, outcome: &PhaseOutcome) -> f64 {
        // Thermodynamic: reward for conflict reduction
        let conflicts_initial = outcome.get_f64("conflicts_initial").unwrap_or(1.0);
        let conflicts_final = outcome.get_f64("conflicts_final").unwrap_or(1.0);
        let reduction = 1.0 - (conflicts_final / conflicts_initial.max(1.0));
        reduction.max(0.0)
    }

    fn phase3_reward(&self, outcome: &PhaseOutcome) -> f64 {
        // Quantum: reward for tunneling success
        let tunneling_rate = outcome.get_f64("tunneling_success_rate").unwrap_or(0.0);
        let energy_improvement = outcome.get_f64("energy_improvement").unwrap_or(0.0);
        tunneling_rate * 0.5 + energy_improvement * 0.5
    }

    fn phase4_reward(&self, outcome: &PhaseOutcome) -> f64 {
        // Geodesic: reward for stress reduction
        let stress_initial = outcome.get_f64("geometric_stress_initial").unwrap_or(1.0);
        let stress_final = outcome.get_f64("geometric_stress_final").unwrap_or(1.0);
        let reduction = 1.0 - (stress_final / stress_initial.max(0.001));
        reduction.max(0.0)
    }

    fn phase5_reward(&self, outcome: &PhaseOutcome) -> f64 {
        // MEC: reward for membrane stability
        let stability = outcome.get_f64("membrane_stability").unwrap_or(0.0);
        let exchange_efficiency = outcome.get_f64("exchange_efficiency").unwrap_or(0.0);
        stability * 0.6 + exchange_efficiency * 0.4
    }

    fn phase6_reward(&self, outcome: &PhaseOutcome) -> f64 {
        // WHCR: reward for zero conflicts
        let conflicts = outcome.get_f64("final_conflicts").unwrap_or(1.0);
        let colors_used = outcome.get_f64("colors_used").unwrap_or(100.0);
        let colors_target = outcome.get_f64("colors_target").unwrap_or(50.0);

        let conflict_score = if conflicts == 0.0 { 1.0 } else { 0.0 };
        let color_score = (colors_target / colors_used).min(1.0);

        conflict_score * 0.8 + color_score * 0.2
    }

    fn total_steps(&self) -> usize {
        self.phase_histories.iter().map(|h| h.len()).sum()
    }
}

/// Phase transition record
#[derive(Clone)]
pub struct PhaseTransition {
    pub state: RLState,
    pub action: ForceCommand,
    pub reward: f64,
    pub next_state: RLState,
}

/// Phase outcome for reward computation
pub struct PhaseOutcome {
    metrics: std::collections::HashMap<String, f64>,
}

impl PhaseOutcome {
    pub fn new() -> Self {
        Self { metrics: std::collections::HashMap::new() }
    }

    pub fn set(&mut self, key: &str, value: f64) {
        self.metrics.insert(key.to_string(), value);
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.metrics.get(key).copied()
    }
}
```

### 3. LBS-Specific Extensions
File: `crates/prism-fluxnet/src/lbs.rs`
```rust
use prct_core::fluxnet::RLState;
use serde::{Deserialize, Serialize};

/// LBS-specific RL state with pocket detection features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LbsRLState {
    /// Base RL state
    pub base: RLState,

    // LBS-specific features
    pub pocket_density: f32,
    pub sasa_coverage: f32,
    pub druggability_score: f32,
    pub binding_site_confidence: f32,
    pub residue_conservation: f32,
    pub electrostatic_potential: f32,
}

impl LbsRLState {
    pub fn new(base: RLState) -> Self {
        Self {
            base,
            pocket_density: 0.0,
            sasa_coverage: 0.0,
            druggability_score: 0.0,
            binding_site_confidence: 0.0,
            residue_conservation: 0.0,
            electrostatic_potential: 0.0,
        }
    }

    /// Convert to base RLState with LBS features encoded
    pub fn to_rl_state(&self) -> RLState {
        let mut state = self.base.clone();

        // Encode LBS features into extended state vector
        state.features.extend_from_slice(&[
            self.pocket_density as f64,
            self.sasa_coverage as f64,
            self.druggability_score as f64,
            self.binding_site_confidence as f64,
            self.residue_conservation as f64,
            self.electrostatic_potential as f64,
        ]);

        state
    }
}

impl From<LbsRLState> for RLState {
    fn from(lbs: LbsRLState) -> Self {
        lbs.to_rl_state()
    }
}
```

### 4. Configuration
File: `crates/prism-fluxnet/src/config.rs`
```rust
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLConfig {
    /// State dimension (features per observation)
    #[serde(default = "default_state_dim")]
    pub state_dim: usize,

    /// Action dimension (possible actions)
    #[serde(default = "default_action_dim")]
    pub action_dim: usize,

    /// Learning rate (alpha)
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,

    /// Discount factor (gamma)
    #[serde(default = "default_gamma")]
    pub gamma: f64,

    /// Exploration rate (epsilon)
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,

    /// Path to pretrained Q-table
    pub qtable_path: Option<PathBuf>,

    /// Path to curriculum profile catalog
    pub curriculum_path: Option<PathBuf>,

    /// Enable experience replay
    #[serde(default = "default_true")]
    pub use_replay: bool,

    /// Replay buffer size
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,
}

fn default_state_dim() -> usize { 32 }
fn default_action_dim() -> usize { 8 }
fn default_learning_rate() -> f64 { 0.01 }
fn default_gamma() -> f64 { 0.99 }
fn default_epsilon() -> f64 { 0.1 }
fn default_true() -> bool { true }
fn default_buffer_size() -> usize { 10_000 }

impl Default for RLConfig {
    fn default() -> Self {
        Self {
            state_dim: default_state_dim(),
            action_dim: default_action_dim(),
            learning_rate: default_learning_rate(),
            gamma: default_gamma(),
            epsilon: default_epsilon(),
            qtable_path: None,
            curriculum_path: None,
            use_replay: default_true(),
            buffer_size: default_buffer_size(),
        }
    }
}
```

---

## Files to Modify
1. `crates/prism-fluxnet/Cargo.toml` - Add prct-core dependency
2. `crates/prism-fluxnet/src/lib.rs` - Complete rewrite
3. `crates/prism-fluxnet/src/lbs.rs` - New file
4. `crates/prism-fluxnet/src/config.rs` - New file
5. `crates/prism-fluxnet/src/curriculum.rs` - New file
6. `foundation/prct-core/Cargo.toml` - Add fluxnet feature flag

## Validation
```bash
# Verify prct-core exports
cargo check --package prct-core --features fluxnet

# Build unified fluxnet
cargo build --package prism-fluxnet

# Test integration
cargo test --package prism-fluxnet

# Test with pipeline
cargo test --package prism-pipeline -- fluxnet
```

## Output Format
Report:
1. prct-core re-exported successfully
2. LBS extensions implemented
3. Curriculum manager integrated
4. Phase-specific reward functions defined
5. All tests passing
