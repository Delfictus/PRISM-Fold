# UltraFluxNetController Integration Guide

This guide explains how to integrate the UltraFluxNetController into the PRISM pipeline using the `IntegratedFluxNet` wrapper.

## Overview

The `IntegratedFluxNet` provides a unified interface that supports three controller modes:

1. **Universal Mode** - Legacy per-phase Q-tables (existing behavior)
2. **Ultra Mode** - New unified RuntimeConfig-based controller (recommended)
3. **Hybrid Mode** - Both controllers running in parallel (experimental)

## Architecture

```text
┌──────────────────────────────────────────────────┐
│           PipelineOrchestrator                   │
├──────────────────────────────────────────────────┤
│                       │                          │
│                       ▼                          │
│           IntegratedFluxNet (Facade)             │
│  ┌────────────────────────────────────────────┐  │
│  │  mode: Universal | Ultra | Hybrid          │  │
│  │  ┌──────────────┐    ┌──────────────────┐ │  │
│  │  │ Universal    │    │ Ultra            │ │  │
│  │  │ (per-phase)  │    │ (unified config) │ │  │
│  │  └──────────────┘    └──────────────────┘ │  │
│  └────────────────────────────────────────────┘  │
│                       │                          │
│                       ▼                          │
│              RuntimeConfig (GPU kernels)         │
└──────────────────────────────────────────────────┘
```

## Usage in Pipeline Orchestrator

### Option 1: Ultra Mode (Recommended)

```rust
use prism_fluxnet::IntegratedFluxNet;
use prism_pipeline::PipelineOrchestrator;

// Create integrated controller in Ultra mode
let fluxnet = IntegratedFluxNet::new_ultra();

// Pass to orchestrator (requires modification of PipelineOrchestrator::new)
// TODO: Update orchestrator to accept IntegratedFluxNet instead of UniversalRLController
```

### Option 2: Universal Mode (Legacy Compatibility)

```rust
use prism_fluxnet::{IntegratedFluxNet, RLConfig, UniversalRLController};

// Create legacy controller
let rl_config = RLConfig::default();
let universal = UniversalRLController::new(rl_config);

// Wrap in IntegratedFluxNet
let fluxnet = IntegratedFluxNet::new_universal(universal);
```

### Option 3: Hybrid Mode (Experimental)

```rust
use prism_fluxnet::{IntegratedFluxNet, RLConfig, UniversalRLController};

// Create both controllers
let rl_config = RLConfig::default();
let universal = UniversalRLController::new(rl_config);

// Hybrid mode runs both in parallel
let fluxnet = IntegratedFluxNet::new_hybrid(universal);
```

## Integration Steps

### 1. Modify PipelineOrchestrator Constructor

**Before:**
```rust
pub struct PipelineOrchestrator {
    rl_controller: UniversalRLController,
    // ...
}

impl PipelineOrchestrator {
    pub fn new(config: PipelineConfig, rl_controller: UniversalRLController) -> Self {
        Self {
            rl_controller,
            // ...
        }
    }
}
```

**After:**
```rust
use prism_fluxnet::IntegratedFluxNet;

pub struct PipelineOrchestrator {
    rl_controller: IntegratedFluxNet,
    // ...
}

impl PipelineOrchestrator {
    pub fn new(config: PipelineConfig, rl_controller: IntegratedFluxNet) -> Self {
        Self {
            rl_controller,
            // ...
        }
    }

    // Alternative: Add factory method for Ultra mode
    pub fn new_ultra(config: PipelineConfig) -> Self {
        Self {
            rl_controller: IntegratedFluxNet::new_ultra(),
            // ...
        }
    }
}
```

### 2. Update Action Selection in execute_phase_with_retry

**Before:**
```rust
// Get RL state before execution
let state_before = self.build_rl_state();

// Select and apply action from RL controller
let action = self.rl_controller.select_action(&state_before, phase_name);
self.apply_action(&action, phase_name);
```

**After (Ultra mode):**
```rust
// Get RL state and telemetry
let state_before = self.build_rl_state();
let telemetry = self.build_telemetry();

// Select action based on controller mode
match self.rl_controller.mode() {
    ControllerMode::Universal => {
        let action = self.rl_controller
            .select_action_universal(&state_before, &telemetry, phase_name)
            .unwrap();
        self.apply_action(&action, phase_name);
    }
    ControllerMode::Ultra | ControllerMode::Hybrid => {
        // Ultra mode: modify RuntimeConfig directly
        if let Some(action) = self.rl_controller.select_action_ultra(&telemetry) {
            self.rl_controller.apply_ultra_action(action);

            // Update PhaseContext with new config
            let updated_config = self.rl_controller.get_config();
            self.context.update_runtime_config(updated_config);
        }

        // Hybrid mode: also apply Universal action
        if self.rl_controller.mode() == ControllerMode::Hybrid {
            if let Some(action) = self.rl_controller
                .select_action_universal(&state_before, &telemetry, phase_name)
            {
                self.apply_action(&action, phase_name);
            }
        }
    }
}
```

### 3. Update Q-Table Updates

**Before:**
```rust
self.rl_controller.update_qtable(
    &state_before,
    &action,
    reward,
    &state_after,
    phase_name,
);
```

**After:**
```rust
let telemetry = self.build_telemetry();

self.rl_controller.update(
    &state_before,
    &action,
    reward,
    &state_after,
    &telemetry,
    phase_name,
);
```

### 4. Build Telemetry Helper

Add a helper method to construct KernelTelemetry from current state:

```rust
impl PipelineOrchestrator {
    fn build_telemetry(&self) -> KernelTelemetry {
        let solution = &self.context.best_solution;

        KernelTelemetry {
            conflicts: solution.conflicts as i32,
            colors_used: solution.num_colors as i32,
            phase_transitions: 0, // Set based on recent transitions
            moves_applied: 0,     // Track from kernel execution
            ..Default::default()
        }
    }
}
```

## Ultra Controller Benefits

1. **Unified Configuration**: Single RuntimeConfig struct for all phases
2. **Simpler Action Space**: 11 discrete actions vs 104 in Universal
3. **Direct GPU Integration**: Actions modify RuntimeConfig that GPU kernels use
4. **Better Exploration**: Epsilon-greedy with decay on unified parameter space
5. **Transfer Learning**: Q-table transfers across different graph instances

## Action Space Comparison

### Universal Actions (104 total)
- Per-phase actions (8 phases × 7 actions = 56)
- Warmstart actions (8)
- Memetic actions (8)
- Geometry actions (8)
- MEC actions (8)
- CMA actions (8)
- NoOp (1)

### Ultra Actions (11 total)
- IncreaseChemicalPotential / DecreaseChemicalPotential
- IncreaseTunneling / DecreaseTunneling
- IncreaseTemperature / DecreaseTemperature
- BoostReservoir / ReduceReservoir
- EnableTransitionResponse / DisableTransitionResponse
- NoOp

## RuntimeConfig Parameters Controlled

The Ultra controller modifies these RuntimeConfig fields:

- `global_temperature` - Simulated annealing temperature
- `chemical_potential` - Chemical potential for phase transitions
- `tunneling_prob_base` - Base tunneling probability
- `tunneling_prob_boost` - Tunneling boost multiplier
- `reservoir_leak_rate` - Dendritic reservoir leak rate

## State Discretization

Ultra controller uses 5 features for state buckets:

1. **Conflict bucket** (0-3): 0-10, 11-50, 51-200, 201+
2. **Color bucket** (0-3): 0-20, 21-40, 41-60, 61+
3. **Temperature bucket** (0-3): <0.1, 0.1-1.0, 1.0-10.0, 10.0+
4. **Transition active** (bool): Phase transitions detected
5. **Stagnation bucket** (0-3): 0-10, 11-50, 51-100, 101+ iterations

State space size: 4 × 4 × 4 × 2 × 4 = 512 states

## Reward Function

```rust
reward = 0.0;

// Primary: conflict reduction
if conflicts == 0 {
    reward += 100.0;  // Big bonus for zero conflicts
} else {
    reward -= conflicts * 0.1;
}

// Secondary: color efficiency
reward -= colors_used * 0.01;

// Bonus for active optimization
reward += moves_applied * 0.001;

// Penalty for stagnation (>100 iterations without improvement)
if stagnation_counter > 100 {
    reward -= 10.0;
}
```

## Saving/Loading Q-Tables

```rust
// Save controller state
controller.save("checkpoints/fluxnet")?;
// Creates: checkpoints/fluxnet.ultra.bin

// Load controller state
controller.load("checkpoints/fluxnet")?;
```

## Example: CLI Integration

```rust
// prism-cli/src/main.rs

use prism_fluxnet::{ControllerMode, IntegratedFluxNet};

#[derive(clap::Parser)]
struct Cli {
    #[arg(long, default_value = "ultra")]
    controller_mode: String,

    #[arg(long)]
    load_qtable: Option<String>,

    // ... other args
}

fn main() -> Result<()> {
    let args = Cli::parse();

    // Create controller based on mode
    let mut fluxnet = match args.controller_mode.as_str() {
        "ultra" => IntegratedFluxNet::new_ultra(),
        "universal" => {
            let config = RLConfig::default();
            let universal = UniversalRLController::new(config);
            IntegratedFluxNet::new_universal(universal)
        }
        "hybrid" => {
            let config = RLConfig::default();
            let universal = UniversalRLController::new(config);
            IntegratedFluxNet::new_hybrid(universal)
        }
        _ => panic!("Invalid controller mode"),
    };

    // Load pre-trained Q-table if specified
    if let Some(path) = args.load_qtable {
        fluxnet.load(&path)?;
        println!("Loaded Q-table from {}", path);
    }

    // Create orchestrator
    let orchestrator = PipelineOrchestrator::new(config, fluxnet);

    // Run pipeline...
}
```

## Performance Considerations

### Ultra Mode
- **Pros**: Faster action selection (11 actions vs 104), simpler Q-table, direct GPU integration
- **Cons**: Less phase-specific guidance, coarser control

### Universal Mode
- **Pros**: Fine-grained phase-specific control, proven in existing implementation
- **Cons**: Larger Q-table, more complex action space

### Hybrid Mode
- **Pros**: Best of both worlds - global optimization + phase-specific tactics
- **Cons**: Higher computational cost, more complex training

## Migration Path

1. **Phase 1**: Test Ultra mode on simple graphs (DIMACS 125, 250 nodes)
2. **Phase 2**: Compare performance against Universal mode on standard benchmarks
3. **Phase 3**: If Ultra performs well, make it the default for new pipelines
4. **Phase 4**: Keep Universal mode as fallback for complex cases

## Future Enhancements

1. **Automatic Mode Selection**: Choose mode based on graph properties
2. **Dynamic Mode Switching**: Start with Universal, switch to Ultra once converged
3. **Multi-Armed Bandit**: Meta-RL to select between modes
4. **MBRL Integration**: Connect UltraFluxNetController to DynaFluxNet world model

## References

- UltraFluxNetController: `crates/prism-fluxnet/src/ultra_controller.rs`
- Integration Layer: `crates/prism-fluxnet/src/integration.rs`
- Example Usage: `crates/prism-fluxnet/examples/ultra_integration.rs`
- Pipeline Orchestrator: `crates/prism-pipeline/src/orchestrator/mod.rs`
