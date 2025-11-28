# 🔧 Complete Config Wire-Up Specification

**Goal:** Make ALL phases fully tunable via TOML configuration files

**Status:** Phase 2, Phase 3, Memetic, and Global are ✅ DONE. Need to wire up: Phase 0, 1, 4, 6, 7, and complete Metaphysical Coupling.

**Pattern to Follow:** Use Phase 3's `::with_config()` pattern (BEST PRACTICE)

---

## Success Criteria

For each phase, the following must be complete:

1. ✅ **Config struct created** in phase file with all parameters
2. ✅ **CLI parsing added** in `prism-cli/src/main.rs`
3. ✅ **Orchestrator setter** or builder method in `prism-pipeline/src/orchestrator/mod.rs`
4. ✅ **Phase constructor** accepts config via `::with_config()` method
5. ✅ **Phase uses values** from config (not hardcoded)
6. ✅ **Default values** defined with `#[serde(default)]`
7. ✅ **Documentation** updated in phase file and AGENT_READY_HYPERTUNING_GUIDE.md
8. ✅ **Build succeeds** with `cargo build --release --features cuda`
9. ✅ **Test run** shows config values in logs

---

## Phase 0: Dendritic Reservoir

### Current Status: ❌ HARDCODED

**Location:** `prism-phases/src/phase0/controller.rs`

**Current Implementation:**
```rust
// Line 69-80: Uses hardcoded defaults
pub fn new() -> Self {
    Self {
        gpu_reservoir: None,
        use_gpu: false,
        // ... all hardcoded
    }
}
```

### Parameters to Expose (from TOML `[phase0_dendritic]`)

Based on typical dendritic reservoir parameters:

```toml
[phase0_dendritic]
num_branches = 10              # Number of dendritic branches
branch_depth = 6               # Depth of each branch
learning_rate = 0.01           # Reservoir learning rate
plasticity = 0.05              # Synaptic plasticity coefficient
activation_threshold = 0.5     # Firing threshold
reservoir_size = 512           # Total reservoir neurons
readout_size = 128             # Readout layer size
gpu_enabled = true             # Enable GPU acceleration
```

### Implementation Checklist

#### Step 1: Create Config Struct

**File:** `prism-phases/src/phase0/controller.rs`

**Add before `Phase0DendriticReservoir` struct:**
```rust
use serde::{Deserialize, Serialize};

/// Configuration for Phase 0 Dendritic Reservoir
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase0Config {
    /// Number of dendritic branches per neuron
    #[serde(default = "default_num_branches")]
    pub num_branches: usize,

    /// Depth of each dendritic branch
    #[serde(default = "default_branch_depth")]
    pub branch_depth: usize,

    /// Learning rate for reservoir training
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,

    /// Synaptic plasticity coefficient
    #[serde(default = "default_plasticity")]
    pub plasticity: f32,

    /// Activation threshold for firing
    #[serde(default = "default_activation_threshold")]
    pub activation_threshold: f32,

    /// Total number of reservoir neurons
    #[serde(default = "default_reservoir_size")]
    pub reservoir_size: usize,

    /// Readout layer size
    #[serde(default = "default_readout_size")]
    pub readout_size: usize,

    /// Enable GPU acceleration
    #[serde(default = "default_gpu_enabled")]
    pub gpu_enabled: bool,
}

// Default value functions
fn default_num_branches() -> usize { 10 }
fn default_branch_depth() -> usize { 6 }
fn default_learning_rate() -> f32 { 0.01 }
fn default_plasticity() -> f32 { 0.05 }
fn default_activation_threshold() -> f32 { 0.5 }
fn default_reservoir_size() -> usize { 512 }
fn default_readout_size() -> usize { 128 }
fn default_gpu_enabled() -> bool { true }

impl Default for Phase0Config {
    fn default() -> Self {
        Self {
            num_branches: default_num_branches(),
            branch_depth: default_branch_depth(),
            learning_rate: default_learning_rate(),
            plasticity: default_plasticity(),
            activation_threshold: default_activation_threshold(),
            reservoir_size: default_reservoir_size(),
            readout_size: default_readout_size(),
            gpu_enabled: default_gpu_enabled(),
        }
    }
}
```

#### Step 2: Add `::with_config()` Constructor

**File:** `prism-phases/src/phase0/controller.rs`

**Add to `impl Phase0DendriticReservoir` block:**
```rust
/// Creates Phase0 controller with custom config
pub fn with_config(config: Phase0Config) -> Self {
    log::info!(
        "Phase0: Initializing with config: branches={}, depth={}, lr={:.3}",
        config.num_branches, config.branch_depth, config.learning_rate
    );

    Self {
        #[cfg(feature = "cuda")]
        gpu_reservoir: None,  // Will be initialized in new_with_gpu if needed
        use_gpu: config.gpu_enabled,
        num_branches: config.num_branches,
        branch_depth: config.branch_depth,
        learning_rate: config.learning_rate,
        plasticity: config.plasticity,
        activation_threshold: config.activation_threshold,
        reservoir_size: config.reservoir_size,
        readout_size: config.readout_size,
        last_difficulty: Vec::new(),
        last_uncertainty: Vec::new(),
        telemetry: None,
        last_execution_time_ms: 0.0,
        reservoir_iterations: 0,
        convergence_loss: 0.0,
    }
}

/// Creates Phase0 controller with config and GPU support
#[cfg(feature = "cuda")]
pub fn with_config_and_gpu(config: Phase0Config, ptx_path: &str) -> Result<Self, PrismError> {
    let mut phase = Self::with_config(config);
    if phase.use_gpu {
        match prism_gpu::DendriticReservoirGpu::new(ptx_path) {
            Ok(gpu) => {
                phase.gpu_reservoir = Some(gpu);
                log::info!("Phase0: GPU acceleration enabled");
            }
            Err(e) => {
                log::warn!("Phase0: GPU initialization failed: {}, using CPU", e);
                phase.use_gpu = false;
            }
        }
    }
    Ok(phase)
}
```

#### Step 3: Add Fields to Phase0DendriticReservoir Struct

**File:** `prism-phases/src/phase0/controller.rs`

**Add to struct definition:**
```rust
pub struct Phase0DendriticReservoir {
    // Existing fields...

    // NEW: Config fields
    num_branches: usize,
    branch_depth: usize,
    learning_rate: f32,
    plasticity: f32,
    activation_threshold: f32,
    reservoir_size: usize,
    readout_size: usize,
}
```

#### Step 4: CLI Parsing

**File:** `prism-cli/src/main.rs`

**Add after Phase 3 parsing (around line 1000):**
```rust
// Parse Phase 0 configuration
let mut phase0_config: Option<prism_phases::phase0::Phase0Config> = None;
if let Some(phase0_table) = toml_config.get("phase0_dendritic") {
    match toml::from_str::<prism_phases::phase0::Phase0Config>(&toml::to_string(phase0_table)?) {
        Ok(cfg) => {
            phase0_config = Some(cfg);
            log::info!("Phase 0 dendritic configuration loaded from TOML");
        }
        Err(e) => {
            log::warn!("Failed to parse phase0_dendritic config: {}, using defaults", e);
        }
    }
}
```

#### Step 5: Orchestrator Setter

**File:** `prism-pipeline/src/orchestrator/mod.rs`

**Add to `PipelineOrchestrator` struct:**
```rust
pub struct PipelineOrchestrator {
    // Existing fields...
    phase0_config: Option<prism_phases::phase0::Phase0Config>,
}
```

**Add setter method:**
```rust
impl PipelineOrchestrator {
    pub fn set_phase0_config(&mut self, config: prism_phases::phase0::Phase0Config) {
        self.phase0_config = Some(config);
    }
}
```

#### Step 6: Orchestrator Initialization

**File:** `prism-pipeline/src/orchestrator/mod.rs`

**Update `initialize_all_phases()` (around line 210):**
```rust
// Phase 0: Dendritic Reservoir
#[cfg(feature = "cuda")]
{
    let ptx_path = ptx_dir.join("dendritic_reservoir.ptx");
    let phase0 = if let Some(ref cfg) = self.phase0_config {
        log::info!("Phase 0: Initializing with custom TOML config");
        Phase0DendriticReservoir::with_config_and_gpu(cfg.clone(), ptx_path.to_str().unwrap())?
    } else {
        Phase0DendriticReservoir::new_with_gpu(ptx_path.to_str().unwrap())
    };
    self.phases.push(Box::new(phase0));
}

#[cfg(not(feature = "cuda"))]
{
    let phase0 = if let Some(ref cfg) = self.phase0_config {
        Phase0DendriticReservoir::with_config(cfg.clone())
    } else {
        Phase0DendriticReservoir::new()
    };
    self.phases.push(Box::new(phase0));
}
```

#### Step 7: CLI Integration

**File:** `prism-cli/src/main.rs`

**After creating orchestrator, add:**
```rust
// Apply Phase 0 config if loaded
if let Some(phase0_cfg) = phase0_config {
    orchestrator.set_phase0_config(phase0_cfg);
}
```

---

## Phase 1: Active Inference

### Current Status: ❌ HARDCODED

**Location:** `prism-phases/src/phase1_active_inference.rs`

### Parameters to Expose

```toml
[phase1_active_inference]
prior_precision = 1.0          # Prior belief precision
likelihood_precision = 2.0     # Likelihood precision
learning_rate = 0.001          # Learning rate for belief updates
free_energy_threshold = 0.01   # Convergence threshold
num_iterations = 1000          # Maximum iterations
hidden_states = 64             # Hidden state dimensionality
policy_depth = 3               # Policy planning depth
exploration_bonus = 0.1        # Exploration coefficient
gpu_enabled = true             # Enable GPU acceleration
```

### Implementation Checklist

(Follow same pattern as Phase 0)

1. ☐ Create `Phase1Config` struct with defaults
2. ☐ Add `::with_config()` and `::with_config_and_gpu()` constructors
3. ☐ Add config fields to `Phase1ActiveInference` struct
4. ☐ Parse in CLI (after line 1000)
5. ☐ Add `phase1_config` field to orchestrator
6. ☐ Add `set_phase1_config()` setter
7. ☐ Update orchestrator initialization (around line 230)
8. ☐ Apply config in CLI after creating orchestrator

---

## Phase 4: Geodesic Distance

### Current Status: ❌ HARDCODED

**Location:** `prism-phases/src/phase4_geodesic.rs`

### Parameters to Expose

```toml
[phase4_geodesic]
distance_threshold = 3.0       # Max geodesic distance for influence
centrality_weight = 1.0        # Betweenness centrality weight
diameter_penalty = 0.5         # Graph diameter penalty
use_betweenness = true         # Use betweenness centrality
use_closeness = false          # Use closeness centrality
use_eigenvector = false        # Use eigenvector centrality
gpu_enabled = true             # Enable GPU Floyd-Warshall
```

### Implementation Checklist

1. ☐ Create `Phase4Config` struct with defaults
2. ☐ Add `::with_config()` and `::with_config_and_gpu()` constructors
3. ☐ Add config fields to `Phase4Geodesic` struct
4. ☐ Parse in CLI
5. ☐ Add `phase4_config` field to orchestrator
6. ☐ Add `set_phase4_config()` setter
7. ☐ Update orchestrator initialization (around line 326)
8. ☐ Apply config in CLI

---

## Phase 6: Topological Data Analysis (TDA)

### Current Status: ❌ HARDCODED

**Location:** `prism-phases/src/phase6_tda.rs`

### Parameters to Expose

```toml
[phase6_tda]
persistence_threshold = 0.1    # Persistence diagram threshold
max_dimension = 2              # Maximum homology dimension
coherence_cv_threshold = 0.3   # Coherence coefficient of variation
vietoris_rips_radius = 2.0     # VR complex radius
num_landmarks = 100            # Number of landmark points
use_witness_complex = false    # Use witness complex instead of VR
gpu_enabled = true             # Enable GPU TDA kernels
```

### Implementation Checklist

1. ☐ Create `Phase6Config` struct with defaults
2. ☐ Add `::with_config()` and `::with_config_and_gpu()` constructors
3. ☐ Add config fields to `Phase6TDA` struct
4. ☐ Parse in CLI
5. ☐ Add `phase6_config` field to orchestrator
6. ☐ Add `set_phase6_config()` setter
7. ☐ Update orchestrator initialization (around line 348)
8. ☐ Apply config in CLI

---

## Phase 7: Ensemble Aggregation

### Current Status: ❌ HARDCODED (Except Memetic)

**Location:** `prism-phases/src/phase7_ensemble.rs`

### Parameters to Expose

**Note:** `[memetic]` section already works. Need to wire up ensemble-specific params.

```toml
[phase7_ensemble]
num_replicas = 64              # Number of ensemble replicas
exchange_interval = 10         # Replica exchange frequency
temperature_range = [0.1, 2.0] # Temperature ladder [min, max]
diversity_weight = 0.1         # Diversity preservation weight
consensus_threshold = 0.7      # Consensus agreement threshold
voting_method = "weighted"     # "majority", "weighted", or "ranked"
replica_selection = "best"     # "best", "diverse", or "all"
gpu_enabled = false            # GPU ensemble (future)
```

### Implementation Checklist

1. ☐ Create `Phase7Config` struct with defaults
2. ☐ Add `::with_config()` constructor
3. ☐ Add config fields to `Phase7Ensemble` struct
4. ☐ Parse in CLI
5. ☐ Add `phase7_config` field to orchestrator
6. ☐ Add `set_phase7_config()` setter
7. ☐ Update orchestrator initialization (around line 362)
8. ☐ Apply config in CLI

---

## Metaphysical Coupling: Complete Implementation

### Current Status: ⚠️ PARTIAL

**Locations:**
- `prism-cli/src/main.rs:1007-1009` (parsing exists)
- `prism-core/src/traits.rs` (GeometryTelemetry exists)
- `prism-pipeline/src/orchestrator/mod.rs:534-572` (partial usage)

### What's Missing

Currently only Phase 1 and RL read geometry metrics. Need to wire up:
- Phase 2: Use geometry stress to adjust temperature schedule
- Phase 3: Use geometry stress to adjust coupling strength
- Phase 7: Use geometry stress for diversity weighting

### Parameters Already Exposed

```toml
[metaphysical_coupling]
enabled = true
geometry_stress_weight = 2.0
feedback_strength = 1.2
hotspot_threshold = 5
stress_decay_rate = 0.6
overlap_penalty = 1.5
```

### Implementation Checklist

#### Phase 2 Integration

**File:** `prism-phases/src/phase2_thermodynamic.rs`

**In `execute()` method, before GPU call:**
```rust
// Adjust temperature based on geometry stress
let mut adjusted_temp_min = self.temp_min;
let mut adjusted_temp_max = self.temp_max;

if let Some(ref geom) = context.geometry_metrics {
    let stress_factor = geom.stress_scalar / 100.0; // Normalize
    if stress_factor > 0.5 {
        // High stress → increase temperature for more exploration
        adjusted_temp_max *= 1.0 + (stress_factor - 0.5);
        log::debug!(
            "Phase2: Geometry stress {:.2} → adjusted temp_max to {:.3}",
            geom.stress_scalar, adjusted_temp_max
        );
    }
}
```

#### Phase 3 Integration

**File:** `prism-phases/src/phase3_quantum.rs`

**In `execute()` method, before quantum evolution:**
```rust
// Adjust coupling based on geometry stress
let mut adjusted_coupling = self.coupling_strength;

if let Some(ref geom) = context.geometry_metrics {
    let stress_factor = geom.stress_scalar / 100.0;
    if stress_factor > 0.5 {
        // High stress → increase coupling to reduce conflicts
        adjusted_coupling *= 1.0 + (stress_factor - 0.5) * 0.5;
        log::debug!(
            "Phase3: Geometry stress {:.2} → adjusted coupling to {:.3}",
            geom.stress_scalar, adjusted_coupling
        );
    }
}
```

#### Phase 7 Integration

**File:** `prism-phases/src/phase7_ensemble.rs`

**In ensemble selection logic:**
```rust
// Adjust diversity weight based on geometry
let mut adjusted_diversity = self.diversity_weight;

if let Some(ref geom) = context.geometry_metrics {
    let hotspot_count = geom.anchor_hotspots.len();
    if hotspot_count > 10 {
        // Many hotspots → increase diversity to escape local minima
        adjusted_diversity *= 1.0 + (hotspot_count as f32 / 20.0);
        log::debug!(
            "Phase7: {} hotspots → adjusted diversity to {:.3}",
            hotspot_count, adjusted_diversity
        );
    }
}
```

---

## Testing Strategy

### For Each Phase

1. **Unit Test** - Verify config struct deserializes from TOML:
```rust
#[test]
fn test_phase0_config_deserialize() {
    let toml = r#"
        num_branches = 12
        branch_depth = 8
        learning_rate = 0.02
    "#;
    let config: Phase0Config = toml::from_str(toml).unwrap();
    assert_eq!(config.num_branches, 12);
    assert_eq!(config.branch_depth, 8);
}
```

2. **Integration Test** - Run with custom config:
```bash
# Create test config
cat > test_phase0.toml <<EOF
[phase0_dendritic]
num_branches = 12
branch_depth = 8
learning_rate = 0.02
EOF

# Run with config
./target/release/prism-cli \
    --config test_phase0.toml \
    --input benchmarks/dimacs/DSJC125.5.col \
    --attempts 1 2>&1 | grep "Phase0.*branches=12"
```

3. **Telemetry Verification** - Check telemetry shows custom values

---

## Agent Task Breakdown

### Task 1: Wire Up Phase 0 ⏱️ 30 min
- Create config struct
- Add constructors
- Wire CLI parsing
- Wire orchestrator
- Test

### Task 2: Wire Up Phase 1 ⏱️ 30 min
- Create config struct
- Add constructors
- Wire CLI parsing
- Wire orchestrator
- Test

### Task 3: Wire Up Phase 4 ⏱️ 30 min
- Create config struct
- Add constructors
- Wire CLI parsing
- Wire orchestrator
- Test

### Task 4: Wire Up Phase 6 ⏱️ 30 min
- Create config struct
- Add constructors
- Wire CLI parsing
- Wire orchestrator
- Test

### Task 5: Wire Up Phase 7 ⏱️ 30 min
- Create config struct
- Add constructors
- Wire CLI parsing
- Wire orchestrator
- Test

### Task 6: Complete Metaphysical Coupling ⏱️ 20 min
- Add Phase 2 geometry integration
- Add Phase 3 geometry integration
- Add Phase 7 geometry integration
- Test

### Task 7: Update Documentation ⏱️ 15 min
- Update AGENT_READY_HYPERTUNING_GUIDE.md
- Update VERIFIED_CONFIG_FLOW.md
- Add example configs

---

## Deliverables

1. ✅ All phases have config structs with defaults
2. ✅ All phases have `::with_config()` constructors
3. ✅ CLI parses all `[phaseX_*]` sections
4. ✅ Orchestrator wires all configs to phases
5. ✅ All phases use config values (not hardcoded)
6. ✅ Metaphysical coupling integrated in all phases
7. ✅ Build succeeds: `cargo build --release --features cuda`
8. ✅ Test configs provided in `configs/full_tunable_example.toml`
9. ✅ Documentation updated
10. ✅ Telemetry shows custom config values in logs

---

## Reference: Working Pattern (Phase 3)

**Config Struct:** `prism-core/src/types.rs:983-1019`
**CLI Parsing:** `prism-cli/src/main.rs:990-999`
**Orchestrator Field:** `prism-pipeline/src/orchestrator/mod.rs:74`
**Orchestrator Setter:** `prism-pipeline/src/orchestrator/mod.rs:82-84`
**Orchestrator Init:** `prism-pipeline/src/orchestrator/mod.rs:263-269`
**Phase Constructor:** `prism-phases/src/phase3_quantum.rs:145-170`

---

**Total Estimated Time:** ~3 hours for all tasks

**Priority Order:**
1. Phase 4 (most impactful - geometry producer)
2. Phase 1 (geometry consumer)
3. Phase 6 (geometry merger)
4. Phase 0 (foundational)
5. Phase 7 (ensemble)
6. Metaphysical coupling completion

**Date Created:** 2025-11-23
**Status:** Ready for Agent Execution
