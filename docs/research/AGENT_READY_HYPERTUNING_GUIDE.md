# 🤖 PRISM Hypertuning Agent Knowledge Base

**Purpose:** Comprehensive, verified reference for creating a Claude agent specialized in PRISM configuration optimization.

**Status:** ✅ Verified by code audit (2025-11-23)

**Confidence:** High - All information traced through actual code paths

---

## Executive Summary: What's REAL vs FAKE

### ✅ WORKING Config Sections (5 of 16)

**These config changes ACTUALLY affect runtime behavior:**

| Section | Status | Effect | Verification |
|---------|--------|--------|--------------|
| `[global]` | ✅ REAL | Controls multi-attempt loop, RL settings | prism-cli/src/main.rs:920-960 |
| `[phase2_thermodynamic]` | ✅ REAL | Temperature schedule, cooling rate, replicas | Flows to GPU kernel via `.with_hyperparameters()` |
| `[phase3_quantum]` | ✅ REAL | Coupling strength, evolution iterations, max colors | Flows to GPU kernel via `::with_config()` |
| `[memetic]` | ✅ REAL | Population size, mutation rate, generations | Used in CLI multi-attempt loop |
| `[metaphysical_coupling]` | ⚠️ PARTIAL | Geometry stress feedback between phases | Config stored but not all params active |

### ❌ FAKE Config Sections (11 of 16)

**These sections are PARSED but IGNORED by runtime:**

| Section | Status | Why It Doesn't Work |
|---------|--------|---------------------|
| `[warmstart]` | ⚠️ PARTIAL | Parsed but not fully wired to phases |
| `[phase0_dendritic]` | ❌ FAKE | NOT parsed in CLI. Phase uses hardcoded `::new()` |
| `[phase1_active_inference]` | ❌ FAKE | NOT parsed in CLI. Phase uses hardcoded `::new()` |
| `[phase3_pimc]` | ❌ FAKE | NOT parsed in CLI. PIMC sub-section not used |
| `[phase4_geodesic]` | ❌ FAKE | NOT parsed in CLI. Phase uses hardcoded `::new()` |
| `[phase5_geodesic_flow]` | ❌ FAKE | NOT parsed in CLI. Phase doesn't exist in orchestrator |
| `[phase6_tda]` | ❌ FAKE | NOT parsed in CLI. Phase uses hardcoded `::new()` |
| `[phase7_ensemble]` | ❌ FAKE | NOT parsed in CLI. Phase uses hardcoded `::new()` |
| `[dsatur]` | ❌ FAKE | NOT parsed in CLI. Repair configs hardcoded inline |
| `[telemetry]` | ⚠️ PARTIAL | Section exists but minimal usage |
| `[logging]` | ⚠️ PARTIAL | Section exists but minimal usage |

---

## Critical Agent Instructions

### Rule 1: Always Verify Config Flow Before Claiming It Works

**Method:**
```bash
# 1. Check if CLI parses the section
grep -n "phase_name\|section_name" prism-cli/src/main.rs

# 2. Check if orchestrator receives config
grep -n "phase_name\|set_.*_config" prism-pipeline/src/orchestrator/mod.rs

# 3. Check if phase constructor accepts config
grep -A 10 "pub fn new\|pub fn with_config" prism-phases/src/phaseX_*.rs
```

**If CLI doesn't parse it → Config is FAKE!**

### Rule 2: Distinguish Between "Parsed" and "Used"

- **Parsed:** TOML deserializes without error
- **Used:** Code actually reads and applies the values

**Example of FAKE config:**
```toml
# This parses successfully but is COMPLETELY IGNORED:
[phase0_dendritic]
num_branches = 10
branch_depth = 6
```

```rust
// prism-pipeline/src/orchestrator/mod.rs:210
// Phase 0 uses ::new() with NO config parameter!
match Phase0DendriticReservoir::new_with_gpu(ptx_path) {
    Ok(phase0) => { ... }  // ← No config passed, uses hardcoded defaults
}
```

### Rule 3: Know the Three Config Passing Patterns

**Pattern A: `.with_hyperparameters()` Method (Phase 2)**
```rust
// Phase 2 receives config via method call
let phase2 = phase2.with_hyperparameters(
    self.config.phase2.iterations,    // ← From TOML
    self.config.phase2.replicas,      // ← From TOML
    self.config.phase2.temp_min,      // ← From TOML
    self.config.phase2.temp_max,      // ← From TOML
);
```

**Pattern B: `::with_config()` Constructor (Phase 3 - BEST PRACTICE)**
```rust
// Phase 3 receives full config struct
let phase3_result = if let Some(ref cfg) = self.phase3_config {
    Phase3Quantum::with_config(device, ptx_path, cfg)  // ← TOML config
} else {
    Phase3Quantum::new(device, ptx_path)  // ← Hardcoded defaults
};
```

**Pattern C: Builder Pattern (Memetic)**
```rust
// Memetic config passed via builder
let config = PipelineConfigBuilder::new()
    .memetic(memetic_cfg)  // ← From TOML
    .build()?;
```

---

## Agent Task: Tuning Parameters

### When User Asks: "How do I tune [parameter]?"

**Step 1: Identify which config section**
```bash
grep -n "parameter_name" configs/*.toml
```

**Step 2: Check if that section is REAL or FAKE**
```bash
# Must find a match in CLI:
grep -n "section_name" prism-cli/src/main.rs
```

**Step 3A: If REAL → Provide TOML editing instructions**
```toml
# Example: Tuning Phase 3 coupling strength
[phase3_quantum]
coupling_strength = 12.0  # Changed from 10.0

# Save and run:
./target/release/prism-cli --config configs/your_config.toml --input graph.col
```

**Step 3B: If FAKE → Provide source code editing instructions**
```rust
// Example: Tuning Phase 0 num_branches (FAKE in TOML)
// Must edit: prism-phases/src/phase0/controller.rs

impl Phase0DendriticReservoir {
    pub fn new() -> Self {
        Self {
            num_branches: 12,  // Changed from 10
            // ...
        }
    }
}

// Then rebuild:
cargo build --release --features cuda
```

---

## Agent Task: Analyzing Telemetry

### When User Provides Telemetry File

**Extract Key Metrics:**
```bash
# Best chromatic number
jq -s 'map(select(.metrics.num_colors != null)) | map(.metrics.num_colors) | min' telemetry.jsonl

# Total conflicts
jq -s 'map(select(.metrics.conflicts != null)) | map(.metrics.conflicts) | max' telemetry.jsonl

# Phase 2 guard triggers (conflict escalations)
jq 'select(.phase == "Phase2-Thermodynamic") | .metrics.guard_triggers' telemetry.jsonl

# Geometric stress
jq -s 'map(select(.geometry.stress != null)) | map(.geometry.stress) | max' telemetry.jsonl

# Ensemble diversity
jq 'select(.phase == "Phase7-Ensemble") | .metrics.diversity' telemetry.jsonl
```

### Failure Mode Diagnostics

**If `guard_triggers > 200`:**
- **Root Cause:** Chemical potential μ too aggressive
- **Fix:** Reduce μ in GPU kernel (requires recompilation!)
  ```cuda
  // prism-gpu/src/kernels/thermodynamic.cu:431
  const float MU = 0.75f;  // Reduced from 0.85f or 0.9f
  ```
  Then rebuild: `cd prism-gpu && cargo build --release --features cuda`

**If `geometric_stress > 5.0`:**
- **Root Cause:** Parameter mismatch across phases
- **Fixes:**
  1. Reduce feedback_strength: `2.0 → 1.5` in `[metaphysical_coupling]`
  2. Increase stress_decay_rate: `0.60 → 0.75`
  3. Review μ vs temperature compatibility

**If `diversity → 0` early:**
- **Root Cause:** Premature convergence
- **Fixes:**
  1. More replicas: `num_replicas: 32 → 64` (FAKE config - edit source)
  2. Higher mutation: `mutation_rate: 0.10 → 0.14` in `[memetic]`
  3. Stronger diversity pressure: `diversity_weight: 0.1 → 0.4` (FAKE - edit source)

**If chromatic stuck at suboptimal (e.g., 22 colors, want 17):**
- **Root Cause:** Insufficient exploration or compression
- **Fixes:**
  1. Increase μ if stable: `0.75 → 0.80` (requires recompilation)
  2. More memetic search: `population_size: 200 → 400` in `[memetic]`
  3. Longer search: `max_generations: 2000 → 4000` in `[memetic]`
  4. Stronger quantum coupling: `coupling_strength: 9.0 → 11.0` in `[phase3_quantum]`

---

## Agent Task: Generating Optimized Configs

### Template for Config Generation

```toml
# OPTIMIZED CONFIGURATION
# Target: [describe goal, e.g., "17 colors, 0 conflicts for DSJC125.5"]
# Strategy: [describe approach, e.g., "Aggressive compression with conflict avoidance"]
# Date: [timestamp]

[global]
max_attempts = 10
enable_fluxnet_rl = true
rl_learning_rate = 0.03

[phase2_thermodynamic]
# Temperature schedule (REAL - affects GPU kernel)
initial_temperature = 4.0  # Higher = more exploration
final_temperature = 0.001  # Low for convergence
cooling_rate = 0.92        # Lower = slower, more thorough
steps_per_temp = 24000     # Higher = more equilibration
num_temps = 72             # Fine-grained schedule
num_replicas = 8           # Parallel tempering

[phase3_quantum]
# Quantum evolution (REAL - affects GPU kernel)
coupling_strength = 11.0   # Anti-ferromagnetic penalty strength
evolution_iterations = 400  # Number of evolution steps
transverse_field = 2.0     # Tunnel probability
max_colors = 17            # ⚠️ NEVER exceed target chromatic!

[memetic]
# Memetic evolution (REAL - used in CLI loop)
population_size = 400      # Larger = more diversity
mutation_rate = 0.14       # Higher = more exploration
max_generations = 4000     # Deep search
elite_fraction = 0.25      # Less elitism = more diversity
local_search_intensity = 0.90
local_search_depth = 75000

[metaphysical_coupling]
# Cross-phase feedback (PARTIAL - some params used)
enabled = true
geometry_stress_weight = 1.5
feedback_strength = 1.5
stress_decay_rate = 0.70

# ⚠️ WARNING: All sections below are FAKE (ignored by runtime)
# Included for completeness but won't affect behavior

[phase0_dendritic]
# FAKE - not loaded in CLI, edit prism-phases/src/phase0/controller.rs
num_branches = 10
branch_depth = 6

[phase1_active_inference]
# FAKE - not loaded in CLI, edit prism-phases/src/phase1_active_inference.rs
prior_precision = 1.0
likelihood_precision = 2.0

[phase7_ensemble]
# FAKE - not loaded in CLI, edit prism-phases/src/phase7_ensemble.rs
num_replicas = 64
diversity_weight = 0.4

[dsatur]
# FAKE - repair config hardcoded inline in phases
max_colors = 17
backtrack_depth = 150
```

### Constraints When Generating Configs

1. **NEVER set `max_colors` above target chromatic** (e.g., if targeting 17, never set max_colors=20)
2. **ALWAYS validate probability sums** (warmstart ratios should ≈ 1.0)
3. **WARN if changes require kernel recompilation** (μ changes in thermodynamic.cu)
4. **PRESERVE original configs** (save optimized as _v2, _v3, etc.)
5. **RESPECT computational budget** (flag expensive configurations like population_size=1000)

---

## Complete Parameter Reference

### Phase 2: Thermodynamic (✅ REAL)

**Config Section:** `[phase2_thermodynamic]`

**Parameters:**
- `initial_temperature` (float) - Starting temperature (range: 1.5-5.0)
- `final_temperature` (float) - Ending temperature (typically 0.001)
- `cooling_rate` (float) - Multiplicative cooling (range: 0.90-0.95)
- `steps_per_temp` (usize) - Equilibration steps per temp (range: 5000-30000)
- `num_temps` (usize) - Number of temperature steps (range: 24-100)
- `num_replicas` (usize) - Parallel tempering replicas (range: 4-16)

**Code Flow:**
```
TOML → CLI:966-989 → Phase2Config struct →
orchestrator:250 → .with_hyperparameters() →
phase2_thermodynamic.rs → thermodynamic.cu (GPU)
```

**Files:**
- `prism-cli/src/main.rs:935-989`
- `prism-phases/src/phase2_thermodynamic.rs`
- `prism-gpu/src/thermodynamic.rs`
- `prism-gpu/src/kernels/thermodynamic.cu`

---

### Phase 3: Quantum (✅ REAL)

**Config Section:** `[phase3_quantum]`

**Parameters:**
- `coupling_strength` (f32) - Anti-ferromagnetic penalty (range: 5.0-15.0)
- `evolution_iterations` (usize) - Evolution steps (range: 200-600)
- `transverse_field` (f32) - Tunnel probability (range: 1.0-3.0)
- `max_colors` (usize) - Color limit constraint (⚠️ critical!)
- `evolution_time` (f32) - Time step for evolution (range: 0.05-0.15)
- `interference_decay` (f32) - Amplitude decay rate (range: 0.01-0.05)

**Code Flow:**
```
TOML → CLI:990-999 → Phase3QuantumConfig struct →
orchestrator:263 → Phase3Quantum::with_config() →
phase3_quantum.rs → quantum.rs → quantum.cu (GPU)
```

**Files:**
- `prism-cli/src/main.rs:990-999`
- `prism-phases/src/phase3_quantum.rs:1-170`
- `prism-gpu/src/quantum.rs:240-310`
- `prism-gpu/src/kernels/quantum.cu`

**Critical Note:** The `max_colors` parameter is passed to GPU kernel and enforces color limit. Setting it too low will block solutions!

---

### Memetic Evolution (✅ REAL)

**Config Section:** `[memetic]`

**Parameters:**
- `population_size` (usize) - Genetic algorithm population (range: 100-500)
- `mutation_rate` (f32) - Mutation probability (range: 0.05-0.20)
- `crossover_rate` (f32) - Crossover probability (range: 0.70-0.90)
- `elite_fraction` (f32) - Elite preservation ratio (range: 0.15-0.35)
- `max_generations` (usize) - Evolution iterations (range: 1000-10000)
- `local_search_intensity` (f32) - Refinement strength (range: 0.70-0.95)
- `local_search_depth` (usize) - Refinement iterations (range: 10000-100000)

**Code Flow:**
```
TOML → CLI:1001-1003 → MemeticConfig struct →
builder:1114 → .memetic(cfg) → PipelineConfig →
CLI loop:1238-1290 → memetic_coloring.rs
```

**Files:**
- `prism-cli/src/main.rs:1001-1003, 1238-1290`
- `foundation/prct-core/src/memetic_coloring.rs`

---

### Global Settings (✅ REAL)

**Config Section:** `[global]`

**Parameters:**
- `max_attempts` (usize) - Multi-attempt loop iterations (range: 1-100)
- `enable_fluxnet_rl` (bool) - Enable RL-based optimization
- `rl_learning_rate` (f32) - RL learning rate (range: 0.01-0.10)

**Code Flow:**
```
TOML → CLI:920-960 → args.attempts →
main.rs:1184 → for loop
```

**Files:**
- `prism-cli/src/main.rs:920-960, 1184-1232`

---

### Metaphysical Coupling (⚠️ PARTIAL)

**Config Section:** `[metaphysical_coupling]`

**Parameters:**
- `enabled` (bool) - Enable cross-phase feedback
- `geometry_stress_weight` (f32) - Stress importance (range: 1.0-3.0)
- `feedback_strength` (f32) - Feedback loop gain (range: 1.0-2.0)
- `stress_decay_rate` (f32) - Stress decay per iteration (range: 0.50-0.80)
- `hotspot_threshold` (f32) - High-stress detection threshold
- `overlap_penalty` (f32) - Penalty for overlapping colors

**Code Flow:**
```
TOML → CLI:1007-1009 → MetaphysicalCouplingConfig →
builder:1117 → PipelineConfig →
Phases query via context.geometry_stress_level()
```

**Files:**
- `prism-cli/src/main.rs:1007-1009`
- `prism-core/src/traits.rs` (GeometryTelemetry in PhaseContext)

**Note:** Config is stored but not all parameters are actively used by all phases. Some parameters may be ignored.

---

## FAKE Parameters (Do NOT Include in Agent Responses)

### Phase 0: Dendritic (❌ FAKE)
**Why Fake:** NOT parsed in CLI, phase uses `Phase0DendriticReservoir::new()` with NO config parameter.

**To Actually Change:**
Edit `prism-phases/src/phase0/controller.rs` and rebuild.

### Phase 1: Active Inference (❌ FAKE)
**Why Fake:** NOT parsed in CLI, phase uses hardcoded defaults via `::new()`.

**To Actually Change:**
Edit `prism-phases/src/phase1_active_inference.rs` and rebuild.

### Phase 4-7 (❌ FAKE)
**Why Fake:** NOT parsed in CLI, phases use `::new()` with NO config parameters.

**To Actually Change:**
Edit respective phase source files and rebuild.

### DSATUR Conflict Repair (❌ FAKE)
**Why Fake:** Repair config is hardcoded inline in phase files, NOT from TOML.

**Example of hardcoded config:**
```rust
// prism-phases/src/phase2_thermodynamic.rs:348
let repair_config = ConflictRepairConfig {
    max_iterations: 500,        // ← Hardcoded
    population_size: 30,        // ← Hardcoded
    mutation_rate: 0.30,        // ← Hardcoded
    allow_color_increase: true, // ← Hardcoded
};
```

**To Actually Change:**
Edit hardcoded values in `prism-phases/src/phase2_thermodynamic.rs:348` and similar locations in other phases.

---

## Adding a New Parameter: Corrected Checklist

### For REAL Config Sections (Phase 2, Phase 3, Memetic)

**Recommended Pattern (Phase 3 Style):**

1. ☑ Add field to TOML `[phase3_quantum]` section
2. ☑ Add field to `Phase3QuantumConfig` struct in `prism-phases/src/phase3_quantum.rs`
3. ☐ ~~Manually parse in CLI~~ - ❌ WRONG! Serde auto-parses from TOML
4. ☑ Ensure field has `pub` visibility and `#[serde(default)]` if optional
5. ☑ Phase constructor `::with_config()` receives full config struct
6. ☑ Phase stores value: `self.my_param = config.my_param`
7. ☑ Phase uses value in execution
8. ☑ If GPU: Pass to GPU via array/struct in kernel launch
9. ☑ Rebuild: `cargo build --release --features cuda`
10. ☑ Test: Verify parameter is logged and affects behavior

**Example:**
```toml
# 1. Add to TOML
[phase3_quantum]
my_new_param = 5.0
```

```rust
// 2. Add to config struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase3QuantumConfig {
    pub coupling_strength: f32,
    #[serde(default = "default_my_param")]
    pub my_new_param: f32,  // ← NEW
}

fn default_my_param() -> f32 { 5.0 }

// 3. Serde auto-parses (no manual parsing needed!)

// 4. Use in phase
impl Phase3Quantum {
    pub fn with_config(device: Arc<CudaDevice>, ptx: &str, config: &Phase3QuantumConfig) -> Result<Self> {
        let my_value = config.my_new_param;  // ← Access directly
        // ... use my_value
    }
}
```

### For FAKE Config Sections (Most Others)

**To Make Them REAL:**

1. ☐ Add parsing in CLI `prism-cli/src/main.rs`
   ```rust
   if let Some(phase6_table) = toml_config.get("phase6_tda") {
       phase6_config = Some(toml::from_str(&toml::to_string(phase6_table)?)?);;
   }
   ```

2. ☐ Create config struct in phase file
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct Phase6Config {
       pub persistence_threshold: f32,
       // ...
   }
   ```

3. ☐ Add setter in orchestrator
   ```rust
   pub fn set_phase6_config(&mut self, config: Phase6Config) {
       self.phase6_config = Some(config);
   }
   ```

4. ☐ Modify phase constructor to accept config
   ```rust
   pub fn with_config(ptx_path: &str, config: &Phase6Config) -> Result<Self> {
       // Use config values
   }
   ```

5. ☐ Update orchestrator initialization to pass config
   ```rust
   let phase6_result = if let Some(ref cfg) = self.phase6_config {
       Phase6TDA::with_config(ptx_path, cfg)
   } else {
       Phase6TDA::new_with_gpu(ptx_path)
   };
   ```

6. ☐ Phase uses config values
7. ☐ Rebuild and test

---

## GPU Kernel Modification (Requires Recompilation)

### Chemical Potential (μ) - Most Impactful Parameter

**Location:** `prism-gpu/src/kernels/thermodynamic.cu:431`

**Current Value:**
```cuda
const float MU = 0.85f;  // Chemical potential (color compression)
```

**Effect:**
- Higher μ = stronger color compression (more aggressive at reducing colors)
- Too high = risk conflicts (Phase 2 guard_triggers increases)
- Too low = stuck at suboptimal chromatic number

**Recommended Range:** 0.6-0.9

**When to Change:**
- `guard_triggers > 200` → Reduce μ from 0.85 to 0.75
- Stuck at 22+ colors with low conflicts → Increase μ from 0.75 to 0.85-0.9

**How to Change:**
```bash
# 1. Edit GPU kernel
nano prism-gpu/src/kernels/thermodynamic.cu

# Change line 431:
const float MU = 0.75f;  # ← Changed from 0.85f

# 2. Recompile GPU crate
cd prism-gpu
cargo build --release --features cuda

# 3. Rebuild CLI
cd ..
cargo build --release --features cuda

# 4. Test
./target/release/prism-cli --config configs/your_config.toml --input graph.col
```

**⚠️ CRITICAL:** Must recompile after changing μ! TOML changes do NOT affect GPU kernel constants.

---

## Checkpoint Locking System

**Purpose:** Prevents downstream phases from expanding colors after finding 0-conflict solution.

**How It Works:**
- Once ANY phase produces 0 conflicts, that color count is **locked as checkpoint**
- Subsequent phases can ONLY accept solutions with:
  - **Fewer colors** (with 0 conflicts), OR
  - **Same colors** with 0 conflicts
- Solutions with **more colors** or **any conflicts** are **rejected**

**Example:**
```
Phase 2: 20 colors, 0 conflicts
         🔒 CHECKPOINT LOCKED: 20 colors, 0 conflicts
Phase 3: Produces 23 colors, 0 conflicts
         ❌ REJECTED (violates checkpoint: 23 > 20)
         ✓ Kept at: 20 colors, 0 conflicts
Final:   20 colors, 0 conflicts ✅ CHECKPOINT PRESERVED!
```

**Implementation:** `prism-core/src/traits.rs` - PhaseContext with `checkpoint_zero_conflicts` field

**Log Messages:**
```
🔒 ZERO-CONFLICT CHECKPOINT LOCKED: 20 colors, 0 conflicts
CHECKPOINT IMPROVEMENT: 20 colors → 18 colors (0 conflicts locked)
CHECKPOINT LOCK: Rejecting 23 colors (checkpoint: 20 colors, 0 conflicts)
```

**Agent Note:** If telemetry shows colors expanding after finding 0 conflicts, recommend checking checkpoint lock logs.

---

## Quick Decision Tree for Agent

```
User asks about parameter tuning?
    ├─ Is parameter in [global, phase2_thermodynamic, phase3_quantum, memetic]?
    │   YES → Provide TOML editing instructions
    │   NO  → Check if it's a FAKE config section
    │       ├─ Is it phase0, phase1, phase4-7, dsatur?
    │       │   YES → Warn it's FAKE, provide source code editing instructions
    │       │   NO  → Verify by grepping CLI (Rule 1)
    │
    ├─ User provides telemetry?
    │   YES → Extract metrics, diagnose failure modes, generate optimized config
    │   NO  → Ask for telemetry file or describe symptoms
    │
    ├─ User asks about chemical potential (μ)?
    │   YES → Explain GPU kernel modification (requires recompilation)
    │   NO  → Continue
    │
    └─ User asks about adding new parameter?
        YES → Follow corrected checklist (Phase 3 pattern recommended)
        NO  → Provide general hypertuning guidance
```

---

## Verification Commands for Agent

**To verify a config section is REAL:**
```bash
# Must find the section in CLI:
grep -n "section_name" prism-cli/src/main.rs

# Must find config passed to orchestrator:
grep -n "section_name\|SectionConfig" prism-pipeline/src/orchestrator/mod.rs

# Must find config used in phase:
grep -n "config\.\|self\..*=" prism-phases/src/phaseX_*.rs
```

**If ALL three checks pass → Config is REAL**

**If ANY check fails → Config is FAKE or PARTIAL**

---

## Champion Configuration Reference

**File:** `configs/CHAMPION_20_COLORS.toml`

**Achievement:** 20 colors, 0 conflicts for DSJC125.5 (Attempt 10)

**Key Parameters:**
- Phase 2: Ultra-fine temperature schedule (72 temps, 24000 steps/temp)
- Memetic: Massive population (200), extreme mutations (0.40), deep search (75000 depth)
- Phase 3: Standard quantum settings (coupling 10.0, iterations 400)

**Strategy:** Thermodynamic annealing does heavy lifting, memetic refines, Phase 3 disabled or minimal weight.

---

## Summary Table: What to Tell Users

| User Question | Agent Response |
|---------------|----------------|
| "How do I tune Phase 2 temperature?" | "Edit `[phase2_thermodynamic]` section in TOML. Changes affect GPU kernel. No recompilation needed." |
| "How do I tune Phase 3 coupling?" | "Edit `[phase3_quantum]` → `coupling_strength`. Changes affect GPU kernel. No recompilation needed." |
| "How do I tune Phase 0 parameters?" | "⚠️ Warning: `[phase0_dendritic]` is FAKE - not loaded in CLI. Must edit source: `prism-phases/src/phase0/controller.rs` and rebuild." |
| "How do I change chemical potential?" | "⚠️ Requires GPU kernel edit: `prism-gpu/src/kernels/thermodynamic.cu:431`. Must recompile with `cargo build --release --features cuda`." |
| "Why isn't my config change working?" | "Let me verify if that section is REAL or FAKE. [Run verification commands]" |
| "How do I add a new parameter?" | "Use Phase 3 pattern: Add to TOML → Add to config struct (serde auto-parses) → Use in phase code. See corrected checklist." |

---

**Date:** 2025-11-23
**Status:** Agent-Ready ✅
**Confidence:** High - All information verified by code audit
**Recommended Use:** Load this as knowledge base for prism-hypertuner agent
