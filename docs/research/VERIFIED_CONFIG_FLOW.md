# ✅ VERIFIED Configuration Flow: What's REAL vs FAKE

## Executive Summary

After auditing the actual codebase, I found that **ONLY 5 of 16 config sections actually work**.

The rest are **ignored** - parsed by TOML but never used by the runtime.

---

## ✅ REAL Config Sections (Actually Used)

| Section | Status | How It Works | Files Involved |
|---------|--------|--------------|----------------|
| `[global]` | ✅ REAL | CLI reads into args, controls multi-attempt loop | `prism-cli/src/main.rs:920-960` |
| `[phase2_thermodynamic]` | ✅ REAL | Loaded into `self.config.phase2`, applied via `.with_hyperparameters()` | CLI → `PipelineConfig` → orchestrator → `.with_hyperparameters()` |
| `[phase3_quantum]` | ✅ REAL | Loaded into `self.phase3_config`, passed to `::with_config()` constructor | CLI → `phase3_config` var → orchestrator → `Phase3Quantum::with_config()` |
| `[memetic]` | ✅ REAL | Loaded via builder pattern, used in CLI multi-attempt loop | CLI → builder → `PipelineConfig.memetic` → used in attempt loop |
| `[metaphysical_coupling]` | ✅ REAL | Loaded via builder pattern, stored in `PipelineConfig` | CLI → builder → `PipelineConfig.metaphysical_coupling` |

---

## ❌ FAKE Config Sections (Ignored)

| Section | Status | Why It's Fake |
|---------|--------|---------------|
| `[warmstart]` | ⚠️ PARTIALLY FAKE | Parsed but not fully wired to phases |
| `[phase0_dendritic]` | ❌ FAKE | **NOT parsed in CLI**. Phase uses hardcoded defaults via `::new()` |
| `[phase1_active_inference]` | ❌ FAKE | **NOT parsed in CLI**. Phase uses hardcoded defaults via `::new()` |
| `[phase3_pimc]` | ❌ FAKE | **NOT parsed in CLI**. PIMC sub-section not used |
| `[phase4_geodesic]` | ❌ FAKE | **NOT parsed in CLI**. Phase uses hardcoded defaults via `::new()` |
| `[phase5_geodesic_flow]` | ❌ FAKE | **NOT parsed in CLI**. Phase doesn't exist in orchestrator |
| `[phase6_tda]` | ❌ FAKE | **NOT parsed in CLI**. Phase uses hardcoded defaults via `::new()` |
| `[phase7_ensemble]` | ❌ FAKE | **NOT parsed in CLI**. Phase uses hardcoded defaults via `::new()` |
| `[dsatur]` | ❌ FAKE | **NOT parsed in CLI**. Repair configs are hardcoded inline in phases |
| `[telemetry]` | ⚠️ PARTIALLY FAKE | Section exists but minimal usage |
| `[logging]` | ⚠️ PARTIALLY FAKE | Section exists but minimal usage |

---

## Detailed Analysis: How Each REAL Config Works

### 1. [global] - ✅ FULLY FUNCTIONAL

**TOML:**
```toml
[global]
max_attempts = 10
enable_fluxnet_rl = true
rl_learning_rate = 0.03
```

**Code Flow:**
```
TOML → CLI args override → multi-attempt loop in main.rs:1184
```

**Files:**
- `prism-cli/src/main.rs:920-960` - Reads global settings
- `prism-cli/src/main.rs:1184-1232` - Uses `args.attempts` for loop

**Verification:**
```rust
// Line 1184
for attempt in 1..=args.attempts {  // ← Uses args.attempts from global config
```

---

### 2. [phase2_thermodynamic] - ✅ FULLY FUNCTIONAL

**TOML:**
```toml
[phase2_thermodynamic]
initial_temperature = 1.5
cooling_rate = 0.95
steps_per_temp = 24000
num_temps = 72
```

**Code Flow:**
```
TOML → Parse in CLI:966-989 → Store in Phase2Config
     → Pass via builder:1107 → self.config.phase2
     → orchestrator:250 → .with_hyperparameters(iterations, replicas, temp_min, temp_max)
     → Phase2Thermodynamic stores values
     → GPU kernel uses values
```

**Files:**
- `prism-cli/src/main.rs:935-989` - Creates `Phase2Config`, parses TOML overrides
- `prism-cli/src/main.rs:1107` - `.phase2(phase2_config)` via builder
- `prism-pipeline/src/orchestrator/mod.rs:240-260` - Applies via `.with_hyperparameters()`
- `prism-phases/src/phase2_thermodynamic.rs` - Stores and uses values
- `prism-gpu/src/thermodynamic.rs` - Passes to GPU kernel

**Verification:**
```rust
// orchestrator.rs:250
let phase2 = phase2.with_hyperparameters(
    self.config.phase2.iterations,  // ← From TOML
    self.config.phase2.replicas,    // ← From TOML
    self.config.phase2.temp_min,    // ← From TOML
    self.config.phase2.temp_max,    // ← From TOML
);
```

---

### 3. [phase3_quantum] - ✅ FULLY FUNCTIONAL

**TOML:**
```toml
[phase3_quantum]
coupling_strength = 10.0
evolution_iterations = 400
transverse_field = 2.0
max_colors = 17
```

**Code Flow:**
```
TOML → Parse in CLI:990-992 → Store in phase3_config variable
     → Pass via setter:1149 → self.phase3_config in orchestrator
     → orchestrator:263 → Phase3Quantum::with_config(device, ptx, config)
     → Phase3Quantum stores config
     → GPU quantum.rs uses config values
     → GPU kernel receives via d_couplings array
```

**Files:**
- `prism-cli/src/main.rs:990-999` - Parses `[phase3_quantum]` section
- `prism-cli/src/main.rs:1149` - `orchestrator.set_phase3_config(cfg)`
- `prism-pipeline/src/orchestrator/mod.rs:74-76` - Stores in `self.phase3_config`
- `prism-pipeline/src/orchestrator/mod.rs:263-269` - Uses `::with_config()` constructor
- `prism-phases/src/phase3_quantum.rs:145-170` - `::with_config()` implementation
- `prism-gpu/src/quantum.rs:273` - Creates `couplings` array from config
- `prism-gpu/src/kernels/quantum.cu` - GPU kernel uses values

**Verification:**
```rust
// orchestrator.rs:263-269
let phase3_result = if let Some(ref cfg) = self.phase3_config {
    log::info!("Phase 3: Initializing with custom TOML config");
    Phase3Quantum::with_config(device, ptx_path, cfg)  // ← Uses TOML config
```

---

### 4. [memetic] - ✅ FULLY FUNCTIONAL

**TOML:**
```toml
[memetic]
population_size = 200
mutation_rate = 0.40
max_generations = 1500
```

**Code Flow:**
```
TOML → Parse in CLI:1001-1003 → Store in memetic_config variable
     → Pass via builder:1114 → .memetic(memetic_cfg)
     → PipelineConfig stores it
     → CLI uses in multi-attempt loop:1238-1290
     → Passes to memetic_coloring.rs
```

**Files:**
- `prism-cli/src/main.rs:1001-1003` - Parses `[memetic]` section
- `prism-cli/src/main.rs:1114` - `.memetic(memetic_cfg)` via builder
- `prism-cli/src/main.rs:1238-1290` - Uses config in memetic evolution loop
- `foundation/prct-core/src/memetic_coloring.rs` - Algorithm uses values

**Verification:**
```rust
// main.rs:1272-1273
let memetic_solver = MemeticColoringSolver::new(
    &graph,
    memetic_config.population_size,  // ← From TOML
    memetic_config.generations,      // ← From TOML
```

---

### 5. [metaphysical_coupling] - ✅ PARTIALLY FUNCTIONAL

**TOML:**
```toml
[metaphysical_coupling]
enabled = true
geometry_stress_weight = 2.0
feedback_strength = 1.2
```

**Code Flow:**
```
TOML → Parse in CLI:1007-1009 → Store in metaphysical_coupling_config
     → Pass via builder:1117 → .metaphysical_coupling(coupling_cfg)
     → PipelineConfig stores it
     → Phases check via context.has_high_geometry_stress()
```

**Files:**
- `prism-cli/src/main.rs:1007-1009` - Parses section
- `prism-cli/src/main.rs:1117` - Passes via builder
- `prism-core/src/traits.rs` - GeometryTelemetry in PhaseContext
- Phases query via `context.geometry_stress_level()`

**Verification:**
Limited - config is stored but not all parameters are actively used by phases.

---

## Why Are So Many Sections FAKE?

### Root Cause Analysis

1. **Historical Artifacts**: Sections were added to TOML but never wired to code
2. **Documentation-Driven Development**: Someone wrote configs before implementing features
3. **Copy-Paste from Other Projects**: Configs copied but not adapted
4. **Incomplete Implementation**: Features planned but never finished

### Evidence

**Example: Phase 0 Dendritic**
```toml
# This section exists in TOML:
[phase0_dendritic]
num_branches = 10
branch_depth = 6
```

But in code:
```rust
// prism-pipeline/src/orchestrator/mod.rs:210
// Phase 0 uses hardcoded ::new() with NO config parameter
match Phase0DendriticReservoir::new_with_gpu(ptx_path) {
    Ok(phase0) => { ... }  // ← No config passed!
}
```

**Example: dsatur**
```toml
# This section exists in TOML:
[dsatur]
max_colors = 17
backtrack_depth = 150
```

But in code:
```rust
// prism-phases/src/phase2_thermodynamic.rs:348
// Repair config is hardcoded inline, NOT from TOML
let repair_config = ConflictRepairConfig {
    max_iterations: 500,        // ← Hardcoded
    population_size: 30,        // ← Hardcoded
    mutation_rate: 0.30,        // ← Hardcoded
    allow_color_increase: true, // ← Hardcoded
    // ...TOML [dsatur] section is completely ignored
};
```

---

## Corrected "Adding a New Parameter" Checklist

### For REAL Config Sections (Phase 2, Phase 3, Memetic)

**Phase 3 Pattern (Recommended):**
1. ☑ Add field to TOML `[phase3_quantum]` section
2. ☑ Add field to `Phase3QuantumConfig` struct in `prism-phases/src/phase3_quantum.rs`
3. ☐ ~~Manually parse in CLI~~ - ❌ WRONG! Serde auto-parses
4. ☑ Ensure field has `pub` visibility and `#[serde(default)]` if optional
5. ☑ Phase constructor `::with_config()` receives full config struct
6. ☑ Phase stores value: `self.my_param = config.my_param`
7. ☑ Phase uses value in execution
8. ☑ If GPU: Pass to GPU via array/struct
9. ☑ Rebuild: `cargo build --release --features cuda`
10. ☑ Test: Verify parameter is logged and affects behavior

**Phase 2 Pattern:**
1. ☑ Add field to TOML `[phase2_thermodynamic]` section
2. ☑ Add field to `Phase2Config` struct
3. ☑ Add parameter to `.with_hyperparameters()` method signature
4. ☑ Update orchestrator call to pass new parameter
5. ☑ Phase stores and uses value
6. ☑ Rebuild and test

### For FAKE Config Sections (Most Others)

**To Make Them REAL:**
1. ☐ Add parsing in CLI `prism-cli/src/main.rs`
2. ☐ Create config struct in phase file
3. ☐ Add builder method OR setter in orchestrator
4. ☐ Modify phase constructor to accept config
5. ☐ Update orchestrator initialization to pass config
6. ☐ Phase uses config values
7. ☐ Rebuild and test

**Example: Making [phase6_tda] REAL:**
```rust
// 1. Add in CLI (prism-cli/src/main.rs):
if let Some(phase6_table) = toml_config.get("phase6_tda") {
    phase6_config = Some(toml::from_str(&toml::to_string(phase6_table)?)?);
}

// 2. Add setter in orchestrator:
pub fn set_phase6_config(&mut self, config: Phase6Config) {
    self.phase6_config = Some(config);
}

// 3. Update initialization in orchestrator:
let phase6_result = if let Some(ref cfg) = self.phase6_config {
    Phase6TDA::with_config(ptx_path, cfg)
} else {
    Phase6TDA::new_with_gpu(ptx_path)
};

// 4. Add constructor in phase:
pub fn with_config(ptx_path: &str, config: &Phase6Config) -> Result<Self> {
    // Use config values
}
```

---

## Quick Reference: What Actually Works

### To Change Phase 2 Behavior
Edit `[phase2_thermodynamic]` → ✅ WORKS

### To Change Phase 3 Behavior
Edit `[phase3_quantum]` → ✅ WORKS

### To Change Memetic Evolution
Edit `[memetic]` → ✅ WORKS

### To Change Phase 0, 1, 4, 6, 7 Behavior
Edit TOML → ❌ DOES NOTHING
Must modify hardcoded defaults in phase source files

### To Change Conflict Repair
Edit `[dsatur]` → ❌ DOES NOTHING
Must edit hardcoded `ConflictRepairConfig` in phase files

---

## Summary Table

| Config Section | Parsed? | Used? | Where to Change |
|----------------|---------|-------|-----------------|
| `[global]` | ✅ Yes | ✅ Yes | TOML works |
| `[warmstart]` | ✅ Yes | ⚠️ Partial | TOML partially works |
| `[phase0_dendritic]` | ❌ No | ❌ No | Edit source: `phase0/controller.rs` |
| `[phase1_active_inference]` | ❌ No | ❌ No | Edit source: `phase1_active_inference.rs` |
| `[phase2_thermodynamic]` | ✅ Yes | ✅ Yes | TOML works |
| `[phase3_quantum]` | ✅ Yes | ✅ Yes | TOML works |
| `[phase3_pimc]` | ❌ No | ❌ No | Edit source: `phase3_quantum.rs` |
| `[phase4_geodesic]` | ❌ No | ❌ No | Edit source: `phase4_geodesic.rs` |
| `[phase5_geodesic_flow]` | ❌ No | ❌ No | Phase doesn't exist |
| `[phase6_tda]` | ❌ No | ❌ No | Edit source: `phase6_tda.rs` |
| `[phase7_ensemble]` | ❌ No | ❌ No | Edit source: `phase7_ensemble.rs` |
| `[metaphysical_coupling]` | ✅ Yes | ⚠️ Partial | TOML partially works |
| `[memetic]` | ✅ Yes | ✅ Yes | TOML works |
| `[dsatur]` | ❌ No | ❌ No | Edit hardcoded values in phase files |
| `[telemetry]` | ⚠️ Partial | ⚠️ Partial | CLI checks `enabled` flag |
| `[logging]` | ⚠️ Partial | ⚠️ Partial | CLI checks `level` |

---

## Recommendations

### For Agent Knowledge Base

**Include:**
- This verified document
- Emphasize that ONLY 5 sections work
- Show grep-based method to verify config usage
- Warn about fake sections

**Agent should:**
1. Always verify config flow with grep before claiming it works
2. Distinguish between "parsed" and "actually used"
3. Show user the actual code that uses (or doesn't use) the parameter

### For Codebase Cleanup

**High Priority:**
1. Remove fake TOML sections OR implement them properly
2. Add comments in TOML: `# NOTE: This section is not yet implemented`
3. Standardize config passing pattern (prefer `::with_config()`)

**Medium Priority:**
4. Create proper config structs for all phases
5. Add validation on config load
6. Log warnings for ignored TOML sections

---

**Date:** 2025-11-23
**Status:** Verified by code audit
**Confidence:** High - traced actual code paths
