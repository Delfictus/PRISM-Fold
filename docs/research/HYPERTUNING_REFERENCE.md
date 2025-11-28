# 🔍 Complete Hypertuning Reference: Finding & Tracing Any Parameter

## TL;DR: How to Find What Files to Touch

**For ANY parameter change:**

```bash
# 1. Find parameter in TOML
grep "parameter_name" configs/CHAMPION_20_COLORS.toml

# 2. Find where it's loaded in CLI
grep -r "parameter_name" prism-cli/src/main.rs

# 3. Find config struct that contains it
grep -r "struct.*Config\|parameter_name.*:" prism-phases/src/ prism-pipeline/src/

# 4. Find where it's used in phase execution
grep -r "self.parameter_name\|parameter_name =" prism-phases/src/

# 5. Find where it's used in GPU kernels (if GPU phase)
grep -r "parameter_name" prism-gpu/src/
```

---

## All Available Config Sections (16 Total)

| Section | Parameters | Used By | Files |
|---------|-----------|---------|-------|
| `[global]` | max_attempts, enable_fluxnet_rl, rl_learning_rate, etc. | CLI + Orchestrator | prism-cli/main.rs, prism-pipeline/orchestrator |
| `[warmstart]` | enabled, greedy_ratio, dsatur_ratio, random_ratio | Phase 0/CLI | prism-phases/phase0_*, prism-cli/main.rs |
| `[phase0_dendritic]` | num_branches, branch_depth, learning_rate, plasticity, gpu_enabled | Phase 0 Dendritic | prism-phases/phase0/controller.rs, prism-gpu/dendritic_reservoir.rs |
| `[phase1_active_inference]` | prior_precision, likelihood_precision, learning_rate, free_energy_threshold, num_iterations | Phase 1 Active Inference | prism-phases/phase1_active_inference.rs, prism-gpu/active_inference.rs |
| `[phase2_thermodynamic]` | initial_temperature, final_temperature, cooling_rate, steps_per_temp, num_temps, num_replicas | Phase 2 Thermodynamic | prism-phases/phase2_thermodynamic.rs, prism-gpu/thermodynamic.rs (GPU kernel) |
| `[phase3_quantum]` | coupling_strength, evolution_iterations, transverse_field, max_colors, evolution_time, interference_decay | Phase 3 Quantum | prism-phases/phase3_quantum.rs, prism-gpu/quantum.rs (GPU kernel) |
| `[phase3_pimc]` | num_replicas, beta, delta_tau, transverse_field, coupling_strength, mc_steps | Phase 3 PIMC | prism-phases/phase3_quantum.rs, prism-gpu/pimc.rs (GPU kernel) |
| `[phase4_geodesic]` | distance_threshold, centrality_weight, gpu_enabled | Phase 4 Geodesic | prism-phases/phase4_geodesic.rs, prism-gpu/floyd_warshall.rs |
| `[phase5_geodesic_flow]` | flow_iterations, flow_strength, diffusion_rate | Phase 5 Geodesic Flow | prism-phases/phase5_geodesic_flow.rs |
| `[phase6_tda]` | persistence_threshold, max_dimension, coherence_cv_threshold, vietoris_rips_radius, gpu_enabled | Phase 6 TDA | prism-phases/phase6_tda.rs, prism-gpu/tda.rs |
| `[phase7_ensemble]` | num_replicas, exchange_interval, temperature_range, diversity_weight, consensus_threshold, gpu_enabled | Phase 7 Ensemble | prism-phases/phase7_ensemble.rs |
| `[metaphysical_coupling]` | enabled, geometry_stress_weight, feedback_strength, hotspot_threshold, stress_decay_rate, overlap_penalty | Multiple phases | prism-phases/*/controller.rs |
| `[memetic]` | enabled, population_size, mutation_rate, crossover_rate, elite_fraction, max_generations, local_search_intensity, local_search_depth | Phase 7 Ensemble | prism-phases/phase7_ensemble.rs, foundation/prct-core/memetic_coloring.rs |
| `[dsatur]` | max_colors, backtrack_depth, early_termination, conflict_penalty | Conflict repair | prism-phases/conflict_repair.rs |
| `[telemetry]` | enabled, capture_all_phases | CLI/Orchestrator | prism-cli/main.rs, prism-pipeline/orchestrator |
| `[logging]` | level | CLI | prism-cli/main.rs |

---

## Step-by-Step: Tracing a Parameter Through the System

### Example 1: Adjust `coupling_strength` in Phase 3

**Step 1: Locate in TOML**
```bash
$ grep -n "coupling_strength" configs/CHAMPION_20_COLORS.toml
68:coupling_strength = 10.0
91:coupling_strength = 2.0
```

Two occurrences: Phase 3 Quantum (line 68) and Phase 3 PIMC (line 91).

**Step 2: Find config struct**
```bash
$ grep -r "coupling_strength.*:" prism-phases/src/ --include="*.rs"
prism-phases/src/phase3_quantum.rs:47:    coupling_strength: f32,
prism-phases/src/phase3_quantum.rs:71:        coupling_strength: 1.0,
```

**Step 3: Find where loaded in CLI**
```bash
$ grep -A 5 "phase3_quantum" prism-cli/src/main.rs | grep -E "from_str|toml"
prism-cli/src/main.rs:991:            phase3_config = Some(toml::from_str(&toml::to_string(phase3_table)?)?);
```

**Step 4: Find struct definition**
```bash
$ grep -B 2 -A 20 "pub struct Phase3QuantumConfig" prism-phases/src/phase3_quantum.rs
pub struct Phase3QuantumConfig {
    pub coupling_strength: f32,
    pub evolution_iterations: usize,
    // ... other fields
}
```

**Step 5: Find where used in phase**
```bash
$ grep -n "coupling_strength" prism-phases/src/phase3_quantum.rs | head -10
47:    coupling_strength: f32,
71:        coupling_strength: 1.0,
109:        coupling_strength: 5.0,
157:        self.quantum_gpu.set_coupling_strength(config.coupling_strength);
```

Line 157 shows it's passed to GPU: `self.quantum_gpu.set_coupling_strength(config.coupling_strength);`

**Step 6: Find where used in GPU**
```bash
$ grep -n "set_coupling_strength\|coupling_strength" prism-gpu/src/quantum.rs | head -20
1162:        pub fn coupling_strength(&self) -> f32 {
1168:        pub fn set_coupling_strength(&mut self, strength: f32) {
273:        let couplings = vec![self.coupling_strength; num_vertices];
```

Line 273 shows it's used to create coupling array and sent to GPU kernel.

**Step 7: Find GPU kernel implementation**
```bash
$ grep -n "coupling\|couplings" prism-gpu/src/kernels/quantum.cu | head -20
202:    float coupling = couplings[idx];
```

The GPU kernel receives `couplings` array and uses it.

**Result:** `coupling_strength` flows: TOML → CLI parse → Phase3QuantumConfig → Phase3Quantum.set_coupling_strength() → quantum_gpu.couplings array → copied to GPU → GPU kernel uses in computation

---

### Example 2: Adjust `population_size` in Memetic

**Step 1-2: Find in TOML and struct**
```bash
$ grep "population_size" configs/CHAMPION_20_COLORS.toml
population_size = 200

$ grep -r "population_size" prism-phases/src/ --include="*.rs"
prism-phases/src/phase7_ensemble.rs:47:    pub population_size: usize,
```

**Step 3: Find memetic config struct**
```bash
$ grep -B 5 -A 15 "pub struct MemeticConfig" prism-phases/src/phase7_ensemble.rs
pub struct MemeticConfig {
    pub enabled: bool,
    pub population_size: usize,
    pub mutation_rate: f32,
    // ...
}
```

**Step 4: Find where used**
```bash
$ grep -n "population_size" prism-phases/src/phase7_ensemble.rs
47:    pub population_size: usize,

$ grep -n "population_size\|self.memetic_config" prism-phases/src/phase7_ensemble.rs
120:        self.memetic_config.population_size,
```

**Step 5: Find in memetic algorithm**
```bash
$ grep -n "population_size" foundation/prct-core/src/memetic_coloring.rs | head -5
89:    pub population_size: usize,
120:        population: Vec::with_capacity(self.population_size),
```

Line 120 shows it's used to create the genetic algorithm population.

**Result:** `population_size` flows: TOML → CLI parse → MemeticConfig → Phase7Ensemble → memetic_coloring.rs → used to initialize population size

---

## ALL Parameters with Their File Paths

### Phase 0: Dendritic Reservoir
```
TOML: [phase0_dendritic]
Config Struct: prism-phases/src/phase0/controller.rs:Phase0Config
Loaded in: prism-cli/src/main.rs (search for phase0)
Used in: prism-phases/src/phase0/controller.rs
GPU Kernel: prism-gpu/src/kernels/dendritic_reservoir.cu
```

Parameters: `num_branches`, `branch_depth`, `learning_rate`, `activation_threshold`, `plasticity`, `gpu_enabled`

### Phase 1: Active Inference
```
TOML: [phase1_active_inference]
Config Struct: prism-phases/src/phase1_active_inference.rs:Phase1Config
Loaded in: prism-cli/src/main.rs
Used in: prism-phases/src/phase1_active_inference.rs
GPU Kernel: prism-gpu/src/kernels/active_inference.cu
```

Parameters: `prior_precision`, `likelihood_precision`, `learning_rate`, `free_energy_threshold`, `num_iterations`

### Phase 2: Thermodynamic Annealing
```
TOML: [phase2_thermodynamic]
Config Struct: prism-phases/src/phase2_thermodynamic.rs:Phase2Config
Loaded in: prism-cli/src/main.rs:966-989
Used in: prism-phases/src/phase2_thermodynamic.rs:execute()
GPU Kernel: prism-gpu/src/kernels/thermodynamic.cu
```

Parameters: `initial_temperature`, `final_temperature`, `cooling_rate`, `steps_per_temp`, `num_temps`, `num_replicas`, `t_min`, `compaction_enabled`, `compaction_threshold`

### Phase 3: Quantum Evolution
```
TOML: [phase3_quantum] + [phase3_pimc]
Config Struct: prism-phases/src/phase3_quantum.rs:Phase3QuantumConfig + Phase3PimcConfig
Loaded in: prism-cli/src/main.rs:990-992
Used in: prism-phases/src/phase3_quantum.rs:execute()
GPU Kernel: prism-gpu/src/kernels/quantum.cu + pimc.cu
```

Parameters: `evolution_time`, `coupling_strength`, `max_colors`, `evolution_iterations`, `transverse_field`, `interference_decay`, `schedule_type`, `stochastic_measurement`, `num_replicas` (PIMC), `beta`, `delta_tau`, `mc_steps`

### Phase 4: Geodesic Distance
```
TOML: [phase4_geodesic]
Config Struct: prism-phases/src/phase4_geodesic.rs:Phase4Config
Loaded in: prism-cli/src/main.rs (search for phase4)
Used in: prism-phases/src/phase4_geodesic.rs
GPU Kernel: prism-gpu/src/kernels/floyd_warshall.cu
```

Parameters: `distance_threshold`, `centrality_weight`, `gpu_enabled`

### Phase 5: Geodesic Flow
```
TOML: [phase5_geodesic_flow]
Config Struct: prism-phases/src/phase5_geodesic_flow.rs (if exists)
Loaded in: prism-cli/src/main.rs
Used in: prism-phases/src/phase5_geodesic_flow.rs
```

Parameters: `flow_iterations`, `flow_strength`, `diffusion_rate`

### Phase 6: Topological Data Analysis (TDA)
```
TOML: [phase6_tda]
Config Struct: prism-phases/src/phase6_tda.rs:Phase6Config
Loaded in: prism-cli/src/main.rs
Used in: prism-phases/src/phase6_tda.rs
GPU Kernel: prism-gpu/src/kernels/tda.cu
```

Parameters: `persistence_threshold`, `max_dimension`, `coherence_cv_threshold`, `vietoris_rips_radius`, `gpu_enabled`

### Phase 7: Ensemble Aggregation
```
TOML: [phase7_ensemble]
Config Struct: prism-phases/src/phase7_ensemble.rs:Phase7Config
Loaded in: prism-cli/src/main.rs
Used in: prism-phases/src/phase7_ensemble.rs
Memetic: foundation/prct-core/src/memetic_coloring.rs
```

Parameters: `num_replicas`, `exchange_interval`, `temperature_range`, `diversity_weight`, `consensus_threshold`, `gpu_enabled`

### Memetic Evolution
```
TOML: [memetic]
Config Struct: prism-phases/src/phase7_ensemble.rs:MemeticConfig
Loaded in: prism-cli/src/main.rs:1001-1003
Used in: foundation/prct-core/src/memetic_coloring.rs
Algorithm: CPU-based genetic algorithm (no GPU)
```

Parameters: `population_size`, `mutation_rate`, `crossover_rate`, `elite_fraction`, `max_generations`, `local_search_intensity`, `local_search_depth`

### Conflict Repair
```
TOML: [dsatur]
Config Struct: prism-phases/src/conflict_repair.rs:ConflictRepairConfig
Loaded in: Phase controllers when repair triggered
Used in: prism-phases/src/conflict_repair.rs
Algorithm: DSATUR coloring with backtracking
```

Parameters: `max_colors`, `backtrack_depth`, `early_termination`, `conflict_penalty`

### Metaphysical Coupling
```
TOML: [metaphysical_coupling]
Config Struct: prism-pipeline/src/config/mod.rs:MetaphysicalCouplingConfig
Loaded in: prism-pipeline/src/config/mod.rs
Used in: prism-pipeline/src/orchestrator/mod.rs
Effect: Feeds geometry stress to Phase 1, 2, 3, 7
```

Parameters: `enabled`, `geometry_stress_weight`, `feedback_strength`, `hotspot_threshold`, `stress_decay_rate`, `overlap_penalty`

---

## Generic "Find and Change" Template

For ANY parameter:

1. **Identify which section in TOML:**
   ```bash
   grep "my_parameter" configs/CHAMPION_20_COLORS.toml
   ```

2. **Find the config struct:**
   ```bash
   grep -r "my_parameter.*:" prism-phases/ prism-pipeline/ --include="*.rs"
   ```

3. **Find CLI loading:**
   ```bash
   grep -n "my_parameter\|my_section" prism-cli/src/main.rs
   ```

4. **Find phase execution:**
   ```bash
   grep -r "self.my_parameter\|config.my_parameter" prism-phases/src/phaseX_*
   ```

5. **Find GPU usage (if GPU phase):**
   ```bash
   grep -r "my_parameter" prism-gpu/src/ --include="*.rs" --include="*.cu"
   ```

6. **Edit TOML, save, run - NO RECOMPILATION NEEDED**

---

## Quick Reference: How to Change Common Things

### Change Algorithm Hyperparameters
- **Phase 2 temperature annealing**: Edit `[phase2_thermodynamic]` → `initial_temperature`, `cooling_rate`, `steps_per_temp`, `num_temps`
- **Phase 3 quantum strength**: Edit `[phase3_quantum]` → `coupling_strength`, `transverse_field`, `evolution_iterations`
- **Memetic evolution**: Edit `[memetic]` → `population_size`, `mutation_rate`, `max_generations`

### Change GPU Usage
- **Enable/disable Phase 2 GPU**: Edit `[phase2_thermodynamic]` → Add/remove `gpu_enabled = true`
- **Enable/disable Phase 3 GPU**: Edit `[phase3_quantum]` → Add/remove `gpu_enabled = true`

### Change Color Constraints
- **Set max colors**: Edit `[phase3_quantum]` → `max_colors = 17`
- **Conflict penalty**: Edit `[dsatur]` → `conflict_penalty = 10000.0`

### Change Repair Behavior
- **Conflict repair depth**: Edit `[dsatur]` → `backtrack_depth = 150`
- **How much to expand colors**: Edit phase-specific repair configs

### Enable/Disable Phases
- Edit `[phaseX_*]` → `enabled = true/false`

---

## Files You'll Touch Most Often (80/20 Rule)

**80% of hypertuning uses these files:**
1. `configs/CHAMPION_20_COLORS.toml` - Config changes
2. `prism-phases/src/phase2_thermodynamic.rs` - If you want to understand Phase 2 behavior
3. `prism-phases/src/phase3_quantum.rs` - If you want to understand Phase 3 behavior
4. `foundation/prct-core/src/memetic_coloring.rs` - If you want to understand memetic evolution

**20% of advanced hypertuning uses:**
- `prism-gpu/src/thermodynamic.rs` - GPU-specific tuning
- `prism-gpu/src/quantum.rs` - GPU-specific tuning
- `prism-gpu/src/kernels/*.cu` - GPU kernel parameter tweaking
- `prism-pipeline/src/orchestrator/mod.rs` - Multi-phase orchestration

---

## Checklist for Adding a New Parameter

1. ☐ Add to TOML file under correct section
2. ☐ Add to config struct (e.g., `Phase3QuantumConfig`)
3. ☐ Add parsing in CLI `prism-cli/src/main.rs`
4. ☐ Add passing to phase controller in `prism-pipeline/src/orchestrator/mod.rs`
5. ☐ Add use in phase execution code
6. ☐ If GPU phase: pass to GPU via `set_*()` method
7. ☐ Use in GPU kernel or CPU algorithm
8. ☐ Add logging: `log::info!("Parameter: {}", value)`
9. ☐ Rebuild: `cargo build --release --features cuda`
10. ☐ Test: `./target/release/prism-cli --config ... --attempts 1`

---

**Date:** 2025-11-23
**Status:** Complete Reference
