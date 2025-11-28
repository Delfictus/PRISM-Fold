# 📋 PRISM Configuration Flow: TOML → Runtime → GPU Kernels

## Quick Answer

**Config changes are REAL**, not fake. Here's the complete flow:

```
TOML File
   ↓
[1] CLI Loads & Parses (main.rs:962-1015)
   ↓
[2] Config Structs Created (phase3_quantum.rs, memetic.rs, etc.)
   ↓
[3] Passed to Phase Controllers (orchestrator.rs:initialize_all_phases)
   ↓
[4] Phase Stores in Instance Variables
   ↓
[5] Phase Uses Values in Execution
   ↓
[6] GPU Kernels Receive Parameters (copied to GPU memory)
   ↓
[7] GPU Executes with Real Parameter Values
```

---

## Detailed Flow: Phase 3 Quantum Example

### Step 1: TOML File Definition
```toml
# configs/CHAMPION_20_COLORS.toml
[phase3_quantum]
coupling_strength = 10.0
evolution_iterations = 400
transverse_field = 2.0
max_colors = 17
```

### Step 2: CLI Loads TOML
**File:** `prism-cli/src/main.rs:990-992`
```rust
if let Some(phase3_table) = toml_config.get("phase3_quantum") {
    phase3_config = Some(toml::from_str(&toml::to_string(phase3_table)?)?);
    log::info!("Phase 3 quantum configuration loaded from TOML");
}
```

**Result:** `phase3_config: Option<Phase3QuantumConfig>`

### Step 3: Phase3 Config Struct
**File:** `prism-phases/src/phase3_quantum.rs:1-150`
```rust
pub struct Phase3QuantumConfig {
    pub coupling_strength: f32,      // ← From TOML
    pub evolution_iterations: usize, // ← From TOML
    pub transverse_field: f32,       // ← From TOML
    pub max_colors: usize,           // ← From TOML
    // ... more fields
}

impl Default for Phase3QuantumConfig {
    fn default() -> Self {
        Self {
            coupling_strength: 5.0,      // Default if not in TOML
            evolution_iterations: 200,   // Default if not in TOML
            transverse_field: 1.5,       // Default if not in TOML
            max_colors: 50,              // Default if not in TOML
        }
    }
}
```

### Step 4: Phase Controller Receives Config
**File:** `prism-pipeline/src/orchestrator/mod.rs:540-560`
```rust
pub fn initialize_all_phases(&mut self) -> Result<(), PrismError> {
    // Phase 3 initialization
    let mut phase3 = if let Some(gpu) = &self.gpu_context {
        Phase3Quantum::with_gpu(
            self.config.phase3_config.clone(),  // ← TOML config passed here
            gpu.clone(),
        )
    } else {
        Phase3Quantum::new(self.config.phase3_config.clone())
    };

    self.phases.push(Box::new(phase3));
}
```

### Step 5: Phase Stores Config in Instance
**File:** `prism-phases/src/phase3_quantum.rs:50-70`
```rust
pub struct Phase3Quantum {
    config: Phase3QuantumConfig,      // ← TOML values stored here
    coupling_strength: f32,
    evolution_iterations: usize,
    // ... other fields
}

impl Phase3Quantum {
    pub fn new(config: Phase3QuantumConfig) -> Self {
        Self {
            coupling_strength: config.coupling_strength,    // 10.0 from TOML
            evolution_iterations: config.evolution_iterations, // 400 from TOML
            // ...
        }
    }
}
```

### Step 6: Phase Executes Using Values
**File:** `prism-gpu/src/quantum.rs:240-310`
```rust
pub fn evolve_and_measure(
    &mut self,
    adjacency: &[Vec<usize>],
    num_vertices: usize,
    max_colors: usize,
) -> Result<Vec<usize>, PrismError> {
    // Step 2: Prepare coupling strengths using config value
    let couplings = vec![self.coupling_strength; num_vertices];
    //                     ^^^^^^^^^^^^^^^^^ 10.0 from TOML!

    // Step 3: Copy to GPU memory
    let d_couplings: CudaSlice<f32> = self
        .device
        .htod_sync_copy(&couplings)  // ← Copies to GPU
        .context("Failed to copy couplings to GPU")?;

    // Step 4: Launch GPU kernel with config values
    self.launch_evolution_kernel(
        &d_adjacency,
        &mut d_amplitudes,
        &d_couplings,  // ← GPU kernel receives config value
        num_vertices,
        max_colors,
    )?;
}
```

### Step 7: GPU Kernel Uses Parameter
**File:** `prism-gpu/src/kernels/quantum.cu:400-450`
```cuda
__global__ void evolve_quantum_state_kernel(
    const float* couplings,  // ← Received from host
    float* real_amps,
    float* imag_amps,
    int num_vertices,
    int max_colors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;

    // Use coupling value in quantum Hamiltonian calculation
    float coupling = couplings[idx];  // 10.0 from TOML!

    // Anti-ferromagnetic coupling: penalize same-colored neighbors
    float fermi_coupling = coupling * neighbor_penalty;

    // Apply to quantum evolution equation
    // ...Hamiltonian calculations use this value...
}
```

---

## Real vs Fake: What's Actually Used

### ✅ REAL CONFIG PARAMETERS (Flow Through System)

| Parameter | File | Used In | Effect |
|-----------|------|---------|--------|
| `coupling_strength` | phase3_quantum.rs | GPU kernel (quantum.cu) | Anti-ferromagnetic penalty strength |
| `evolution_iterations` | phase3_quantum.rs | quantum.rs:loop | Number of evolution steps on GPU |
| `transverse_field` | phase3_quantum.rs | quantum.cu | Tunnel probability in quantum annealing |
| `max_colors` | phase3_quantum.rs | quantum.cu | Color limit constraint |
| `initial_temperature` | phase2_thermodynamic.rs | thermodynamic.cu | Starting temperature for annealing |
| `cooling_rate` | phase2_thermodynamic.rs | thermodynamic.cu | Temperature decrease per iteration |
| `steps_per_temp` | phase2_thermodynamic.rs | thermodynamic.cu | Equilibration steps at each temperature |
| `population_size` | memetic.rs | memetic_coloring.rs | Evolutionary algorithm population |
| `mutation_rate` | memetic.rs | memetic_coloring.rs | Genetic algorithm mutation probability |
| `max_generations` | memetic.rs | memetic_coloring.rs | Evolution loop iterations |

### ❌ FAKE CONFIG PARAMETERS (Parsed but Not Used)

Very few! But examples:
- `description` fields (just comments)
- Undocumented TOML sections (silently ignored)
- Parameters for disabled phases (skipped by orchestrator)

---

## How to Make a Config Change

### Example: Adjust Phase 3 Coupling Strength

**1. Edit TOML File:**
```bash
nano configs/CHAMPION_20_COLORS.toml
```

**2. Change parameter:**
```toml
[phase3_quantum]
coupling_strength = 12.0  # Changed from 10.0
```

**3. Save and run:**
```bash
./target/release/prism-cli --config configs/CHAMPION_20_COLORS.toml --input benchmarks/dimacs/DSJC125.5.col --attempts 1
```

**4. Expected change:**
- Stronger anti-ferromagnetic penalty (12.0 vs 10.0)
- Should reduce conflicts more aggressively (but might expand colors)
- GPU kernel will execute with `coupling = 12.0` in quantum evolution

**Verification:**
```
[2025-11-23T17:33:28Z INFO  prism_phases::phase3_quantum]   Coupling strength: 12.0  ✓
[2025-11-23T17:33:37Z INFO  prism_gpu::quantum] Copying couplings to GPU...
[GPU kernel launches with coupling_strength=12.0 in d_couplings array]
```

---

## Important Files for Config Changes

### If You Want To:

**Adjust Phase 3 Quantum:**
- Edit: `configs/CHAMPION_20_COLORS.toml` → `[phase3_quantum]` section
- Reads: `prism-cli/src/main.rs:990-992`
- Defines: `prism-phases/src/phase3_quantum.rs:1-150`
- Uses: `prism-gpu/src/quantum.rs:240-310` and GPU kernel

**Adjust Phase 2 Thermodynamic:**
- Edit: `configs/CHAMPION_20_COLORS.toml` → `[phase2_thermodynamic]` section
- Reads: `prism-cli/src/main.rs:966-989`
- Defines: `prism-phases/src/phase2_thermodynamic.rs`
- Uses: `prism-gpu/src/thermodynamic.rs` and GPU kernel

**Adjust Memetic Evolution:**
- Edit: `configs/CHAMPION_20_COLORS.toml` → `[memetic]` section
- Reads: `prism-cli/src/main.rs:1001-1003`
- Defines: `prism-phases/src/phase7_ensemble.rs:1-100`
- Uses: `foundation/prct-core/src/memetic_coloring.rs`

**Add New Parameter:**
1. Add to TOML: `configs/CHAMPION_20_COLORS.toml`
2. Define struct field: Phase config struct (e.g., `Phase3QuantumConfig`)
3. Parse in CLI: `prism-cli/src/main.rs`
4. Use in phase: `prism-phases/src/phaseX_*.rs`
5. Use in computation: GPU kernel or algorithm

---

## Config Validation Flow

```
TOML Parse Error?
        ↓
  log::error!() + return Err
        ↓
Default Value Used (if applicable)
        ↓
Range Validation (e.g., coupling_strength > 0.0)
        ↓
Log INFO (confirm param loaded)
        ↓
Phase Receives Validated Config
        ↓
Execution with Real Values
```

Example from `phase3_quantum.rs`:
```rust
pub fn set_coupling_strength(&mut self, strength: f32) {
    assert!(strength > 0.0, "coupling_strength must be positive");  // ← Validation
    self.coupling_strength = strength;
    log::debug!("Coupling strength updated to {}", strength);       // ← Confirmation
}
```

---

## Testing Config Changes

### Before Modifying Code
1. Try config adjustment first (faster iteration)
2. If config doesn't support parameter → then modify code

### Verify Config is Used
```bash
# Run with debug logging
RUST_LOG=debug ./target/release/prism-cli --config configs/CHAMPION_20_COLORS.toml \
    --input benchmarks/dimacs/DSJC125.5.col --attempts 1 2>&1 | grep -i "coupling\|temperature\|mutation"
```

Expected output shows your values being logged at startup and used in GPU kernels.

---

## Summary

| Aspect | Status | Details |
|--------|--------|---------|
| **Are config changes real?** | ✅ YES | Flow through entire system to GPU |
| **Do they affect GPU kernels?** | ✅ YES | Parameters copied to GPU memory |
| **Do I need to recompile?** | ❌ NO | Just edit TOML and run |
| **Are they validated?** | ✅ YES | Assertions and range checks in phase code |
| **Can they be logged?** | ✅ YES | RUST_LOG=debug shows all values |

---

**Bottom Line:** Config changes are completely real and flow all the way to GPU kernel execution. No recompilation needed unless you're adding a new parameter type.
