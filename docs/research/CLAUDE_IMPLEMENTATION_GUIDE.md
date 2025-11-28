# PRISM Integration Master Guide

**Read this entire file before starting work on each subsystem.** Work on one subsystem at a time, deliver real functionality, telemetry, and CLI proof per the Task Completion Requirements.

## Subsystem Order
1. CMA GPU Pipeline (kernels + phase)
2. MEC Phase (MD + telemetry)
3. PhaseContext & Telemetry updates
4. FluxNet RL state/action wiring + retraining
5. Biomolecular & Material adapters
6. GNN/ONNX inference integration
7. CLI modes & validation commands

Do not move to the next subsystem until you have implemented, tested, and demonstrated the current one.

---

## 1. CMA GPU Pipeline
- Kernels: `prism-gpu/src/kernels/{pimc.cu, transfer_entropy.cu, ensemble_exchange.cu}`.
- Wrappers: `prism-gpu/src/{pimc.rs, transfer_entropy.rs, cma.rs}`.
- Phase: `prism-phases/src/phaseX_cma.rs`.
- Config: `prism-pipeline/src/config/mod.rs`, expose `[cma]` struct with defaults.
- Context: add `PhaseContext::update_cma_state(CmaState { ... })` in `prism-core/src/traits.rs`.
- Telemetry: emit real metrics (TE mean/max, pac_bayes_bound, acceptance_rate) via `prism-pipeline/src/telemetry/mod.rs`.
- PTX loading: add modules in `prism-gpu/src/context.rs` (`load_module("pimc", "target/ptx/pimc.ptx")`).
- CLI proof: run `./target/release/prism-cli --input benchmarks/dimacs/DSJC250.5.col --config configs/dsjc250_cma.toml --attempts 2 --gpu`, paste log snippet showing CMA metrics.

## 2. MEC Phase
- Implement MD logic in `prism-mec/src/molecular.rs` (CPU fallback) and `prism-gpu/src/kernels/molecular_dynamics.cu` for GPU.
- `MecPhaseController::execute` must run MD, compute metrics (total_energy, temperature, pattern_index).
- Add `PhaseContext::update_mec_state(MecState)` in `prism-core/src/traits.rs`.
- Telemetry entry for Phase M with real values.
- CLI proof: run the same DSJC run with `mec.enabled=true`; show telemetry JSON line with MEC metrics.

## 3. PhaseContext & Telemetry
- Add setters: `update_ontology_state`, `update_mec_state`, `update_cma_state`, `update_biomolecular_state`, `update_materials_state`, `update_gnn_state`, `update_md_state`.
- Ensure each phase calls its setter with real data.
- Extend JSON telemetry schema to include sections for ontology/mec/cma/biomolecular/materials/gnn/molecular_dynamics.
- Provide sample telemetry snippets after each phase implementation.

## 4. FluxNet RL Wiring
- Update `prism-fluxnet/src/core/state.rs` to include new metrics (mec_free_energy, cma_te_mean, ontology_conflicts, etc.).
- Extend `actions.rs` with MEC/CMA/adapter actions.
- Update `controller.rs` reward shaping to use new signals.
- Retrain Q-table: `RUST_LOG=info cargo run --release --bin fluxnet_train benchmarks/dimacs/DSJC250.5.col 1500 artifacts/fluxnet/curriculum_bank_v4_geometry.bin --config configs/fluxnet_training.toml`.
- Provide training log excerpt showing completion.

## 5. Biomolecular & Material Adapters
- Move protein parser to `prism-core/src/domain/protein.rs` with `ProteinContactGraph::from_pdb_file`.
- Implement `BiomolecularAdapter` and `MaterialsAdapter` as callable modules in pipeline.
- Add CLI modes: `--mode biomolecular --sequence <fa> --ligand <smi>`, `--mode materials --config <toml>`.
- Update PhaseContext/telemetry with binding stats and material scores.
- Provide run output (command + telemetry snippet).

## 6. GNN/ONNX Integration
- Create `prism-gnn` crate exposing `E3EquivariantGnn` and `OnnxGnn`.
- Add `[gnn]` config section with `onnx_model_path`.
- Invoke GNN in an appropriate phase (e.g., after CMA) and store results via `update_gnn_state`.
- Telemetry entry with predicted chromatic number / manifold metrics.
- Provide CLI run showing GNN telemetry.

## 7. CLI Modes & Validation
- Add CLI parsing for modes (coloring, biomolecular, materials, mec-only).
- Provide example commands and ensure they run end-to-end.
- Add integration tests in `tests/` covering CMA, MEC, biomolecular adapters.
- Final validation: run the benchmark and attach telemetry/log proof.

---

## Phase 3 Quantum Evolution Configuration

Phase 3 (Quantum Evolution) performs quantum-inspired annealing to refine graph colorings. The following configuration parameters control the quantum evolution dynamics and are specified in the `[phase3_quantum]` section of TOML config files.

### Basic Parameters

- **evolution_time** (f32, default: 1.0)
  - Controls quantum fluctuation magnitude
  - Lower values (0.3-0.7) reduce violent quantum fluctuations and prevent collapse
  - Higher values (1.0-2.0) increase exploration but risk instability
  - Recommended: 0.5 for dense graphs (DSJC500.5)

- **coupling_strength** (f32, default: 1.0)
  - Conflict penalty weight in Hamiltonian
  - Higher values (2.0-3.0) strongly penalize edge conflicts
  - Lower values (0.5-1.0) allow more exploration with conflicts
  - Recommended: 2.5 for preventing collapse to single color

- **max_colors** (usize, default: 50)
  - Maximum number of colors available in quantum state space
  - Should match or exceed warmstart color space
  - Typical range: 50-100 depending on graph density

- **num_qubits** (usize, default: 500)
  - Number of qubits (typically equals number of vertices)
  - Automatically set based on graph size

### Complex Quantum Evolution Parameters (New)

- **use_complex_amplitudes** (bool, default: true)
  - Enable complex-valued quantum amplitudes for interference effects
  - `true`: Full quantum interference, better exploration, requires more GPU memory
  - `false`: Real-only amplitudes (legacy mode), faster but less effective
  - Recommended: `true` for all use cases

- **evolution_iterations** (usize, default: 100)
  - Number of multi-step evolution iterations with annealing schedule
  - Range: 50-200 (higher for large graphs)
  - Each iteration applies Hamiltonian evolution with decaying parameters
  - Recommended: 150 for DSJC500.5

- **transverse_field** (f32, default: 1.0)
  - Initial quantum tunneling strength (σ_x coupling)
  - Controls quantum superposition and tunneling through barriers
  - Range: 0.5-2.0 (higher = more tunneling)
  - Decays according to schedule_type during evolution
  - Recommended: 1.2 for dense graphs

- **interference_decay** (f32, default: 0.01)
  - Decoherence rate per iteration (imaginary part decay)
  - Controls how quickly quantum coherence is lost
  - Range: 0.01-0.05 (higher = faster decoherence)
  - Balances exploration vs classical convergence
  - Recommended: 0.02 for stable evolution

- **schedule_type** (String, default: "linear")
  - Annealing schedule for transverse_field decay
  - Options:
    - `"linear"`: σ_x(t) = σ_x₀ * (1 - t/T)
    - `"exponential"`: σ_x(t) = σ_x₀ * exp(-λt)
    - `"custom"`: User-defined schedule (requires code changes)
  - Recommended: `"exponential"` for smoother quantum-to-classical transition

- **stochastic_measurement** (bool, default: false)
  - Use RNG-based quantum measurement collapse
  - `true`: Stochastic collapse proportional to |amplitude|²
  - `false`: Deterministic collapse (argmax of probabilities)
  - Requires CUDA RNG initialization (curandStatePhilox4_32_10_t)
  - Recommended: `false` for deterministic results, `true` for ensemble methods

### Example Configuration

```toml
[phase3_quantum]
enabled = true

# Basic parameters
evolution_time = 0.5            # Reduced fluctuations
coupling_strength = 2.5         # Strong conflict penalty
max_colors = 60                 # Match warmstart space
num_qubits = 500

# Complex quantum evolution
use_complex_amplitudes = true   # Enable interference
evolution_iterations = 150      # Multi-step annealing
transverse_field = 1.2          # Tunneling strength
interference_decay = 0.02       # Decoherence rate
schedule_type = "exponential"   # Smooth annealing
stochastic_measurement = false  # Deterministic
```

### Parameter Tuning Guidelines

**For Quantum Collapse Issues (chromatic_number = 1):**
1. Reduce `evolution_time` (0.3-0.5)
2. Increase `coupling_strength` (2.0-3.0)
3. Enable `use_complex_amplitudes = true`
4. Use `schedule_type = "exponential"`
5. Set `evolution_iterations = 100-200`

**For Insufficient Refinement (chromatic_number too high):**
1. Increase `evolution_iterations` (150-300)
2. Increase `transverse_field` (1.5-2.0)
3. Decrease `interference_decay` (0.005-0.01)
4. Try `stochastic_measurement = true` with ensemble methods

**For Performance Optimization:**
- Complex amplitudes use 2x GPU memory vs real-only
- Higher `evolution_iterations` linearly increase Phase 3 time
- Stochastic measurement requires RNG state per vertex (~40 bytes/vertex)

### Telemetry Metrics

Phase 3 emits the following quantum-specific telemetry:
- `purity`: Quantum state purity (0-1, higher = more classical)
- `entanglement`: Entanglement measure (0-1, higher = more correlated)
- `coupling_strength`: Final coupling strength used
- `evolution_time`: Evolution time parameter
- `max_colors`: Color space size
- `gpu_enabled`: GPU acceleration flag (0 or 1)

Additional metrics for complex evolution mode:
- `amplitude_variance`: Variance of complex amplitudes
- `coherence`: Quantum coherence measure
- `complex_mode`: Flag indicating complex amplitudes used

### Implementation Architecture (Complex Quantum Evolution)

**CUDA Kernels** (`prism-gpu/src/kernels/quantum.cu`):
1. `quantum_evolve_complex_kernel` - Complex amplitude evolution with transverse field
   - Separate real/imaginary amplitude buffers (Rust FFI compatible)
   - Complex phase rotation: `(r, i) → (r*cos(φ) - i*sin(φ), r*sin(φ) + i*cos(φ))`
   - Transverse field mixing: `r += σ_x * cos(φ_t) / √(max_colors)`
   - Interference decay: `i *= (1.0 - decay)`
   - Per-vertex normalization: `Σ(r² + i²) = 1`

2. `quantum_measure_stochastic_kernel` - RNG-based probabilistic measurement
   - cuRAND Philox4_32_10 (high-quality, reproducible)
   - Probability distribution: `P(color) = real[c]² + imag[c]²`
   - Cumulative distribution sampling with inverse CDF
   - Per-vertex independent RNG state

3. `init_rng_states_kernel` - Initialize cuRAND Philox4_32_10 states
4. `init_complex_amplitudes_kernel` - Equal superposition initialization

**Legacy Kernels (Preserved):**
- `quantum_evolve_kernel` - Real-only evolution (original)
- `quantum_measure_kernel` - Deterministic measurement (original)
- `quantum_evolve_measure_fused_kernel` - Fused version (original)
- `init_amplitudes_kernel` - Real-only initialization (original)

**Rust Wrappers** (`prism-gpu/src/quantum.rs`):
- `evolve_complex_and_measure()` - High-level API matching controller interface
- `evolve_complex()` - Low-level multi-iteration evolution loop
- `init_complex_amplitudes()` - Buffer initialization
- `init_rng_states()` - RNG state setup
- `launch_measurement_stochastic()` - Stochastic measurement
- `compute_amplitude_variance()` - Telemetry metric
- `compute_coherence()` - Telemetry metric

**Controller Integration** (`prism-phases/src/phase3_quantum.rs`):
- Config-based path selection: `if config.use_complex_amplitudes { complex_path() } else { legacy_path() }`
- Telemetry retrieval: `get_amplitude_variance()`, `get_coherence()`
- FluxNet RL state population: `state.quantum_amplitude_variance`, `state.quantum_coherence`

**FluxNet RL Integration** (`prism-fluxnet/src/core/state.rs`):
- `UniversalRLState` fields: `quantum_amplitude_variance`, `quantum_coherence`
- Discretization: Hash-based mapping to 4096 or 65536 states

### Troubleshooting Guide

**Issue: Quantum Collapse (chromatic_number = 1)**
- **Symptoms:** Phase 3 outputs single color with many conflicts
- **Root Cause:** Excessive quantum fluctuations or insufficient conflict penalty
- **Solutions:**
  1. Set `use_complex_amplitudes = true` (if not already)
  2. Reduce `evolution_time` to 0.3-0.5
  3. Increase `coupling_strength` to 2.5-3.0
  4. Use `schedule_type = "exponential"`
  5. Increase `evolution_iterations` to 150-200

**Issue: Insufficient Refinement (chromatic_number too high)**
- **Symptoms:** Phase 3 outputs exceed target chromatic number
- **Root Cause:** Insufficient quantum exploration or early decoherence
- **Solutions:**
  1. Increase `evolution_iterations` to 200-300
  2. Increase `transverse_field` to 1.5-2.0
  3. Decrease `interference_decay` to 0.005-0.01
  4. Try `stochastic_measurement = true` with ensemble methods

**Issue: GPU Out of Memory**
- **Symptoms:** CUDA allocation errors during Phase 3
- **Root Cause:** Complex amplitudes require 2× memory vs real-only
- **Memory Requirements:**
  - Real-only: `n × max_colors × 4 bytes`
  - Complex: `n × max_colors × 8 bytes` (amplitudes) + `n × 40 bytes` (RNG states)
- **Solutions:**
  1. Reduce `max_colors` (e.g., 48 instead of 64)
  2. Disable complex mode: `use_complex_amplitudes = false` (legacy fallback)
  3. Reduce graph size or use graph partitioning

**Issue: Slow Phase 3 Performance**
- **Symptoms:** Phase 3 takes > 1 minute for 500-vertex graphs
- **Root Cause:** High `evolution_iterations` or inefficient GPU utilization
- **Solutions:**
  1. Reduce `evolution_iterations` to 100 (minimum)
  2. Use deterministic measurement: `stochastic_measurement = false`
  3. Check GPU utilization with `nvidia-smi` (should be > 80%)
  4. Verify PTX kernels loaded correctly (check logs for "Loading PTX module")

**Issue: Non-Reproducible Results**
- **Symptoms:** Different outputs with same config and seed
- **Root Cause:** Stochastic measurement or RNG seed not set
- **Solutions:**
  1. Set `stochastic_measurement = false` for deterministic results
  2. If using stochastic mode, set consistent `seed` in config
  3. Verify RNG state initialization in logs

**Issue: Compilation Errors with CUDA**
- **Symptoms:** `cargo build --features cuda` fails with PTX errors
- **Root Cause:** CUDA_HOME not set or nvcc version mismatch
- **Solutions:**
  1. Set `CUDA_HOME=/usr/local/cuda-12.6` (or your CUDA path)
  2. Verify nvcc version: `nvcc --version` (requires 12.x)
  3. Check compute capability: PTX requires sm_86 (RTX 3060) or higher
  4. Inspect `target/ptx/quantum.ptx` for symbol exports

### Rolling Back to Legacy Mode

If complex quantum evolution causes issues, revert to the original real-only mode:

**Config Change:**
```toml
[phase3_quantum]
use_complex_amplitudes = false  # Disable complex mode
```

**Verification:**
- Check logs for "Using legacy real-only quantum evolution"
- Telemetry: `amplitude_variance` and `coherence` will be 0.0
- Memory usage should drop by ~50%

**Legacy Mode Characteristics:**
- ✅ Faster (single-shot evolution, no iteration loop)
- ✅ Lower memory usage (no imaginary amplitudes or RNG states)
- ✅ Deterministic (no stochastic measurement)
- ❌ Less effective refinement (no quantum interference)
- ❌ Higher risk of collapse (no transverse field tunneling)

**When to Use Legacy Mode:**
- GPU memory constraints (< 4 GB VRAM)
- Quick prototyping or debugging
- Baseline performance comparison
- Systems without CUDA support (CPU fallback)

---

**Remember:** no task is complete without real code, telemetry, and command output proof. μην mark any checklist items done until the evidence is captured.
