# PRISM Ultra Implementation Plan - Part 4: Agent Instructions, Verification & Benchmarks

## Document Metadata
- **Version**: 1.0.0
- **Created**: 2025-11-29
- **Purpose**: Agent-specific instructions and verification criteria
- **Scope**: All agents, PTX compilation, benchmarks, validation

---

# SECTION J: AGENT-SPECIFIC INSTRUCTIONS

## J.1 prism-gpu-specialist Agent

### J.1.1 Responsibilities

The `prism-gpu-specialist` agent is responsible for:

1. **cudarc 0.18+ Migration** (Phase 0)
   - Update all Cargo.toml files
   - Transform API calls from 0.9 to 0.18+
   - Implement stream-based operations
   - Test basic kernel functionality

2. **Ultra Kernel Implementation** (Phase 1A)
   - Create `dr_whcr_ultra.cu` with all 8 components
   - Implement shared memory layout
   - Wire up all device functions

3. **TPTP Module** (Phase 1B)
   - Create `tptp.cu` for persistent homology
   - Implement simplicial complex construction
   - Implement phase transition detection

4. **AATGS Scheduler** (Phase 1C)
   - Create `aatgs.rs` with circular buffers
   - Implement non-blocking config upload
   - Implement non-blocking telemetry download

5. **Stream Management** (Phase 2)
   - Create `stream_manager.rs`
   - Implement triple-buffering
   - Coordinate async operations

6. **Multi-GPU** (Phase 3)
   - Modify `multi_device_pool.rs` for P2P
   - Implement cross-GPU replica exchange
   - Enable NVLink/PCIe P2P transfers

7. **PTX Compilation** (Phase 4)
   - Update `build.rs` for new kernels
   - Compile all PTX files
   - Verify kernel signatures

### J.1.2 File Ownership

| File | Action | Priority |
|------|--------|----------|
| `crates/prism-gpu/Cargo.toml` | Modify | P0 |
| `crates/prism-gpu/src/context.rs` | Modify | P0 |
| `crates/prism-gpu/src/whcr.rs` | Modify | P0 |
| `crates/prism-gpu/src/thermodynamic.rs` | Modify | P0 |
| `crates/prism-gpu/src/quantum.rs` | Modify | P0 |
| `crates/prism-gpu/src/dendritic_reservoir.rs` | Modify | P0 |
| `crates/prism-gpu/src/kernels/dr_whcr_ultra.cu` | Create | P1 |
| `crates/prism-gpu/src/kernels/tptp.cu` | Create | P1 |
| `crates/prism-gpu/src/kernels/runtime_config.cuh` | Create | P0 |
| `crates/prism-gpu/src/aatgs.rs` | Create | P1 |
| `crates/prism-gpu/src/stream_manager.rs` | Create | P2 |
| `crates/prism-gpu/src/multi_device_pool.rs` | Modify | P3 |
| `crates/prism-gpu/build.rs` | Modify | P4 |
| `foundation/quantum/Cargo.toml` | Modify | P0 |
| `foundation/neuromorphic/Cargo.toml` | Modify | P0 |

### J.1.3 Code Patterns to Follow

**Pattern 1: Stream-based Kernel Launch**
```rust
// CORRECT: cudarc 0.18+ pattern
let stream = ctx.new_stream()?;
unsafe {
    stream.launch(&kernel, config, (param1, param2, param3))?;
}
// Non-blocking - returns immediately

// INCORRECT: cudarc 0.9 pattern (DO NOT USE)
// device.launch(&kernel, config, params)?; // Blocks until complete
```

**Pattern 2: Async Memory Copy**
```rust
// CORRECT: Async with stream
stream.memcpy_htod(&d_buffer, &h_data)?;
// Check completion later with stream.query() or stream.synchronize()

// INCORRECT: Sync copy (DO NOT USE)
// device.htod_sync_copy_into(&h_data, &mut d_buffer)?; // Blocks
```

**Pattern 3: Struct Parameter Passing**
```rust
// CORRECT: Single struct parameter
#[repr(C)]
struct KernelParams { /* all params */ }
let params = KernelParams { ... };
stream.launch(&kernel, config, (&d_params,))?;

// INCORRECT: Many individual parameters (hits 12-param limit)
// stream.launch(&kernel, config, (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13))?;
```

### J.1.4 Verification Commands

```bash
# After each file modification:
PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.6 \
cargo check -p prism-gpu

# After all GPU crate changes:
PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.6 \
cargo build --release -p prism-gpu

# Compile PTX:
for kernel in dr_whcr_ultra tptp runtime_config; do
    /usr/local/cuda-12.6/bin/nvcc --ptx \
        -o target/ptx/${kernel}.ptx \
        crates/prism-gpu/src/kernels/${kernel}.cu \
        -arch=sm_86 --std=c++14 -Xcompiler -fPIC \
        -I crates/prism-gpu/src/kernels
done
```

---

## J.2 prism-architect Agent

### J.2.1 Responsibilities

The `prism-architect` agent is responsible for:

1. **RuntimeConfig Definition** (Phase 0)
   - Create `crates/prism-core/src/runtime_config.rs`
   - Define 50+ parameter struct
   - Implement flag accessors

2. **MBRL World Model** (Phase 1D)
   - Create `crates/prism-fluxnet/src/mbrl.rs`
   - Implement GNN-based prediction
   - Implement MCTS action selection

3. **Ultra FluxNet Controller** (Phase 1E)
   - Create `crates/prism-fluxnet/src/ultra_controller.rs`
   - Implement discrete state/action spaces
   - Implement Q-learning with MBRL integration

4. **Config System Updates**
   - Modify `crates/prism-cli/src/config.rs` to use RuntimeConfig
   - Update pipeline integration

5. **Telemetry Schema Updates**
   - Update telemetry structs for new metrics
   - Add TPTP metrics (Betti numbers)
   - Add reservoir metrics

### J.2.2 File Ownership

| File | Action | Priority |
|------|--------|----------|
| `crates/prism-core/src/runtime_config.rs` | Create | P0 |
| `crates/prism-core/src/lib.rs` | Modify | P0 |
| `crates/prism-fluxnet/src/mbrl.rs` | Create | P1 |
| `crates/prism-fluxnet/src/ultra_controller.rs` | Create | P1 |
| `crates/prism-fluxnet/src/lib.rs` | Modify | P1 |
| `crates/prism-cli/src/config.rs` | Modify | P1 |
| `crates/prism-pipeline/src/telemetry.rs` | Modify | P2 |

### J.2.3 Code Patterns to Follow

**Pattern 1: #[repr(C)] for FFI Structs**
```rust
// CORRECT: C-compatible struct for GPU transfer
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeConfig {
    pub field: f32,
    // ...
}

// INCORRECT: Rust-only struct (will break GPU transfer)
// pub struct RuntimeConfig {
//     pub field: String, // Not FFI-safe!
// }
```

**Pattern 2: Deterministic State Discretization**
```rust
// CORRECT: Consistent discretization
impl DiscreteState {
    pub fn from_telemetry(t: &KernelTelemetry) -> Self {
        Self {
            bucket: match t.conflicts {
                0..=10 => 0,   // Low
                11..=50 => 1,  // Medium
                _ => 2,        // High
            },
        }
    }
}

// INCORRECT: Non-deterministic (random)
// bucket: rand::random::<u8>() % 3
```

**Pattern 3: Serializable Q-Tables**
```rust
// CORRECT: Use HashMap with bincode
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct QTable(HashMap<(State, Action), f64>);

impl QTable {
    pub fn save(&self, path: &str) -> Result<()> {
        let data = bincode::serialize(&self.0)?;
        std::fs::write(path, data)?;
        Ok(())
    }
}
```

### J.2.4 Verification Commands

```bash
# After RuntimeConfig:
cargo check -p prism-core

# After MBRL/FluxNet:
cargo check -p prism-fluxnet

# Full build:
cargo build --release
```

---

## J.3 prism-hypertuner Agent

### J.3.1 Responsibilities

The `prism-hypertuner` agent is responsible for:

1. **Configuration Analysis**
   - Analyze telemetry to identify bottlenecks
   - Recommend RuntimeConfig adjustments
   - Explain which parameters affect runtime

2. **Performance Diagnostics**
   - Interpret GPU utilization metrics
   - Identify phase transitions
   - Diagnose stagnation causes

3. **Parameter Tuning Guidance**
   - Chemical potential tuning
   - Temperature schedule optimization
   - Reservoir hyperparameter adjustment

### J.3.2 Knowledge Base

**Key Parameters and Effects**:

| Parameter | Increase Effect | Decrease Effect | Recommended Range |
|-----------|-----------------|-----------------|-------------------|
| `chemical_potential` | Stronger color compression | More color exploration | 0.5 - 5.0 |
| `tunneling_prob_base` | More escape moves | More stability | 0.01 - 0.3 |
| `global_temperature` | More exploration | More exploitation | 0.01 - 10.0 |
| `reservoir_leak_rate` | Faster adaptation | More memory | 0.1 - 0.5 |
| `belief_weight` | Stronger prior influence | More data-driven | 0.1 - 0.5 |
| `spectral_radius` | Richer dynamics | More stable | 0.8 - 0.99 |

**Diagnostic Patterns**:

| Symptom | Likely Cause | Recommended Action |
|---------|--------------|-------------------|
| High conflicts, stable | Stuck in local minimum | Increase tunneling, temperature |
| Oscillating conflicts | Cycling between states | Increase reservoir memory, decrease leak rate |
| Many colors, few conflicts | Over-exploration | Increase chemical potential |
| Slow convergence | Conservative parameters | Increase all exploration params |
| GPU underutilization | CPU bottleneck | Check async pipeline, increase batch |

---

# SECTION K: PTX COMPILATION

## K.1 Build Script Updates

**File**: `crates/prism-gpu/build.rs`

```rust
//! Build script for PRISM GPU kernels
//!
//! Compiles CUDA kernels to PTX for runtime loading.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/");

    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = format!("{}/bin/nvcc", cuda_home);

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_dir = out_dir.join("ptx");
    std::fs::create_dir_all(&ptx_dir).unwrap();

    // Kernel list with dependencies
    let kernels = vec![
        ("runtime_config", vec![]),  // Header only, compile first
        ("whcr", vec!["runtime_config.cuh"]),
        ("quantum", vec!["runtime_config.cuh"]),
        ("thermodynamic", vec!["runtime_config.cuh"]),
        ("dendritic_whcr", vec!["runtime_config.cuh"]),
        ("tda", vec!["runtime_config.cuh"]),
        ("dr_whcr_ultra", vec!["runtime_config.cuh"]),
        ("tptp", vec!["runtime_config.cuh"]),
    ];

    let include_dir = PathBuf::from("src/kernels");

    for (kernel, _deps) in kernels {
        let cu_file = format!("src/kernels/{}.cu", kernel);
        let ptx_file = ptx_dir.join(format!("{}.ptx", kernel));

        // Skip header-only files
        if !std::path::Path::new(&cu_file).exists() {
            continue;
        }

        let status = Command::new(&nvcc)
            .args(&[
                "--ptx",
                "-o", ptx_file.to_str().unwrap(),
                &cu_file,
                "-arch=sm_86",
                "--std=c++14",
                "-Xcompiler", "-fPIC",
                "-I", include_dir.to_str().unwrap(),
                "--use_fast_math",
                "-O3",
            ])
            .status()
            .expect("Failed to execute nvcc");

        if !status.success() {
            panic!("Failed to compile {}", cu_file);
        }

        println!("cargo:warning=Compiled {} -> {}", cu_file, ptx_file.display());
    }

    // Copy PTX files to target/ptx for runtime access
    let target_ptx = PathBuf::from("../../target/ptx");
    std::fs::create_dir_all(&target_ptx).unwrap();

    for entry in std::fs::read_dir(&ptx_dir).unwrap() {
        let entry = entry.unwrap();
        let dest = target_ptx.join(entry.file_name());
        std::fs::copy(entry.path(), dest).unwrap();
    }
}
```

## K.2 Manual PTX Compilation Commands

```bash
#!/bin/bash
# compile_ptx.sh - Manual PTX compilation script

CUDA_HOME=/usr/local/cuda-12.6
NVCC=$CUDA_HOME/bin/nvcc
PTX_DIR=target/ptx
KERNEL_DIR=crates/prism-gpu/src/kernels

mkdir -p $PTX_DIR

# Compilation flags
FLAGS="--ptx -arch=sm_86 --std=c++14 -Xcompiler -fPIC -I $KERNEL_DIR --use_fast_math -O3"

# Compile each kernel
for kernel in whcr quantum thermodynamic dendritic_whcr tda dr_whcr_ultra tptp; do
    CU_FILE=$KERNEL_DIR/${kernel}.cu
    PTX_FILE=$PTX_DIR/${kernel}.ptx

    if [ -f "$CU_FILE" ]; then
        echo "Compiling $kernel..."
        $NVCC $FLAGS -o $PTX_FILE $CU_FILE

        if [ $? -eq 0 ]; then
            echo "  ✓ $PTX_FILE"
        else
            echo "  ✗ Failed to compile $kernel"
            exit 1
        fi
    fi
done

echo "All PTX files compiled successfully!"
ls -la $PTX_DIR/*.ptx
```

---

# SECTION L: VERIFICATION CRITERIA

## L.1 Phase-by-Phase Verification

### L.1.1 Phase 0: cudarc Migration

**Checklist**:
- [ ] `Cargo.toml` updated to cudarc 0.18+
- [ ] `cargo check` passes with no errors
- [ ] Basic kernel launch test passes
- [ ] Stream creation works
- [ ] Async copy compiles
- [ ] All existing tests pass

**Test Command**:
```bash
cargo test -p prism-gpu -- --nocapture
```

### L.1.2 Phase 1A: Ultra Kernel

**Checklist**:
- [ ] `dr_whcr_ultra.cu` compiles to PTX
- [ ] Shared memory layout fits 98KB
- [ ] All 8 components implemented
- [ ] Kernel launches without error
- [ ] Conflicts decrease over iterations

**Test Command**:
```bash
# Compile PTX
$CUDA_HOME/bin/nvcc --ptx -o target/ptx/dr_whcr_ultra.ptx \
    crates/prism-gpu/src/kernels/dr_whcr_ultra.cu \
    -arch=sm_86 --std=c++14

# Test kernel
cargo test -p prism-gpu test_ultra_kernel -- --nocapture
```

### L.1.3 Phase 1B: TPTP

**Checklist**:
- [ ] `tptp.cu` compiles to PTX
- [ ] Betti numbers computed correctly
- [ ] Phase transitions detected
- [ ] Integration with ultra kernel works

### L.1.4 Phase 1C: AATGS

**Checklist**:
- [ ] `aatgs.rs` compiles
- [ ] Circular buffer logic correct
- [ ] Non-blocking operations work
- [ ] Triple-buffer synchronization correct

### L.1.5 Phase 1D: MBRL

**Checklist**:
- [ ] `mbrl.rs` compiles
- [ ] ONNX model loads
- [ ] Prediction returns valid output
- [ ] MCTS selects reasonable actions

### L.1.6 Phase 1E: Ultra FluxNet

**Checklist**:
- [ ] `ultra_controller.rs` compiles
- [ ] Q-table updates correctly
- [ ] Epsilon decay works
- [ ] Best config tracked

### L.1.7 Phase 2: Async Pipeline

**Checklist**:
- [ ] Stream manager creates streams
- [ ] Triple-buffer swaps correctly
- [ ] Pipeline stages overlap
- [ ] No blocking operations

### L.1.8 Phase 3: Multi-GPU

**Checklist**:
- [ ] Multi-device pool initializes
- [ ] P2P enabled where available
- [ ] Replica exchange works
- [ ] Cross-GPU sync correct

---

## L.2 Integration Tests

### L.2.1 Full Pipeline Test

**File**: `tests/integration/full_pipeline.rs`

```rust
#[test]
fn test_full_pipeline_dsjc500() {
    let graph = load_dimacs("benchmarks/dimacs/DSJC500.5.col");
    let config = RuntimeConfig::production();

    let mut pipeline = UltraPipeline::new(config)?;
    let result = pipeline.solve(&graph, 1000)?; // 1000 iterations

    assert!(result.conflicts == 0, "Should find valid coloring");
    assert!(result.colors_used <= 50, "DSJC500.5 chromatic number is ~48");
}
```

### L.2.2 Async Pipeline Test

```rust
#[test]
fn test_async_pipeline_no_blocking() {
    let ctx = CudaContext::new(0)?;
    let pipeline = AsyncPipelineCoordinator::new(Arc::new(ctx))?;

    let start = std::time::Instant::now();

    // Queue 100 configs rapidly
    for i in 0..100 {
        let config = RuntimeConfig { iteration: i, ..Default::default() };
        pipeline.begin_config_upload(config)?;
    }

    let queue_time = start.elapsed();

    // Should be very fast (< 10ms) if truly async
    assert!(queue_time.as_millis() < 10, "Queueing should be non-blocking");
}
```

---

# SECTION M: BENCHMARKS

## M.1 DIMACS Benchmark Suite

### M.1.1 Target Graphs

| Graph | Vertices | Edges | Density | χ (Best Known) | Target Colors |
|-------|----------|-------|---------|----------------|---------------|
| DSJC125.1 | 125 | 736 | 0.094 | 5 | 5 |
| DSJC125.5 | 125 | 3,891 | 0.500 | 17 | 17 |
| DSJC125.9 | 125 | 6,961 | 0.898 | 44 | 44 |
| DSJC250.1 | 250 | 3,218 | 0.103 | 8 | 8 |
| DSJC250.5 | 250 | 15,668 | 0.503 | 28 | 28 |
| DSJC250.9 | 250 | 27,897 | 0.896 | 72 | 72 |
| DSJC500.1 | 500 | 12,458 | 0.100 | 12 | 12 |
| DSJC500.5 | 500 | 62,624 | 0.501 | 48 | 48 |
| DSJC500.9 | 500 | 112,437 | 0.900 | 126 | 126 |
| DSJC1000.1 | 1000 | 49,629 | 0.099 | 20 | 20 |
| DSJC1000.5 | 1000 | 249,826 | 0.500 | 83 | 83 |
| DSJC1000.9 | 1000 | 449,449 | 0.899 | 223 | 223 |

### M.1.2 Benchmark Script

```bash
#!/bin/bash
# benchmark.sh - Run DIMACS benchmarks

PRISM_CLI=./target/release/prism-cli
RESULTS_DIR=benchmark_results/$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULTS_DIR

GRAPHS=(
    "DSJC125.1" "DSJC125.5" "DSJC125.9"
    "DSJC250.1" "DSJC250.5" "DSJC250.9"
    "DSJC500.1" "DSJC500.5" "DSJC500.9"
    "DSJC1000.1" "DSJC1000.5" "DSJC1000.9"
)

echo "PRISM Ultra Benchmark Suite"
echo "=========================="
echo ""

for graph in "${GRAPHS[@]}"; do
    echo "Testing $graph..."

    GRAPH_FILE="benchmarks/dimacs/${graph}.col"
    RESULT_FILE="$RESULTS_DIR/${graph}.json"

    # Run with timeout
    timeout 300 $PRISM_CLI \
        --graph $GRAPH_FILE \
        --config configs/production.toml \
        --telemetry $RESULT_FILE \
        --max-iterations 10000

    if [ $? -eq 0 ]; then
        COLORS=$(jq '.colors_used' $RESULT_FILE)
        CONFLICTS=$(jq '.final_conflicts' $RESULT_FILE)
        TIME=$(jq '.total_time_ms' $RESULT_FILE)
        GPU_UTIL=$(jq '.avg_gpu_utilization' $RESULT_FILE)

        echo "  Colors: $COLORS | Conflicts: $CONFLICTS | Time: ${TIME}ms | GPU: ${GPU_UTIL}%"
    else
        echo "  TIMEOUT or ERROR"
    fi

    echo ""
done

echo "Results saved to $RESULTS_DIR"
```

## M.2 Performance Targets

### M.2.1 Speed Targets

| Graph | CPU Baseline | GPU Target | Speedup Target |
|-------|-------------|------------|----------------|
| DSJC500.5 | ~60s | <6s | >10x |
| DSJC1000.5 | ~300s | <30s | >10x |

### M.2.2 Quality Targets

| Metric | Target |
|--------|--------|
| Zero conflicts | 100% of runs |
| Within 5% of χ | >90% of runs |
| GPU utilization | >80% |
| Memory efficiency | <4GB for DSJC1000 |

### M.2.3 Async Pipeline Targets

| Metric | Target |
|--------|--------|
| CPU-GPU overlap | >90% |
| Config latency | <1ms |
| Telemetry latency | <1ms |
| Zero-copy transfers | Where possible |

---

# SECTION N: EXECUTION ORDER

## N.1 Recommended Execution Sequence

```
Phase 0 (Foundation) ─────────────────────────────────────────────
   │
   ├── 0.1: Update Cargo.toml (cudarc 0.18+)
   ├── 0.2: Create RuntimeConfig struct
   ├── 0.3: Create runtime_config.cuh header
   ├── 0.4: Migrate context.rs
   ├── 0.5: Migrate whcr.rs
   ├── 0.6: Migrate thermodynamic.rs
   ├── 0.7: Migrate quantum.rs
   ├── 0.8: Migrate dendritic_reservoir.rs
   └── 0.9: VERIFY: cargo check passes
   │
Phase 1 (Kernel Components) ──────────────────────────────────────
   │
   ├── 1A: Create dr_whcr_ultra.cu
   │     ├── 1A.1: Shared memory layout
   │     ├── 1A.2: W-cycle multigrid
   │     ├── 1A.3: Dendritic reservoir
   │     ├── 1A.4: Quantum tunneling
   │     ├── 1A.5: Parallel tempering
   │     ├── 1A.6: Active inference
   │     ├── 1A.7: WHCR moves
   │     └── 1A.8: VERIFY: PTX compiles
   │
   ├── 1B: Create tptp.cu
   │     ├── 1B.1: Simplicial complex
   │     ├── 1B.2: Homology computation
   │     ├── 1B.3: Transition detection
   │     └── 1B.4: VERIFY: PTX compiles
   │
   ├── 1C: Create aatgs.rs
   │     ├── 1C.1: Circular buffers
   │     ├── 1C.2: Async operations
   │     └── 1C.3: VERIFY: cargo check
   │
   ├── 1D: Create mbrl.rs
   │     ├── 1D.1: World model
   │     ├── 1D.2: MCTS
   │     └── 1D.3: VERIFY: cargo check
   │
   └── 1E: Create ultra_controller.rs
         ├── 1E.1: State discretization
         ├── 1E.2: Q-learning
         └── 1E.3: VERIFY: cargo check
   │
Phase 2 (Async Pipeline) ─────────────────────────────────────────
   │
   ├── 2.1: Create stream_manager.rs
   ├── 2.2: Implement triple-buffering
   ├── 2.3: Wire up pipeline coordinator
   └── 2.4: VERIFY: async tests pass
   │
Phase 3 (Multi-GPU) ──────────────────────────────────────────────
   │
   ├── 3.1: Update multi_device_pool.rs
   ├── 3.2: Enable P2P
   ├── 3.3: Implement replica exchange
   └── 3.4: VERIFY: multi-GPU tests pass
   │
Phase 4 (PTX + Integration) ──────────────────────────────────────
   │
   ├── 4.1: Update build.rs
   ├── 4.2: Compile all PTX
   ├── 4.3: Create Rust wrappers
   └── 4.4: VERIFY: full build succeeds
   │
Phase 5 (Benchmarks) ─────────────────────────────────────────────
   │
   ├── 5.1: Run DSJC500.5
   ├── 5.2: Run DSJC1000.5
   └── 5.3: Collect metrics
   │
Phase 6 (Validation) ─────────────────────────────────────────────
   │
   ├── 6.1: Verify >80% GPU utilization
   ├── 6.2: Verify >10x speedup
   ├── 6.3: Verify zero conflicts
   └── 6.4: DONE
```

## N.2 Parallel Execution Opportunities

The following can be executed in parallel:

**Parallel Group 1** (Phase 0):
- prism-gpu-specialist: Cargo.toml + context.rs migration
- prism-architect: RuntimeConfig struct

**Parallel Group 2** (Phase 1):
- prism-gpu-specialist: dr_whcr_ultra.cu + tptp.cu
- prism-architect: mbrl.rs + ultra_controller.rs

**Parallel Group 3** (Phase 2-3):
- prism-gpu-specialist: stream_manager.rs + multi_device_pool.rs

---

# END OF IMPLEMENTATION PLAN

## Document Summary

| Part | Sections | Content |
|------|----------|---------|
| Part 1 | A-B | Architecture, Phase 0 cudarc migration |
| Part 2 | C-E | Ultra kernel, TPTP, AATGS |
| Part 3 | F-I | MBRL, FluxNet, Async, Multi-GPU |
| Part 4 | J-N | Agent instructions, verification, benchmarks |

**Total Implementation Scope**:
- ~3000 LOC CUDA (Ultra kernel)
- ~800 LOC CUDA (TPTP)
- ~2000 LOC Rust modifications (cudarc migration)
- ~1600 LOC Rust new (AATGS, MBRL, FluxNet, Stream Manager)
- ~600 LOC Rust modifications (Multi-GPU)

**Estimated Effort**: 4 phases, highly parallelizable with specialized agents.
