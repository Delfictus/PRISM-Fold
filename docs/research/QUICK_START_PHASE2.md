# Phase 2 Thermodynamic Quick Start Guide

## TL;DR: Run DSJC250 with Full GPU Acceleration

```bash
# 1. Build and verify
./scripts/verify_build.sh

# 2. Train Q-table (1000 epochs, ~5 minutes)
./scripts/train_fluxnet_dsjc250.sh

# 3. Run aggressive mode
./target/release/prism-cli \
  --input benchmarks/dimacs/DSJC250.5.col \
  --warmstart \
  --warmstart-anchor-fraction 0.15 \
  --gpu \
  --fluxnet-qtable profiles/curriculum/qtable_dsjc250.bin \
  --verbose
```

---

## What Was Implemented

### Phase 2 Thermodynamic Simulated Annealing

**Before**: CPU-only stub with no functionality
**After**: GPU-accelerated parallel tempering with 16 temperature replicas

**Key Features**:
- 50,000 annealing iterations (configurable)
- Metropolis-Hastings acceptance criterion
- Replica exchange every 100 iterations
- Warmstart support (uses prior coloring)
- FluxNet RL parameter tuning

---

## File Locations

### Core Implementation
```
prism-gpu/src/
├── kernels/thermodynamic.cu          # CUDA kernel (978 KB PTX)
└── thermodynamic.rs                  # Rust GPU wrapper

prism-phases/src/
└── phase2_thermodynamic.rs           # Phase controller with RL

prism-fluxnet/src/
└── bin/train.rs                      # Q-table training binary
```

### Configuration & Scripts
```
configs/
└── dsjc250_aggressive.toml           # Production config

scripts/
├── verify_build.sh                   # Build verification
├── train_fluxnet_dsjc250.sh         # Train Q-table
└── test_dsjc250_aggressive.sh       # Integration test
```

### Compiled Artifacts
```
target/
├── ptx/thermodynamic.ptx            # GPU kernel (978 KB)
└── release/
    ├── prism-cli                    # Main CLI
    └── fluxnet_train                # Training binary
```

---

## Command Reference

### Build Commands

```bash
# Full build with GPU features
cargo build --release --features "cuda,gpu"

# Build just the CLI
cargo build --release --bin prism-cli

# Build training binary
cargo build --release --bin fluxnet_train

# Run tests (requires GPU)
cargo test --features cuda
```

### PTX Compilation

```bash
# Compile thermodynamic kernel
/usr/local/cuda-12.6/bin/nvcc --ptx \
  -o target/ptx/thermodynamic.ptx \
  prism-gpu/src/kernels/thermodynamic.cu \
  -arch=sm_86 --use_fast_math -O3
```

### Training Commands

```bash
# Train Q-table (default: 1000 epochs)
./scripts/train_fluxnet_dsjc250.sh

# Or manually:
./target/release/fluxnet_train \
  benchmarks/dimacs/DSJC250.5.col \
  1000 \
  profiles/curriculum/qtable_dsjc250.bin
```

### Benchmark Commands

```bash
# Baseline (no warmstart, CPU)
./target/release/prism-cli --input benchmarks/dimacs/DSJC250.5.col

# With warmstart
./target/release/prism-cli \
  --input benchmarks/dimacs/DSJC250.5.col \
  --warmstart

# With warmstart + GPU
./target/release/prism-cli \
  --input benchmarks/dimacs/DSJC250.5.col \
  --warmstart \
  --gpu

# Full aggressive mode (warmstart + GPU + Q-table)
./target/release/prism-cli \
  --input benchmarks/dimacs/DSJC250.5.col \
  --warmstart \
  --warmstart-anchor-fraction 0.15 \
  --gpu \
  --fluxnet-qtable profiles/curriculum/qtable_dsjc250.bin \
  --fluxnet-epsilon 0.2 \
  --verbose
```

---

## CLI Options (Phase 2 & FluxNet)

### Warmstart Options
```
--warmstart                          Enable warmstart system
--warmstart-anchor-fraction 0.15     Use 15% structural anchors
--warmstart-flux-weight 0.5          Weight for reservoir priors
--warmstart-ensemble-weight 0.3      Weight for ensemble methods
--warmstart-random-weight 0.2        Weight for randomization
```

### GPU Options
```
--gpu                                Enable GPU acceleration
--gpu-devices 0                      CUDA device ID
--gpu-ptx-dir target/ptx            PTX kernel directory
```

### FluxNet RL Options
```
--fluxnet-qtable <path>              Load pretrained Q-table (binary)
--fluxnet-epsilon 0.2                Exploration rate (0-1)
--fluxnet-alpha 0.1                  Learning rate (0-1)
--fluxnet-gamma 0.95                 Discount factor (0-1)
```

---

## Expected Results (DSJC250.5)

### Baseline (CPU, no warmstart)
- **Chromatic number**: ~50-60 colors
- **Time**: 5-10 seconds
- **Method**: Greedy DSATUR

### Warmstart + GPU (no Q-table)
- **Chromatic number**: ~38-45 colors
- **Time**: 30-60 seconds
- **Method**: Dendritic reservoir + parallel tempering

### Aggressive (warmstart + GPU + Q-table)
- **Chromatic number**: ~32-40 colors (target: 28)
- **Time**: 1-2 minutes
- **Method**: Full stack with RL parameter tuning

---

## Verification Checklist

### Build Verification
- [ ] Rust toolchain installed (`cargo --version`)
- [ ] CUDA 12.6+ installed (`nvcc --version`)
- [ ] PTX kernel compiled (`ls target/ptx/thermodynamic.ptx`)
- [ ] CLI binary built (`./target/release/prism-cli --help`)
- [ ] Training binary built (`./target/release/fluxnet_train --help`)

### Runtime Verification
- [ ] Phase 2 logs "GPU: true"
- [ ] No "falling back to CPU" warnings
- [ ] Warmstart produces < 45 colors (initial)
- [ ] Phase 2 reduces chromatic number
- [ ] Q-table loads successfully (if specified)

### Output Verification
- [ ] Final coloring is valid (no conflicts)
- [ ] Chromatic number ≤ initial coloring
- [ ] Telemetry shows temperature, compaction_ratio
- [ ] GPU metrics available (if --enable-metrics)

---

## Troubleshooting

### Problem: Build fails with "nvcc not found"

**Solution**: Install CUDA Toolkit 12.6 or add to PATH:
```bash
export PATH="/usr/local/cuda-12.6/bin:$PATH"
```

### Problem: "PTX not found" at runtime

**Solution**: Compile PTX kernels:
```bash
./scripts/verify_build.sh
```

### Problem: Phase 2 falls back to CPU

**Cause**: GPU initialization failed or PTX missing

**Solution**:
1. Check CUDA device: `nvidia-smi`
2. Verify PTX: `ls -l target/ptx/thermodynamic.ptx`
3. Check logs: look for "GPU init failed" message

### Problem: Q-table fails to load

**Cause**: File not found or incompatible format

**Solution**:
```bash
# Train new Q-table
./scripts/train_fluxnet_dsjc250.sh

# Or use JSON format
./target/release/prism-cli \
  --fluxnet-qtable profiles/curriculum/qtable_dsjc250.json
```

### Problem: Out of GPU memory

**Cause**: Graph too large or too many replicas

**Solution**: Reduce num_replicas in configuration:
```toml
[phase2_thermodynamic]
num_replicas = 8  # Reduce from 16
```

---

## Performance Tuning

### For Speed (Faster Results)
```toml
[phase2_thermodynamic]
iterations = 10000      # Reduce from 50000
num_replicas = 8        # Reduce from 16
```

### For Quality (Better Chromatic Number)
```toml
[phase2_thermodynamic]
iterations = 100000     # Increase to 100K
num_replicas = 32       # More temperature schedules
temp_min = 0.0001      # Lower minimum
temp_max = 50.0        # Higher maximum
```

### For World Record Attempts
```toml
[phase2_thermodynamic]
iterations = 1000000    # 1 million iterations
num_replicas = 64       # Maximum parallelism
cooling_rate = 0.995    # Very slow cooling
```

---

## Next Steps

1. **Verify Installation**
   ```bash
   ./scripts/verify_build.sh
   ```

2. **Train Q-Table**
   ```bash
   ./scripts/train_fluxnet_dsjc250.sh
   ```

3. **Run Integration Test**
   ```bash
   ./scripts/test_dsjc250_aggressive.sh
   ```

4. **Benchmark Your Hardware**
   ```bash
   ./target/release/prism-cli \
     --input benchmarks/dimacs/DSJC250.5.col \
     --warmstart --gpu \
     --fluxnet-qtable profiles/curriculum/qtable_dsjc250.bin \
     --verbose
   ```

5. **Tune for World Record**
   - Edit `configs/dsjc250_aggressive.toml`
   - Increase iterations to 500K-1M
   - Add more replicas (32-64)
   - Run extended benchmark

---

## Support

- **Full Documentation**: `PHASE2_IMPLEMENTATION_SUMMARY.md`
- **Specification**: `docs/spec/prism_gpu_plan.md` (§4.2)
- **Test Suite**: `prism-phases/tests/phase2_gpu_smoke.rs`
- **Example Config**: `configs/dsjc250_aggressive.toml`

---

**Last Updated**: 2025-11-18
**Version**: PRISM v0.2.0
