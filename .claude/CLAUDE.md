# PRISM-Fold Project Configuration

## Project Overview
PRISM-Fold: Phase Resonance Integrated Solver Machine for Molecular Folding
- GPU-accelerated graph coloring and ligand binding site prediction
- 7-phase optimization pipeline with FluxNet RL
- Production-grade CUDA kernels (18 PTX files)

## Environment
```bash
CUDA_HOME=/usr/local/cuda-12.6
PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
```

## Build Commands
```bash
# Standard build
cargo build --release --features cuda

# With CUDA environment
CUDA_HOME=/usr/local/cuda-12.6 cargo build --release --features cuda

# Check only
cargo check --features cuda

# Run tests
cargo test --all-features

# Compile PTX
nvcc --ptx -o kernels/ptx/whcr.ptx prism-gpu/src/kernels/whcr.cu -arch=sm_86 --std=c++14
```

---

## Specialized Agent Registry

This project has custom slash commands that serve as specialized agents for implementation tasks.

### Technical Debt Agents

| Command | Purpose | Invocation |
|---------|---------|------------|
| `/prism-cudarc-migrator` | Upgrade cudarc 0.9→0.18.1 | Phase 0.5.1 |
| `/prism-whcr-refactor` | WHCR parameter struct pattern | Phase 0.5.2 |
| `/prism-config-serde` | Serde config parsing migration | Phase 0.5.3 |

### Integration Agents

| Command | Purpose | Invocation |
|---------|---------|------------|
| `/prism-onnx-integrator` | Wire ONNX GNN model | Phase 1 |
| `/prism-lbs-optimizer` | Optimize LBS GPU kernels | Phase 2 |
| `/prism-fluxnet-unifier` | Consolidate FluxNet RL | Phase 3 |

### Verification Agents

| Command | Purpose | Invocation |
|---------|---------|------------|
| `/prism-benchmark` | DIMACS/PDBBind/DUD-E benchmarks | Phase 4 |
| `/prism-master-orchestrator` | Coordinate parallel execution | Meta |

---

## Implementation Plan

### Phase 0.5: Technical Debt Resolution (Pre-requisite)
- [ ] 0.5.1: Cudarc 0.9 → 0.18.1 upgrade (400-600 LOC)
- [ ] 0.5.2: WHCR parameter struct pattern (150-250 LOC)
- [ ] 0.5.3: Serde config parsing migration (200-300 LOC)

### Phase 1: ONNX GNN Integration (Days 3-5)
- [ ] Add `ort` crate with CUDA EP
- [ ] Implement real inference in prism-gnn
- [ ] Wire to LBS pocket detection

### Phase 2: LBS Kernel Optimization (Days 6-12)
- [ ] SASA spatial grid (O(N²) → O(N×27))
- [ ] Jones-Plassmann pocket clustering
- [ ] Batched distance matrix

### Phase 3: FluxNet Unification (Days 13-16)
- [ ] Re-export prct-core/fluxnet
- [ ] Add LBS-specific state extensions
- [ ] Integrate curriculum learning

### Phase 4: WHCR Cross-Integration (Days 17-19)
- [ ] Wire WHCR to LBS pocket repair
- [ ] Add LBS-specific configuration

### Phase 5: Testing & Benchmarks (Days 20-23)
- [ ] DIMACS benchmark suite
- [ ] PDBBind/DUD-E validation
- [ ] GPU performance profiling

### Phase 6: Production Hardening (Days 24-28)
- [ ] Error recovery & fallbacks
- [ ] Telemetry integration
- [ ] Documentation

---

## Key File Locations

### Core Implementation
- `crates/prism-gpu/src/whcr.rs` - WHCR GPU (1,060 LOC)
- `foundation/prct-core/src/gpu_thermodynamic.rs` - GPU SA (1,640 LOC)
- `foundation/prct-core/src/fluxnet/controller.rs` - FluxNet RL (960 LOC)
- `crates/prism-phases/src/dendritic_reservoir_whcr.rs` - Neuromorphic (876 LOC)

### PTX Kernels
- `kernels/ptx/whcr.ptx` - 97 KB
- `kernels/ptx/thermodynamic.ptx` - 1 MB
- `kernels/ptx/quantum.ptx` - 1.2 MB
- `kernels/ptx/dendritic_whcr.ptx` - 1 MB

### Models
- `models/gnn/gnn_model.onnx` - 440 KB
- `models/gnn/gnn_model.onnx.data` - 5.4 MB

---

## Critical Rules

1. **GPU-First**: All compute-intensive operations MUST use GPU
2. **PTX Required**: No proceeding without compiled PTX kernels
3. **Zero Conflicts**: Pipeline must achieve 0 conflicts on DIMACS
4. **Performance**: GPU utilization must exceed 80%
5. **No Regressions**: All existing tests must continue passing

---

## Parallel Execution Strategy

For maximum efficiency, launch independent agents in parallel:

```
Phase 0.5 Parallel Tracks:
├─ Track A: cudarc-migrator → whcr-refactor
├─ Track B: config-serde (independent)
└─ Track C: fluxnet-unifier (independent)

Phase 1 Parallel Tracks:
├─ Track A: onnx-integrator
└─ Track B: lbs-optimizer
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| DSJC500.5 Colors | ≤48 |
| GPU Utilization | ≥80% |
| PDBBind Success | ≥70% DCC<4Å |
| Build Time | <2 min |
| Memory Usage | <4GB VRAM |

---

## Copyright
```
Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
Los Angeles, CA 90013
Contact: IS@Delfictus.com
All Rights Reserved.
```
