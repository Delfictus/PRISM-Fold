# PRISM-LBS Implementation Handoff Document

**Date**: 2025-11-29
**Session Focus**: Full LBS Engine Implementation with Complex<f32> Quantum Fix

---

## Executive Summary

This session focused on deploying parallel specialized agents to implement a flagship Ligand Binding Site (LBS) prediction engine for PRISM. The implementation was interrupted but comprehensive specifications were prepared for all components.

---

## Critical Bug Identified & Fix Specified

### Complex<f32> Quantum Amplitudes (CRITICAL)

**Problem**: All quantum kernels use real-only f32 amplitudes, destroying phase information:
```cuda
// WRONG - missing imaginary component
float amp = amplitudes[idx];
amplitudes[idx] = amp * cosf(phase);  // WHERE IS SIN?
```

**Solution**: Dual-buffer Complex<f32> with proper rotation:
```cuda
// CORRECT
amplitudes_real[idx] = real * cos_phase - imag * sin_phase;
amplitudes_imag[idx] = real * sin_phase + imag * cos_phase;
```

**Files Requiring Update**:
- `crates/prism-gpu/src/quantum.rs`
- `foundation/prct-core/src/gpu_quantum.rs`
- `foundation/prct-core/src/gpu_quantum_multi.rs`
- `foundation/prct-core/src/gpu_quantum_annealing.rs`
- `crates/prism-phases/src/phase3_quantum.rs`
- All related PTX kernels

---

## Implementation Tasks (8 Parallel Workstreams)

### 1. Complex<f32> Quantum Fix
**Status**: Specification complete, implementation pending
**Priority**: CRITICAL - Mathematical correctness bug

Key changes:
- Add `d_amplitudes_imag` buffer alongside `d_amplitudes_real`
- Update all evolution kernels for complex rotation
- Add norm preservation tests
- Recompile all quantum PTX

### 2. LBS GPU Kernel Optimization
**Status**: Specification complete, implementation pending
**Target Performance**:
| Kernel | Current | Target |
|--------|---------|--------|
| SASA (10K atoms) | ~2000ms | <20ms |
| Pocket Clustering | Race conditions | Race-free |
| Distance Matrix | O(N²) memory | <4GB VRAM |

**New Kernels to Create**:
- `crates/prism-gpu/src/kernels/lbs/surface_accessibility.cu` (Spatial grid SASA)
- `crates/prism-gpu/src/kernels/lbs/pocket_clustering.cu` (Jones-Plassmann)
- `crates/prism-gpu/src/kernels/lbs/distance_matrix.cu` (Batched tiled)

### 3. WHCR Refinement Implementation
**Status**: Specification complete, implementation pending
**Current State**: Placeholder at `crates/prism-lbs/src/phases/whcr_refinement.rs`

Required implementation:
- Wavelet-based conflict detection (Haar decomposition)
- Cavity-aware refinement (merge connected regions)
- Topological persistence filtering
- Reservoir priority integration
- Iterative conflict resolution loop

### 4. Phase 3 Quantum for LBS
**Status**: Specification complete, implementation pending
**File to Create**: `crates/prism-lbs/src/phases/phase3_quantum.rs`

Features:
- Path Integral Monte Carlo (PIMC) replica evolution
- Quantum tunneling for escaping local minima
- Superposition for boundary uncertainty
- Annealing schedule for global optimization

### 5. ONNX GNN Integration
**Status**: Specification complete, implementation pending

Changes:
- Add `ort` crate with CUDA EP to `crates/prism-gnn/Cargo.toml`
- Create `crates/prism-gnn/src/onnx_runtime.rs`
- Wire into `crates/prism-lbs/src/pocket/detector.rs`

### 6. FluxNet GPU Training
**Status**: Specification complete, implementation pending

New components:
- GPU reward kernel at `crates/prism-gpu/src/kernels/fluxnet_reward.cu`
- Batch training in `crates/prism-lbs/src/training/trainer.rs`
- Momentum SGD with GPU-accelerated gradient computation

### 7. Benchmark Infrastructure
**Status**: Specification complete, implementation pending

Files to create:
- `crates/prism/benches/dimacs.rs`
- `crates/prism-lbs/benches/industry.rs`
- `crates/prism-gpu/benches/kernels.rs`
- `tests/integration/pipeline_e2e.rs`

### 8. TUI LBS Integration
**Status**: Specification complete, implementation pending

Updates needed:
- Add LBS events to `crates/prism/src/runtime/events.rs`
- Add LbsState to `crates/prism/src/runtime/state.rs`
- Extend `crates/prism/src/runtime/pipeline_bridge.rs`
- Add biomolecular mode to `crates/prism/src/main.rs`

---

## Specialized Agent Commands Available

| Command | Purpose |
|---------|---------|
| `/prism-lbs-optimizer` | LBS GPU kernel optimization |
| `/prism-whcr-refactor` | WHCR parameter struct pattern |
| `/prism-onnx-integrator` | ONNX GNN integration |
| `/prism-fluxnet-unifier` | FluxNet consolidation |
| `/prism-benchmark` | Benchmark infrastructure |
| `/prism-master-orchestrator` | Coordinate parallel execution |

---

## File References

### Already Read This Session
- `crates/prism/src/runtime/pipeline_bridge.rs` - Pipeline execution bridge
- `crates/prism/src/runtime/events.rs` - Event system
- `crates/prism-lbs/src/phases/whcr_refinement.rs` - PLACEHOLDER (needs replacement)
- `crates/prism-lbs/src/training/trainer.rs` - Training loop
- `crates/prism-lbs/src/phases/mod.rs` - Phase exports (missing Phase 3)

### Command Specifications Read
- `.claude/commands/prism-lbs-optimizer.md`
- `.claude/commands/prism-onnx-integrator.md`
- `.claude/commands/prism-fluxnet-unifier.md`
- `.claude/commands/prism-whcr-refactor.md`
- `.claude/commands/prism-benchmark.md`
- `.claude/commands/prism-master-orchestrator.md`

---

## Dendritic Reservoir Wiring (COMPLETED)

Previous session completed wiring Phase 0 outputs to all phases:

| Phase | Integration |
|-------|-------------|
| Phase 1 Active Inference | `apply_dendritic_difficulty()` boosts exploration for hard vertices |
| Phase 2 Thermodynamic | Temperature adjustment based on mean_difficulty |
| Phase 3 Quantum | Coupling strength modulation from uncertainty |
| Phase 4 Geodesic | Centrality boosting for high-difficulty vertices |
| Phase 6 TDA | Persistence threshold scaling from uncertainty |
| Phase 7 Ensemble | Diversity weight adjustment |

---

## Build Environment

```bash
CUDA_HOME=/usr/local/cuda-12.6
PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"

# Build command
CUDA_HOME=/usr/local/cuda-12.6 cargo build --release --features cuda

# PTX compilation
/usr/local/cuda-12.6/bin/nvcc --ptx -o kernels/ptx/NAME.ptx \
    crates/prism-gpu/src/kernels/NAME.cu \
    -arch=sm_86 --std=c++14 -Xcompiler -fPIC
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| DSJC500.5 Colors | ≤48 |
| GPU Utilization | ≥80% |
| PDBBind Success | ≥70% DCC<4Å |
| Quantum Norm Drift | <1e-4 after 10K steps |
| SASA Latency (10K atoms) | <20ms |
| Build Time | <2 min |

---

## Next Steps (Priority Order)

1. **CRITICAL**: Implement Complex<f32> quantum fix across all kernels
2. Create LBS GPU kernels (SASA, clustering, distance)
3. Replace WHCR refinement placeholder with full implementation
4. Add Phase 3 Quantum for LBS
5. Wire ONNX GNN integration
6. Implement benchmark infrastructure
7. Integrate LBS into TUI
8. Run full test suite
9. Commit and push on green

---

## Notes

- cudarc 0.9 is retained (0.18.1 migration deferred - adds complexity without perf gain)
- WHCR parameter struct pattern already completed
- FluxNet already unified with prct-core re-exports
- All specifications in this document are ready for direct implementation
