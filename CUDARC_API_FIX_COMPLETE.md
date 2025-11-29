# cudarc 0.18.1 API Migration - Status Report

**Date**: 2025-11-29  
**Task**: Fix all `get_func` and `load_ptx` method calls for cudarc 0.18.1 compatibility

## Executive Summary

- **Total Errors**: ~29 locations across 31 files
- **Status**: 2 files fixed (6.5%), 29 files remaining (93.5%)
- **Critical Path**: 4 P0 files blocking pipeline compilation

## What Changed

### cudarc 0.9/0.11 → 0.18.1 API Breaking Changes

| Old API | New API |
|---------|---------|
| `context.load_ptx(ptx, "module", &["kernels"])` | `context.load_module(Ptx::Image(bytes))` |
| `context.get_func("module", "kernel")` | `module.load_function("kernel")` |
| Returns `()` | Returns `CudaModule` |
| Module name required | Module name not used |
| Kernel list required | Kernels loaded on-demand |

### Key Differences

1. **Module Handle**: New API returns `CudaModule` which must be stored
2. **Lazy Loading**: Kernels loaded when needed, not at module load time
3. **Byte Arrays**: `Ptx::Image(&[u8])` instead of string-based PTX
4. **No Module Names**: Module names no longer part of API

## Files Fixed ✅

### 1. crates/prism-gpu/src/whcr.rs
```rust
// OLD (lines 118-167):
context.load_ptx(ptx, "whcr", &[...kernels...])?;
let kernel = context.get_func("whcr", "count_conflicts_f32")?;

// NEW (lines 122-145):
let module = device.load_module(Ptx::Image(&ptx_data))?;
let count_conflicts_f32 = module.load_function("count_conflicts_f32")?;
```
- **Impact**: Core WHCR repair engine
- **Kernels**: 8 kernels (conflict counting, wavelet, move evaluation, locking)

### 2. crates/prism-gpu/src/dendritic_whcr.rs
```rust
// OLD (lines 132-191):
context.load_ptx(ptx_content.into(), "dendritic_whcr", &[...kernels...])?;
let kernel = context.get_func("dendritic_whcr", "init_vertex_states")?;

// NEW (lines 130-160):
let module = device.load_module(Ptx::Image(&ptx_data))?;
let init_vertex_states = module.load_function("_Z18init_vertex_statesP...")?;
```
- **Impact**: Neuromorphic reservoir for adaptive repair
- **Kernels**: 7 kernels (dendritic processing, soma integration, priority modulation)

## Files Requiring Fixes ⚠️

### Priority 0 - Critical (Blocks Pipeline) 🔥

1. **foundation/prct-core/src/gpu_thermodynamic.rs**
   - Lines ~96-254: load_ptx + 8 get_func calls
   - Kernels: initialize_oscillators, compute_coupling_forces, evolve_oscillators, etc.
   - Impact: Phase 2 thermodynamic annealing

2. **foundation/prct-core/src/gpu_quantum.rs**
   - Lines ~156-200: compile_ptx + 4 get_func calls
   - Kernels: complex_matvec, complex_axpy, complex_norm_squared, complex_normalize
   - Impact: Phase 3 quantum evolution

3. **crates/prism-gpu/src/thermodynamic.rs**
   - Lines ~62-95: load_ptx + 2 get_func calls
   - Kernels: parallel_tempering_step, replica_swap
   - Impact: GPU-accelerated simulated annealing

4. **crates/prism-gpu/src/quantum.rs**
   - Estimated similar pattern
   - Impact: Quantum Hamiltonian operations

### Priority 1 - High (Core Functionality) ⭐

5. **crates/prism-gpu/src/floyd_warshall.rs** (lines ~70-250)
6. **crates/prism-gpu/src/lbs.rs** (lines ~19-316) **SPECIAL CASE**
7. **crates/prism-gpu/src/cma.rs** (lines ~150-514)
8. **crates/prism-gpu/src/active_inference.rs**

### Priority 2 - Medium (Supporting) 📦

9. crates/prism-gpu/src/aatgs.rs
10. crates/prism-gpu/src/tda.rs
11. foundation/neuromorphic/src/cuda_kernels.rs
12. foundation/quantum/src/gpu_coloring.rs
13-22. (10 more files)

### Priority 3 - Low (Integration/Utils) 📋

23-31. (9 files: multi_gpu, stream management, etc.)

## Special Cases

### LBS Module (Multiple PTX Files)
**File**: `crates/prism-gpu/src/lbs.rs`

**Challenge**: Loads 4 separate PTX modules:
- lbs_surface_accessibility.ptx
- lbs_distance_matrix.ptx  
- lbs_pocket_clustering.ptx
- lbs_druggability_scoring.ptx

**Solution**: Store all modules in struct:
```rust
pub struct LbsGpu {
    device: Arc<CudaContext>,
    modules: Vec<CudaModule>,  // Store all loaded modules
}
```

Then load functions on-demand from appropriate module.

### Inline NVRTC Compilation
**Files**: gpu_quantum.rs, potentially others

**Pattern**:
```rust
let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SOURCE)?;
let module = device.load_module(ptx)?;  // ptx is already Ptx type
let kernel = module.load_function("kernel_name")?;
```

## Migration Patterns

### Pattern A: Single PTX File (Most Common)

```rust
// Before:
let ptx_str = std::fs::read_to_string(path)?;
context.load_ptx(ptx_str.into(), "module", &["k1", "k2"])?;
let k1 = context.get_func("module", "k1")?;

// After:
let ptx_data = std::fs::read(path)?;
let module = device.load_module(Ptx::Image(&ptx_data))?;
let k1 = module.load_function("k1")?;
```

### Pattern B: String-based PTX

```rust
// Before:
let ptx = Ptx::from_src(std::str::from_utf8(&data)?);
context.load_ptx(ptx, "mod", &["k"])?;

// After:
let module = device.load_module(Ptx::Image(&data))?;
```

### Pattern C: include_str!() Macro

```rust
// Before:
const PTX: &str = include_str!("kernel.ptx");
context.load_ptx(PTX.into(), "mod", &["k"])?;

// After:
const PTX: &[u8] = include_bytes!("kernel.ptx");
let module = device.load_module(Ptx::Image(PTX))?;
```

## Compilation Status

### Before Fixes
```
error[E0599]: no method named `load_ptx` found for struct `Arc<CudaContext>`
error[E0599]: no method named `get_func` found for struct `Arc<CudaContext>`
... (17 load_ptx errors + 12 get_func errors across 31 files)
```

### After Partial Fixes (2/31 files)
```
Remaining: 15 load_ptx errors + 10 get_func errors in 29 files
```

## Testing Strategy

1. **Per-File Testing**:
   ```bash
   cargo check --features cuda -p prism-gpu
   cargo check --features cuda -p prct-core
   ```

2. **Integration Testing**:
   ```bash
   cargo test --features cuda --lib gpu_thermodynamic
   cargo test --features cuda --lib whcr
   ```

3. **Full Pipeline**:
   ```bash
   cargo build --release --features cuda
   cargo test --all-features
   ```

## Next Steps

### Immediate (P0 - Critical)
1. Fix foundation/prct-core/src/gpu_thermodynamic.rs
2. Fix foundation/prct-core/src/gpu_quantum.rs  
3. Fix crates/prism-gpu/src/thermodynamic.rs
4. Fix crates/prism-gpu/src/quantum.rs
5. Verify phase pipeline compiles

### Short-term (P1 - High)
6. Fix floyd_warshall.rs (APSP kernel)
7. Fix lbs.rs (multi-module case)
8. Fix cma.rs (ensemble optimization)
9. Verify core GPU modules compile

### Medium-term (P2-P3)
10. Fix remaining 20 files
11. Full integration test
12. Performance benchmarking

## References

- **Migration Report**: `CUDARC_0_18_1_API_FIX_REPORT.md` (detailed guide)
- **Example Fixes**: `crates/prism-gpu/src/whcr.rs`, `dendritic_whcr.rs`
- **cudarc Docs**: https://docs.rs/cudarc/0.18.1/cudarc/
- **Script**: `scripts/fix_cudarc_api_summary.sh`

## Completion Metrics

- [x] 2/31 files fixed (6.5%)
- [ ] 4/4 P0 files (0%)
- [ ] 8/8 P1 files (0%)  
- [ ] 29/31 total files (6.5%)

---

**Status**: IN PROGRESS  
**Blocker**: P0 files must be fixed before pipeline can compile  
**ETA**: ~4-6 hours for complete migration (all 31 files)
