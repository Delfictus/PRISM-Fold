# cudarc 0.18.1 Migration - Quick Reference

## Current Status

✅ **Completed:** Partial migration of `gpu_coloring.rs` (~40% done)
🔴 **Remaining:** Complete `gpu_coloring.rs` and migrate `gpu_tsp.rs` (~60% remaining)

## What's Been Fixed

### gpu_coloring.rs - Partial (Commits  Made)
- ✅ Imports updated (added `CudaStream`)
- ✅ Struct has `stream` field
- ✅ Constructor creates stream
- ✅ `build_adjacency_gpu()` signature updated
- ✅ Some memory operations migrated
- ✅ Some module loading fixed
- 🔴 Still needs: function signatures, remaining memory ops, module loading

### gpu_tsp.rs - Not Started
- 🔴 Needs full migration

## Complete API Translation Guide

### Memory Operations
```rust
// OLD (0.9)                              → NEW (0.18.1)
context.alloc_zeros::<T>(n)              → stream.alloc_zeros::<T>(n)
context.htod_sync_copy_into(&h, &mut d)  → let d = stream.clone_htod(&h)?
context.dtoh_sync_copy_into(&d, &mut h)  → let h = stream.clone_dtoh(&d)?
context.memset_zeros(&mut d)             → stream.memset_zeros(&mut d)
```

### Module/Kernel Management
```rust
// OLD (0.9)
context.load_ptx(ptx, "module_name", &["k1", "k2"])?;
let func = context.get_func("module_name", "k1")?;

// NEW (0.18.1)
let module = context.load_module(&ptx, &["k1", "k2"])?;
let func = module.get_function("k1")?;
```

### Stream Management
```rust
// OLD (0.9) - WRONG in 0.18.1
let stream = context.fork_default_stream()?;  // ❌ Doesn't exist

// NEW (0.18.1) - CORRECT
// In constructor:
let stream = Arc::new(context.default_stream());

// In methods:
// Just use self.stream or function parameter - don't create new ones
```

### Struct Pattern
```rust
pub struct GpuSolver {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,   // ← ADD THIS
    // ...
}

impl GpuSolver {
    pub fn new() -> Result<Self> {
        let context = Arc::new(CudaContext::new(0)?);
        let stream = Arc::new(context.default_stream());  // ← ADD THIS

        Ok(Self {
            context,
            stream,  // ← ADD THIS
            // ...
        })
    }
}
```

## Remaining Work Breakdown

### gpu_coloring.rs (22 fixes needed)

**Function Signatures (3):**
1. `jones_plassmann_gpu()` - add stream parameter
2. `find_optimal_threshold_gpu()` - add stream parameter
3. All call sites - pass stream argument

**Stream Management (5):**
4-8. Lines 376, 407, 429, 431, 671 - remove `fork_default_stream()`, use existing stream

**Memory Operations (5):**
9-10. Lines 429, 636 - `htod_sync_copy_into` → `clone_htod`
11-13. Lines 444, 453, 690 - `dtoh_sync_copy_into` → `clone_dtoh`

**Module Loading (4):**
14. Line 306 - `load_ptx()` → `load_module()`
15. Line 660 - `load_ptx()` → `load_module()`
16-19. Lines 340, 356, 361, 366 - `get_func()` → `module.get_function()`
20. Line 665 - `get_func()` → `module.get_function()`

**Allocation Operations (~5 more):**
21-22. Various `context.alloc_zeros()` → `stream.alloc_zeros()`

### gpu_tsp.rs (31 fixes needed)

**Imports & Types (2):**
1. Add `CudaStream` to imports
2. Replace all `CudaDevice` → `CudaContext` (~10 occurrences)

**Struct (2):**
3. Add `stream: Arc<CudaStream>` field
4. Initialize stream in constructor

**Signatures (2):**
5. `compute_distance_matrix_gpu()` - add stream parameter
6. Update all call sites

**Stream Management (~4):**
7-10. Remove all `fork_default_stream()` calls

**Memory Operations (~15):**
11-17. `context.alloc_zeros()` → `stream.alloc_zeros()`
18-23. `context.htod_sync_copy_into()` → `stream.clone_htod()`
24-27. `context.dtoh_sync_copy_into()` → `stream.clone_dtoh()`

**Module Loading (~8):**
28-29. `load_ptx()` → `load_module()`
30-31. `get_func()` → `module.get_function()`

## Quick Fix Commands

### For gpu_coloring.rs
```bash
# Function signature (jones_plassmann_gpu)
# Line 274 - add stream parameter after context

# Function signature (find_optimal_threshold_gpu)
# Line 694 - add stream parameter after context

# Remove fork_default_stream (lines 376, 407, 429, 431, 671)
sed -i 's/let stream = .*fork_default_stream.*$/\/\/ Using existing stream/' \
    foundation/quantum/src/gpu_coloring.rs

# Fix memory operations
sed -i 's/\.htod_sync_copy_into(&\([^,]*\), &mut \([^)]*\))/\.clone_htod(\&\1)/g' \
    foundation/quantum/src/gpu_coloring.rs

sed -i 's/\.dtoh_sync_copy_into(&\([^,]*\), &mut \([^)]*\))/\.clone_dtoh(\&\1)/g' \
    foundation/quantum/src/gpu_coloring.rs
```

### For gpu_tsp.rs
```bash
# Replace CudaDevice
sed -i 's/CudaDevice/CudaContext/g' foundation/quantum/src/gpu_tsp.rs

# Add CudaStream to imports
sed -i 's/use cudarc::driver::{CudaContext,/use cudarc::driver::{CudaContext, CudaStream,/' \
    foundation/quantum/src/gpu_tsp.rs

# Fix memory operations (same patterns as gpu_coloring.rs)
```

## Testing Strategy

```bash
# 1. Syntax check after each major fix
cargo check --features cuda

# 2. Focus on one file at a time
cargo check -p quantum --features cuda 2>&1 | grep gpu_coloring

# 3. Full build when both files done
cargo build --release --features cuda

# 4. Run tests
cargo test -p quantum --all-features
```

## Verification Checklist

After completion, these should all be empty:
```bash
grep -n "CudaDevice" foundation/quantum/src/gpu_*.rs
grep -n "fork_default_stream" foundation/quantum/src/gpu_*.rs
grep -n "htod_sync_copy_into" foundation/quantum/src/gpu_*.rs
grep -n "dtoh_sync_copy_into" foundation/quantum/src/gpu_*.rs
grep -n 'get_func("' foundation/quantum/src/gpu_*.rs
grep -n 'load_ptx(' foundation/quantum/src/gpu_*.rs
```

And these should have matches:
```bash
grep -n "CudaStream" foundation/quantum/src/gpu_*.rs
grep -n "clone_htod" foundation/quantum/src/gpu_*.rs
grep -n "clone_dtoh" foundation/quantum/src/gpu_*.rs
grep -n "load_module" foundation/quantum/src/gpu_*.rs
grep -n "get_function" foundation/quantum/src/gpu_*.rs
```

## Documentation Created

1. **CUDARC_FIX_PLAN.md** - Detailed implementation plan
2. **MIGRATION_STATUS_FINAL.md** - Comprehensive status report
3. **CUDARC_MIGRATION_COMPLETE.md** - Complete API reference
4. **This file** - Quick reference guide

## Scripts Available

1. **scripts/fix_quantum_cudarc.sh** - Automated sed replacements (partial)
2. **scripts/fix_quantum_cudarc.py** - Python migration script (not executed)

## Next Steps

**Option 1: Continue Manual Edits (Recommended)**
- Most precise control
- ~15-20 strategic edits remaining
- Can test incrementally

**Option 2: Run Automation Script**
```bash
./scripts/fix_quantum_cudarc.sh
# Then manually fix struct definitions and function signatures
```

**Option 3: Systematic File-by-File**
1. Complete gpu_coloring.rs fully
2. Test: `cargo check -p quantum --features cuda`
3. Then start gpu_tsp.rs
4. Test: `cargo check -p quantum --features cuda`

## Contact

If issues arise, check:
- `MIGRATION_STATUS_FINAL.md` for current status
- `CUDARC_MIGRATION_COMPLETE.md` for API reference
- Official cudarc docs: https://docs.rs/cudarc/0.18.1

---

**Created:** 2025-11-29
**cudarc Version:** 0.18.1
**Files Affected:** 2 (gpu_coloring.rs, gpu_tsp.rs)
**Progress:** 40% complete
