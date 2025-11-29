# cudarc 0.18.1 Migration Status - foundation/quantum/src

## Files Status

| File | Status | Notes |
|------|--------|-------|
| `gpu_coloring.rs` | 🟡 PARTIAL | Struct updated, many fixes done, some remain |
| `gpu_tsp.rs` | 🔴 TODO | Needs full migration |
| `gpu_k_opt.rs` | ✅ OK | Already uses correct API |
| `hamiltonian.rs` | ✅ OK | No GPU code |
| `lib.rs` | ✅ OK | Just re-exports |
| `prct_coloring.rs` | ✅ OK | CPU only |
| `prct_tsp.rs` | ✅ OK | CPU only |
| `qubo.rs` | ✅ OK | No cudarc usage |
| `robust_eigen.rs` | ✅ OK | CPU only (ndarray-linalg) |
| `security.rs` | ✅ OK | No GPU code |
| `types.rs` | ✅ OK | Type definitions only |

## Changes Completed

### ✅ gpu_coloring.rs - Partially Fixed

**Completed:**
1. Added `CudaStream` to imports (line 7)
2. Added `stream: Arc<CudaStream>` field to struct (line 19)
3. Updated `new_adaptive()` to create context + stream (lines 35-36)
4. Updated `new()` signature to accept stream parameter (line 49)
5. Stored stream in struct initialization (line 68)
6. Updated `build_adjacency_gpu()` signature with stream param (line 86)
7. Fixed memory operations in `build_adjacency_gpu()`:
   - `stream.clone_htod()` instead of alloc+copy (line 126)
   - `stream.alloc_zeros()` instead of `context.alloc_zeros()` (line 132)
8. Fixed module loading pattern (lines 171-179):
   - `context.load_module()` instead of `load_ptx()`
   - `module.get_function()` instead of `get_func()`
9. Removed one `fork_default_stream()` call (line 187)
10. Updated `download_adjacency()` signature with stream (line 208)
11. Fixed download to use `stream.clone_dtoh()` (line 215)

**Remaining Issues (22 patterns found):**
1. Line 306: `load_ptx()` → needs `load_module()`
2. Lines 340, 356, 361, 366: `get_func()` → needs `module.get_function()`
3. Lines 376, 407, 429, 431: `fork_default_stream()` → use existing `self.stream`
4. Lines 429, 636: `htod_sync_copy_into()` → `stream.clone_htod()`
5. Lines 444, 453, 690: `dtoh_sync_copy_into()` → `stream.clone_dtoh()`
6. Line 660: `load_ptx()` → needs `load_module()`
7. Line 665: `get_func()` → needs `module.get_function()`
8. Line 671: `fork_default_stream()` → use existing `self.stream`

Additionally:
- `jones_plassmann_gpu()` needs stream parameter added
- `find_optimal_threshold_gpu()` needs stream parameter added
- All their call sites need stream argument added

### 🔴 gpu_tsp.rs - Not Started

**Required Changes:**
1. Add `CudaStream` to imports
2. Replace ALL `CudaDevice` → `CudaContext` (appears ~10 times)
3. Add `stream: Arc<CudaStream>` field to `GpuTspSolver` struct
4. Update constructor to create stream: `Arc::new(context.default_stream())`
5. Add stream to struct initialization
6. Update `compute_distance_matrix_gpu()` signature with stream
7. Replace ALL `context.alloc_zeros()` → `stream.alloc_zeros()` (~8 instances)
8. Replace ALL `context.htod_sync_copy_into()` → `stream.clone_htod()` (~6 instances)
9. Replace ALL `context.dtoh_sync_copy_into()` → `stream.clone_dtoh()` (~4 instances)
10. Remove ALL `fork_default_stream()` calls (~4 instances)
11. Fix module loading: `load_ptx()` → `load_module()`
12. Fix kernel retrieval: `get_func()` → `module.get_function()`

## Automation Tools Created

### 1. Migration Plan Document
**File:** `CUDARC_FIX_PLAN.md`
- Detailed API change reference
- Line-by-line fix guide
- Testing instructions

### 2. Shell Script (Partial automation)
**File:** `../scripts/fix_quantum_cudarc.sh`
- Automated sed replacements for common patterns
- Backs up files before modification
- Reports changes made

**Limitations:**
- Can only handle simple regex patterns
- Cannot add function parameters
- Cannot modify struct definitions
- Manual review still required

### 3. Python Script (Not run)
**File:** `../scripts/fix_quantum_cudarc.py`
- More sophisticated pattern matching
- Handles complex transformations
- Not executed (would need manual review)

## Critical Remaining Work

### Priority 1: gpu_coloring.rs Function Signatures

**jones_plassmann_gpu():**
```rust
// Current (line 274)
fn jones_plassmann_gpu(
    context: &Arc<CudaContext>,
    gpu_adjacency: &CudaSlice<u8>,
    n: usize,
    max_colors: usize,
) -> Result<Vec<usize>>

// Needs to be:
fn jones_plassmann_gpu(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    gpu_adjacency: &CudaSlice<u8>,
    n: usize,
    max_colors: usize,
) -> Result<Vec<usize>>
```

**find_optimal_threshold_gpu():**
```rust
// Current (line 694)
fn find_optimal_threshold_gpu(
    context: &Arc<CudaContext>,
    coupling_matrix: &Array2<Complex64>,
    target_colors: usize,
) -> Result<f64>

// Needs to be:
fn find_optimal_threshold_gpu(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    coupling_matrix: &Array2<Complex64>,
    target_colors: usize,
) -> Result<f64>
```

### Priority 2: Replace fork_default_stream()

**Pattern to replace:**
```rust
// WRONG (throughout file):
let stream = self.context.fork_default_stream()?;
let stream = context.fork_default_stream()?;

// CORRECT:
// Just use self.stream or stream parameter - no need to create new one
```

**Locations:** Lines 376, 407, 429, 431, 671

### Priority 3: Fix Memory Operations

**htod_sync_copy_into patterns:**
```rust
// WRONG:
context.htod_sync_copy_into(&data, &mut gpu_buffer)?;
self.context.htod_sync_copy_into(&data, &mut gpu_buffer)?;

// CORRECT:
let gpu_buffer = stream.clone_htod(&data)?;
let gpu_buffer = self.stream.clone_htod(&data)?;
```

**dtoh_sync_copy_into patterns:**
```rust
// WRONG:
context.dtoh_sync_copy_into(&gpu_buffer, &mut cpu_data)?;
self.context.dtoh_sync_copy_into(&gpu_buffer, &mut cpu_data)?;

// CORRECT:
let cpu_data = stream.clone_dtoh(&gpu_buffer)?;
let cpu_data = self.stream.clone_dtoh(&gpu_buffer)?;
```

### Priority 4: Fix Module Loading

**Pattern in jones_plassmann_gpu() (line 302-340):**
```rust
// WRONG:
let ptx_parsed = Ptx::from_src(&ptx);
context.load_ptx(ptx_parsed, "parallel_coloring", &[...])?;
let init_priorities = context.get_func("parallel_coloring", "init_priorities")?;

// CORRECT:
let ptx_parsed = Ptx::from_src(&ptx);
let module = context.load_module(&ptx_parsed, &[...])?;
let init_priorities = module.get_function("init_priorities")?;
```

**Pattern in count_conflicts_gpu() (line 656-665):**
```rust
// WRONG:
self.context.load_ptx(ptx_parsed, "graph_coloring_conflicts", &["count_conflicts"])?;
let count_conflicts = self.context.get_func("graph_coloring_conflicts", "count_conflicts")?;

// CORRECT:
let module = self.context.load_module(&ptx_parsed, &["count_conflicts"])?;
let count_conflicts = module.get_function("count_conflicts")?;
```

## Recommended Approach

### Option A: Manual Completion (Recommended)
1. Continue systematic Edit commands for remaining patterns
2. Focus on high-impact changes first (function signatures)
3. Test incrementally with `cargo check`
4. Estimated time: 15-20 more edits

### Option B: Script + Manual Review
1. Run `scripts/fix_quantum_cudarc.sh`
2. Review all changes carefully
3. Fix remaining compilation errors manually
4. Estimated time: Same as Option A (script isn't comprehensive enough)

### Option C: Complete Rewrite of Both Files
1. Create corrected versions from scratch
2. Preserve all logic, update only cudarc API calls
3. Most thorough but time-intensive
4. Estimated time: 1-2 hours

## Testing Checklist

After all fixes:
```bash
# 1. Check syntax
cd foundation/quantum
cargo check --all-features

# 2. Build with CUDA
cargo build --release --features cuda

# 3. Run tests
cargo test --all-features

# 4. Integration test
cd ../../
cargo test -p quantum --all-features
```

## Success Metrics

- ✅ No cudarc API compile errors
- ✅ No warnings about unused variables/imports
- ✅ All tests pass
- ✅ GPU operations execute successfully

## Verification Commands

```bash
# Check for old API patterns (should return nothing):
grep -n "CudaDevice" foundation/quantum/src/*.rs
grep -n "LaunchAsync" foundation/quantum/src/*.rs
grep -n "fork_default_stream" foundation/quantum/src/*.rs
grep -n "htod_sync_copy_into" foundation/quantum/src/*.rs
grep -n "dtoh_sync_copy_into" foundation/quantum/src/*.rs
grep -n "get_func" foundation/quantum/src/*.rs
grep -n "load_ptx" foundation/quantum/src/*.rs

# Check for correct patterns (should find many):
grep -n "CudaStream" foundation/quantum/src/*.rs
grep -n "default_stream" foundation/quantum/src/*.rs
grep -n "clone_htod" foundation/quantum/src/*.rs
grep -n "clone_dtoh" foundation/quantum/src/*.rs
grep -n "load_module" foundation/quantum/src/*.rs
grep -n "get_function" foundation/quantum/src/*.rs
```

---

**Status:** 🟡 IN PROGRESS
**Completion:** ~40% (gpu_coloring.rs partially done, gpu_tsp.rs not started)
**Remaining Work:** ~60% (systematic completion of both files)
**Estimated Effort:** 15-20 more strategic edits OR 30-40 automated sed replacements + manual review

**Next Immediate Steps:**
1. Fix `jones_plassmann_gpu()` signature + all calls
2. Fix `find_optimal_threshold_gpu()` signature + all calls
3. Replace all `fork_default_stream()` with existing stream usage
4. Fix all remaining htod/dtoh patterns
5. Move to gpu_tsp.rs and apply same pattern

