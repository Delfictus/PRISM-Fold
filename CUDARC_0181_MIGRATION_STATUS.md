# cudarc 0.18.1 Migration Status

## Overview
Systematic migration of all `crates/prism-gpu/src/*.rs` files from cudarc 0.9 API to cudarc 0.18.1 API.

## Automated Fixes Applied

### Phase 1: Import Fixes
✅ **Completed**
- Replaced all `use cudarc::driver::CudaDevice` with `use cudarc::driver::CudaContext`
- Removed all `LaunchAsync` imports (doesn't exist in 0.18.1)
- Added `CudaStream` imports where needed
- Added `Ptx` imports where needed
- Fixed duplicate `CudaContext` import in `whcr.rs`

### Phase 2: Type Replacements
✅ **Completed**
- Replaced all `Arc<CudaDevice>` with `Arc<CudaContext>`
- Replaced all `CudaDevice::new` with `CudaContext::new`

### Phase 3: Stream API Fixes
✅ **Completed**
- Replaced all `.fork_default_stream()` with `.default_stream()`

### Phase 4: Code Structure Fixes
✅ **Completed**
- Removed `use` statements incorrectly placed inside `impl` blocks in:
  - `dendritic_reservoir.rs`
  - `floyd_warshall.rs`
  - `tda.rs`
  - `molecular.rs`

## Remaining Manual Fixes Needed

### Critical API Changes (per file)

#### Files with Variable Name Issues
These files reference `device` or `stream` variables that don't exist in scope:

1. **active_inference.rs** (2 errors)
   - Line 438, 468: `stream` not found
   - Fix: Define `let stream = self.context.default_stream();`

2. **cma.rs** (13+ errors)
   - Multiple lines: `device` not found
   - Fix: Replace `device` with `self.context` or get stream first

3. **cma_es.rs** (1+ errors)
   - Line 366: `stream` not found
   - Fix: Define stream variable before use

#### Files Needing Module Loading Pattern Updates

All files that call `.load_ptx()` need to be updated to use the new pattern:

**Old Pattern (cudarc 0.9):**
```rust
device.load_ptx(ptx_str.into(), "module_name", &["kernel1", "kernel2"])?;
let kernel = device.get_func("module_name", "kernel1")?;
```

**New Pattern (cudarc 0.18.1):**
```rust
let ptx_bytes = std::fs::read(ptx_path)?;
let module = device.load_module(Ptx::Image(&ptx_bytes))?;
let kernel = module.load_function("kernel1")?;
```

Files affected:
- `context.rs` (load_ptx_module method)
- `quantum.rs` (new method)
- `thermodynamic.rs` (new method)
- `whcr.rs` (new method)
- All other kernel-loading files

#### Files Needing Memory Operation Updates

Memory operations moved from `CudaContext` to `CudaStream`:

**Old Pattern (cudarc 0.9):**
```rust
let d_buffer = device.alloc_zeros::<f32>(size)?;
device.htod_sync_copy_into(&host_data, &mut d_buffer)?;
let host_result = device.dtoh_sync_copy(&d_buffer)?;
```

**New Pattern (cudarc 0.18.1):**
```rust
let stream = device.default_stream();
let d_buffer = stream.alloc_zeros::<f32>(size)?;
stream.memcpy_htod(&host_data, &mut d_buffer)?;
let host_result = stream.memcpy_dtoh(&d_buffer)?;
```

**Note:** Also:
- `htod_sync_copy` → `memcpy_htod` or `clone_htod`
- `dtoh_sync_copy` → `memcpy_dtoh` or `clone_dtoh`
- `memset_zeros` is on stream, not context

All files with GPU operations are affected.

## Implementation Strategy

Due to the large scope of changes, I recommend:

### Option A: Complete Manual Migration (Recommended)
1. Create a separate git worktree for cudarc 0.18.1 migration
2. Fix each file systematically following the patterns above
3. Test incrementally with `cargo check --features cuda`
4. Once all compile errors are resolved, run full test suite
5. Merge back when verified working

### Option B: Continue with cudarc 0.9
The CLAUDE.md documentation states that cudarc 0.9 was retained because:
> "0.18.1 stream-centric API adds complexity without perf gain"

If 0.9 is sufficient for current needs, migration can be deferred.

## Key cudarc 0.18.1 API Facts (Reference)

1. **NO CudaDevice** - use `CudaContext`
2. **NO LaunchAsync trait** - removed from imports
3. **NO fork_default_stream()** - use `default_stream()`
4. **Memory ops on CudaStream** - `alloc_zeros`, `clone_htod`, `memcpy_*`, `memset_zeros`
5. **NO dtoh_sync_copy, htod_sync_copy** - use `clone_dtoh`, `clone_htod` on STREAM
6. **NO get_func on context** - use `context.load_module()` then `module.load_function()`
7. **NO load_ptx** - use `load_module(Ptx::Image(bytes))`
8. **Kernel launch** - `stream.launch(&func, cfg, params)` NOT `func.launch()`

## Current Compilation Status

After automated fixes:
- ✅ All import errors resolved
- ✅ All type name errors resolved
- ✅ All duplicate import errors resolved
- ⚠️ **~20-30 variable scope errors remaining** (device/stream not in scope)
- ⚠️ **Module loading patterns need updating** (load_ptx → load_module)
- ⚠️ **Memory operations need updating** (move to stream)

## Files Modified (Automated Phase)

All `*.rs` files in `crates/prism-gpu/src/`:
- aatgs.rs
- aatgs_integration.rs
- active_inference.rs
- cma.rs
- cma_es.rs
- context.rs
- dendritic_reservoir.rs
- dendritic_whcr.rs
- floyd_warshall.rs
- lbs.rs
- lib.rs
- molecular.rs
- multi_device_pool.rs
- multi_gpu.rs
- multi_gpu_integration.rs
- pimc.rs
- quantum.rs
- stream_integration.rs
- stream_manager.rs
- tda.rs
- thermodynamic.rs
- transfer_entropy.rs
- whcr.rs

## Next Steps

1. **Decision Point**: Confirm whether to proceed with full 0.18.1 migration or stay on 0.9
2. If proceeding:
   - Create git worktree for migration work
   - Systematically fix each file following the patterns above
   - Test incrementally
   - Document any discovered issues
3. If staying on 0.9:
   - Revert automated changes
   - Document decision in project README
   - Focus on other implementation priorities

## Time Estimate

Full manual migration of all remaining issues: **4-6 hours** of focused work, assuming:
- ~23 files to fix
- ~10-15 minutes per file on average
- Additional time for testing and debugging

Automated migration with script: Could reduce to **2-3 hours** but requires careful scripting to avoid breaking working code.
