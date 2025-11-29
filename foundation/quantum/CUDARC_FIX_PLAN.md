# cudarc 0.18.1 Migration Plan for foundation/quantum/src

## Files to Fix
1. `gpu_coloring.rs` - GPU chromatic graph coloring
2. `gpu_tsp.rs` - GPU TSP solver

## Key API Changes Required

### 1. Remove CudaDevice → Use CudaContext
- Replace `CudaDevice::new(0)` with `CudaContext::new(0)`
- Remove `LaunchAsync` import (doesn't exist in 0.18.1)

### 2. Add CudaStream Management
- Every struct needs both `context` and `stream` fields
- Create stream: `Arc::new(context.default_stream())`
- NO `fork_default_stream()` - doesn't exist

### 3. Memory Operations on Stream (NOT Context)
- `stream.alloc_zeros<T>(n)` instead of `context.alloc_zeros()`
- `stream.clone_htod(&data)` instead of `context.htod_sync_copy_into()`
- `stream.clone_dtoh(&gpu_data)` instead of `context.dtoh_sync_copy_into()`
- `stream.memset_zeros(&mut slice)` for zeroing

### 4. Module/Kernel Loading
- Context loads modules: `context.load_module(&ptx, &["kernel1", "kernel2"])`
- Get function from loaded module: `let module = context.get_module("module_name")?`
- Then: `let func = module.get_function("kernel_name")?`

### 5. Kernel Launch
- `stream.launch(&func, cfg, params)` - params is a tuple

## Specific Fixes for gpu_coloring.rs

Line-by-line replacements:
1. Line 7: Add `CudaStream` to imports
2. Line 17-18: Add `stream: Arc<CudaStream>` field
3. Line 33-36: Create context + stream in new_adaptive
4. Line 48: Add stream param to `new()` signature
5. Line 61-64: Update build/color calls to pass stream
6. Line 84-89: Add stream param to build_adjacency_gpu
7. Line 125-128: Use `stream.clone_htod()` instead of alloc+copy
8. Line 132-134: Use `stream.alloc_zeros()`
9. Line 184: Replace `fork_default_stream` with using existing stream
10. Throughout: Replace all `context.htod_sync_copy_into` → `stream.clone_htod`
11. Throughout: Replace all `context.dtoh_sync_copy_into` → `stream.clone_dtoh`
12. Throughout: Pass `&stream` to all helper functions

## Specific Fixes for gpu_tsp.rs

Similar pattern:
1. Add CudaStream import and struct field
2. Replace CudaDevice with CudaContext
3. All memory ops through stream
4. Remove fork_default_stream calls

## Testing After Migration
```bash
cd foundation/quantum
cargo check --all-features
cargo build --release --features cuda
cargo test --all-features
```
