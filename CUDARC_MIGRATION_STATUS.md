# cudarc 0.18.1 Migration Status

## Overview
Migrating ALL GPU files in `foundation/prct-core/src/` from cudarc 0.9 API to cudarc 0.18.1 stream-based API.

## Migration Pattern

### Required Changes for Each File:

1. **Import Updates**
   - OLD: `use cudarc::driver::{CudaDevice, ...}`
   - NEW: `use cudarc::driver::{CudaContext, CudaStream, ...}`

2. **Struct Field Addition**
   - Add `stream: CudaStream` field to all GPU solver structs

3. **Initialization**
   - OLD: `device.load_ptx(...)` and `device.get_func(...)`
   - NEW: `stream.load_ptx(...)` and `stream.get_func(...)`

4. **Memory Operations**
   - OLD: `device.htod_sync_copy(...)`
   - NEW: `stream.htod_sync_copy(...)`

   - OLD: `device.clone_htod(...)`
   - NEW: `stream.clone_htod(...)`

   - OLD: `device.alloc_zeros(...)`
   - NEW: `stream.alloc_zeros(...)`

   - OLD: `device.dtoh_sync_copy(...)`
   - NEW: `stream.dtoh_sync_copy(...)`

5. **Kernel Launches**
   - OLD: `(*kernel_fn).clone().launch(cfg, params)`
   - NEW: `stream.launch(&kernel_fn, cfg, params)`

## Files Requiring Updates

### ✅ COMPLETED:
1. ✅ `gpu_thermodynamic.rs` - Already migrated (1640 LOC)
2. ✅ `gpu_thermodynamic_multi.rs` - Already migrated (uses default_stream)
3. ✅ `gpu_thermodynamic_streams.rs` - Already migrated (multi-stream support)
4. ✅ `gpu/stream_pool.rs` - Already migrated (fork_stream API)
5. ✅ `gpu_quantum.rs` - JUST FIXED (stream-based)

### 🔄 IN PROGRESS:
6. `gpu_quantum_annealing.rs` - 496 LOC, needs full migration
7. `gpu_quantum_multi.rs` - 255 LOC, needs stream for gpu_quantum_annealing calls
8. `gpu_kuramoto.rs` - 366 LOC, needs full migration
9. `gpu_transfer_entropy.rs` - 670 LOC, needs full migration
10. `gpu_active_inference.rs` - 357 LOC, needs full migration

### ⏳ PENDING:
11. `gpu/state.rs` - Review for compatibility
12. `gpu/multi_device_pool.rs` - Review for compatibility
13. `world_record_pipeline_gpu.rs` - Update caller code
14. `adapters/*.rs` - Review all adapter files

## Critical Notes

1. **Stream Strategy**:
   - Single default stream per context: `device.default_stream()`
   - Multi-stream for parallelism: `device.fork_stream()`
   - stream_pool.rs provides managed pools

2. **Kernel Launch Syntax**:
   - Must use `stream.launch(&kernel, cfg, params)`
   - Remove all `(*kernel).clone().launch(...)` patterns
   - Remove all `LaunchAsync` imports (deprecated)

3. **Test Files**:
   - Update test helper: `CudaContext::new(0)` instead of `CudaDevice::new(0)`

## Next Steps

1. Fix remaining 5 GPU solver files (quantum_annealing, kuramoto, transfer_entropy, active_inference)
2. Verify gpu/state.rs and gpu/multi_device_pool.rs compatibility
3. Update world_record_pipeline_gpu.rs caller code
4. Run `cargo check --features cuda` to verify compilation
5. Run GPU tests to verify functionality

## Verification Command
```bash
cd /mnt/c/Users/Predator/Desktop/PRISM
cargo check --features cuda 2>&1 | grep -E "(error|warning)"
```

## Status: ~60% COMPLETE
- 5 files fully migrated
- 5 files need migration
- 4 files need review
