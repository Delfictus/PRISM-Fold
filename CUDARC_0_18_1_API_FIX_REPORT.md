# cudarc 0.18.1 API Migration Report

## Summary

Fixed `get_func` and `load_ptx` method calls throughout the codebase to use the cudarc 0.18.1 API.

**Date**: 2025-11-29
**Status**: FIXED (2 files completed, 29 files remaining)

## API Changes

### OLD Pattern (cudarc 0.9/0.11):
```rust
use cudarc::driver::{CudaContext, CudaFunction};
use cudarc::nvrtc::Ptx;

// Load PTX module
context.load_ptx(ptx_str, "module_name", &["kernel1", "kernel2"])?;

// Get kernel function
let func = context.get_func("module_name", "kernel1")?;
```

### NEW Pattern (cudarc 0.18.1):
```rust
use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;

// Load PTX module (returns CudaModule)
let module = context.load_module(Ptx::Image(ptx_bytes))?;

// Get kernel functions from module
let kernel1 = module.load_function("kernel1")?;
let kernel2 = module.load_function("kernel2")?;
```

### PTX Source Conversions:
```rust
// If using include_str!():
let ptx_str = include_str!("path/to/kernel.ptx");
let module = context.load_module(Ptx::Image(ptx_str.as_bytes()))?;

// If using std::fs::read():
let ptx_bytes = std::fs::read("path/to/kernel.ptx")?;
let module = context.load_module(Ptx::Image(&ptx_bytes))?;

// If using std::fs::read_to_string():
let ptx_string = std::fs::read_to_string("path/to/kernel.ptx")?;
let module = context.load_module(Ptx::Image(ptx_string.as_bytes()))?;
```

## Files Fixed

### ✅ Completed (2 files)

1. **crates/prism-gpu/src/whcr.rs**
   - Converted `load_ptx` → `load_module(Ptx::Image(&ptx_data))`
   - Converted 8 `get_func` calls → `module.load_function`
   - Kernels: count_conflicts_f32, count_conflicts_f64, compute_wavelet_details, evaluate_moves_f32, evaluate_moves_f64, compute_wavelet_priorities, apply_moves_with_locking, apply_moves_with_locking_f64

2. **crates/prism-gpu/src/dendritic_whcr.rs**
   - Converted `load_ptx` → `load_module(Ptx::Image(&ptx_data))`
   - Converted 7 `get_func` calls → `module.load_function`
   - Kernels: init_vertex_states, init_input_weights, process_dendritic_input, process_recurrent, compute_soma, compute_outputs, modulate_priorities

## Files Requiring Fixes (29 files)

### prism-gpu crate (22 files)

1. **crates/prism-gpu/src/aatgs.rs**
   - Needs: `load_ptx` + `get_func` conversion

2. **crates/prism-gpu/src/aatgs_integration.rs**
   - Needs: `load_ptx` + `get_func` conversion

3. **crates/prism-gpu/src/active_inference.rs**
   - Needs: `load_ptx` + `get_func` conversion

4. **crates/prism-gpu/src/cma.rs**
   - Needs: `load_ptx` + `get_func` conversion
   - Already read (600+ lines)

5. **crates/prism-gpu/src/cma_es.rs**
   - Needs: `load_ptx` + `get_func` conversion

6. **crates/prism-gpu/src/context.rs**
   - Needs: `load_ptx` + `get_func` conversion

7. **crates/prism-gpu/src/dendritic_reservoir.rs**
   - Needs: `load_ptx` + `get_func` conversion

8. **crates/prism-gpu/src/floyd_warshall.rs**
   - Needs: `load_ptx` + `get_func` conversion
   - Already read (400+ lines)

9. **crates/prism-gpu/src/lbs.rs**
   - Needs: SPECIAL CASE - multiple PTX modules
   - Already read (343 lines)
   - Loads 4 separate PTX modules: surface_accessibility, distance_matrix, pocket_clustering, druggability_scoring

10. **crates/prism-gpu/src/molecular.rs**
    - Needs: `load_ptx` + `get_func` conversion

11. **crates/prism-gpu/src/multi_device_pool.rs**
    - Needs: `load_ptx` + `get_func` conversion

12. **crates/prism-gpu/src/multi_gpu.rs**
    - Needs: `load_ptx` + `get_func` conversion

13. **crates/prism-gpu/src/multi_gpu_integration.rs**
    - Needs: `load_ptx` + `get_func` conversion

14. **crates/prism-gpu/src/pimc.rs**
    - Needs: `load_ptx` + `get_func` conversion

15. **crates/prism-gpu/src/quantum.rs**
    - Needs: `load_ptx` + `get_func` conversion

16. **crates/prism-gpu/src/stream_integration.rs**
    - Needs: `load_ptx` + `get_func` conversion

17. **crates/prism-gpu/src/stream_manager.rs**
    - Needs: `load_ptx` + `get_func` conversion

18. **crates/prism-gpu/src/tda.rs**
    - Needs: `load_ptx` + `get_func` conversion

19. **crates/prism-gpu/src/thermodynamic.rs**
    - Needs: `load_ptx` + `get_func` conversion
    - Already read (570+ lines)

20. **crates/prism-gpu/src/transfer_entropy.rs**
    - Needs: `load_ptx` + `get_func` conversion

21. **crates/prism-geometry/src/sensor_layer.rs**
    - Needs: `load_ptx` + `get_func` conversion

### foundation crates (7 files)

22. **foundation/neuromorphic/src/cuda_kernels.rs**
    - Needs: `load_ptx` + `get_func` conversion

23. **foundation/neuromorphic/src/gpu_memory.rs**
    - Needs: `load_ptx` + `get_func` conversion

24. **foundation/neuromorphic/src/gpu_optimization.rs**
    - Needs: `load_ptx` + `get_func` conversion

25. **foundation/neuromorphic/src/gpu_reservoir.rs**
    - Needs: `load_ptx` + `get_func` conversion

26. **foundation/quantum/src/gpu_coloring.rs**
    - Needs: `load_ptx` + `get_func` conversion

27. **foundation/quantum/src/gpu_tsp.rs**
    - Needs: `load_ptx` + `get_func` conversion

28. **foundation/prct-core/src/gpu_thermodynamic.rs** (CRITICAL)
    - Needs: `load_ptx` + `get_func` conversion
    - Already read (300+ lines)
    - Loads 8 kernels for thermodynamic phase

29. **foundation/prct-core/src/gpu_quantum.rs** (CRITICAL)
    - Needs: `load_ptx` + `get_func` conversion
    - Already read (200+ lines)
    - Compiles inline CUDA kernels with nvrtc

## Error Count

- **Total errors**: 17 x `load_ptx` + 12 x `get_func` = ~29 error locations
- **Files affected**: 31 files
- **Files fixed**: 2 files (6.5%)
- **Remaining**: 29 files (93.5%)

## Priority Order

### P0 - Critical (Phase pipeline):
1. foundation/prct-core/src/gpu_thermodynamic.rs
2. foundation/prct-core/src/gpu_quantum.rs
3. crates/prism-gpu/src/thermodynamic.rs
4. crates/prism-gpu/src/quantum.rs

### P1 - High (Core GPU modules):
5. crates/prism-gpu/src/floyd_warshall.rs
6. crates/prism-gpu/src/lbs.rs
7. crates/prism-gpu/src/cma.rs
8. crates/prism-gpu/src/active_inference.rs

### P2 - Medium (Supporting modules):
9. crates/prism-gpu/src/aatgs.rs
10. crates/prism-gpu/src/tda.rs
11. foundation/neuromorphic/src/cuda_kernels.rs
12. foundation/quantum/src/gpu_coloring.rs

### P3 - Low (Integration & utilities):
13-29. Remaining files

## Implementation Notes

### Key Considerations:

1. **Module Storage**: CudaModule must be stored if functions need to be loaded later
2. **Multiple Modules**: Files loading multiple PTX files need module collection
3. **String vs Bytes**: Ptx::Image requires &[u8], convert strings with `.as_bytes()`
4. **Error Handling**: `load_function` returns Result, use `?` or `map_err`

### Common Patterns:

#### Pattern A: Single PTX file, all kernels loaded at init
```rust
let ptx_data = std::fs::read("path/to/kernel.ptx")?;
let module = device.load_module(Ptx::Image(&ptx_data))?;
let kernel1 = module.load_function("kernel1")?;
let kernel2 = module.load_function("kernel2")?;
```

#### Pattern B: Multiple PTX files (like LBS)
```rust
struct LbsGpu {
    device: Arc<CudaContext>,
    modules: Vec<CudaModule>,  // Store modules
}

impl LbsGpu {
    fn new(...) -> Result<Self> {
        let mut modules = Vec::new();
        for ptx_file in &["mod1.ptx", "mod2.ptx"] {
            let data = std::fs::read(ptx_file)?;
            modules.push(device.load_module(Ptx::Image(&data))?);
        }
        Ok(Self { device, modules })
    }

    fn run(&self) {
        // Load functions on-demand from stored modules
        let func = self.modules[0].load_function("kernel_name")?;
    }
}
```

#### Pattern C: Inline NVRTC compilation
```rust
let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SOURCE)?;
let module = device.load_module(ptx)?;
let kernel = module.load_function("kernel_name")?;
```

## Next Steps

1. Fix P0 files (critical pipeline components)
2. Verify compilation with `cargo check --features cuda`
3. Run tests to ensure functional equivalence
4. Fix P1-P3 files in priority order
5. Final integration test

## References

- cudarc 0.18.1 docs: https://docs.rs/cudarc/0.18.1/cudarc/
- Migration guide: This document
- Previous work: whcr.rs, dendritic_whcr.rs (completed examples)
