# Stream Launch API Fix Report - CUDARC 0.18.1 Migration

## ⚠️ CRITICAL DISCOVERY

The user's instruction to replace `stream.launch()` with `kernel.launch_on_stream()` was **INCORRECT**.

### Actual cudarc 0.18.1 API

The correct cudarc 0.18.1 kernel launch API is:

```rust
// OLD (cudarc 0.9):
unsafe {
    stream.launch(&kernel, cfg, (param1, param2, param3))
}

// CORRECT NEW (cudarc 0.18.1):
unsafe {
    stream.launch_builder(&kernel)
        .arg(&param1)
        .arg(&param2)
        .arg(&param3)
        .launch(cfg)?
}
```

### What I Did (INCORRECT)

I replaced all occurrences with:
```rust
unsafe {
    kernel.launch_on_stream(&stream, cfg, (params))
}
```

**This is wrong!** `CudaFunction` does not have a `launch_on_stream` method in cudarc 0.18.1.

### Why the Confusion

The user may have confused cudarc with other CUDA Rust bindings that use different APIs.

## Files Modified (Need Reversion + Correct Fix)

### Foundation

1. `foundation/prct-core/src/gpu_quantum.rs`
   - Lines 301, 321: matvec_fn, axpy_fn launches

2. `foundation/neuromorphic/src/gpu_reservoir.rs`
   - Lines 445, 491: custom GEMV kernel launches

3. `foundation/neuromorphic/src/cuda_kernels.rs`
   - Lines 343, 412, 454, 499: leaky_integration, spike_encoding, pattern_detection, spectral_radius

4. `foundation/prct-core/src/gpu_thermodynamic_streams.rs`
   - Line 185: kernel_evolve_osc

### Crates

5. `crates/prism-gpu/src/tda.rs`
   - Lines 217, 235, 251, 264, 392, 413, 425, 447: Union-find, persistence, importance kernels

6. `crates/prism-gpu/src/dendritic_whcr.rs`
   - Lines 290, 305, 318, 328, 355: Dendritic processing kernels

7. `crates/prism-gpu/src/pimc.rs`
   - Lines 217, 270, 315, 365, 410, 522: PIMC kernels

8. `crates/prism-gpu/src/molecular.rs`
   - Lines 153, 257, 289, 319, 339, 421: MD simulation kernels

9. `crates/prism-gpu/src/cma.rs`
   - Lines 284, 326, 368, 412, 459, 522: CMA-ES kernels

## Action Required

### Option 1: Stay on cudarc 0.9

The original cudarc 0.9 API works fine. Revert all changes and stay on 0.9.

### Option 2: Properly Migrate to 0.18.1

Need to:

1. **Revert all current changes**
2. **Properly migrate each `stream.launch()` call to the builder pattern**
3. **Decompose tuple parameters into individual `.arg()` calls**
4. **Test thoroughly**

Example transformation:

```rust
// Before (cudarc 0.9):
unsafe {
    self.stream.launch(
        &self.matvec_fn,
        cfg,
        (
            &d_h_re,
            &d_h_im,
            &d_state_re,
            &d_state_im,
            alpha_re,
            alpha_im,
            &mut d_temp_re,
            &mut d_temp_im,
            n as i32,
        ),
    )?;
}

// After (cudarc 0.18.1 - CORRECT):
unsafe {
    self.stream.launch_builder(&self.matvec_fn)
        .arg(&d_h_re)
        .arg(&d_h_im)
        .arg(&d_state_re)
        .arg(&d_state_im)
        .arg(&alpha_re)
        .arg(&alpha_im)
        .arg(&mut d_temp_re)
        .arg(&mut d_temp_im)
        .arg(&(n as i32))
        .launch(cfg)?;
}
```

## Recommendation

**DO NOT** proceed with cudarc 0.18.1 migration until the user confirms they want the builder pattern API.

The current codebase works with cudarc 0.9. Upgrading to 0.18.1 requires:
- 100+ launch call conversions
- Extensive testing
- Potential debugging of subtle parameter passing issues

## Files That Need Immediate Attention

ALL files I modified need to be reverted or properly fixed before compilation will succeed.

Currently, **compilation is broken** due to incorrect API usage.
