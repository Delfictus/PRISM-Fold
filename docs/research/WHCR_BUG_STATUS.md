# WHCR Bug Status Report

## Current State (F32 VALIDATION FAILED)
**Date**: 2025-11-27
**Status**: ❌ F32 on-device revalidation FAILED - oscillation persists
**Severity**: CRITICAL - WHCR makes solutions 13x worse

## Latest Test Results (With F32 On-Device Revalidation)
- **Input**: DSJC125.5 graph (125 vertices)
- **Initial State**: 11 colors, 147 conflicts
- **After WHCR Level 4**: Oscillating between 1930 and 958 conflicts
- **Pattern**: 147 → 1930 → 958 → 1930 → 958 (repeating)
- **Moves Applied**: 89 out of 97 conflict vertices (still too many!)
- **Test Command**: `prism-cli --input DSJC125.5.col --config WHCR_FIX_TEST.toml`

## All Fixes Applied (None Working)
✅ Re-enabled all WHCR phases (2, 3, 5, 7)
✅ Fixed Rust borrow checker issue in belief buffer allocation
✅ Reordered operations to avoid mutable/immutable borrow conflicts
✅ Added buffer synchronization between f64/f32 kernels
✅ Implemented proper geometry binding with zero-copy
✅ Corrected V-cycle iteration (coarse→fine)
✅ Added safe belief fallback allocation
✅ GPU-side validation in apply_moves_with_locking_f64 kernel
✅ CSR pointers passed to kernel for conflict revalidation
✅ f64 conflict counts mirrored to f32 buffer
✅ **LATEST**: Added f32 on-device revalidation (same as f64)
❌ **Core oscillation issue STILL UNRESOLVED after 5 fix attempts**

## ROOT CAUSE IDENTIFIED!

### The Bug: Uninitialized Memory Read
**Location**: `prism-gpu/src/kernels/whcr.cu`, line 318
**Issue**: The `evaluate_moves_f32` kernel skips writing delta for current color

```cuda
// Line 317-318 - THE BUG
for (int new_color = 0; new_color < num_colors; new_color++) {
    if (new_color == current_color) continue;  // NEVER writes move_deltas[tid * num_colors + current_color]!
}
```

### Why This Causes Oscillation
1. Kernel evaluates all colors EXCEPT current (line 318 skips)
2. If no color improves, `best_color = current_color`
3. But `move_deltas[tid * num_colors + current_color]` was never written
4. Apply kernel reads uninitialized/zero value from that slot
5. Zero or garbage may satisfy `delta < -0.001f` check
6. All 97 vertices get "moved" causing conflict explosion

### Proof
- ALL 97 conflict vertices are moved (shouldn't happen)
- Conflicts explode identically every time: 147 → 2308
- Pattern is deterministic due to zeroed buffer

## THE FIX

### Primary Fix: Write Zero for Current Color
In `prism-gpu/src/kernels/whcr.cu`, line 318:
```cuda
// evaluate_moves_f32 kernel (around line 317)
for (int new_color = 0; new_color < num_colors; new_color++) {
    if (new_color == current_color) {
        move_deltas[tid * num_colors + new_color] = 0.0f;  // FIX: Explicitly write zero
        continue;
    }
    // ... rest of evaluation
}
```

### Same fix needed in evaluate_moves_f64
Around line 420 (in the f64 evaluation kernel), apply the same fix.

### Alternative Fix: Check in Apply Kernel
Add safeguard in `apply_moves_with_locking_f32`:
```cuda
if (new_color == coloring[vertex]) {
    return;  // Skip no-op moves
}
```

## Evidence of Impact
- DSJC125.5: Conflicts increase from 147 → 2308 (should decrease to ~0)
- DSJC500.5: Conflicts increase from 1534 → 50892 (should decrease to <100)

## Validation After Fixes
Expected behavior after proper fixes:
1. Level 4: 147 → ~100 conflicts
2. Level 3: ~100 → ~50 conflicts
3. Level 2: ~50 → ~20 conflicts
4. Level 1: ~20 → ~5 conflicts
5. Level 0: ~5 → 0 conflicts

## Testing Command
```bash
PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.6 \
RUST_LOG=info \
./target/release/prism-cli \
  --input benchmarks/dimacs/DSJC125.5.col \
  --config configs/WHCR_FIX_TEST.toml \
  --gpu --attempts 1 --verbose 2>&1 | \
  grep "WHCR Level" | head -20
```

## Recommendation
Until the buffer synchronization and V-cycle issues are properly fixed, WHCR should be disabled in production as it makes solutions worse rather than better.