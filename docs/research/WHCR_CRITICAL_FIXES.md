# WHCR Critical Fixes Required

## Status: DISABLED in Production
**Date**: 2025-11-27
**Severity**: CRITICAL - Makes solutions 13-15x worse

## Immediate Actions Taken
✅ Disabled WHCR in configs/WHCR_FIX_TEST.toml
✅ Fixed uninitialized memory bug (lines 318, 447-450)
❌ Core algorithmic issues remain unresolved

## Critical Fixes Required (Priority Order)

### 1. Fix Delta Calculation (HIGHEST PRIORITY)
**Location**: whcr.cu lines 454, 324-338
**Problem**: Delta sign is inverted - positive deltas are treated as improvements
**Fix Required**:
```cuda
// Current (WRONG):
double delta = neighbor_weights[new_color] - neighbor_weights[current_color];

// Should be:
double delta = neighbor_weights[current_color] - neighbor_weights[new_color];
// OR check for delta > 0 instead of delta < 0
```

### 2. Implement Proper Graph Coarsening
**Location**: prism-gpu/src/whcr.rs
**Problem**: V-cycle uses same graph at all levels
**Fix Required**:
- Implement edge contraction for coarsening
- Build hierarchy of progressively smaller graphs
- Add restriction/interpolation operators
- Properly map solutions between levels

### 3. Fix Geometry Weight Application
**Location**: whcr.cu lines 436-438, 464-466
**Problem**: High-stress vertices get higher weights, making them harder to fix
**Fix Required**:
```cuda
// Current (WRONG):
double weight = 1.0 + (my_stress + n_stress) * c_stress_weight;

// Should prioritize high-stress vertices:
double weight = 1.0 - (my_stress + n_stress) * c_stress_weight;
// OR increase delta reduction for hotspots
```

### 4. Validate Move Application Threshold
**Location**: whcr.cu lines 577-580 (apply_moves kernels)
**Problem**: Threshold of -0.001f too permissive
**Fix Required**:
```cuda
// Increase threshold to prevent bad moves:
if (delta < -1.0f) {  // Much stricter threshold
    // Apply move
}
```

### 5. Add Conflict Validation Before Moves
**Location**: whcr.cu apply_moves kernels
**Problem**: No validation that moves actually reduce conflicts
**Fix Required**:
- Count actual conflicts before/after move
- Only apply if conflicts truly decrease
- Add atomic conflict counter update

## Testing Protocol After Fixes

1. **Baseline Test** (WHCR disabled):
```bash
./target/release/prism-cli --input benchmarks/dimacs/DSJC125.5.col \
  --config configs/WHCR_FIX_TEST.toml --gpu --attempts 1
```
Expected: ~17 colors, 0 conflicts

2. **WHCR Enabled Test** (after fixes):
```bash
# Re-enable WHCR in config
# Same command as above
```
Expected behavior:
- Level 4: 147 → ~100 conflicts
- Level 3: ~100 → ~50 conflicts
- Level 2: ~50 → ~20 conflicts
- Level 1: ~20 → ~5 conflicts
- Level 0: ~5 → 0 conflicts

## Recommendation: Keep WHCR Disabled Until Fixed

The WHCR system should remain **DISABLED** in all production configs until:
1. Delta calculation is fixed
2. Proper V-cycle with coarsening is implemented
3. Geometry weights correctly prioritize problem vertices
4. Move validation prevents conflict explosion

## Alternative: Simplify to Basic Conflict Repair

If full WHCR is too complex, consider a simpler approach:
1. Remove V-cycle complexity
2. Use single-level conflict-focused repair
3. Apply only moves that provably reduce conflicts
4. Skip geometry/belief integration until core works

## Performance Impact

Currently with WHCR enabled:
- DSJC125.5: 147 → 2308 conflicts (15.7x worse)
- DSJC500.5: 1534 → 50892 conflicts (33x worse)

After proper fixes, expected:
- DSJC125.5: 147 → 0 conflicts (100% improvement)
- DSJC500.5: 1534 → <100 conflicts (93% improvement)
