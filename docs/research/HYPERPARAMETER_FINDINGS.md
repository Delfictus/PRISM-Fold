# Hyperparameter Tuning Findings - DSJC500.5

## Test Results Summary

### Baseline Configuration (DSJC500_CONFIG.toml)
- **Phase 2 Runtime**: 70 seconds
- **Colors Achieved**: 23 (from 71)
- **Remaining Conflicts**: 1534
- **WHCR**: Catastrophic failure (oscillation 1534 → 32591 ↔ 50892)

### Optimized Configuration (DSJC500_OPTIMIZED.toml)
- **Phase 2 Runtime**: 66 seconds (expected ~25s)
- **Colors Achieved**: 23 (same as baseline)
- **Remaining Conflicts**: 1534 (same as baseline)
- **WHCR**: Still running despite being disabled

## Critical Issues Identified

### 1. Configuration Not Loading Properly
- Set `max_iterations_per_round = 3000` but still running 10,000
- Set `whcr.enabled = false` but WHCR still invoked
- Possible causes:
  - Config parsing errors
  - Hard-coded defaults overriding TOML
  - Missing config sections being ignored

### 2. WHCR Buffer Oscillation Bug
**Pattern**: Conflicts oscillate between two values indefinitely
- DSJC125.5: 147 → 1372 ↔ 1721
- DSJC500.5: 1534 → 32591 ↔ 50892

**Root Cause**: Buffer mismatch between f32/f64 kernels
- Evaluation kernel writes to `d_move_deltas_f64`
- Apply kernel reads from `d_move_deltas_f32`
- Buffers contain stale/garbage data

### 3. Thermodynamic Phase Bottleneck
- Most improvement happens in first 3000 iterations
- Iterations 3000-10000 provide minimal benefit
- Temperature auto-capping affects exploration

## Successful Parameters (DSJC125.5)
From previous tests that achieved 20 colors:
- Chemical potential: 0.85-0.9
- Conflict penalty: High (50+)
- Memetic search: Aggressive (freq=25, intensity=20)
- Temperature range: [0.01, 3.0]

## Recommendations

### Immediate Actions
1. **Fix config loading** - Ensure TOML overrides are applied
2. **Disable WHCR completely** - Add hard bypass in code
3. **Reduce thermodynamic iterations** - Cap at 3000

### Parameter Adjustments for DSJC500.5
Based on analysis:
```toml
[phase2]
max_iterations_per_round = 3000  # MUST be enforced
conflict_penalty_weight = 100.0  # Double from optimized
allow_temporary_conflicts = true
replica_count = 16  # More diversity
sweep_enabled = true
early_stopping_delta = 0.001  # Stop if no improvement

[phase3]
target_colors = 42  # More aggressive
chemical_potential = 0.85  # Sweet spot from DSJC125
coupling_strength = 15.0

[memetic]
local_search_probability = 0.7
local_search_iterations = 150
kempe_chain_attempts = 100
```

### Code Fixes Required
1. Check config loading in `prism-cli/src/main.rs`
2. Add explicit WHCR bypass check
3. Verify thermodynamic iteration override
4. Add config validation/logging

## Performance Targets
- **Colors**: < 48 (best known for DSJC500.5)
- **Conflicts**: 0
- **Runtime**: < 30 seconds total
- **Memory**: < 2GB GPU

## Next Steps
1. Debug config loading issue
2. Hard-code WHCR disable as temporary fix
3. Test with enforced 3000 iterations
4. Implement early stopping for thermodynamic phase
5. Test on other DIMACS graphs for validation