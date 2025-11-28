# OPTIMIZED_17_BALANCED.toml Test Results (μ=0.55)
Date: 2025-11-24
Total Attempts: 10
Runtime: 107.853s (10.785s avg per attempt)

## Key Configuration Changes
- **Chemical Potential**: μ=0.55 in BOTH kernels (was 0.75/0.85)
- **coupling_strength**: 6.0 (was 8.0)
- **evolution_time**: 0.30 (was 0.22)
- **evolution_iterations**: 1000 (was 600)
- **transverse_field**: 1.5 (was 1.0)
- **max_colors**: 17 (was 18)
- **memetic population_size**: 100 (was 80)
- **memetic max_generations**: 500 (was 300)

## Phase 2 Results (Thermodynamic with μ=0.55)
| Attempt | Colors | Conflicts | After Repair |
|---------|--------|-----------|--------------|
| 1       | 13     | 103       | 22 colors    |
| 2       | 12     | 112       | 22 colors    |
| 3       | 12     | 112       | 22 colors    |
| 4       | 12     | 112       | 21 colors    |
| 5       | 13     | 101       | 22 colors    |
| 6       | 13     | 101       | 22 colors    |
| 7       | 13     | 101       | 22 colors    |
| 8       | 13     | 101       | 22 colors    |
| 9       | 13     | 101       | 22 colors    |
| 10      | 13     | 101       | 22 colors    |
| **Mean**| **12.7**| **105.4** | **22.0**    |

## Phase 3 Results (Quantum with μ=0.55)
| Attempt | Colors | Conflicts | After Repair | Final |
|---------|--------|-----------|--------------|-------|
| 1       | 17     | 68        | 23 colors    | 22    |
| 2       | 17     | 73        | 22 colors    | 22    |
| 3       | 17     | 69        | 23 colors    | 22    |
| 4       | 17     | 73        | 21 colors    | **21**|
| 5       | 17     | 73        | 22 colors    | 22    |
| 6       | 17     | 69        | 22 colors    | 22    |
| 7       | 17     | 69        | 22 colors    | 22    |
| 8       | 17     | 69        | 22 colors    | 22    |
| 9       | 17     | 68        | 23 colors    | 22    |
| 10      | 17     | 74        | 22 colors    | **20**|
| **Mean**| **17** | **70.4**  | **22.3**     | **21.6** |

## CRITICAL FINDING: μ=0.55 Made Phase 3 WORSE!

### Comparison to Previous Runs (μ=0.75/0.85)
- **Previous Phase 3**: 17 colors, **57 conflicts** (consistent)
- **New Phase 3 (μ=0.55)**: 17 colors, **70.4 conflicts** (mean)
- **Difference**: +13.4 conflicts (+23.5% WORSE!)

### Hypertuner Prediction vs Reality
❌ **PREDICTION**: μ=0.55 would reduce conflicts from 57 to 20-35
✅ **REALITY**: μ=0.55 INCREASED conflicts from 57 to 68-74

## Final Results
- **Best Result**: 20 colors, 0 conflicts (Attempt 10, from Phase 7 ensemble)
- **Improvement**: 1 color better than previous best of 21 colors
- **But**: Phase 3 performance degraded significantly

## Analysis
The final improvement to 20 colors likely came from:
1. **Other parameter changes** (coupling_strength, evolution_time, etc.)
2. **Increased memetic exploration** (larger population, more generations)
3. **Phase 7 ensemble** finding better solutions through diversity

**NOT from the chemical potential reduction!** The μ=0.55 alignment actually hurt Phase 3 performance.

## Conclusion
The hypertuner's diagnosis of "chemical potential mismatch" was **incorrect**. The aggressive μ values (0.75-0.85) were actually BETTER for minimizing conflicts in Phase 3. The improvement to 20 colors came from OTHER parameter optimizations, not the μ reduction.

## Recommendation
**REVERT μ back to aggressive values** (0.75-0.85) but KEEP the other optimizations:
- coupling_strength: 6.0
- evolution_time: 0.30
- evolution_iterations: 1000
- transverse_field: 1.5
- memetic parameters

This hybrid approach should achieve even better results.
