# DSJC500.5 Hyperparameter Tuning Report

## Executive Summary
Analysis of DSJC500.5 graph coloring performance to optimize hyperparameters for achieving < 50 colors (best known: ~48).

## Graph Characteristics
- **Vertices**: 500
- **Edges**: 62,624
- **Density**: ~0.5
- **Target Colors**: < 50 (best known ~48)

## Initial Test Results

### Phase Performance (Original Config)
1. **Phase 1 (Active Inference)**: 71 colors in 1.14ms
2. **Phase 2 (Thermodynamic)**:
   - Colors: 71 → 23 (67.6% reduction)
   - Conflicts: 1534 remaining
   - Runtime: ~70 seconds
   - Temperature: Auto-capped at 1.89
3. **WHCR**: FAILED - Oscillation bug (1534 → 32,591 ↔ 50,892 conflicts)

### Key Issues Identified
1. **Thermodynamic Phase**:
   - Too many iterations (10,000) with slow convergence
   - Insufficient conflict penalty weight (10.0)
   - Temperature range not optimal
   - Most improvement in first 3000 iterations

2. **WHCR Module**:
   - Critical bug: Buffer mismatch between f32/f64 kernels
   - Oscillating conflicts making problem worse
   - Must be disabled until fixed

## Hyperparameter Optimizations

### Critical Changes
| Parameter | Original | Optimized | Rationale |
|-----------|----------|-----------|-----------|
| **Phase 2 Iterations** | 10,000 | 3,000 | 70% reduction - most gains early |
| **Conflict Penalty** | 10.0 | 50.0 | 5x increase to reduce conflicts |
| **Initial Temperature** | 2.0 | 3.0 | Better exploration |
| **Annealing Factor** | 0.95 | 0.92 | Faster cooling |
| **Compaction Factor** | 0.95 | 0.90 | More aggressive |
| **Replicas** | 8 | 12 | More diversity |
| **WHCR Enabled** | true | false | Critical bug |

### Phase-Specific Optimizations

#### Phase 3 (Quantum)
- Target colors: 48 → 45 (more aggressive)
- Coupling strength: 8.0 → 12.0
- Chemical potential: 0.6 → 0.75
- Added tunneling field: 0.3

#### Phase 7 (Ensemble)
- Population: 200 → 300
- Generations: 1000 → 2000
- Memetic frequency: 50 → 25 (more frequent)
- Memetic intensity: 10 → 20

#### Memetic Search
- Local search probability: 0.3 → 0.5
- Local search iterations: 50 → 100
- Added Kempe chain attempts: 50

## Expected Improvements
1. **Runtime**: ~70s → ~25s (64% reduction)
2. **Conflict Resolution**: Better with 5x penalty weight
3. **Color Reduction**: More aggressive targeting
4. **Exploration**: Better temperature schedule

## Testing Protocol
1. Run with optimized config (DSJC500_OPTIMIZED.toml)
2. Compare against baseline
3. Track:
   - Final colors achieved
   - Remaining conflicts
   - Phase execution times
   - Memory usage

## Next Steps
1. Complete optimized test run
2. Analyze telemetry for bottlenecks
3. Fine-tune based on results
4. Test on other DIMACS graphs
5. Fix WHCR oscillation bug (priority)

## Commands
```bash
# Baseline test
PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.6 \
./target/release/prism-cli \
  --input benchmarks/dimacs/DSJC500.5.col \
  --config configs/DSJC500_CONFIG.toml \
  --gpu --attempts 1 --verbose

# Optimized test
PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.6 \
./target/release/prism-cli \
  --input benchmarks/dimacs/DSJC500.5.col \
  --config configs/DSJC500_OPTIMIZED.toml \
  --gpu --attempts 1 --verbose
```

## Telemetry Analysis Points
- Phase 2 convergence curve
- Conflict reduction rate
- Temperature vs. acceptance ratio
- Memory pool utilization
- GPU kernel efficiency

## Success Metrics
- **Primary**: Achieve < 50 colors with 0 conflicts
- **Secondary**: Runtime < 30 seconds
- **Tertiary**: Consistent results across runs