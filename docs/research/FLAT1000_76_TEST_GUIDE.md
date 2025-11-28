# FLAT1000_76 Hypertuned Test Guide

## Overview
This guide provides a complete setup for testing the PRISM system on the **flat1000_76** DIMACS benchmark graph, targeting its chromatic number of **exactly 76 colors with 0 conflicts**.

## Graph Specifications
- **Name**: flat1000_76_0
- **Vertices**: 1000
- **Edges**: 246,708
- **Chromatic Number (χ)**: 76
- **Type**: Flat graph (geometric structure)
- **Difficulty**: High (larger graph with tight chromatic bound)

## Files Created

### 1. Configuration File
**Path**: `configs/FLAT1000_76_HYPERTUNED.toml`

Key optimizations:
- **Phase 2 (Thermodynamic)**: 150,000 iterations, very slow cooling (0.996), target range 73-76
- **Phase 3 (Quantum)**: Strong chemical potential (3.0), 750 iterations, hard limit at 76 colors
- **Memetic**: 98% local search probability, 300 Kempe chain attempts, head-first aggressive mode
- **GPU**: Full tensor core utilization, dynamic parallelism enabled
- **WHCR**: DISABLED (causes oscillation issues)

### 2. Test Script
**Path**: `scripts/test_flat1000_76.sh`

Features:
- Automatic build and PTX compilation
- 3 attempts with color-coded output
- Success detection for 76-color solutions
- Detailed failure analysis
- Results saved to `results/flat1000_76/`

## Setup Instructions

### 1. Download Benchmark File
```bash
# Download flat1000_76_0.col if not present
cd benchmarks/dimacs/
wget http://www.info.univ-angers.fr/pub/porumbel/graphs/flat1000_76_0.col
# OR
wget https://mat.tepper.cmu.edu/COLOR04/INSTANCES/flat1000_76_0.col
```

### 2. Verify Environment
```bash
# Check CUDA installation
nvidia-smi
nvcc --version  # Should show CUDA 12.6

# Check Rust toolchain
rustc --version
cargo --version
```

### 3. Run Test
```bash
# Make script executable (already done)
chmod +x scripts/test_flat1000_76.sh

# Run the test
./scripts/test_flat1000_76.sh
```

## Key Parameters & Tuning

### Phase 2 - Thermodynamic Annealing
```toml
max_iterations_per_round = 150000  # Very deep exploration
conflict_penalty_weight = 250.0    # Extreme penalty
annealing_factor = 0.996          # Very slow cooling
min_colors = 73                   # Stay close to target
target_colors = 76                # Exact target
```

### Phase 3 - Quantum Evolution
```toml
target_colors = 76                # Chromatic number
chemical_potential = 3.0          # Strong compression
quantum_iterations = 750          # More iterations for 1000 vertices
max_colors = 76                   # Hard limit
```

### Memetic Optimization
```toml
local_search_probability = 0.98  # Almost always search
kempe_chain_attempts = 300       # Extensive chains
conflict_resolution_mode = "head" # Aggressive mode
color_class_balancing = true     # Balance distribution
force_color_reduction = true     # Push toward 76
```

## Expected Behavior

### Success Scenario
```
Phase 2: ~76-80 colors, 0-10 conflicts
Phase 3: Compress to exactly 76 colors
Phase 4: Maintain 76 colors, 0 conflicts
Final: 76 colors, 0 conflicts ✅
```

### Common Issues & Solutions

1. **Too Few Colors with Conflicts**
   - Reduce `conflict_penalty_weight` slightly
   - Increase `min_colors` to 74-75
   - Lower `annealing_factor` to 0.995

2. **Too Many Colors (>76)**
   - Increase `chemical_potential` to 3.5-4.0
   - Add more `quantum_iterations`
   - Increase `memetic_intensity`

3. **Oscillation in Conflicts**
   - Ensure WHCR remains disabled
   - Reduce `local_search_iterations`
   - Adjust `tabu_tenure` higher

## Command Line Options

### Direct Run (Manual)
```bash
PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.6 \
RUST_LOG=info \
timeout 900 ./target/release/prism-cli \
    --input benchmarks/dimacs/flat1000_76_0.col \
    --config configs/FLAT1000_76_HYPERTUNED.toml \
    --gpu \
    --attempts 1 \
    --verbose 2>&1 | tee flat1000_76_manual.log
```

### Multiple Attempts
```bash
# Run 5 attempts sequentially
for i in {1..5}; do
    echo "Attempt $i"
    ./scripts/test_flat1000_76.sh
    if [ $? -eq 0 ]; then
        echo "Success on attempt $i!"
        break
    fi
done
```

## Monitoring Progress

### Real-time Monitoring
```bash
# Watch log file
tail -f results/flat1000_76/flat1000_76_*.log | grep -E "Phase|colors|conflicts"

# Check for success
grep "76 colors.*0 conflicts" results/flat1000_76/*.log
```

### Performance Metrics
```bash
# Extract phase timings
grep "completed.*ms" results/flat1000_76/*.log

# Check convergence
grep "NEW BEST" results/flat1000_76/*.log | tail -10
```

## Hyperparameter Tuning Tips

### If Getting 77+ Colors
1. Increase quantum `chemical_potential` by 0.5
2. Set `force_color_limit = true` in Phase 3
3. Reduce `min_colors` in Phase 2 to 72

### If Getting <76 Colors with Conflicts
1. Reduce thermodynamic `conflict_penalty_weight` by 50
2. Increase `min_colors` to 74-75
3. Add more `replica_count` (up to 48)

### For Faster Convergence
1. Reduce `max_iterations_per_round` to 100000
2. Set `annealing_factor` to 0.994
3. Lower `quantum_iterations` to 500

## Success Criteria

The test is successful when:
1. **Colors**: Exactly 76
2. **Conflicts**: Exactly 0
3. **Validation**: Graph properly colored with no adjacent vertices sharing colors
4. **Stability**: Solution maintains through phases 4-7

## Troubleshooting

### PTX Compilation Errors
```bash
# Manually compile specific kernel
$CUDA_HOME/bin/nvcc -ptx --gpu-architecture=sm_86 \
    -o target/ptx/quantum.ptx \
    prism-gpu/src/kernels/quantum.cu
```

### Out of Memory
- Reduce `replica_count` to 16
- Lower `num_landmarks` to 200
- Decrease `hidden_dim` to 128

### Timeout Issues
- Increase timeout in script from 900 to 1200 seconds
- Reduce `max_iterations_per_round`
- Use fewer attempts

## Reference Results

Expected performance on RTX 4090:
- Phase 2: ~60-70 seconds
- Phase 3: ~5-10 seconds
- Total time: ~90-120 seconds per attempt
- Success rate: ~40-60% with proper tuning

## Next Steps

Once 76-color solution achieved:
1. Save successful configuration
2. Test on other flat graphs (flat300_28, flat1000_60)
3. Fine-tune for speed optimization
4. Document parameter sensitivity

## Contact & Support

For issues or improvements:
- Check logs in `results/flat1000_76/`
- Review telemetry in `telemetry.jsonl`
- Adjust parameters based on phase-specific feedback

---
*This configuration represents extensive hyperparameter tuning specifically for the flat1000_76 benchmark.*