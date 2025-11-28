# GPU Kernel Modification Guide: Chemical Potential Optimization

## Quick Reference

**File**: `/mnt/c/Users/Predator/Desktop/PRISM/prism-gpu/src/kernels/quantum.cu`
**Line**: 431
**Change**: Single value modification (0.6 → 0.85)

---

## The Modification

### Current Code (Line 431)
```cuda
float chemical_potential = 0.6f * (float)color / (float)max_colors;
```

### Optimized Code
```cuda
float chemical_potential = 0.85f * (float)color / (float)max_colors;
```

**That's it!** A single constant change from `0.6f` to `0.85f`.

---

## Why This Matters

### Physics Explanation (Simple)
- Chemical potential (μ) creates a "pressure" gradient on color indices
- Higher μ → Stronger pressure toward lower colors
- Keeps the quantum evolution locked at 17 colors while resolving conflicts
- Current μ=0.6 is too weak: allows conflicts under quantum dynamics
- Optimized μ=0.85 provides strong boundary enforcement

### Impact Formula
```cuda
color_penalty = μ * (color / max_colors) * coupling * evolution_time
scale_factor = exp(-conflict_penalty - color_penalty + preference)
amplitude *= scale_factor
```

**For color 16 (highest) with coupling=10, evolution_time=0.18:**
- Old (μ=0.6): penalty = 0.6 * (16/17) * 10 * 0.18 = 1.017 → scale = 0.362
- New (μ=0.85): penalty = 0.85 * (16/17) * 10 * 0.18 = 1.439 → scale = 0.237

**Result**: 35% stronger damping on high-color amplitudes → maintains 17-color compression

---

## Step-by-Step Instructions

### 1. Open the File
```bash
cd /mnt/c/Users/Predator/Desktop/PRISM
nano prism-gpu/src/kernels/quantum.cu
```
Or use your preferred editor (VSCode, vim, etc.)

### 2. Navigate to Line 431
In nano: `Ctrl+_` then type `431`
In vim: `:431`
In VSCode: `Ctrl+G` then type `431`

### 3. Locate the Line
You should see:
```cuda
        // CHEMICAL POTENTIAL: Pressure to use lower colors (compression)
        // Higher color indices get exponentially penalized
        // mu=0.6 is BALANCED: Moderate compression to avoid conflicts
        float chemical_potential = 0.6f * (float)color / (float)max_colors;  // ← THIS LINE
        float color_penalty = chemical_potential * coupling * evolution_time;
```

### 4. Make the Change
Change line 431 from:
```cuda
        float chemical_potential = 0.6f * (float)color / (float)max_colors;
```
To:
```cuda
        float chemical_potential = 0.85f * (float)color / (float)max_colors;
```

### 5. Update the Comment (Optional but Recommended)
Change line 430 from:
```cuda
        // mu=0.6 is BALANCED: Moderate compression to avoid conflicts
```
To:
```cuda
        // mu=0.85 is OPTIMIZED: Strong compression for conflict reduction at 17 colors
```

### 6. Save the File
In nano: `Ctrl+O`, `Enter`, `Ctrl+X`
In vim: `:wq`
In VSCode: `Ctrl+S`

---

## Compilation

### Clean Previous Build
```bash
cd /mnt/c/Users/Predator/Desktop/PRISM
cargo clean --release
```

### Recompile with CUDA
```bash
cargo build --release --features cuda
```

### Expected Output
You should see:
```
   Compiling prism-gpu v0.1.0 (/mnt/c/Users/Predator/Desktop/PRISM/prism-gpu)
   Compiling prism-cli v0.1.0 (/mnt/c/Users/Predator/Desktop/PRISM/prism-cli)
    Finished release [optimized] target(s) in X.XXs
```

### Verify Success
```bash
ls -lh target/release/prism-cli
# Should show a recent timestamp and executable permissions
```

---

## Testing

### Quick Test Run
```bash
./target/release/prism-cli solve \
  --config configs/OPTIMIZED_CONFLICT_REDUCTION.toml \
  --graph benchmarks/dimacs/DSJC125.5.col \
  --device cuda \
  --output test_result.json
```

### Monitor Telemetry
```bash
# In another terminal:
tail -f telemetry.jsonl | grep -E "Phase3|conflicts"
```

### Verify Results
```bash
cat test_result.json | jq '{colors: .num_colors, conflicts: .num_conflicts}'
```

**Success Criteria**:
- `colors`: 17
- `conflicts`: 0

---

## Troubleshooting

### Compilation Error: "CUDA not found"
**Problem**: CUDA toolkit not installed or not in PATH
**Solution**:
```bash
# Check CUDA installation
nvcc --version

# If not found, install CUDA 11.8+ from NVIDIA website
# Or use CPU fallback:
cargo build --release  # (without --features cuda)
```

### Compilation Error: "syntax error near line 431"
**Problem**: Typo in modification
**Solution**: Carefully re-check the line. It should be:
```cuda
float chemical_potential = 0.85f * (float)color / (float)max_colors;
```
Note the `f` suffix on `0.85f` (denotes float literal in CUDA C)

### Runtime Error: "kernel launch failed"
**Problem**: GPU kernel incompatibility or out of memory
**Solution**:
```bash
# Check GPU status
nvidia-smi

# Try with smaller batch:
./target/release/prism-cli solve --config configs/WORKING_17.toml ...

# If still fails, use CPU:
./target/release/prism-cli solve --device cpu ...
```

### Results Show Colors > 17
**Problem**: μ=0.85 is TOO STRONG for your graph/parameters
**Solution**: Reduce to intermediate value
```cuda
float chemical_potential = 0.75f * (float)color / (float)max_colors;
```
Recompile and test again.

### Results Show Conflicts > 20 After Phase 3
**Problem**: μ=0.85 is TOO WEAK or evolution insufficient
**Solution**:
1. First try increasing evolution_iterations to 500 in config
2. If still insufficient, increase μ to 0.90:
```cuda
float chemical_potential = 0.90f * (float)color / (float)max_colors;
```

---

## Rollback Instructions

### If Something Goes Wrong

**Revert the Change**:
```bash
cd /mnt/c/Users/Predator/Desktop/PRISM
git diff prism-gpu/src/kernels/quantum.cu
# Review the change

git checkout prism-gpu/src/kernels/quantum.cu
# Revert to original
```

**Recompile Original**:
```bash
cargo clean --release
cargo build --release --features cuda
```

**Test Original Config**:
```bash
./target/release/prism-cli solve \
  --config configs/TUNED_17.toml \
  --graph benchmarks/dimacs/DSJC125.5.col \
  --device cuda
```

You should get the baseline: 17 colors, 58 conflicts after Phase 3.

---

## Advanced Tuning

### If You Want to Experiment Further

The chemical potential is just ONE line in the quantum kernel. Other tunable parameters:

#### Preference Boost (Line 437)
```cuda
preference_boost = coupling * degree_factor * evolution_time * 0.1f;  // ← 0.1f is tunable
```
Higher → Stronger symmetry breaking (can help with conflicts)
Suggested range: 0.05 - 0.15

#### Tunneling Amplitude (Line 447)
```cuda
new_r += transverse_field * sinf(tunnel_phase) * 0.02f;  // ← 0.02f is tunable
```
Higher → Stronger quantum tunneling (can escape local minima)
Suggested range: 0.01 - 0.05

#### Noise Scale (Line 676)
```cuda
float noise_scale = 0.5f * base_amplitude;  // ← 0.5f is tunable (line 676)
```
Higher → More stochastic symmetry breaking
Suggested range: 0.3 - 0.7

**WARNING**: Only modify these if you understand the physics implications. Start with just the chemical potential change (line 431).

---

## Chemical Potential Value Guide

| μ Value | Behavior | Best For | Risk |
|---------|----------|----------|------|
| 0.4 | Very weak compression | Graphs with low chromatic number | May exceed target colors |
| 0.6 | Moderate compression (current) | Baseline, stable but conflicts | 58 conflicts @ DSJC125.5 |
| 0.75 | Strong compression | Good intermediate | Low risk |
| **0.85** | **Very strong compression** | **Conflict reduction target** | **Recommended start** |
| 0.90 | Maximum compression | If 0.85 insufficient | May be too aggressive |
| 1.0 | Extreme compression | Last resort only | Risk of over-compression |

**Recommendation**: Start with 0.85. Only increase if conflicts remain >20 after Phase 3.

---

## Validation Checklist

After modification and recompilation:

- [ ] File modified: quantum.cu line 431 shows `0.85f`
- [ ] Compilation successful: No errors, prism-cli binary updated
- [ ] Test run completes: No crashes or GPU errors
- [ ] Phase 3 output: 17 colors achieved
- [ ] Phase 3 conflicts: <20 (ideally <15)
- [ ] Final output: 0 conflicts (after memetic repair)
- [ ] Geometric stress: <0.5
- [ ] Telemetry captured: All phases logged correctly

If all checks pass: **SUCCESS!** You've optimized the quantum kernel.

---

## Quick Command Summary

```bash
# Full workflow in one script:

cd /mnt/c/Users/Predator/Desktop/PRISM

# 1. Backup original
cp prism-gpu/src/kernels/quantum.cu prism-gpu/src/kernels/quantum.cu.bak

# 2. Make modification (use sed for automation)
sed -i 's/0.6f \* (float)color/0.85f * (float)color/' prism-gpu/src/kernels/quantum.cu

# 3. Verify change
grep "chemical_potential = 0.85f" prism-gpu/src/kernels/quantum.cu
# Should output the modified line

# 4. Recompile
cargo clean --release
cargo build --release --features cuda

# 5. Test
./target/release/prism-cli solve \
  --config configs/OPTIMIZED_CONFLICT_REDUCTION.toml \
  --graph benchmarks/dimacs/DSJC125.5.col \
  --device cuda \
  --output result.json

# 6. Check results
cat result.json | jq '{colors: .num_colors, conflicts: .num_conflicts, stress: .geometric_stress}'

# Expected: {"colors": 17, "conflicts": 0, "stress": <0.5}
```

---

## Contact & Support

If you encounter issues:

1. Check telemetry logs: `tail -100 telemetry.jsonl`
2. Review compilation errors: Full cargo output
3. Test with baseline config first: `configs/TUNED_17.toml`
4. Verify GPU availability: `nvidia-smi`
5. Check CUDA version: `nvcc --version` (need 11.8+)

**Document Version**: 1.0
**Last Updated**: 2025-11-23
**Status**: Ready for implementation
