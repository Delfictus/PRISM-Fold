# WHCR Fix Testing Guide

## Quick Test Commands

### 1. Basic Compilation Test
```bash
cargo build --release --features cuda
```
**Expected**: No errors, warnings OK

---

### 2. Run WHCR on DSJC125.5
```bash
RUST_LOG=debug ./target/release/prism-cli solve \
  --graph instances/DSJC125.5.col \
  --config configs/TUNED_17.toml \
  --output solution_125.txt \
  2>&1 | tee whcr_test_output.log
```

---

### 3. Extract WHCR Logs
```bash
grep "WHCR Level" whcr_test_output.log
```

**Expected Output Pattern**:
```
[DEBUG] WHCR Level 4: Before repair - 2308 conflicts
[INFO]  WHCR Level 4: Complete - conflicts 2308 → 1842 (delta: -466, precision: f32)
[DEBUG] WHCR Level 3: Before repair - 1842 conflicts
[INFO]  WHCR Level 3: Complete - conflicts 1842 → 1156 (delta: -686, precision: f32)
[DEBUG] WHCR Level 2: Before repair - 1156 conflicts
[INFO]  WHCR Level 2: Complete - conflicts 1156 → 523 (delta: -633, precision: f64)
[DEBUG] WHCR Level 1: Before repair - 523 conflicts
[INFO]  WHCR Level 1: Complete - conflicts 523 → 89 (delta: -434, precision: f64)
[DEBUG] WHCR Level 0: Before repair - 89 conflicts
[INFO]  WHCR Level 0: Complete - conflicts 89 → 0 (delta: -89, precision: f64)
```

---

### 4. Check for Warning Flags
```bash
grep "No moves applied but" whcr_test_output.log
```

**Expected**: NO OUTPUT (if warnings appear, buffer selection may still have issues)

---

### 5. Validate Final Solution
```bash
# Check for zero conflicts
grep "final_conflicts: 0" whcr_test_output.log

# Validate solution file
python scripts/validate_coloring.py \
  instances/DSJC125.5.col \
  solution_125.txt
```

**Expected**: `Valid coloring with N colors` (N ≤ 17 for world-record attempt)

---

## Success Criteria

### ✅ PASS Indicators
- [ ] Build completes with no errors
- [ ] All WHCR levels show **negative deltas** (conflicts decreasing)
- [ ] Final conflicts reach **0**
- [ ] Fine levels (0-2) use **f64 precision**
- [ ] Coarse levels (3-4) use **f32 precision**
- [ ] No "No moves applied" warnings
- [ ] Solution validates as proper coloring

### ❌ FAIL Indicators
- [ ] Conflicts stuck at same value (e.g., 2308 → 2308)
- [ ] Positive or zero deltas
- [ ] Warnings: "No moves applied but X conflicts remain"
- [ ] Final conflicts > 0
- [ ] Solution validation fails

---

## Debugging Failed Tests

### If conflicts don't decrease:
```bash
# Check which buffer is being used
grep "Converting f64 move deltas to f32" whcr_test_output.log

# Should appear for fine levels (0-2) when use_precise=true
```

### If synchronization issues suspected:
```bash
# Check for GPU errors in log
grep -i "cuda\|gpu\|synchronize" whcr_test_output.log
```

### If move evaluation seems wrong:
```bash
# Check move counts per iteration
grep "Applied.*color changes" whcr_test_output.log
```

---

## Performance Benchmarks

### Target Metrics (DSJC125.5)
- **Total WHCR time**: < 2 seconds
- **Conflicts per level**: Reduce by 30-50% per level
- **Moves per iteration**: 10-100 at fine levels, 1-20 at coarse levels
- **Final conflicts**: 0

### Collect Timing Data
```bash
# Enable performance logging
RUST_LOG=info,prism_gpu=debug ./target/release/prism-cli solve \
  --graph instances/DSJC125.5.col \
  --config configs/TUNED_17.toml \
  --output solution_125.txt \
  2>&1 | grep -E "WHCR|elapsed|conflicts"
```

---

## Quick Validation Script

Save as `test_whcr_fix.sh`:

```bash
#!/bin/bash
set -e

echo "=== WHCR Fix Validation ==="

# Build
echo "[1/5] Building with CUDA..."
cargo build --release --features cuda > /dev/null 2>&1 || {
  echo "❌ Build failed"
  exit 1
}
echo "✅ Build successful"

# Run solver
echo "[2/5] Running WHCR on DSJC125.5..."
RUST_LOG=debug ./target/release/prism-cli solve \
  --graph instances/DSJC125.5.col \
  --config configs/TUNED_17.toml \
  --output solution_125.txt \
  2>&1 | tee whcr_test.log > /dev/null

# Check for decreasing conflicts
echo "[3/5] Checking conflict reduction..."
DELTAS=$(grep "delta:" whcr_test.log | grep -oE "delta: -[0-9]+" | wc -l)
if [ "$DELTAS" -ge 4 ]; then
  echo "✅ Conflicts decreasing ($DELTAS negative deltas found)"
else
  echo "❌ Conflicts not decreasing properly"
  exit 1
fi

# Check for warnings
echo "[4/5] Checking for buffer warnings..."
WARNINGS=$(grep "No moves applied but" whcr_test.log | wc -l)
if [ "$WARNINGS" -eq 0 ]; then
  echo "✅ No buffer mismatch warnings"
else
  echo "❌ Found $WARNINGS buffer warnings"
  exit 1
fi

# Check final conflicts
echo "[5/5] Validating final solution..."
FINAL=$(grep -oE "final_conflicts: [0-9]+" whcr_test.log | tail -1 | grep -oE "[0-9]+")
if [ "$FINAL" -eq 0 ]; then
  echo "✅ Final conflicts: 0"
else
  echo "❌ Final conflicts: $FINAL (expected 0)"
  exit 1
fi

echo ""
echo "🎉 ALL TESTS PASSED - WHCR fix verified!"
```

Usage:
```bash
chmod +x test_whcr_fix.sh
./test_whcr_fix.sh
```

---

## Troubleshooting

### "PTX file not found"
```bash
# Build PTX kernels first
cargo build --release --features cuda
# Or set PTX path explicitly
export PRISM_PTX_PATH=/mnt/c/Users/Predator/Desktop/PRISM/target/ptx/whcr.ptx
```

### "CUDA driver error"
```bash
# Check CUDA availability
nvidia-smi
# Ensure CUDA drivers are loaded
```

### Stuck at 2308 conflicts (original bug)
This means the fix didn't work. Check:
1. Did you rebuild after applying the fix? (`cargo clean && cargo build --release --features cuda`)
2. Is the correct binary being run? (`which prism-cli`)
3. Are logs showing f64→f32 conversion at fine levels? (`grep "Converting" whcr_test.log`)

---

## Expected Timeline

- **Build**: ~20-30 seconds
- **WHCR execution (DSJC125.5)**: 1-2 seconds
- **Full solve (with Phase 2, 3, 5)**: 5-15 seconds
- **Validation**: < 1 second

Total test cycle: **< 1 minute**

---

*Quick Reference - Keep this open during testing*
*Status: Ready for immediate use after fix implementation*
