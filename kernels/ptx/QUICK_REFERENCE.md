# PTX Kernel Quick Reference

**Last Updated**: 2025-11-29
**Status**: All kernels compiled successfully

## Quick Stats

- **Total Kernels**: 20 CUDA files → 24 PTX files
- **Total Size**: 9.6 MB
- **Architecture**: sm_86 (Ampere/Ada)
- **Compilation**: 100% success, 0 critical errors

## File Locations

```
Primary: /mnt/c/Users/Predator/Desktop/PRISM/kernels/ptx/
Backup:  /mnt/c/Users/Predator/Desktop/PRISM/target/ptx/
```

## Kernel Categories

### Core Graph Coloring (4 kernels)
- `whcr.ptx` - 96 KB - Main WHCR algorithm
- `dendritic_whcr.ptx` - 1.1 MB - Neuromorphic variant
- `dr_whcr_ultra.ptx` - 1.2 MB - Ultra reservoir
- `dendritic_reservoir.ptx` - 990 KB - Multi-branch

### Optimization (5 kernels)
- `thermodynamic.ptx` - 978 KB - Simulated annealing
- `quantum.ptx` - 1.2 MB - Quantum annealing
- `pimc.ptx` - 1.1 MB - Path integral MC
- `ensemble_exchange.ptx` - 1.0 MB - Parallel tempering
- `cma_es.ptx` - 1.1 MB - Evolution strategies

### LBS Detection (4 kernels)
- `lbs/distance_matrix.ptx` - 2.1 KB
- `lbs/surface_accessibility.ptx` - 5.0 KB
- `lbs/pocket_clustering.ptx` - 2.1 KB
- `lbs/druggability_scoring.ptx` - 3.2 KB

### Molecular/Graph (4 kernels)
- `molecular_dynamics.ptx` - 1.0 MB
- `gnn_inference.ptx` - 69 KB
- `floyd_warshall.ptx` - 9.9 KB
- `tda.ptx` - 8.6 KB

### Specialized (3 kernels)
- `active_inference.ptx` - 23 KB
- `transfer_entropy.ptx` - 45 KB
- `tptp.ptx` - 39 KB

## Recompilation Commands

### Single Kernel
```bash
/usr/local/cuda-12.6/bin/nvcc --ptx \
    -o kernels/ptx/<kernel>.ptx \
    crates/prism-gpu/src/kernels/<kernel>.cu \
    -arch=sm_86 --std=c++14 -Xcompiler -fPIC \
    -I crates/prism-gpu/src/kernels \
    --use_fast_math -O3
```

### All Kernels
```bash
for cu in crates/prism-gpu/src/kernels/*.cu; do
    name=$(basename "$cu" .cu)
    /usr/local/cuda-12.6/bin/nvcc --ptx \
        -o "kernels/ptx/${name}.ptx" "$cu" \
        -arch=sm_86 --std=c++14 -Xcompiler -fPIC \
        -I crates/prism-gpu/src/kernels \
        --use_fast_math -O3
done
```

### LBS Kernels
```bash
for cu in crates/prism-gpu/src/kernels/lbs/*.cu; do
    name=$(basename "$cu" .cu)
    /usr/local/cuda-12.6/bin/nvcc --ptx \
        -o "kernels/ptx/lbs/${name}.ptx" "$cu" \
        -arch=sm_86 --std=c++14 -Xcompiler -fPIC \
        -I crates/prism-gpu/src/kernels \
        --use_fast_math -O3
done
```

## Verification

### Check All Files Exist
```bash
find kernels/ptx -name "*.ptx" -type f | wc -l
# Should output: 24
```

### Verify Checksums
```bash
sha256sum -c kernels/ptx/SHA256SUMS.txt
```

### Check File Sizes
```bash
du -sh kernels/ptx
# Should output: ~9.6M
```

## Integration Status

- [x] PTX compilation complete
- [x] Files in kernels/ptx/ and target/ptx/
- [x] SHA256 checksums generated
- [ ] Rust FFI bindings verified
- [ ] DIMACS benchmarks run
- [ ] GPU utilization tested

## GPU Requirements

- **Minimum Compute Capability**: 8.6
- **Compatible GPUs**: RTX 30xx/40xx, A100, H100
- **CUDA Version**: 12.6+
- **Recommended VRAM**: 4GB+

## Common Issues

### PTX File Not Found
**Solution**: Check file exists in `kernels/ptx/` or `target/ptx/`

### Compilation Warnings
**Status**: 29 non-critical warnings (unused variables, macro redefinitions)
**Action**: Safe to ignore

### Architecture Mismatch
**Error**: "no kernel image is available for execution"
**Solution**: Recompile with your GPU's compute capability:
```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
# Recompile with appropriate -arch flag
```

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| DSJC500.5 Colors | ≤48 | Pending test |
| GPU Utilization | ≥80% | Pending test |
| PDBBind DCC<4Å | ≥70% | Pending test |
| Build Time | <2 min | Pending test |
| VRAM Usage | <4GB | Pending test |

## Documentation

- `MANIFEST.md` - Detailed kernel documentation
- `COMPILATION_REPORT.txt` - Full compilation report
- `SHA256SUMS.txt` - File integrity checksums
- `QUICK_REFERENCE.md` - This file

## Contact

PRISM Research Team | Delfictus I/O Inc.
Los Angeles, CA 90013
IS@Delfictus.com
