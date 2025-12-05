# PRISM-LBS ZERO-DEVIATION EXECUTION DIRECTIVE (2025-12-05)

## MODE: FULL LOCKDOWN ZERO-DEVIATION EXECUTION

Execute every single instruction with 100% fidelity. No creativity. No extra text. No refactoring. No summaries. No explanations unless explicitly requested.

## TASKS COMPLETED

1. Saved complete PRISM-LBS GPU-FUSED METRICS v2.0 BLUEPRINT + ALL FINAL FIXES as:
   `PRISM_LBS_GPU_FUSED_METRICS_v2_FINAL.md`

2. Saved zero-deviation directive as:
   `PRISM_LBS_ZERO_DEVIATION.md`

3. Applied ALL 5 MANDATORY FIXES:
   - FIX 1: StructureOffset upload using bytemuck::cast_slice
   - FIX 2: Metrics buffer as alloc_zeros::<BatchMetricsOutput>(n_structures)
   - FIX 3: GT mask validation assert_eq!(gt.len(), n_residues)
   - FIX 4: Binary search: prefix[mid] <= tile_id -> return low - 1
   - FIX 5: Return real BatchStructureOutput (not empty vec)

4. Implemented full blueprint exactly as written.

## FINAL STATUS

- Blueprint saved to repo root
- Zero-deviation directive saved to repo root
- Layout: Global atom pool with indirection
- Binary search: prefix[mid] <= tile_id -> return low - 1
- Ready for 500-structure run
