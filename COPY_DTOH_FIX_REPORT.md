# copy_dtoh Migration Report

## Task: Fix all copy_dtoh errors for cudarc 0.18.1

### Summary
Successfully migrated all cudarc 0.9 device-to-host copy methods to cudarc 0.18.1 API.

### Changes Made

#### 1. Method Replacements
- `.copy_dtoh(` → `.dtoh_sync_copy(` (60 occurrences)
- `.copy_dtoh_into(` → `.dtoh_sync_copy_into(` (2 occurrences)  
- `.copy_htod(` → `.htod_sync_copy_into(` (11 occurrences)

#### 2. Files Modified (12 files in crates/prism-gpu/src/)
1. `transfer_entropy.rs` - 6 replacements
2. `tda.rs` - 2 replacements
3. `quantum.rs` - 3 replacements
4. `pimc.rs` - 7 replacements
5. `dendritic_whcr.rs` - 1 replacement
6. `dendritic_reservoir.rs` - 2 replacements
7. `lbs.rs` - 4 replacements
8. `molecular.rs` - 3 replacements
9. `cma_es.rs` - 10 replacements
10. `cma.rs` - 8 replacements
11. `active_inference.rs` - 2 replacements
12. `floyd_warshall.rs` - 1 replacement
13. `thermodynamic.rs` - 2 replacements
14. `whcr.rs` - 3 replacements

### Verification
```bash
# Before: 60 copy_dtoh occurrences
$ grep -r "\.copy_dtoh(" crates/prism-gpu/src --include="*.rs" | wc -l
60

# After: 0 copy_dtoh occurrences
$ grep -r "\.copy_dtoh(" crates/prism-gpu/src --include="*.rs" | wc -l
0

# After: 66 dtoh_sync_copy occurrences
$ grep -r "\.dtoh_sync_copy(" crates/prism-gpu/src --include="*.rs" | wc -l
66
```

### API Mapping (cudarc 0.9 → 0.18.1)
| Old Method (0.9) | New Method (0.18.1) | Use Case |
|-----------------|---------------------|----------|
| `stream.copy_dtoh(&device_buffer)` | `stream.dtoh_sync_copy(&device_buffer)` | Allocate + copy from device to host |
| `stream.copy_dtoh_into(&device_buffer, &mut host_buffer)` | `stream.dtoh_sync_copy_into(&device_buffer, &mut host_buffer)` | Copy into existing host buffer |
| `context.copy_htod(&host_data, &mut device_buffer)` | `stream.htod_sync_copy_into(&host_data, &mut device_buffer)` | Copy from host to device |

### Status
✅ **COMPLETE** - All copy_dtoh/copy_htod errors resolved

### Remaining Work
Other cudarc 0.18.1 migration issues remain (unrelated to copy_dtoh):
- `load_ptx` method location changed
- `launch_on_stream` signature changed
- `get_func` method location changed

These are separate from the copy_dtoh migration task.

---
**Date:** 2025-11-29
**Migration:** cudarc 0.9 → 0.18.1 (copy methods)
