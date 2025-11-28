# 🔒 PRISM Checkpoint Locking System

## Overview

**Checkpoint Locking** is a critical new feature that prevents downstream phases from expanding the color count once a **zero-conflict solution** is found.

## Problem Solved

Previously, when a phase produced a great solution (e.g., 20 colors, 0 conflicts), downstream phases would sometimes expand it unnecessarily (e.g., Phase 3 expanding 20→23 colors).

**With Checkpoint Locking:**
- Once ANY phase produces 0 conflicts, that color count is **locked as a checkpoint**
- Subsequent phases can ONLY accept solutions that:
  - Have **fewer colors** (with 0 conflicts), OR
  - Have **same colors** with 0 conflicts
- Phases that produce **more colors** or **any conflicts** are **rejected**

## How It Works

### 1. Checkpoint Creation (Automatic)
When a phase produces a solution with **0 conflicts**, the checkpoint is automatically locked:

```
Phase 2 produces: 20 colors, 0 conflicts
  ↓
🔒 CHECKPOINT LOCKED: 20 colors, 0 conflicts
  ↓
Subsequent phases cannot expand beyond 20 colors
```

### 2. Checkpoint Validation
Each phase checks the checkpoint before accepting a new solution:

```rust
// In PhaseContext::is_solution_allowed()
if let Some(checkpoint) = checkpoint_zero_conflicts {
    // REJECT: If new solution has conflicts
    if solution.conflicts > 0 { return false; }

    // REJECT: If new solution has MORE colors
    if solution.chromatic_number > checkpoint.chromatic_number { return false; }

    // ACCEPT: If fewer or same colors with 0 conflicts
    return true;
}
```

### 3. Logging
The system logs checkpoint events:

```
🔒 ZERO-CONFLICT CHECKPOINT LOCKED: 20 colors, 0 conflicts
CHECKPOINT IMPROVEMENT: 20 colors → 18 colors (0 conflicts locked)
CHECKPOINT LOCK: Rejecting 23 colors (checkpoint: 20 colors, 0 conflicts)
```

## API for Phase Controllers

Phases can check the checkpoint status:

```rust
// Check if a solution is allowed
if context.is_solution_allowed(&candidate_solution) {
    context.update_best_solution(candidate_solution);
} else {
    log::warn!("Solution violates checkpoint lock");
}

// Get checkpoint info
if let Some((colors, conflicts)) = context.get_checkpoint() {
    log::info!("Checkpoint: {} colors, {} conflicts", colors, conflicts);
}
```

## Implementation Details

### Files Modified
- `prism-core/src/traits.rs` - PhaseContext struct and checkpoint logic

### New PhaseContext Fields
```rust
pub struct PhaseContext {
    // Existing fields...

    /// CHECKPOINT: Best solution with ZERO conflicts (locked once achieved)
    /// Prevents downstream phases from expanding colors unnecessarily
    pub checkpoint_zero_conflicts: Option<ColoringSolution>,
}
```

### New Methods
- `update_best_solution()` - Updated with checkpoint locking logic
- `is_solution_allowed(&solution)` - Check if solution respects checkpoint
- `get_checkpoint()` - Get current checkpoint color count and conflicts

## Example Scenario

### Without Checkpoint Locking (Previous Behavior)
```
Phase 2: 13 colors, 103 conflicts → Repair → 20 colors, 0 conflicts
Phase 3: 17 colors, 72 conflicts → Repair → 23 colors, 0 conflicts ❌ EXPANSION!
Phase 4: 24 colors, 0 conflicts ❌ MORE EXPANSION!
Final:   24 colors, 0 conflicts (lost the good 20-color solution)
```

### With Checkpoint Locking (New Behavior)
```
Phase 2: 13 colors, 103 conflicts → Repair → 20 colors, 0 conflicts
         🔒 CHECKPOINT LOCKED: 20 colors, 0 conflicts
Phase 3: 17 colors, 72 conflicts → Repair → 23 colors, 0 conflicts
         ❌ REJECTED (violates checkpoint: 23 > 20)
         ✓ Kept at: 20 colors, 0 conflicts
Phase 4: Produces 24 colors, 0 conflicts
         ❌ REJECTED (violates checkpoint: 24 > 20)
         ✓ Kept at: 20 colors, 0 conflicts
Final:   20 colors, 0 conflicts ✅ CHECKPOINT PRESERVED!
```

## Configuration

The checkpoint system is **automatic and always enabled**. No configuration needed!

However, phases can be modified to use `is_solution_allowed()` for stricter enforcement:

```rust
// In phase code:
if solution.conflicts == 0 && !context.is_solution_allowed(&solution) {
    log::warn!("Solution locked by checkpoint, skipping");
    return PhaseOutcome::success(); // Keep checkpoint
}
```

## Behavior When Checkpoint Locked

When a checkpoint exists with 0 conflicts:

| Scenario | Acceptance | Action |
|----------|-----------|--------|
| New solution: 17 colors, 0 conflicts (checkpoint: 20 colors) | ✅ YES | Improves checkpoint to 17 colors |
| New solution: 20 colors, 0 conflicts (checkpoint: 20 colors) | ✅ YES | Maintains checkpoint (no update) |
| New solution: 22 colors, 0 conflicts (checkpoint: 20 colors) | ❌ NO  | Rejects - keeps 20 colors |
| New solution: 20 colors, 1 conflict (checkpoint: 20 colors) | ❌ NO  | Rejects - keeps 0 conflicts |
| New solution: 18 colors, 3 conflicts (checkpoint: 20 colors) | ❌ NO  | Rejects - keeps 0 conflicts |

## Performance Impact

- **Minimal overhead**: Simple O(1) comparisons
- **Memory**: One additional optional `ColoringSolution` stored per context
- **Logic**: ~50 lines of code in core traits

## Future Enhancements

### Phase-Specific Checkpoints
```rust
// Lock best solution for each phase separately
context.set_phase_checkpoint("Phase2", solution);
context.get_phase_checkpoint("Phase2");
```

### Flexible Checkpoint Policies
```rust
// Allow specific phases to ignore checkpoint
context.bypass_checkpoint_for_phase("PhaseM-MEC");

// Set color expansion threshold
context.set_max_expansion_colors(25); // Allow up to 25 colors max
```

### Checkpoint Reporting
```rust
// Export checkpoint state for analysis
context.checkpoint_violations // Counter of rejected solutions
context.checkpoint_improvements // How many times checkpoint improved
```

## CHAMPION_20_COLORS Config

The `configs/CHAMPION_20_COLORS.toml` configuration was specifically tuned to achieve the checkpoint-locked 20-color solution.

**Key Parameters:**
- Phase 2: Ultra-fine temperature schedule (72 temps, 24000 steps/temp)
- Memetic: Massive population (200), extreme mutations (0.40), deep search (75000 depth)
- Phase 3: Disabled via ensemble weighting (or skip entirely)

**Result:** 20 colors, 0 conflicts achieved in Attempt 10 and maintained via checkpoint

## Testing Checkpoint Lock

Run checkpoint test with logging:

```bash
timeout 300 ./target/release/prism-cli \
    --config configs/CHAMPION_20_COLORS.toml \
    --input benchmarks/dimacs/DSJC125.5.col \
    --attempts 2 2>&1 | grep "CHECKPOINT\|ZERO-CONFLICT"
```

Expected output:
```
🔒 ZERO-CONFLICT CHECKPOINT LOCKED: 20 colors, 0 conflicts
CHECKPOINT IMPROVEMENT: 20 colors → 18 colors (0 conflicts locked)
CHECKPOINT LOCK: Rejecting 23 colors (checkpoint: 20 colors, 0 conflicts)
```

## See Also

- `configs/CHAMPION_20_COLORS.toml` - Configuration that achieves 20-color solution
- `CHAMPION_20_COLORS.toml` - Documentation of champion config parameters
- `prism-core/src/traits.rs` - PhaseContext implementation with checkpoint logic

---

**Date:** 2025-11-23
**Status:** Implemented and Tested
**Impact:** Prevents color expansion beyond checkpoint-locked solution
