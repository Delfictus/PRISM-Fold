# Bug Fix: FluxNet RL Controller Phase Name Mismatch

## Problem Description

The prism-fluxnet RL controller had a phase name mismatch causing runtime panic during phase execution:

- **Q-tables initialized with keys**: `"Phase0"`, `"Phase1"`, `"Phase2"`, etc. (simplified names)
- **Phase controllers return names**: `"Phase0-DendriticReservoir"`, `"Phase1-ActiveInference"`, etc. (full names)
- **Result**: When orchestrator called `select_action(&state, "Phase0-DendriticReservoir")`, Q-table lookup failed → panic

## Root Cause

Inconsistent phase naming between:
1. `prism-fluxnet/src/core/controller.rs` - Q-table initialization
2. `prism-fluxnet/src/core/actions.rs` - Action selection methods
3. `prism-phases/src/phase*/controller.rs` - Phase name() implementations

## Files Modified

### 1. `/mnt/c/Users/Predator/Desktop/PRISM-v2/prism-fluxnet/src/core/controller.rs`

**Line 154-168**: Updated Q-table initialization to use full phase names:
```rust
// BEFORE (broken):
for phase in &[
    "Phase0", "Phase1", "Phase2", "Phase3", "Phase4", "Phase5", "Phase6", "Phase7",
] {
    phase_qtables.insert(phase.to_string(), vec![vec![0.0; NUM_ACTIONS]; num_states]);
}

// AFTER (fixed):
for phase in &[
    "Phase0-DendriticReservoir",
    "Phase1-ActiveInference",
    "Phase2-Thermodynamic",
    "Phase3-QuantumClassical",
    "Phase4-Geodesic",
    "Phase6-TDA",
    "Phase7-Ensemble",
] {
    phase_qtables.insert(phase.to_string(), vec![vec![0.0; NUM_ACTIONS]; num_states]);
}
```

**Line 404-424**: Updated `initialize_all_phases_from_curriculum()` method

**Test updates** (lines 440-560): Updated all test cases to use full phase names

### 2. `/mnt/c/Users/Predator/Desktop/PRISM-v2/prism-fluxnet/src/core/actions.rs`

**Line 58-71**: Updated `from_index()` method:
```rust
// BEFORE:
match phase {
    "Phase0" => DendriticAction::from_index(index).map(UniversalAction::Phase0),
    "Phase1" => ActiveInferenceAction::from_index(index - 8).map(UniversalAction::Phase1),
    ...
}

// AFTER:
match phase {
    "Phase0-DendriticReservoir" => DendriticAction::from_index(index).map(UniversalAction::Phase0),
    "Phase1-ActiveInference" => ActiveInferenceAction::from_index(index - 8).map(UniversalAction::Phase1),
    ...
}
```

**Line 74-106**: Updated `all_actions_for_phase()` method

**Test updates** (lines 510-522): Updated test cases to use full phase names

## Verification

### Unit Tests
All prism-fluxnet unit tests pass:
```bash
cargo test --package prism-fluxnet --lib
# Result: ok. 25 passed; 0 failed; 0 ignored
```

### Integration Test
Successfully executed full pipeline without panic:
```bash
./target/release/prism-cli --input benchmarks/dimacs/DSJC250.5.col --verbose
```

**Key log evidence**:
- `[INFO] Executing phase: Phase0-DendriticReservoir`
- `[DEBUG] Applying action NoOp for phase Phase0-DendriticReservoir`
- `[INFO] Phase Phase0-DendriticReservoir completed successfully`
- `[INFO] Executing phase: Phase1-ActiveInference`
- `[INFO] Executing phase: Phase2-Thermodynamic`
- `[DEBUG] Applying action Phase2(DecreaseTemperature) for phase Phase2-Thermodynamic`
- All 7 phases executed successfully

**No more "Phase Q-table not found" panic!**

## Phase Naming Convention (Canonical Reference)

The following phase names are now standardized across the codebase:

| Phase ID | Full Name (Canonical)         | Controller Type           |
|----------|-------------------------------|---------------------------|
| Phase 0  | `Phase0-DendriticReservoir`   | Phase0DendriticReservoir  |
| Phase 1  | `Phase1-ActiveInference`      | Phase1ActiveInference     |
| Phase 2  | `Phase2-Thermodynamic`        | Phase2Thermodynamic       |
| Phase 3  | `Phase3-QuantumClassical`     | Phase3Quantum             |
| Phase 4  | `Phase4-Geodesic`             | Phase4Geodesic            |
| Phase 5  | `Phase5-NetworkTopology`      | (alias for Phase4)        |
| Phase 6  | `Phase6-TDA`                  | Phase6TDA                 |
| Phase 7  | `Phase7-Ensemble`             | Phase7Ensemble            |

## Implementation Notes

1. **Phase 5 handling**: Phase 5 (NetworkTopology) is an alias for Phase 4 (Geodesic) and shares the same Q-table and actions.

2. **Consistency requirement**: Any future code that references phases must use the full canonical names listed above.

3. **Spec alignment**: This fix aligns with PRISM GPU Plan spec requirements for phase naming consistency.

4. **Backward compatibility**: No breaking changes to public APIs - this is purely an internal naming fix.

## Related Spec Sections

- §3.2: UniversalAction enum specification
- §3.3: UniversalRLController architecture
- §4.1-4.7: Individual phase specifications

## Status

**FIXED** - All tests passing, integration verified, production-ready.
