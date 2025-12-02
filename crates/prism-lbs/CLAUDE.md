# CLAUDE.md - PRISM-LBS Development Agent Configuration

> **Purpose**: This file configures Claude Code to act as a specialized ligand binding site prediction expert when working on the PRISM-LBS module.

## Agent Activation

When working on files in `crates/prism-lbs/`, Claude should:
1. Read the full agent specification at `docs/agents/PRISM_LBS_AGENT.md`
2. Follow all performance contracts strictly
3. Run validation tests before committing

## Quick Reference

### The Bug We're Fixing

```
CURRENT OUTPUT (BROKEN):
  Pocket 1: 1 atom, 60 Å³    ← Single atom is not a pocket
  Pocket 5: 135 residues, 75,868 Å³  ← That's the whole protein

EXPECTED OUTPUT (CORRECT):
  Pocket 1: 30+ atoms, 450-600 Å³  ← The actual active site
  All pockets: 50-5000 Å³, 5-500 atoms
```

### Validation Command

```bash
# This MUST pass before any PR
cargo test -p prism-lbs -- --include-ignored
```

### Critical Files

| File | Purpose | Status |
|------|---------|--------|
| `pocket/alpha_sphere.rs` | Cavity detection | REWRITE |
| `pocket/clustering.rs` | DBSCAN grouping | REWRITE |
| `pocket/detector.rs` | Orchestration | REWRITE |
| `pocket/scoring.rs` | Druggability | REFACTOR |
| `constants.rs` | Physical parameters | CREATE |
| `types.rs` | Data structures | REFACTOR |

### Performance Contracts

```
VOLUME BOUNDS:     50 Å³ ≤ V ≤ 5000 Å³
ATOM BOUNDS:       5 ≤ atoms ≤ 500
DRUGGABILITY:      0.0 ≤ score ≤ 1.0
TIME (5k atoms):   < 2 seconds
MEMORY:            < 4× structure size
```

### The Litmus Test

HIV-1 Protease (PDB: 4HVP) must produce:
- Active site detected (residues 25, 50, 82)
- Volume: 400-700 Å³
- Druggability: > 0.7
- Classification: Druggable or HighlyDruggable

If this test fails, the implementation is wrong.

## Code Patterns

### Error Handling
```rust
use anyhow::{Result, Context, bail, ensure};

pub fn function() -> Result<T> {
    ensure!(condition, "Error message with {}", context);
    operation().context("What failed")?;
    Ok(result)
}
```

### Validation
```rust
impl Pocket {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.volume >= 50.0, "Volume too small");
        ensure!(self.volume <= 5000.0, "Volume too large");
        ensure!(self.atom_indices.len() >= 5, "Too few atoms");
        Ok(())
    }
}
```

### Logging
```rust
log::info!("Starting pocket detection for {} atoms", atoms.len());
log::debug!("Generated {} alpha spheres", spheres.len());
log::warn!("Pocket {} has unusual volume: {}", id, volume);
```

## Dependencies

```toml
# Use EXACTLY these versions
cudarc = "0.18.1"
nalgebra = "0.32"
rayon = "1.7"
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
rand = "0.8"
log = "0.4"
```

## Testing Strategy

1. **Unit tests**: Each module tests its own functions
2. **Integration tests**: `tests/integration/hiv1_protease.rs`
3. **Property tests**: Volume always in bounds, scores in [0,1]
4. **Benchmark tests**: Performance contracts verified

## Do NOT

- Return single-atom pockets
- Return pockets > 5000 Å³
- Skip validation before returning
- Use `unwrap()` in production code
- Ignore the HIV-1 protease test

## Do

- Log all major operations
- Validate all outputs
- Test incrementally
- Check bounds explicitly
- Handle edge cases (empty input, tiny structures)

---

For the complete implementation guide with code, see:
`docs/agents/PRISM_LBS_AGENT.md`
