# PRISM-LBS Validation Suite

## Overview

This validation suite tests PRISM-LBS pocket detection against gold-standard benchmarks to establish:

1. **Correctness** — Does it find known binding sites?
2. **Novelty** — Does it find sites that other tools miss?
3. **Value** — Is it faster, more accurate, or more useful?

---

## Benchmarks

### Tier 1: Basic Correctness (Must Pass)

| Test | Structure | Expected Result | Pass Criteria |
|------|-----------|-----------------|---------------|
| HIV-1 Protease | 4HVP | Active site (ASP25, ILE50) | Pocket with res 25, 50 detected |
| Trypsin | 3PTB | S1 pocket (ASP189, SER190) | Pocket with res 189, 190 detected |
| DHFR | 4DFR | Folate binding site | Top pocket contains active site |

```bash
# Run basic validation
./scripts/validate_basic.sh
```

### Tier 2: CryptoSite Benchmark (Novelty Test)

The CryptoSite benchmark tests detection of **cryptic binding sites** — pockets that are hidden in the unbound (apo) structure and only appear when a ligand binds.

**This is the key novelty test.** fpocket achieves ~50% detection on this benchmark. If PRISM beats that, you have a publishable result.

| Protein | PDB (apo) | Difficulty | Site Description |
|---------|-----------|------------|------------------|
| TEM-1 β-lactamase | 1JWP | Hard | Allosteric site 16Å from active site |
| Interleukin-2 | 1M47 | Hard | Groove between helices |
| Protein Kinase A | 1J3H | Medium | Myristate binding pocket |
| p38 MAP Kinase | 1K8K | Medium | DFG-out allosteric pocket |
| Bcl-xL | 1FJS | Hard | BH3 binding groove |
| HSP90 | 2FGU | Hard | ATP lid cryptic pocket |
| ... | ... | ... | 18 total test cases |

```bash
# Setup and run CryptoSite benchmark
./scripts/setup_cryptosite_benchmark.sh
cd benchmark/cryptosite
./run_benchmark.sh
```

**Interpreting Results:**

| PRISM Detection Rate | Meaning |
|---------------------|---------|
| > 65% | 🏆 Excellent — publishable result |
| 55-65% | ✅ Good — significant improvement over fpocket |
| 45-55% | ⚠️ Competitive — matches state-of-art |
| < 45% | ❌ Needs work — below baseline |

### Tier 3: Speed Benchmark

Test on large structures to validate GPU acceleration value.

```bash
# Download ribosome (huge structure)
wget https://files.rcsb.org/download/4V9D.pdb -O ribosome.pdb

# Time comparison
echo "=== fpocket ==="
time fpocket -f ribosome.pdb

echo "=== PRISM ==="
time cargo run --release -p prism-lbs -- ribosome.pdb
```

**Success criteria:** PRISM should be >5x faster on large structures.

---

## Running the Full Validation Suite

```bash
# 1. Build PRISM
cargo build --release -p prism-lbs

# 2. Run all validations
./scripts/run_full_validation.sh

# Expected output:
# ✅ Basic correctness: PASS
# ✅ CryptoSite: 67% (beats fpocket 52%)
# ✅ Speed: 8x faster on ribosome
```

---

## Expected Results for Key Structures

### HIV-1 Protease (4HVP)

```json
{
  "pocket_1": {
    "volume": "400-800 Å³",
    "residues_must_include": [25, 27, 49, 50, 80, 81, 82, 84],
    "druggability": ">0.6",
    "location": "catalytic cleft"
  }
}
```

### TEM-1 β-lactamase Cryptic Site (1JWP)

```json
{
  "cryptic_site": {
    "residues": [238, 240, 244, 276],
    "visible_in_apo": false,
    "visible_in_holo": true,
    "distance_from_active_site": "16 Å",
    "fpocket_detects": false,
    "prism_should_detect": true
  }
}
```

---

## What Makes a Result Publishable?

### Minimum Requirements

1. **CryptoSite detection rate > 60%** (fpocket baseline: ~50%)
2. **No false positives** on negative controls
3. **Reproducible** across multiple runs
4. **Validated** on held-out test set

### Strong Publication

1. **CryptoSite > 70%**
2. **Novel algorithmic contribution** (not just parameter tuning)
3. **Speed improvement** on large structures
4. **Case study** on pharmaceutically relevant target

### Target Journals

| Journal | Impact | Focus |
|---------|--------|-------|
| J. Chem. Inf. Model. | 5.6 | Methods + validation |
| Bioinformatics | 5.8 | Software tools |
| Nucleic Acids Research | 16.9 | Web servers/databases |
| Nature Methods | 47.9 | Major methodological advance |

---

## Troubleshooting

### "PRISM misses known binding sites"

Check:
1. Is the structure complete? (no missing residues)
2. Are HETATM records being parsed? (ligands/cofactors)
3. Is the pocket too shallow? (adjust depth threshold)

### "Too many false positives"

Check:
1. Volume filtering — reject pockets > 2000 Å³
2. Atom count filtering — reject pockets < 10 atoms
3. Surface vs buried — check enclosure ratio

### "CryptoSite performance worse than fpocket"

The cryptic sites are hard because they don't exist in the apo structure. PRISM's potential advantages:

1. **Multi-resolution analysis** — may detect subtle cavities
2. **Persistence scoring** — may identify stable sub-cavities
3. **Flexibility modeling** — may account for local motion

If PRISM isn't beating fpocket on cryptic sites, focus algorithm improvements on these areas.

---

## Files

```
scripts/
├── setup_cryptosite_benchmark.sh   # Download and setup CryptoSite
├── validate_basic.sh               # Basic correctness tests
└── run_full_validation.sh          # Complete validation suite

benchmark/
└── cryptosite/
    ├── ground_truth.csv            # Cryptic site definitions
    ├── structures/
    │   ├── apo/                    # Test inputs (unbound)
    │   └── holo/                   # Reference (bound)
    ├── results/
    │   ├── prism/                  # PRISM outputs
    │   └── fpocket/                # fpocket outputs
    ├── run_benchmark.sh            # Main benchmark runner
    └── analyze_results.py          # Detailed analysis
```
