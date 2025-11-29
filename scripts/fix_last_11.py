#!/usr/bin/env python3
"""Fix the last 11 files with load_ptx."""
import re
from pathlib import Path

base = Path("/mnt/c/Users/Predator/Desktop/PRISM")

files = [
    "crates/prism-geometry/src/sensor_layer.rs",
    "crates/prism-gpu/src/pimc.rs",
    "crates/prism-gpu/src/transfer_entropy.rs",
    "foundation/quantum/src/gpu_k_opt.rs",
    "crates/prism-gpu/src/tda.rs",
    "crates/prism-gpu/src/floyd_warshall.rs",
    "crates/prism-gpu/src/context.rs",
    "crates/prism-gpu/src/molecular.rs",
    "crates/prism-gpu/src/dendritic_whcr.rs",
    "crates/prism-gpu/src/dendritic_reservoir.rs",
    "foundation/prct-core/src/gpu_quantum_annealing.rs",
]

for rel_path in files:
    filepath = base / rel_path
    print(f"Processing: {rel_path}")

    text = filepath.read_text()
    orig = text

    # Show the current .load_ptx pattern
    matches = list(re.finditer(r'\.load_ptx\(', text))
    if matches:
        for m in matches:
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 200)
            snippet = text[start:end]
            print(f"  Found: ...{snippet}...")
    else:
        print(f"  No .load_ptx( found")

print("\nDone analyzing")
