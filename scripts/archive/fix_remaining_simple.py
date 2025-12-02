#!/usr/bin/env python3
"""Simple fix for remaining load_ptx calls."""

import re
from pathlib import Path

def fix_file(path):
    text = path.read_text()
    orig = text

    # Pattern 1: stream.load_ptx(...) → let module = context.load_module(...)
    text = re.sub(
        r'stream\s*\n\s*\.load_ptx\(\s*\n\s*([^,]+),\s*"[^"]+",\s*&\[[^\]]+\]\s*\n\s*\)',
        r'let module = context.load_module(\1)',
        text,
        flags=re.MULTILINE | re.DOTALL
    )

    # Pattern 2: device.load_ptx(...)  → let module = device.load_module(...)
    text = re.sub(
        r'device\s*\n\s*\.load_ptx\(\s*\n\s*([^,]+),\s*"[^"]+",\s*&\[[^\]]+\]\s*\n\s*\)',
        r'let module = device.load_module(\1)',
        text,
        flags=re.MULTILINE | re.DOTALL
    )

    # Pattern 3: stream.load_ptx on single line
    text = re.sub(
        r'stream\.load_ptx\(([^,]+),\s*"[^"]+",\s*&\[[^\]]+\]\)',
        r'let module = context.load_module(\1)',
        text
    )

    # Pattern 4: device.load_ptx on single line
    text = re.sub(
        r'device\.load_ptx\(([^,]+),\s*"[^"]+",\s*&\[[^\]]+\]\)',
        r'let module = device.load_module(\1)',
        text
    )

    # Pattern 5: stream.get_func("module", "kernel") → module.get_fn("kernel")
    text = re.sub(
        r'stream\.get_func\("([^"]+)",\s*"([^"]+)"\)',
        r'module.get_fn("\2")',
        text
    )

    # Pattern 6: device.get_func("module", "kernel") → module.get_fn("kernel")
    text = re.sub(
        r'device\.get_func\("([^"]+)",\s*"([^"]+)"\)',
        r'module.get_fn("\2")',
        text
    )

    # Pattern 7: context.get_func("module", "kernel") → module.get_fn("kernel")
    text = re.sub(
        r'context\.get_func\("([^"]+)",\s*"([^"]+)"\)',
        r'module.get_fn("\2")',
        text
    )

    if text != orig:
        path.write_text(text)
        return True
    return False

files = [
    "crates/prism-geometry/src/sensor_layer.rs",
    "crates/prism-gpu/src/cma_es.rs",
    "crates/prism-gpu/src/context.rs",
    "crates/prism-gpu/src/dendritic_reservoir.rs",
    "crates/prism-gpu/src/dendritic_whcr.rs",
    "crates/prism-gpu/src/floyd_warshall.rs",
    "crates/prism-gpu/src/molecular.rs",
    "crates/prism-gpu/src/pimc.rs",
    "crates/prism-gpu/src/tda.rs",
    "crates/prism-gpu/src/transfer_entropy.rs",
    "foundation/prct-core/src/gpu_quantum_annealing.rs",
    "foundation/quantum/src/gpu_k_opt.rs",
]

base = Path("/mnt/c/Users/Predator/Desktop/PRISM")
fixed = 0

for rel in files:
    p = base / rel
    if p.exists():
        if fix_file(p):
            print(f"✓ {rel}")
            fixed += 1
        else:
            print(f"- {rel}")
    else:
        print(f"✗ {rel} NOT FOUND")

print(f"\n{fixed}/{len(files)} files fixed")
