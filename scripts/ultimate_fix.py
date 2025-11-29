#!/usr/bin/env python3
"""Ultimate fix for ALL remaining load_ptx patterns."""
import re
from pathlib import Path

def fix_multiline_load_ptx(text):
    """Fix multiline load_ptx patterns."""

    # Pattern: (context|device|stream).load_ptx(arg1, "name", &[...])
    # This pattern can span multiple lines
    # We need to match greedily until we find the closing )

    pattern = re.compile(
        r'((?:context|device|stream))\s*\n?\s*\.load_ptx\s*\(\s*'
        r'([^,]+?),\s*'               # First arg (PTX)
        r'"[^"]*",\s*'                # Module name
        r'&\[[^\]]*?\]'               # Kernel list
        r'\s*\)',
        re.MULTILINE | re.DOTALL
    )

    def replace(match):
        obj = match.group(1)
        ptx_arg = match.group(2).strip()

        # Use context for stream/device, otherwise use the object itself
        if obj in ('stream', 'device'):
            obj = 'context'

        return f'let _module = {obj}.load_module({ptx_arg})'

    return pattern.sub(replace, text)

def fix_get_func(text):
    """Fix get_func calls to use module.get_fn."""
    # Pattern: object.get_func("module", "kernel") → module.get_fn("kernel")
    text = re.sub(
        r'\b(context|device|stream)\.get_func\s*\(\s*"[^"]*",\s*"([^"]*)"\s*\)',
        r'module.get_fn("\2")',
        text
    )
    return text

def main():
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

    fixed = 0
    for rel_path in files:
        filepath = base / rel_path
        print(f"Processing: {rel_path}")

        text = filepath.read_text()
        orig = text

        # Apply fixes
        text = fix_multiline_load_ptx(text)
        text = fix_get_func(text)

        if text != orig:
            filepath.write_text(text)
            print(f"  ✓ FIXED")
            fixed += 1
        else:
            print(f"  - No changes")

    print(f"\n{'='*60}")
    print(f"Fixed {fixed}/{len(files)} files")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
