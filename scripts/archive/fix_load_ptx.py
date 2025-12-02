#!/usr/bin/env python3
"""
Automatically fix all load_ptx calls to use the cudarc 0.18.1 API.

The fix transforms:
  context.load_ptx(ptx, "module_name", &["kernel1", "kernel2"])?;
  stream.get_func("module_name", "kernel_name")

To:
  let module = context.load_module(ptx)?;
  module.get_fn("kernel_name")
"""

import re
import sys
from pathlib import Path


def fix_load_ptx_in_file(filepath):
    """Fix load_ptx calls in a single file."""
    print(f"Processing: {filepath}")

    content = filepath.read_text()
    original_content = content

    # Pattern 1: Find and replace load_ptx calls
    # Matches: context.load_ptx(ptx, "module", &["k1", "k2"])
    # or device.load_ptx(ptx, "module", &["k1"])
    load_ptx_pattern = r'(\s+)(context|device|stream)\s*\n\s*\.load_ptx\([^)]+\)[^;]*;'

    def replace_load_ptx(match):
        indent = match.group(1)
        obj = match.group(2)
        if obj == "stream":
            obj = "context"  # Streams don't have load_module either, use context
        return f'{indent}let module = {obj}.load_module(ptx)?;'

    content = re.sub(load_ptx_pattern, replace_load_ptx, content, flags=re.MULTILINE)

    # Pattern 2: Fix inline load_ptx calls
    content = re.sub(
        r'(context|device)\s*\.load_ptx\(([^,]+),\s*"[^"]+",\s*&\[[^\]]+\]\)',
        r'let module = \1.load_module(\2)',
        content
    )

    # Pattern 3: Replace stream.get_func("module", "kernel") with module.get_fn("kernel")
    content = re.sub(
        r'(stream|context|device)\.get_func\("([^"]+)",\s*"([^"]+)"\)',
        r'module.get_fn("\3")',
        content
    )

    if content != original_content:
        filepath.write_text(content)
        print(f"  ✓ Fixed {filepath}")
        return True
    else:
        print(f"  - No changes needed for {filepath}")
        return False


def main():
    # Files to fix based on the grep results
    files_to_fix = [
        "crates/prism-gpu/src/active_inference.rs",
        "crates/prism-gpu/src/cma.rs",
        "crates/prism-gpu/src/cma_es.rs",
        "crates/prism-gpu/src/context.rs",
        "crates/prism-gpu/src/dendritic_reservoir.rs",
        "crates/prism-gpu/src/dendritic_whcr.rs",
        "crates/prism-gpu/src/floyd_warshall.rs",
        "crates/prism-gpu/src/lbs.rs",
        "crates/prism-gpu/src/molecular.rs",
        "crates/prism-gpu/src/pimc.rs",
        "crates/prism-gpu/src/quantum.rs",
        "crates/prism-gpu/src/tda.rs",
        "crates/prism-gpu/src/thermodynamic.rs",
        "crates/prism-gpu/src/transfer_entropy.rs",
        "crates/prism-geometry/src/sensor_layer.rs",
        "foundation/prct-core/src/gpu_quantum.rs",
        "foundation/prct-core/src/gpu_thermodynamic.rs",
        "foundation/prct-core/src/gpu_thermodynamic_streams.rs",
        "foundation/prct-core/src/gpu_kuramoto.rs",
        "foundation/prct-core/src/gpu_transfer_entropy.rs",
        "foundation/prct-core/src/gpu_quantum_annealing.rs",
        "foundation/quantum/src/gpu_tsp.rs",
    ]

    base_dir = Path("/mnt/c/Users/Predator/Desktop/PRISM")
    fixed_count = 0

    for rel_path in files_to_fix:
        filepath = base_dir / rel_path
        if filepath.exists():
            if fix_load_ptx_in_file(filepath):
                fixed_count += 1
        else:
            print(f"  ✗ File not found: {filepath}")

    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} out of {len(files_to_fix)} files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
