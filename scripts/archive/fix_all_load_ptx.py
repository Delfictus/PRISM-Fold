#!/usr/bin/env python3
"""
Fix all load_ptx calls to use cudarc 0.18.1 load_module API.

This script handles multiple patterns:
1. Simple load_ptx: context.load_ptx(...) → let module = context.load_module(...)
2. get_func calls: stream.get_func("module", "kernel") → module.get_fn("kernel")
3. device.load_ptx → device.load_module
"""

import re
import sys
from pathlib import Path


def fix_file(filepath):
    """Fix a single Rust file."""
    print(f"Processing: {filepath.relative_to(Path.cwd())}")

    content = filepath.read_text()
    original = content

    # Step 1: Find load_ptx blocks and extract module name for later use
    # We'll store the module name mapping
    module_names = {}

    # Pattern to match load_ptx with module name
    load_ptx_pattern = re.compile(
        r'([ \t]*)((?:context|device|stream))\s*\n?\s*\.load_ptx\(\s*([^,]+),\s*"([^"]+)",\s*&\[[^\]]+\]\s*\)',
        re.MULTILINE
    )

    def replace_load_ptx(match):
        indent = match.group(1)
        obj_type = match.group(2)
        ptx_var = match.group(3).strip()
        module_name = match.group(4)

        # Map old module name to new variable (we'll use just 'module')
        module_names[module_name] = "module"

        # Use context if it was stream or device
        if obj_type == "stream":
            obj_type = "context"

        return f'{indent}let module = {obj_type}.load_module({ptx_var})'

    content = load_ptx_pattern.sub(replace_load_ptx, content)

    # Step 2: Replace .get_func("module_name", "kernel_name") with .get_fn("kernel_name")
    # This handles both stream.get_func and context.get_func
    get_func_pattern = re.compile(
        r'(stream|context|device)\.get_func\("([^"]+)",\s*"([^"]+)"\)'
    )

    def replace_get_func(match):
        old_module = match.group(2)
        kernel_name = match.group(3)
        return f'module.get_fn("{kernel_name}")'

    content = get_func_pattern.sub(replace_get_func, content)

    # Step 3: Handle multi-line load_ptx patterns
    multiline_load_ptx = re.compile(
        r'([ \t]*)((?:context|device|stream))\s*\n\s*\.load_ptx\(\s*\n([^)]+)\n\s*\)',
        re.MULTILINE | re.DOTALL
    )

    def replace_multiline(match):
        indent = match.group(1)
        obj_type = match.group(2)
        args = match.group(3)

        # Extract just the PTX variable (first argument)
        first_arg = args.split(',')[0].strip()

        if obj_type == "stream":
            obj_type = "context"

        return f'{indent}let module = {obj_type}.load_module({first_arg})'

    content = multiline_load_ptx.sub(replace_multiline, content)

    if content != original:
        filepath.write_text(content)
        print(f"  ✓ Fixed")
        return True
    else:
        print(f"  - No changes")
        return False


def main():
    base = Path("/mnt/c/Users/Predator/Desktop/PRISM")

    # Find all Rust files with load_ptx
    files = [
        "crates/prism-geometry/src/sensor_layer.rs",
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
        "crates/prism-gpu/src/tda.rs",
        "crates/prism-gpu/src/transfer_entropy.rs",
        "foundation/prct-core/src/gpu_active_inference.rs",
        "foundation/prct-core/src/gpu_kuramoto.rs",
        "foundation/prct-core/src/gpu_quantum.rs",
        "foundation/prct-core/src/gpu_quantum_annealing.rs",
        "foundation/prct-core/src/gpu_thermodynamic.rs",
        "foundation/prct-core/src/gpu_thermodynamic_streams.rs",
        "foundation/prct-core/src/gpu_transfer_entropy.rs",
        "foundation/quantum/src/gpu_coloring.rs",
        "foundation/quantum/src/gpu_k_opt.rs",
        "foundation/quantum/src/gpu_tsp.rs",
    ]

    fixed = 0
    for rel_path in files:
        filepath = base / rel_path
        if filepath.exists():
            if fix_file(filepath):
                fixed += 1
        else:
            print(f"✗ Not found: {rel_path}")

    print(f"\n{'='*60}")
    print(f"Fixed {fixed}/{len(files)} files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
