#!/usr/bin/env python3
"""
Fix all stream.alloc_zeros to context.alloc_zeros across the codebase.
"""
import re
import sys
from pathlib import Path

def fix_file(filepath):
    """Fix alloc_zeros calls in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original = content

        # Pattern 1: self.stream.alloc_zeros -> self.context.alloc_zeros
        content = re.sub(
            r'self\.stream\.alloc_zeros',
            r'self.context.alloc_zeros',
            content
        )

        # Pattern 2: stream.alloc_zeros -> device.alloc_zeros (for gpu_tsp.rs where device is Arc<CudaContext>)
        # Only for foundation/quantum files
        if 'foundation/quantum' in str(filepath):
            content = re.sub(
                r'(?<!self\.)stream\.alloc_zeros',
                r'device.alloc_zeros',
                content
            )

        # Pattern 3: stream.alloc_zeros -> context.alloc_zeros (for other cases)
        # This catches remaining standalone stream.alloc_zeros calls
        content = re.sub(
            r'(?<!self\.)stream\.alloc_zeros',
            r'context.alloc_zeros',
            content
        )

        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed {filepath}")
            return True
        else:
            print(f"  No changes needed for {filepath}")
            return False

    except Exception as e:
        print(f"✗ Error fixing {filepath}: {e}", file=sys.stderr)
        return False

def main():
    root = Path("/mnt/c/Users/Predator/Desktop/PRISM")

    # Files to fix based on grep results
    files_to_fix = [
        # Foundation neuromorphic
        "foundation/neuromorphic/src/gpu_memory.rs",

        # Foundation quantum
        "foundation/quantum/src/gpu_tsp.rs",

        # Prism GPU
        "crates/prism-gpu/src/lbs.rs",
        "crates/prism-gpu/src/tda.rs",
        "crates/prism-gpu/src/quantum.rs",
        "crates/prism-gpu/src/cma_es.rs",
        "crates/prism-gpu/src/cma.rs",
        "crates/prism-gpu/src/dendritic_reservoir.rs",
        "crates/prism-gpu/src/active_inference.rs",
        "crates/prism-gpu/src/dendritic_whcr.rs",
        "crates/prism-gpu/src/molecular.rs",
        "crates/prism-gpu/src/pimc.rs",
        "crates/prism-gpu/src/whcr.rs",

        # Prism geometry
        "crates/prism-geometry/src/sensor_layer.rs",
    ]

    fixed_count = 0
    for file_path in files_to_fix:
        full_path = root / file_path
        if full_path.exists():
            if fix_file(full_path):
                fixed_count += 1
        else:
            print(f"⚠ File not found: {full_path}", file=sys.stderr)

    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} file(s)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
