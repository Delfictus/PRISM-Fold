#!/usr/bin/env python3
"""
Fix cudarc 0.18.1 dtoh_sync_copy migration.
Replaces:
  - self.stream.dtoh_sync_copy(&buffer) → self.stream.clone_dtoh(&buffer)
  - self.device.dtoh_sync_copy(&buffer) → stream.clone_dtoh(&buffer) (with stream creation)
  - stream.dtoh_sync_copy(&buffer) → stream.clone_dtoh(&buffer)
"""

import re
import sys
from pathlib import Path

def fix_stream_dtoh(content):
    """Fix dtoh_sync_copy calls that already use a stream."""
    # Pattern: self.stream.dtoh_sync_copy or stream.dtoh_sync_copy
    pattern = r'(self\.stream|stream)\.dtoh_sync_copy\('
    replacement = r'\1.clone_dtoh('
    return re.sub(pattern, replacement, content)

def fix_device_dtoh(content):
    """Fix dtoh_sync_copy calls that use device - needs stream creation."""
    # This is more complex - for now just replace the method name
    # The user will need to add stream creation manually if needed
    pattern = r'\.dtoh_sync_copy\('
    replacement = r'.clone_dtoh('
    return re.sub(pattern, replacement, content)

def process_file(filepath):
    """Process a single file."""
    print(f"Processing {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # First fix stream-based calls
    content = fix_stream_dtoh(content)

    # Then fix device-based calls (simpler replacement)
    content = fix_device_dtoh(content)

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Updated {filepath}")
        return True
    else:
        print(f"  - No changes needed in {filepath}")
        return False

def main():
    # List of files to process
    files = [
        "foundation/prct-core/src/gpu_quantum_annealing.rs",
        "crates/prism-gpu/src/tda.rs",
        "crates/prism-gpu/src/transfer_entropy.rs",
        "crates/prism-gpu/src/pimc.rs",
        "crates/prism-gpu/src/molecular.rs",
        "crates/prism-gpu/src/floyd_warshall.rs",
        "crates/prism-gpu/src/dendritic_reservoir.rs",
        "crates/prism-gpu/src/dendritic_whcr.rs",
        "foundation/quantum/src/gpu_tsp.rs",
        "foundation/quantum/src/gpu_coloring.rs",
        "foundation/prct-core/src/gpu_transfer_entropy.rs",
        "foundation/prct-core/src/gpu_thermodynamic.rs",
        "foundation/prct-core/src/gpu_quantum.rs",
        "foundation/prct-core/src/gpu_kuramoto.rs",
        "foundation/prct-core/src/gpu_active_inference.rs",
        "foundation/neuromorphic/src/gpu_reservoir.rs",
        "foundation/neuromorphic/src/gpu_memory.rs",
        "crates/prism-gpu/src/whcr.rs",
        "crates/prism-gpu/src/quantum.rs",
        "crates/prism-gpu/src/lbs.rs",
    ]

    base_path = Path("/mnt/c/Users/Predator/Desktop/PRISM")
    updated = 0

    for file_rel in files:
        filepath = base_path / file_rel
        if filepath.exists():
            if process_file(filepath):
                updated += 1
        else:
            print(f"  ✗ File not found: {filepath}")

    print(f"\n✓ Updated {updated} files")

if __name__ == "__main__":
    main()
