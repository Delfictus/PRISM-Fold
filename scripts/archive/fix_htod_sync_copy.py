#!/usr/bin/env python3
"""
Fix htod_sync_copy calls for cudarc 0.18.1 migration.

In cudarc 0.18.1:
- CudaDevice::htod_sync_copy() → CudaStream::clone_htod()
- context.htod_sync_copy(&data) → stream.clone_htod(&data)
- self.context.htod_sync_copy(&data) → self.stream.clone_htod(&data)
- device.htod_sync_copy(&data) → stream.clone_htod(&data)
"""

import os
import re
import sys
from pathlib import Path

def fix_htod_sync_copy(file_path):
    """Replace htod_sync_copy with clone_htod on stream objects."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Pattern 1: context.htod_sync_copy → stream.clone_htod
    content = re.sub(
        r'\bcontext\.htod_sync_copy\(',
        r'stream.clone_htod(',
        content
    )

    # Pattern 2: self.context.htod_sync_copy → self.stream.clone_htod
    content = re.sub(
        r'\bself\.context\.htod_sync_copy\(',
        r'self.stream.clone_htod(',
        content
    )

    # Pattern 3: device.htod_sync_copy → stream.clone_htod
    content = re.sub(
        r'\bdevice\.htod_sync_copy\(',
        r'stream.clone_htod(',
        content
    )

    # Pattern 4: self.stream.htod_sync_copy → self.stream.clone_htod (already correct object)
    content = re.sub(
        r'\bself\.stream\.htod_sync_copy\(',
        r'self.stream.clone_htod(',
        content
    )

    # Pattern 5: stream.htod_sync_copy → stream.clone_htod
    content = re.sub(
        r'\bstream\.htod_sync_copy\(',
        r'stream.clone_htod(',
        content
    )

    # Pattern 6: Generic pattern for any variable.htod_sync_copy
    # This catches cases where the stream variable might have different names
    content = re.sub(
        r'\.htod_sync_copy\(',
        r'.clone_htod(',
        content
    )

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    # Find all Rust files with htod_sync_copy
    base_dir = Path('/mnt/c/Users/Predator/Desktop/PRISM')

    patterns = [
        'crates/prism-gpu/src/*.rs',
        'crates/prism-whcr/src/*.rs',
        'crates/prism-geometry/src/*.rs',
        'foundation/prct-core/src/*.rs',
        'foundation/prct-core/src/gpu/*.rs',
        'foundation/prct-core/src/fluxnet/*.rs',
        'foundation/neuromorphic/src/*.rs',
        'foundation/neuromorphic/benches/*.rs',
        'foundation/quantum/src/*.rs',
    ]

    files_modified = 0
    for pattern in patterns:
        for file_path in base_dir.glob(pattern):
            if file_path.is_file():
                # Check if file contains htod_sync_copy
                with open(file_path, 'r', encoding='utf-8') as f:
                    if 'htod_sync_copy' in f.read():
                        if fix_htod_sync_copy(file_path):
                            print(f"✓ Fixed: {file_path.relative_to(base_dir)}")
                            files_modified += 1

    print(f"\n{files_modified} files modified")
    return 0

if __name__ == '__main__':
    sys.exit(main())
