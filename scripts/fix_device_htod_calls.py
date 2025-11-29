#!/usr/bin/env python3
"""
Fix remaining .device.htod_sync_copy() calls for cudarc 0.18.1.
Also fixes .device to .context where CudaContext is the correct type.
"""

import re
from pathlib import Path

def fix_file(file_path):
    """Fix htod_sync_copy patterns in a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Fix struct field: device: Arc<CudaDevice> → context: Arc<CudaContext>
    content = re.sub(
        r'(\s+)device: Arc<CudaDevice>,',
        r'\1context: Arc<CudaContext>,',
        content
    )

    # Fix imports if needed
    if 'CudaContext' not in content and 'Arc<CudaContext>' in content:
        content = re.sub(
            r'use cudarc::driver::\{([^}]+)\};',
            lambda m: f'use cudarc::driver::{{{m.group(1).replace("CudaDevice", "CudaDevice, CudaContext")}}};',
            content
        )

    # Fix field access: self.device → self.context (when appropriate)
    # This is tricky - we need context clues

    # Fix .device.htod_sync_copy multiline patterns
    # Pattern: .device\n<spaces>.htod_sync_copy
    content = re.sub(
        r'\.device\s*\n\s+\.htod_sync_copy\(',
        r'.context.fork_default_stream()?\n            .clone_htod(',
        content
    )

    # Fix .device.htod_sync_copy single-line
    content = re.sub(
        r'\.device\.htod_sync_copy\(',
        r'.context.fork_default_stream()?.clone_htod(',
        content
    )

    # Fix self.device.htod_sync_copy multiline
    content = re.sub(
        r'self\s*\.\s*device\s*\n\s+\.htod_sync_copy\(',
        r'self.context.fork_default_stream()?\n            .clone_htod(',
        content
    )

    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    base = Path('/mnt/c/Users/Predator/Desktop/PRISM')

    # Files that need fixing
    files_to_fix = [
        'crates/prism-gpu/src/dendritic_reservoir.rs',
        'crates/prism-gpu/src/tda.rs',
        'crates/prism-gpu/src/transfer_entropy.rs',
    ]

    count = 0
    for file_rel in files_to_fix:
        file_path = base / file_rel
        if file_path.exists():
            if fix_file(file_path):
                print(f"✓ Fixed: {file_rel}")
                count += 1

    print(f"\n{count} files modified")

if __name__ == '__main__':
    main()
