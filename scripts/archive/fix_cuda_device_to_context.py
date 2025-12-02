#!/usr/bin/env python3
"""
Fix CudaDevice → CudaContext migration for cudarc 0.18.1.
Handles:
1. Import statements
2. Struct field declarations
3. Field access patterns
4. htod_sync_copy → stream.clone_htod
"""

import re
from pathlib import Path

def fix_file(file_path):
    """Migrate CudaDevice to CudaContext in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # 1. Fix imports: CudaDevice → CudaContext
    content = re.sub(
        r'use cudarc::driver::\{([^}]*?)CudaDevice([^}]*?)\};',
        lambda m: f'use cudarc::driver::{{{m.group(1)}CudaContext{m.group(2)}}};',
        content
    )

    # 2. Fix struct field type: device: Arc<CudaDevice> → context: Arc<CudaContext>
    content = re.sub(
        r'(\s+)///\s*CUDA device handle\s*\n\s+device: Arc<CudaDevice>',
        r'\1/// CUDA context handle\n\1context: Arc<CudaContext>',
        content
    )

    # Also fix without comment
    content = re.sub(
        r'(\s+)device: Arc<CudaDevice>',
        r'\1context: Arc<CudaContext>',
        content
    )

    # 3. Fix field access in methods: self.device → self.context
    content = re.sub(
        r'\bself\.device\b',
        r'self.context',
        content
    )

    # 4. Fix .htod_sync_copy on device/context across multiple lines
    # This handles patterns like:
    #   self.context
    #       .htod_sync_copy(&data)
    # Should become:
    #   let stream = self.context.fork_default_stream()?;
    #   stream.clone_htod(&data)

    # For now, let's do a simpler inline fix
    # Pattern: .htod_sync_copy( becomes .fork_default_stream()?.clone_htod(
    # But this creates invalid code - we need stream stored separately

    # Better approach: just fix the method call, manual stream creation later
    content = re.sub(
        r'\.htod_sync_copy\(',
        r'.clone_htod(',
        content
    )

    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    base = Path('/mnt/c/Users/Predator/Desktop/PRISM')

    # Files that still use CudaDevice
    files_to_fix = [
        'crates/prism-gpu/src/dendritic_reservoir.rs',
        'crates/prism-gpu/src/tda.rs',
        'crates/prism-gpu/src/floyd_warshall.rs',
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
