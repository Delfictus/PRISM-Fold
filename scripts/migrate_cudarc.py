#!/usr/bin/env python3
"""
PRISM cudarc 0.11 → 0.18.1 Migration Script

Systematically migrates all GPU files from cudarc 0.11 API to 0.18.1 API.

Key changes:
1. CudaDevice → CudaContext
2. Remove LaunchAsync import
3. device.htod_sync_copy → stream.clone_htod
4. device.dtoh_sync_copy → stream.copy_dtoh
5. device.htod_copy_into → stream.copy_htod
6. device.alloc_zeros → stream.alloc_zeros
7. device.memset_zeros → stream.memset_zeros
8. func.clone().launch(cfg, params) → stream.launch(&func, cfg, params)
9. device.synchronize → stream.synchronize
10. Add stream: CudaStream field to structs
11. Add stream = context.default_stream() in constructors
"""

import os
import re
from pathlib import Path

GPU_SRC_DIR = Path("/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-gpu/src")

# Files already migrated
MIGRATED_FILES = ["context.rs", "whcr.rs"]

# Files to migrate
FILES_TO_MIGRATE = [
    "thermodynamic.rs",
    "aatgs.rs",
    "active_inference.rs",
    "cma.rs",
    "cma_es.rs",
    "dendritic_reservoir.rs",
    "dendritic_whcr.rs",
    "floyd_warshall.rs",
    "lbs.rs",
    "molecular.rs",
    "multi_device_pool.rs",
    "pimc.rs",
    "quantum.rs",
    "stream_integration.rs",
    "stream_manager.rs",
    "tda.rs",
    "transfer_entropy.rs",
]

def migrate_file(filepath):
    """Migrate a single file from cudarc 0.11 to 0.18.1"""
    print(f"Migrating {filepath.name}...")

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # 1. Update imports
    content = re.sub(
        r'use cudarc::driver::\{([^}]*?)CudaDevice([^}]*?)\}',
        lambda m: f'use cudarc::driver::{{{m.group(1)}CudaContext{m.group(2)}}}',
        content
    )

    # Remove LaunchAsync import
    content = re.sub(
        r',\s*LaunchAsync\s*,',
        ', ',
        content
    )
    content = re.sub(
        r',\s*LaunchAsync\s*}',
        '}',
        content
    )
    content = re.sub(
        r'{\s*LaunchAsync\s*,',
        '{',
        content
    )

    # Add CudaStream import if not present
    if 'CudaStream' not in content and 'CudaContext' in content:
        content = re.sub(
            r'(use cudarc::driver::\{[^}]*?)(CudaContext[^}]*?)\}',
            r'\1\2, CudaStream}',
            content
        )

    # 2. Update struct fields: Arc<CudaDevice> → Arc<CudaContext>
    content = re.sub(
        r'device:\s*Arc<CudaDevice>',
        'context: Arc<CudaContext>',
        content
    )

    # 3. Update function parameters: Arc<CudaDevice> → Arc<CudaContext>
    content = re.sub(
        r'device:\s*Arc<CudaDevice>',
        'context: Arc<CudaContext>',
        content
    )

    # 4. CudaDevice::new → CudaContext::new
    content = re.sub(
        r'CudaDevice::new\(',
        'CudaContext::new(',
        content
    )

    # 5. Memory operations
    content = re.sub(
        r'\.htod_sync_copy\(',
        '.clone_htod(',
        content
    )
    content = re.sub(
        r'\.dtoh_sync_copy\(',
        '.copy_dtoh(',
        content
    )
    content = re.sub(
        r'\.htod_copy_into\(',
        '.copy_htod(',
        content
    )

    # 6. Kernel launches - this is tricky, needs context
    # Pattern: func.clone().launch(cfg, params)
    # Replace with: stream.launch(&func, cfg, params)
    content = re.sub(
        r'(\w+)\.clone\(\)\.launch\(([^,]+),\s*([^)]+)\)',
        r'self.stream.launch(&self.\1, \2, \3)',
        content
    )

    # 7. Device references to context references
    content = re.sub(
        r'self\.device\.',
        'self.stream.',
        content
    )
    content = re.sub(
        r'&self\.device\b',
        '&self.context',
        content
    )

    # 8. Update comments and documentation
    content = re.sub(
        r'CudaDevice',
        'CudaContext',
        content
    )
    content = re.sub(
        r'CUDA device',
        'CUDA context',
        content,
        flags=re.IGNORECASE
    )

    # Only write if content changed
    if content != original_content:
        # Create backup
        backup_path = filepath.with_suffix('.rs.bak')
        with open(backup_path, 'w') as f:
            f.write(original_content)
        print(f"  Created backup: {backup_path.name}")

        # Write migrated content
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Migrated {filepath.name}")
        return True
    else:
        print(f"  - No changes needed for {filepath.name}")
        return False

def main():
    """Main migration entry point"""
    print("="*70)
    print("PRISM cudarc 0.11 → 0.18.1 Migration")
    print("="*70)
    print()

    migrated_count = 0
    skipped_count = 0

    for filename in FILES_TO_MIGRATE:
        filepath = GPU_SRC_DIR / filename

        if not filepath.exists():
            print(f"⚠ File not found: {filename}")
            skipped_count += 1
            continue

        if filename in MIGRATED_FILES:
            print(f"✓ Already migrated: {filename}")
            skipped_count += 1
            continue

        try:
            if migrate_file(filepath):
                migrated_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"✗ Error migrating {filename}: {e}")
            skipped_count += 1

    print()
    print("="*70)
    print(f"Migration complete: {migrated_count} files migrated, {skipped_count} skipped")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Review changes manually (especially kernel launches)")
    print("2. Add 'stream: CudaStream' field to struct definitions")
    print("3. Add 'stream = context.default_stream()' in constructors")
    print("4. cargo check --features cuda")
    print("5. Fix any remaining compilation errors")

if __name__ == "__main__":
    main()
