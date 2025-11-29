#!/usr/bin/env python3
"""
Fix .context.clone_htod calls for cudarc 0.18.1 migration.
Handles multi-line patterns where context appears on one line and clone_htod on next.
"""

import os
import re
import sys
from pathlib import Path

def fix_context_clone_htod(file_path):
    """Replace .context.clone_htod with .stream.clone_htod, handling multiline."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Multi-line pattern: match across newlines with whitespace
    # Pattern: .context\n<whitespace>.clone_htod
    content = re.sub(
        r'\.context\s*\n\s*\.clone_htod',
        r'.stream\n            .clone_htod',
        content
    )

    # Also handle single-line patterns
    content = re.sub(
        r'\.context\.clone_htod',
        r'.stream.clone_htod',
        content
    )

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    # Find all Rust files with .context followed by .clone_htod
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
                # Check if file contains .context before clone_htod
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '.context' in content and '.clone_htod' in content:
                        if fix_context_clone_htod(file_path):
                            print(f"✓ Fixed: {file_path.relative_to(base_dir)}")
                            files_modified += 1

    print(f"\n{files_modified} files modified")
    return 0

if __name__ == '__main__':
    sys.exit(main())
