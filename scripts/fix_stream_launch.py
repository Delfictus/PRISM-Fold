#!/usr/bin/env python3
"""
Fix stream.launch() to kernel.launch_on_stream() for cudarc 0.18.1
"""

import re
import sys
from pathlib import Path

def fix_stream_launch(content):
    """
    Replace all occurrences of:
        self.stream.launch(&kernel, cfg, params)
    with:
        kernel.launch_on_stream(&self.stream, cfg, params)

    Also handles variations like:
        self.stream.launch(&self.kernel_name, cfg, params)
    with:
        self.kernel_name.launch_on_stream(&self.stream, cfg, params)
    """

    # Pattern 1: self.stream.launch(&kernel,
    pattern1 = re.compile(
        r'(\s+)self\.stream\.launch\(&kernel,',
        re.MULTILINE
    )
    content = pattern1.sub(r'\1kernel.launch_on_stream(&self.stream,', content)

    # Pattern 2: self.stream.launch(&self.kernel_name,
    pattern2 = re.compile(
        r'(\s+)self\.stream\.launch\(&(self\.\w+),',
        re.MULTILINE
    )
    content = pattern2.sub(r'\1\2.launch_on_stream(&self.stream,', content)

    # Pattern 3: stream.launch(&kernel_func, (for local stream variables)
    pattern3 = re.compile(
        r'(\s+)stream\.launch\(&(\w+),',
        re.MULTILINE
    )
    content = pattern3.sub(r'\1\2.launch_on_stream(&stream,', content)

    return content

def process_file(filepath):
    """Process a single file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    content = fix_stream_launch(content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    # Find all Rust files in relevant directories
    base_path = Path(__file__).parent.parent

    search_paths = [
        base_path / "crates" / "prism-gpu" / "src",
        base_path / "foundation" / "neuromorphic" / "src",
        base_path / "foundation" / "prct-core" / "src",
    ]

    files_fixed = 0

    for search_path in search_paths:
        if not search_path.exists():
            continue

        for rs_file in search_path.rglob("*.rs"):
            # Skip backup files
            if '.bak' in str(rs_file):
                continue

            if process_file(rs_file):
                print(f"✓ Fixed: {rs_file}")
                files_fixed += 1

    print(f"\n{'='*60}")
    print(f"Total files fixed: {files_fixed}")
    print(f"{'='*60}")

    return 0 if files_fixed > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
