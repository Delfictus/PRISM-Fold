#!/usr/bin/env python3
"""
Systematic fix for cudarc 0.18.1 memory operations API changes.

Key changes:
1. Memory ops (alloc_zeros, clone_htod, etc.) move from CudaContext to CudaStream
2. Need to call .default_stream() first
3. htod_sync_copy → memcpy_htod or clone_htod
4. dtoh_sync_copy → memcpy_dtoh or clone_dtoh
"""

import re
import sys
from pathlib import Path

def fix_memory_operations(content: str, filename: str) -> str:
    """Fix memory operation patterns in a file."""

    # Pattern 1: self.context.alloc_zeros → stream.alloc_zeros
    # Need to ensure stream variable exists
    content = re.sub(
        r'self\.context\.alloc_zeros',
        r'stream.alloc_zeros',
        content
    )

    # Pattern 2: self.device.alloc_zeros → stream.alloc_zeros
    content = re.sub(
        r'self\.device\.alloc_zeros',
        r'stream.alloc_zeros',
        content
    )

    # Pattern 3: device.alloc_zeros → stream.alloc_zeros (local variable)
    content = re.sub(
        r'(\s+)device\.alloc_zeros',
        r'\1stream.alloc_zeros',
        content
    )

    # Pattern 4: self.context.clone_htod → stream.clone_htod
    content = re.sub(
        r'self\.context\.clone_htod',
        r'stream.clone_htod',
        content
    )

    # Pattern 5: self.device.clone_htod → stream.clone_htod
    content = re.sub(
        r'self\.device\.clone_htod',
        r'stream.clone_htod',
        content
    )

    # Pattern 6: device.clone_htod → stream.clone_htod (local variable)
    content = re.sub(
        r'(\s+)device\.clone_htod',
        r'\1stream.clone_htod',
        content
    )

    # Pattern 7: self.context.htod_sync_copy_into → stream.memcpy_htod
    content = re.sub(
        r'self\.context\.htod_sync_copy_into\(([^,]+),\s*([^)]+)\)',
        r'stream.memcpy_htod(\1, \2)',
        content
    )

    # Pattern 8: self.device.htod_sync_copy_into → stream.memcpy_htod
    content = re.sub(
        r'self\.device\.htod_sync_copy_into\(([^,]+),\s*([^)]+)\)',
        r'stream.memcpy_htod(\1, \2)',
        content
    )

    # Pattern 9: device.htod_sync_copy_into → stream.memcpy_htod
    content = re.sub(
        r'(\s+)device\.htod_sync_copy_into\(([^,]+),\s*([^)]+)\)',
        r'\1stream.memcpy_htod(\2, \3)',
        content
    )

    # Pattern 10: self.context.dtoh_sync_copy → stream.memcpy_dtoh
    content = re.sub(
        r'self\.context\.dtoh_sync_copy\(&([^)]+)\)',
        r'stream.memcpy_dtoh(&\1)',
        content
    )

    # Pattern 11: self.device.dtoh_sync_copy → stream.memcpy_dtoh
    content = re.sub(
        r'self\.device\.dtoh_sync_copy\(&([^)]+)\)',
        r'stream.memcpy_dtoh(&\1)',
        content
    )

    # Pattern 12: device.dtoh_sync_copy → stream.memcpy_dtoh
    content = re.sub(
        r'(\s+)device\.dtoh_sync_copy\(&([^)]+)\)',
        r'\1stream.memcpy_dtoh(&\2)',
        content
    )

    # Pattern 13: self.context.dtoh_sync_copy_into → stream.memcpy_dtoh_into
    content = re.sub(
        r'self\.context\.dtoh_sync_copy_into\(([^,]+),\s*([^)]+)\)',
        r'stream.memcpy_dtoh_into(\1, \2)',
        content
    )

    # Pattern 14: self.device.dtoh_sync_copy_into → stream.memcpy_dtoh_into
    content = re.sub(
        r'self\.device\.dtoh_sync_copy_into\(([^,]+),\s*([^)]+)\)',
        r'stream.memcpy_dtoh_into(\1, \2)',
        content
    )

    # Pattern 15: self.context.memset_zeros → stream.memset_zeros
    content = re.sub(
        r'self\.context\.memset_zeros',
        r'stream.memset_zeros',
        content
    )

    # Pattern 16: self.device.memset_zeros → stream.memset_zeros
    content = re.sub(
        r'self\.device\.memset_zeros',
        r'stream.memset_zeros',
        content
    )

    # Pattern 17: device.memset_zeros → stream.memset_zeros
    content = re.sub(
        r'(\s+)device\.memset_zeros',
        r'\1stream.memset_zeros',
        content
    )

    return content


def ensure_stream_variable(content: str) -> str:
    """Add stream variable definition in methods that use stream operations."""

    # Find all methods that use stream.XXX but don't define stream
    lines = content.split('\n')
    new_lines = []

    in_function = False
    function_indent = 0
    has_stream_use = False
    has_stream_def = False
    function_start_line = 0

    for i, line in enumerate(lines):
        # Detect function/method start
        if re.match(r'(\s*)pub fn |(\s*)fn ', line):
            in_function = True
            function_indent = len(line) - len(line.lstrip())
            has_stream_use = False
            has_stream_def = False
            function_start_line = i

        # Detect stream usage
        if in_function and re.search(r'\bstream\.', line):
            has_stream_use = True

        # Detect stream definition
        if in_function and re.search(r'let\s+stream\s*=', line):
            has_stream_def = True

        # Detect end of function (closing brace at same indent level)
        if in_function and line.strip() == '}' and len(line) - len(line.lstrip()) == function_indent:
            # If we used stream but never defined it, insert definition
            if has_stream_use and not has_stream_def:
                # Find the line right after function signature (after opening brace)
                for j in range(function_start_line, i):
                    if '{' in new_lines[j]:
                        # Insert stream definition after opening brace
                        indent = ' ' * (function_indent + 4)
                        new_lines.insert(j + 1, f'{indent}let stream = self.context.default_stream();')
                        break

            in_function = False

        new_lines.append(line)

    return '\n'.join(new_lines)


def main():
    gpu_src = Path('crates/prism-gpu/src')

    if not gpu_src.exists():
        print(f"Error: {gpu_src} not found. Run from PRISM root directory.")
        sys.exit(1)

    files_to_fix = [
        'active_inference.rs',
        'cma.rs',
        'cma_es.rs',
        'whcr.rs',
        'quantum.rs',
        'thermodynamic.rs',
        'lbs.rs',
        'molecular.rs',
        'pimc.rs',
        'tda.rs',
        'floyd_warshall.rs',
        'dendritic_reservoir.rs',
        'dendritic_whcr.rs',
        'transfer_entropy.rs',
    ]

    for filename in files_to_fix:
        filepath = gpu_src / filename

        if not filepath.exists():
            print(f"Skipping {filename} (not found)")
            continue

        print(f"Processing {filename}...")

        with open(filepath, 'r') as f:
            content = f.read()

        # Apply fixes
        content = fix_memory_operations(content, filename)
        content = ensure_stream_variable(content)

        # Write back
        with open(filepath, 'w') as f:
            f.write(content)

        print(f"  ✓ Fixed {filename}")

    print("\n✓ Memory operation fixes applied to all files")
    print("Note: Some manual fixes may still be needed for complex cases.")
    print("Run 'cargo check --features cuda' to verify.")


if __name__ == '__main__':
    main()
