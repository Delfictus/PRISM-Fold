#!/usr/bin/env python3
"""
Fix htod_copy method calls for cudarc 0.18.1

In cudarc 0.18.1:
- CudaDevice → CudaContext
- device.htod_copy(data) → stream.clone_htod(&data)
- Need to call context.default_stream() to get stream
"""

import re
import sys
from pathlib import Path

def fix_file(filepath):
    """Fix htod_copy calls in a single file"""
    print(f"Processing {filepath}...")

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # Step 1: Find all functions that use htod_copy
    # We'll track whether each function needs a stream variable

    lines = content.split('\n')
    new_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this line has htod_copy
        if '.htod_copy(' in line and 'htod_copy_into' not in line and not line.strip().startswith('//'):
            # Determine the variable being called (device, context, self.context, etc)
            match = re.search(r'(\w+(?:\.\w+)*)\.htod_copy\(', line)
            if match:
                var_name = match.group(1)
                indent = len(line) - len(line.lstrip())

                # Check if we need to add stream declaration
                # Look back up to 20 lines
                need_stream = True
                for j in range(max(0, i-20), i):
                    if 'let stream =' in lines[j]:
                        need_stream = False
                        break

                if need_stream:
                    # Add stream declaration before this line
                    if var_name == 'self.context':
                        new_lines.append(' ' * indent + 'let stream = self.context.default_stream();')
                    elif var_name == 'context':
                        new_lines.append(' ' * indent + 'let stream = context.default_stream();')
                    elif var_name == 'device':
                        new_lines.append(' ' * indent + 'let stream = device.default_stream();')
                    elif var_name == 'cuda_device':
                        new_lines.append(' ' * indent + 'let stream = cuda_device.default_stream();')
                    else:
                        new_lines.append(' ' * indent + f'let stream = {var_name}.default_stream();')

                # Replace .htod_copy( with .clone_htod(
                line = line.replace(f'{var_name}.htod_copy(', 'stream.clone_htod(')

        new_lines.append(line)
        i += 1

    content = '\n'.join(new_lines)

    # Step 2: Replace all CudaDevice with CudaContext in type signatures
    # But only if CudaDevice is not already imported from cudarc
    content = re.sub(r'Arc<CudaDevice>', 'Arc<CudaContext>', content)
    content = re.sub(r'&Arc<CudaDevice>', '&Arc<CudaContext>', content)

    # Step 3: Replace device variable with context in function parameters
    # Only in new() functions where parameter is called 'device'
    # This is tricky - we need to be careful

    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Fixed {filepath}")
        return True
    else:
        print(f"  - No changes needed for {filepath}")
        return False

def main():
    files_to_fix = [
        'crates/prism-gpu/src/whcr.rs',
        'crates/prism-gpu/src/dendritic_whcr.rs',
        'crates/prism-gpu/src/lbs.rs',
        'foundation/prct-core/src/gpu_transfer_entropy.rs',
        'foundation/prct-core/src/gpu_thermodynamic.rs',
        'foundation/prct-core/src/gpu_quantum_annealing.rs',
        'foundation/prct-core/src/gpu_quantum.rs',
        'foundation/prct-core/src/gpu_kuramoto.rs',
        'foundation/prct-core/src/gpu_active_inference.rs',
    ]

    fixed_count = 0
    for filepath in files_to_fix:
        path = Path(filepath)
        if path.exists():
            if fix_file(filepath):
                fixed_count += 1
        else:
            print(f"  ⚠ File not found: {filepath}")

    print(f"\n✓ Fixed {fixed_count} files")

if __name__ == '__main__':
    main()
