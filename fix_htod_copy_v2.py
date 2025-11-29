#!/usr/bin/env python3
"""
Fix htod_copy method calls for cudarc 0.18.1

In cudarc 0.18.1:
- device.htod_copy(data) → stream.clone_htod(&data)
- Need to call context.default_stream() to get stream
- Also replace Arc<CudaDevice> with Arc<CudaContext>
"""

import re
from pathlib import Path

def fix_file(filepath):
    """Fix htod_copy calls in a single file"""
    print(f"Processing {filepath}...")

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # Step 1: In each function, add a single stream declaration at the start if htod_copy is used
    # First, find functions that contain htod_copy
    functions_with_htod = []
    in_function = False
    fn_start_line = 0
    brace_depth = 0
    lines = content.split('\n')

    for i, line in enumerate(lines):
        if re.search(r'^\s*(pub\s+)?fn\s+\w+', line):
            in_function = True
            fn_start_line = i
            brace_depth = 0

        if in_function:
            brace_depth += line.count('{') - line.count('}')

            if '.htod_copy(' in line and 'htod_copy_into' not in line and '//' not in line[:line.find('.htod_copy(')]:
                if fn_start_line not in [f[0] for f in functions_with_htod]:
                    functions_with_htod.append((fn_start_line, i))

            if brace_depth == 0 and '{' in ''.join(lines[max(0, fn_start_line):i]):
                in_function = False

    # Step 2: For each function with htod_copy, add stream declaration after the first line with '{'
    # and replace htod_copy calls
    if functions_with_htod:
        new_lines = []
        stream_added_at = set()

        for i, line in enumerate(lines):
            # Check if we should add stream declaration
            should_add_stream = False
            for fn_start, first_htod in functions_with_htod:
                if fn_start <= i < first_htod and '{' in line and fn_start not in stream_added_at:
                    should_add_stream = True
                    stream_added_at.add(fn_start)
                    break

            new_lines.append(line)

            if should_add_stream:
                # Determine indent (add 4 spaces from current line's indent)
                indent = len(line) - len(line.lstrip()) + 4
                #new_lines.append(' ' * indent + '// Stream for GPU memory transfers')
                new_lines.append(' ' * indent + 'let stream = context.default_stream();')

        lines = new_lines

    # Step 3: Replace htod_copy calls with clone_htod
    new_lines = []
    for line in lines:
        if '.htod_copy(' in line and 'htod_copy_into' not in line and not line.strip().startswith('//'):
            # Replace any_var.htod_copy( with stream.clone_htod(
            line = re.sub(r'(\w+(?:\.\w+)*)\.htod_copy\(', r'stream.clone_htod(', line)

        new_lines.append(line)

    content = '\n'.join(new_lines)

    # Step 4: Replace Arc<CudaDevice> with Arc<CudaContext>
    content = re.sub(r'Arc<CudaDevice>', 'Arc<CudaContext>', content)
    content = re.sub(r'&Arc<CudaDevice>', '&Arc<CudaContext>', content)

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
