#!/usr/bin/env python3
"""
Fix device/context variable references after cudarc migration.

This script fixes function bodies where the parameter was renamed from
device to context but the code still references device.
"""

import re
from pathlib import Path

GPU_SRC = Path("/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-gpu/src")

def fix_function_bodies(filepath):
    """Fix device references in functions with context parameter"""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Pattern: pub fn new(context: Arc<CudaContext>) -> Result<Self>
    # Then fix all 'device' references in that function to 'context'

    # Simple approach: Replace standalone device references with context
    # But preserve self.device() method calls

    # Replace device references that are not method calls or struct fields
    lines = content.split('\n')
    result_lines = []

    for line in lines:
        # Skip if it's a struct field definition
        if re.match(r'\s*(context|device):\s*Arc<Cuda', line):
            result_lines.append(line)
            continue

        # Replace device variable references (not self.device or device:)
        # Pattern: variable usage like device.something, Arc::new(device), etc
        if 'device' in line and 'self.device' not in line and not re.search(r'^\s*(context|device):', line):
            # Replace device with context in various contexts
            line = re.sub(r'\bdevice\.', 'context.', line)
            line = re.sub(r'\(device\)', '(context)', line)
            line = re.sub(r'\(device,', '(context,', line)
            line = re.sub(r',\s*device\)', ', context)', line)
            line = re.sub(r'= device$', '= context', line)
            line = re.sub(r'= device\b', '= context', line)
            line = re.sub(r'AATGSScheduler::new\(device\)', 'AATGSScheduler::new(context.clone())', line)
            line = re.sub(r'AsyncPipeline::new\(device', 'AsyncPipeline::new(context', line)

        result_lines.append(line)

    content = '\n'.join(result_lines)

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix all GPU source files"""
    print("Fixing device/context variable references...")

    fixed_count = 0
    for rs_file in sorted(GPU_SRC.glob("*.rs")):
        if fix_function_bodies(rs_file):
            print(f"  ✓ Fixed {rs_file.name}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()
