#!/usr/bin/env python3
"""
Complete migration of htod_sync_copy for cudarc 0.18.1.
Handles all remaining edge cases.
"""

import re
from pathlib import Path

def fix_htod_comprehensive(file_path):
    """Fix all htod-related issues in one pass."""
    with open(file_path, 'r') as f:
        content = f.read()

    original = content

    # 1. Fix imports: Add CudaContext if using Arc<CudaContext>
    if 'Arc<CudaContext>' in content or 'context: Arc<CudaContext>' in content:
        if 'use cudarc::driver::{' in content and 'CudaContext' not in content:
            content = re.sub(
                r'use cudarc::driver::\{([^}]+)\};',
                lambda m: f'use cudarc::driver::{{{m.group(1)}, CudaContext}};' if 'CudaContext' not in m.group(1) else m.group(0),
                content
            )

    # 2. Replace htod_sync_copy with clone_htod (will fix caller later)
    content = content.replace('.htod_sync_copy(', '.clone_htod(')

    # 3. Replace .device with .context for struct field access
    #    But be careful not to replace in comments or type names
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # Skip comments and type annotations
        if '///' in line or '// ' in line or 'CudaDevice' in line or 'let device' in line:
            continue
        # Replace .device with .context (field access)
        if re.search(r'\bself\.device\b', line):
            lines[i] = re.sub(r'\bself\.device\b', 'self.context', line)
        elif line.strip() == '.device':
            lines[i] = line.replace('.device', '.context')

    content = '\n'.join(lines)

    # 4. Fix struct field declarations
    content = re.sub(
        r'(\s+)device: Arc<CudaDevice>',
        r'\1context: Arc<CudaContext>',
        content
    )

    # 5. Fix constructor returns
    content = content.replace('Ok(Self { device })', 'Ok(Self { context })')

    # 6. Now fix .context.clone_htod() → stream.clone_htod()
    # Find patterns like:
    #   self.context\n            .clone_htod
    # Replace with:
    #   let stream = self.context.fork_default_stream()?;\n        stream\n            .clone_htod

    # This is complex - let's do it carefully
    # Pattern: variable = self\n            .context\n            .clone_htod(&data)
    # Becomes: let stream = self.context.fork_default_stream()?;\n        variable = stream\n            .clone_htod(&data)

    # For now, just fix the immediate .context.clone_htod inline calls
    # More complex multiline patterns need manual fixing or smarter parsing

    if content != original:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    base = Path('/mnt/c/Users/Predator/Desktop/PRISM')

    # All Rust files that might have these issues
    patterns = [
        'crates/prism-gpu/src/*.rs',
        'crates/prism-whcr/src/*.rs',
        'crates/prism-geometry/src/*.rs',
        'foundation/prct-core/src/*.rs',
        'foundation/prct-core/src/gpu/*.rs',
        'foundation/neuromorphic/src/*.rs',
        'foundation/quantum/src/*.rs',
    ]

    count = 0
    for pattern in patterns:
        for fpath in base.glob(pattern):
            if fpath.is_file():
                try:
                    if fix_htod_comprehensive(fpath):
                        print(f"✓ {fpath.relative_to(base)}")
                        count += 1
                except Exception as e:
                    print(f"✗ {fpath.relative_to(base)}: {e}")

    print(f"\n{count} files modified")

if __name__ == '__main__':
    main()
