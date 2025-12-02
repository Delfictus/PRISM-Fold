#!/usr/bin/env python3
"""
Fix incorrect 2-arg clone_dtoh usage in quantum.rs.
These should be 1-arg form that returns a Vec, not 2-arg memcpy form.
"""

import re
from pathlib import Path

def fix_quantum_rs():
    """Fix quantum.rs clone_dtoh 2-arg calls."""
    filepath = Path("/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-gpu/src/quantum.rs")

    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern 1: clone_dtoh with &mut second arg (incorrect 2-arg form)
    # Replace:  .clone_dtoh(&d_colors, &mut colors_i32)
    # With:     .clone_dtoh(&d_colors)
    # And remove the vec! allocation line before it

    # Fix line 309
    pattern1 = r'let mut colors_i32 = vec!\[0i32; num_vertices\];\s+self\.stream\s+\.clone_dtoh\(&d_colors, &mut colors_i32\)'
    replacement1 = 'let colors_i32 = self.stream\n .clone_dtoh(&d_colors)'
    content = re.sub(pattern1, replacement1, content)

    # Fix line 895, 899 - these might be legitimate 1-arg forms, check context
    # Pattern: .clone_dtoh(real_amps) without &mut - these are OK

    # Fix line 1434
    pattern2 = r'\.clone_dtoh\(&d_colors, &mut [a-z_]+\)'
    replacement2 = '.clone_dtoh(&d_colors)'

    # More general pattern for 2-arg clone_dtoh
    # (.clone_dtoh\([^,)]+),\s*&mut\s+[^)]+\)
    general_pattern = r'(\.clone_dtoh\([^,)]+),\s*&mut\s+[^)]+\)'
    general_replacement = r'\1)'

    content = re.sub(general_pattern, general_replacement, content)

    # Remove "let mut X = vec![...]; " if followed by X being used as &mut in next line
    # This is more complex, skip for now

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"✓ Fixed {filepath}")

def main():
    fix_quantum_rs()

if __name__ == "__main__":
    main()
