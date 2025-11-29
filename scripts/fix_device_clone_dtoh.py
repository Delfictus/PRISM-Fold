#!/usr/bin/env python3
"""
Fix device.clone_dtoh calls to use stream API in cudarc 0.18.1.

For structs without a stream field, we create a temporary stream.
"""

import re
from pathlib import Path

def fix_dendritic_whcr():
    """Fix dendritic_whcr.rs - add stream creation."""
    filepath = Path("/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-gpu/src/dendritic_whcr.rs")

    with open(filepath, 'r') as f:
        content = f.read()

    # Replace: let result = self.device.clone_dtoh(&d_final)?;
    # With:    let stream = self.device.fork_default_stream()?;
    #          let result = stream.clone_dtoh(&d_final)?;

    pattern = r'let result = self\.device\.clone_dtoh\(&d_final\)\?;'
    replacement = '''let stream = self.device.fork_default_stream()?;
        let result = stream.clone_dtoh(&d_final)?;'''

    content = re.sub(pattern, replacement, content)

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"✓ Fixed {filepath}")

def fix_transfer_entropy():
    """Fix transfer_entropy.rs - replace context.clone_dtoh with stream.clone_dtoh."""
    filepath = Path("/mnt/c/Users/Predator/Desktop/PRISM/crates/prism-gpu/src/transfer_entropy.rs")

    with open(filepath, 'r') as f:
        content = f.read()

    # First, check if struct has a stream field
    # If not, we need to add one or create temp streams

    # Simple replacement: self.context.clone_dtoh -> stream.clone_dtoh
    # But first need to create stream

    # For each method that uses clone_dtoh, add stream creation at the start
    # This is complex - let's just replace context with a local stream

    # Simpler approach: replace all self.context.clone_dtoh with temp stream
    # Pattern: self.context.clone_dtoh(&...)
    # Replace with: self.context.fork_default_stream()?.clone_dtoh(&...)

    pattern = r'self\.context\.clone_dtoh\('
    replacement = 'self.context.fork_default_stream()?.clone_dtoh('

    content = re.sub(pattern, replacement, content)

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"✓ Fixed {filepath}")

def main():
    fix_dendritic_whcr()
    fix_transfer_entropy()

if __name__ == "__main__":
    main()
