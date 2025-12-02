#!/usr/bin/env python3
"""Final comprehensive fix for ALL load_ptx patterns."""
import re
from pathlib import Path

def fix_load_ptx(text):
    """Fix all load_ptx patterns in text."""

    # Pattern 1: Multi-line context.load_ptx(...)
    text = re.sub(
        r'(\s+)context\s*\.\s*load_ptx\s*\(\s*([^,]+?),\s*"[^"]*",\s*&\[[^\]]*\]\s*\)',
        r'\1let _module = context.load_module(\2)',
        text,
        flags=re.MULTILINE | re.DOTALL
    )

    # Pattern 2: Multi-line device.load_ptx(...)
    text = re.sub(
        r'(\s+)device\s*\.\s*load_ptx\s*\(\s*([^,]+?),\s*"[^"]*",\s*&\[[^\]]*\]\s*\)',
        r'\1let _module = device.load_module(\2)',
        text,
        flags=re.MULTILINE | re.DOTALL
    )

    # Pattern 3: Multi-line stream.load_ptx(...)
    text = re.sub(
        r'(\s+)stream\s*\.\s*load_ptx\s*\(\s*([^,]+?),\s*"[^"]*",\s*&\[[^\]]*\]\s*\)',
        r'\1let _module = context.load_module(\2)',
        text,
        flags=re.MULTILINE | re.DOTALL
    )

    # Pattern 4: Fix get_func calls
    text = re.sub(
        r'\b(context|device|stream)\.get_func\s*\(\s*"[^"]*",\s*"([^"]*)"\s*\)',
        r'module.get_fn("\2")',
        text
    )

    return text

def main():
    base = Path("/mnt/c/Users/Predator/Desktop/PRISM")

    # Find ALL .rs files with load_ptx
    files = list(base.glob("**/*.rs"))
    fixed_count = 0

    for filepath in files:
        try:
            if "target/" in str(filepath) or ".git/" in str(filepath):
                continue

            content = filepath.read_text()
            if ".load_ptx(" not in content:
                continue

            new_content = fix_load_ptx(content)

            if new_content != content:
                filepath.write_text(new_content)
                print(f"✓ Fixed: {filepath.relative_to(base)}")
                fixed_count += 1
            else:
                print(f"- No change: {filepath.relative_to(base)}")

        except Exception as e:
            print(f"✗ Error in {filepath.relative_to(base)}: {e}")

    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} files")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
