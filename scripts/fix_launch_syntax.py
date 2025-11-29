#!/usr/bin/env python3
"""Fix launch_builder().launch() syntax errors in cudarc 0.18.1 migration"""

import re
import sys

def fix_launch_syntax(content):
    """Fix launch_builder patterns that have extra closing paren"""

    # Pattern: .launch(config)\n )\n }\n .context(...)
    # Replace with: .launch(config)?;\n }
    pattern = r'(\.launch\(config\))\s*\)\s*\}\s*\.context\([^)]+\)\?;'
    replacement = r'\1?;\n }'

    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    return content

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_launch_syntax.py <file>")
        sys.exit(1)

    filepath = sys.argv[1]

    with open(filepath, 'r') as f:
        content = f.read()

    fixed_content = fix_launch_syntax(content)

    with open(filepath, 'w') as f:
        f.write(fixed_content)

    print(f"Fixed: {filepath}")

if __name__ == '__main__':
    main()
