#!/bin/bash

# Fix all remaining load_ptx calls in Rust files
# This handles the pattern: stream.load_ptx(...) or context.load_ptx(...)

set -e

FILES=(
    "crates/prism-geometry/src/sensor_layer.rs"
    "crates/prism-gpu/src/cma.rs"
    "crates/prism-gpu/src/cma_es.rs"
    "crates/prism-gpu/src/context.rs"
    "crates/prism-gpu/src/dendritic_reservoir.rs"
    "crates/prism-gpu/src/dendritic_whcr.rs"
    "crates/prism-gpu/src/floyd_warshall.rs"
    "crates/prism-gpu/src/molecular.rs"
    "crates/prism-gpu/src/pimc.rs"
    "crates/prism-gpu/src/tda.rs"
    "crates/prism-gpu/src/transfer_entropy.rs"
    "foundation/prct-core/src/gpu_quantum_annealing.rs"
    "foundation/quantum/src/gpu_k_opt.rs"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing: $file"

        # Use Python for more robust multi-line replacement
        python3 << 'PYTHON_EOF' "$file"
import sys
import re

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    content = f.read()

original = content

# Pattern 1: stream.load_ptx(...) on multiple lines
content = re.sub(
    r'(\s+)(stream|device)\s*\n\s*\.load_ptx\(',
    r'\1let module = context.load_module(',
    content,
    flags=re.MULTILINE
)

# Pattern 2: Remove module name and kernel array arguments from load_module
# This is tricky - we need to keep only the first argument (PTX)
# Match: load_module(arg1, "module_name", &["kernel1", "kernel2"])
# Replace with: load_module(arg1)

content = re.sub(
    r'(load_module\([^,)]+),\s*"[^"]+",\s*&\[[^\]]+\]',
    r'\1',
    content
)

# Pattern 3: stream.get_func("module", "kernel") → module.get_fn("kernel")
content = re.sub(
    r'(stream|device)\.get_func\("([^"]+)",\s*"([^"]+)"\)',
    r'module.get_fn("\3")',
    content
)

# Pattern 4: context.get_func("module", "kernel") → module.get_fn("kernel")
content = re.sub(
    r'context\.get_func\("([^"]+)",\s*"([^"]+)"\)',
    r'module.get_fn("\2")',
    content
)

if content != original:
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  ✓ Fixed {filepath}")
else:
    print(f"  - No changes for {filepath}")
PYTHON_EOF

    else
        echo "  ✗ File not found: $file"
    fi
done

echo ""
echo "============================================"
echo "All remaining files processed"
echo "============================================"
