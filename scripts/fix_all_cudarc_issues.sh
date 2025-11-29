#!/bin/bash

# Comprehensive cudarc 0.18.1 migration fixes

set -e

echo "=== Comprehensive cudarc 0.18.1 fixes ==="

# 1. Add missing CudaFunction imports
echo "1. Adding missing CudaFunction imports..."

FILES_NEED_CUDAFUNCTION=(
  "crates/prism-gpu/src/floyd_warshall.rs"
)

for FILE in "${FILES_NEED_CUDAFUNCTION[@]}"; do
  if [ -f "$FILE" ]; then
    echo "  Fixing: $FILE"
    # Check if CudaFunction is already imported
    if ! grep -q "use cudarc::driver::CudaFunction" "$FILE"; then
      # Add it to existing cudarc import line
      sed -i 's/use cudarc::driver::{CudaContext,/use cudarc::driver::{CudaContext, CudaFunction,/g' "$FILE"
    fi
  fi
done

# 2. Fix Arc<CudaModule>.get_fn() calls
echo "2. Fixing Arc<CudaModule>.get_fn() calls..."

# Replace module.get_fn with (*module).get_fn() for Arc<CudaModule>
FILES_WITH_MODULE_GETFN=(
  "crates/prism-geometry/src/sensor_layer.rs"
)

for FILE in "${FILES_WITH_MODULE_GETFN[@]}"; do
  if [ -f "$FILE" ]; then
    echo "  Fixing: $FILE"
    # Find patterns like:  let kernel = module\n  .get_fn
    # Replace with:  let kernel = (*module)\n  .get_fn

    # First, backup the file
    cp "$FILE" "$FILE.fixbak"

    # Use perl for multi-line regex
    perl -i -pe 's/let (\w+) = module$/let $1 = (*module)/g' "$FILE"
  fi
done

echo ""
echo "=== Comprehensive fixes complete! ==="
