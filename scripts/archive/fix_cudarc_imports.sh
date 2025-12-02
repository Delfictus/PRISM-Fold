#!/bin/bash

# Fix cudarc 0.18.1 migration errors
# CudaDevice -> CudaContext
# Remove LaunchAsync imports

set -e

echo "=== Fixing cudarc 0.18.1 CudaDevice -> CudaContext migration ==="

# Find all Rust files with CudaDevice
FILES=$(grep -rl "CudaDevice" --include="*.rs" \
  crates/prism-gpu/src/ \
  crates/prism-geometry/src/ \
  crates/prism-whcr/src/ \
  crates/prism-gnn/src/ \
  crates/prism-phases/src/ \
  crates/prism-mec/src/ \
  crates/prism-pipeline/src/ \
  crates/prism-core/src/ \
  foundation/prct-core/src/ \
  foundation/neuromorphic/src/ \
  foundation/quantum/src/ \
  2>/dev/null || true)

for FILE in $FILES; do
  echo "Processing: $FILE"

  # Skip backup files
  if [[ "$FILE" == *.bak ]]; then
    echo "  Skipping backup file"
    continue
  fi

  # Replace CudaDevice with CudaContext
  sed -i 's/CudaDevice/CudaContext/g' "$FILE"

  # Remove LaunchAsync from imports (handle various formats)
  sed -i 's/LaunchAsync, //g' "$FILE"
  sed -i 's/, LaunchAsync//g' "$FILE"
  sed -i 's/LaunchAsync,//g' "$FILE"

  # Clean up double commas and spaces
  sed -i 's/,,/,/g' "$FILE"
  sed -i 's/{ ,/{/g' "$FILE"
  sed -i 's/, }/}/g' "$FILE"
  sed -i 's/  */ /g' "$FILE"

  echo "  ✓ Fixed"
done

# Also fix test files
TEST_FILES=$(grep -rl "CudaDevice" --include="*.rs" \
  crates/prism-gpu/tests/ \
  crates/prism-geometry/tests/ \
  crates/prism-phases/tests/ \
  foundation/prct-core/tests/ \
  foundation/prct-core/examples/ \
  foundation/neuromorphic/benches/ \
  2>/dev/null || true)

for FILE in $TEST_FILES; do
  echo "Processing test: $FILE"

  sed -i 's/CudaDevice/CudaContext/g' "$FILE"
  sed -i 's/LaunchAsync, //g' "$FILE"
  sed -i 's/, LaunchAsync//g' "$FILE"
  sed -i 's/LaunchAsync,//g' "$FILE"
  sed -i 's/,,/,/g' "$FILE"
  sed -i 's/{ ,/{/g' "$FILE"
  sed -i 's/, }/}/g' "$FILE"

  echo "  ✓ Fixed"
done

echo ""
echo "=== Migration complete! ==="
echo ""
echo "Files processed: $(echo "$FILES $TEST_FILES" | wc -w)"
echo ""
echo "Next steps:"
echo "  1. cargo check --features cuda"
echo "  2. Review any remaining compilation errors"
echo "  3. Update any device.method() calls to context.method() if needed"
