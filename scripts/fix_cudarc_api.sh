#!/bin/bash

# Fix cudarc 0.18.1 API changes
# 1. CudaModule.get_func() -> get_fn()
# 2. CudaStream fields need Arc<CudaStream>
# 3. Fix stream usage patterns

set -e

echo "=== Fixing cudarc 0.18.1 API changes ==="

# Fix get_func -> get_fn in all Rust files
echo "Fixing CudaModule.get_func() -> get_fn()..."
find crates/ foundation/ -name "*.rs" -type f ! -name "*.bak" -exec sed -i 's/\.get_func(/\.get_fn(/g' {} +

# Fix stream field declarations: stream: CudaStream -> stream: Arc<CudaStream>
echo "Fixing stream field types..."

# Files that need stream field fixes
FILES_WITH_STREAM_FIELD=(
  "crates/prism-geometry/src/sensor_layer.rs"
  "crates/prism-whcr/src/geometry_sync.rs"
  "crates/prism-whcr/src/geometry_accumulator.rs"
)

for FILE in "${FILES_WITH_STREAM_FIELD[@]}"; do
  if [ -f "$FILE" ]; then
    echo "  Processing: $FILE"
    # Fix field declaration
    sed -i 's/stream: CudaStream,/stream: Arc<CudaStream>,/g' "$FILE"
  fi
done

echo ""
echo "=== API migration complete! ==="
echo ""
echo "Next: Review sensor_layer.rs for manual fixes (clone_htod usage)"
