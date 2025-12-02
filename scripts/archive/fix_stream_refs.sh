#!/bin/bash

# Fix Arc<CudaStream> reference patterns
# Replace self.stream references with proper dereferencing

set -e

echo "=== Fixing Arc<CudaStream> usage patterns ==="

FILE="crates/prism-geometry/src/sensor_layer.rs"

if [ -f "$FILE" ]; then
  echo "Fixing: $FILE"

  # Fix launch_on_stream calls: &self.stream -> &*self.stream
  sed -i 's/launch_on_stream(&self\.stream,/launch_on_stream(\&*self.stream,/g' "$FILE"

  echo "  ✓ Fixed launch_on_stream calls"
fi

echo ""
echo "=== Stream reference fixes complete! ==="
