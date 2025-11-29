#!/bin/bash
# Systematically fix all cudarc 0.18.1 API issues in prism-gpu

set -e

PRISM_GPU_DIR="crates/prism-gpu/src"

echo "=== Fixing cudarc 0.18.1 API issues in $PRISM_GPU_DIR ==="

# Fix 1: Update imports - remove CudaDevice, LaunchAsync, add CudaContext, CudaStream
find "$PRISM_GPU_DIR" -name "*.rs" -exec sed -i \
  -e 's/use cudarc::driver::CudaDevice;/use cudarc::driver::CudaContext;/g' \
  -e 's/use cudarc::driver::{CudaDevice,/use cudarc::driver::{CudaContext,/g' \
  -e 's/use cudarc::driver::{CudaDevice /use cudarc::driver::{CudaContext /g' \
  -e 's/, CudaDevice/, CudaContext/g' \
  -e 's/CudaDevice,/CudaContext,/g' \
  -e 's/, LaunchAsync//g' \
  -e 's/LaunchAsync, //g' \
  {} \;

# Fix 2: Add missing imports where needed
for file in "$PRISM_GPU_DIR"/*.rs; do
  # Add CudaStream import if file uses stream operations
  if grep -q "default_stream\|CudaStream" "$file" 2>/dev/null; then
    if ! grep -q "use cudarc::driver::CudaStream" "$file" 2>/dev/null; then
      sed -i '/use cudarc::driver::CudaContext;/a use cudarc::driver::CudaStream;' "$file"
    fi
  fi

  # Add Ptx import if file loads PTX
  if grep -q "load_module\|Ptx::" "$file" 2>/dev/null; then
    if ! grep -q "use cudarc::nvrtc::Ptx" "$file" 2>/dev/null; then
      sed -i '/use cudarc::driver::CudaContext;/a use cudarc::nvrtc::Ptx;' "$file"
    fi
  fi
done

# Fix 3: Replace Arc<CudaDevice> with Arc<CudaContext>
find "$PRISM_GPU_DIR" -name "*.rs" -exec sed -i \
  -e 's/Arc<CudaDevice>/Arc<CudaContext>/g' \
  {} \;

# Fix 4: Replace CudaDevice::new with CudaContext::new
find "$PRISM_GPU_DIR" -name "*.rs" -exec sed -i \
  -e 's/CudaDevice::new/CudaContext::new/g' \
  {} \;

# Fix 5: Replace device.get_func patterns - this needs context
# Pattern: device.get_func("module", "kernel") -> module.load_function("kernel")
# This requires more sophisticated handling - skip for now, will be caught by compiler

# Fix 6: Replace fork_default_stream() with default_stream()
find "$PRISM_GPU_DIR" -name "*.rs" -exec sed -i \
  -e 's/\.fork_default_stream()/.default_stream()/g' \
  {} \;

# Fix 7: Fix load_ptx calls - replace with load_module pattern
# Pattern: device.load_ptx(ptx_str.into(), "name", &[...])
# Becomes: let module = device.load_module(Ptx::Image(&ptx_bytes))?;
# This is complex and file-specific - will be caught by compiler

echo "=== Basic fixes applied. Compile errors will identify remaining issues. ==="
echo "=== Next steps: ==="
echo "1. Run 'cargo check --features cuda' to identify remaining issues"
echo "2. Fix module loading patterns (load_ptx -> load_module)"
echo "3. Fix memory operations (ensure they use stream, not context)"
echo "4. Fix kernel loading (get_func -> load_function)"
