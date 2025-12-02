#!/bin/bash
# PRISM PTX Kernel Build Script
# Compiles all CUDA kernels to PTX format

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PTX_DIR="$PROJECT_ROOT/target/ptx"
KERNEL_DIR="$PROJECT_ROOT/crates/prism-gpu/src/kernels"

# CUDA configuration
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="$CUDA_HOME/bin/nvcc"
ARCH="${CUDA_ARCH:-sm_86}"

echo "=========================================="
echo "PRISM PTX Kernel Build"
echo "=========================================="
echo "CUDA Home: $CUDA_HOME"
echo "Architecture: $ARCH"
echo "Output: $PTX_DIR"
echo ""

# Create output directory
mkdir -p "$PTX_DIR"

# Find all .cu files and compile them
compile_kernel() {
    local cu_file="$1"
    local name=$(basename "$cu_file" .cu)
    local ptx_file="$PTX_DIR/$name.ptx"

    echo "Compiling $name.cu..."
    "$NVCC" --ptx \
        -o "$ptx_file" \
        "$cu_file" \
        -arch="$ARCH" \
        --std=c++14 \
        -Xcompiler -fPIC \
        -O3

    echo "  -> $ptx_file ($(stat -c%s "$ptx_file" 2>/dev/null || stat -f%z "$ptx_file") bytes)"
}

# Compile all kernels
if [ -d "$KERNEL_DIR" ]; then
    find "$KERNEL_DIR" -name "*.cu" | while read cu_file; do
        compile_kernel "$cu_file"
    done
else
    echo "Warning: Kernel directory not found: $KERNEL_DIR"
fi

echo ""
echo "=========================================="
echo "PTX Compilation Complete"
echo "=========================================="
ls -la "$PTX_DIR"/*.ptx 2>/dev/null || echo "No PTX files found"
