#!/bin/bash
# Comprehensive cudarc 0.18.1 migration for foundation/quantum/src
# Applies ALL systematic API fixes in one pass

set -e

QUANTUM_SRC="foundation/quantum/src"

echo "🔧 cudarc 0.18.1 Migration for foundation/quantum/src"
echo "=================================================="

# Backup original files
echo "📦 Creating backups..."
for file in gpu_coloring.rs gpu_tsp.rs; do
    if [ -f "$QUANTUM_SRC/$file" ]; then
        cp "$QUANTUM_SRC/$file" "$QUANTUM_SRC/$file.bak"
        echo "  ✅ Backed up $file"
    fi
done

echo ""
echo "🔧 Fixing gpu_coloring.rs..."

# Fix gpu_coloring.rs systematically
sed -i 's/let stream = self\.context\.fork_default_stream()\?;/\/\/ Using existing self.stream/g' "$QUANTUM_SRC/gpu_coloring.rs"
sed -i 's/let stream = context\.fork_default_stream()\?;/\/\/ Using existing stream/g' "$QUANTUM_SRC/gpu_coloring.rs"
sed -i 's/let stream = device\.fork_default_stream()\?;/\/\/ Using existing stream/g' "$QUANTUM_SRC/gpu_coloring.rs"

# Fix htod_sync_copy_into (this pattern is more complex, needs careful handling)
# We'll handle the most common patterns
sed -i 's/context\.htod_sync_copy_into(&\([^,]*\), &mut \([^)]*\))\?;/stream.clone_htod(\&\1);  \/\/ TODO: assign to \2/g' "$QUANTUM_SRC/gpu_coloring.rs"
sed -i 's/self\.context\.htod_sync_copy_into(&\([^,]*\), &mut \([^)]*\))\?;/self.stream.clone_htod(\&\1);  \/\/ TODO: assign to \2/g' "$QUANTUM_SRC/gpu_coloring.rs"

# Fix dtoh_sync_copy_into
sed -i 's/context\.dtoh_sync_copy_into(&\([^,]*\), &mut \([^)]*\))\?;/let \2 = stream.clone_dtoh(\&\1)\?;/g' "$QUANTUM_SRC/gpu_coloring.rs"
sed -i 's/self\.context\.dtoh_sync_copy_into(&\([^,]*\), &mut \([^)]*\))\?;/let \2 = self.stream.clone_dtoh(\&\1)\?;/g' "$QUANTUM_SRC/gpu_coloring.rs"

# Fix alloc_zeros
sed -i 's/context\.alloc_zeros::\([^(]*\)(\([^)]*\))/stream.alloc_zeros::\1(\2)/g' "$QUANTUM_SRC/gpu_coloring.rs"
sed -i 's/self\.context\.alloc_zeros::\([^(]*\)(\([^)]*\))/self.stream.alloc_zeros::\1(\2)/g' "$QUANTUM_SRC/gpu_coloring.rs"

# Fix load_ptx patterns
sed -i 's/\.load_ptx(\([^,]*\), "\([^"]*\)", &\[\([^\]]*\)\])/\.load_module(\&\1, \&[\3])  \/\/ Module: \2/g' "$QUANTUM_SRC/gpu_coloring.rs"

# Fix get_func patterns
sed -i 's/\.get_func("\([^"]*\)", "\([^"]*\)")/\.get_function("\2")  \/\/ Was module: \1/g' "$QUANTUM_SRC/gpu_coloring.rs"

echo "  ✅ Fixed gpu_coloring.rs"

echo ""
echo "🔧 Fixing gpu_tsp.rs..."

# Fix gpu_tsp.rs
sed -i 's/CudaDevice/CudaContext/g' "$QUANTUM_SRC/gpu_tsp.rs"
sed -i 's/let stream = self\.context\.fork_default_stream()\?;/\/\/ Using existing self.stream/g' "$QUANTUM_SRC/gpu_tsp.rs"
sed -i 's/let stream = context\.fork_default_stream()\?;/\/\/ Using existing stream/g' "$QUANTUM_SRC/gpu_tsp.rs"
sed -i 's/let stream = device\.fork_default_stream()\?;/\/\/ Using existing stream/g' "$QUANTUM_SRC/gpu_tsp.rs"

# Memory operations
sed -i 's/context\.htod_sync_copy_into(&\([^,]*\), &mut \([^)]*\))\?;/stream.clone_htod(\&\1);  \/\/ TODO: assign to \2/g' "$QUANTUM_SRC/gpu_tsp.rs"
sed -i 's/device\.htod_sync_copy_into(&\([^,]*\), &mut \([^)]*\))\?;/stream.clone_htod(\&\1);  \/\/ TODO: assign to \2/g' "$QUANTUM_SRC/gpu_tsp.rs"
sed -i 's/self\.context\.htod_sync_copy_into(&\([^,]*\), &mut \([^)]*\))\?;/self.stream.clone_htod(\&\1);  \/\/ TODO: assign to \2/g' "$QUANTUM_SRC/gpu_tsp.rs"

sed -i 's/context\.dtoh_sync_copy_into(&\([^,]*\), &mut \([^)]*\))\?;/let \2 = stream.clone_dtoh(\&\1)\?;/g' "$QUANTUM_SRC/gpu_tsp.rs"
sed -i 's/device\.dtoh_sync_copy_into(&\([^,]*\), &mut \([^)]*\))\?;/let \2 = stream.clone_dtoh(\&\1)\?;/g' "$QUANTUM_SRC/gpu_tsp.rs"
sed -i 's/self\.context\.dtoh_sync_copy_into(&\([^,]*\), &mut \([^)]*\))\?;/let \2 = self.stream.clone_dtoh(\&\1)\?;/g' "$QUANTUM_SRC/gpu_tsp.rs"

sed -i 's/context\.alloc_zeros::\([^(]*\)(\([^)]*\))/stream.alloc_zeros::\1(\2)/g' "$QUANTUM_SRC/gpu_tsp.rs"
sed -i 's/device\.alloc_zeros::\([^(]*\)(\([^)]*\))/stream.alloc_zeros::\1(\2)/g' "$QUANTUM_SRC/gpu_tsp.rs"
sed -i 's/self\.context\.alloc_zeros::\([^(]*\)(\([^)]*\))/self.stream.alloc_zeros::\1(\2)/g' "$QUANTUM_SRC/gpu_tsp.rs"

# Module loading
sed -i 's/\.load_ptx(\([^,]*\), "\([^"]*\)", &\[\([^\]]*\)\])/\.load_module(\&\1, \&[\3])  \/\/ Module: \2/g' "$QUANTUM_SRC/gpu_tsp.rs"
sed -i 's/\.get_func("\([^"]*\)", "\([^"]*\)")/\.get_function("\2")  \/\/ Was module: \1/g' "$QUANTUM_SRC/gpu_tsp.rs"

echo "  ✅ Fixed gpu_tsp.rs"

echo ""
echo "📊 Summary of Changes"
echo "====================="

for file in gpu_coloring.rs gpu_tsp.rs; do
    if [ -f "$QUANTUM_SRC/$file.bak" ]; then
        changes=$(diff -u "$QUANTUM_SRC/$file.bak" "$QUANTUM_SRC/$file" | grep -E '^\+|^-' | grep -v '+++\|---' | wc -l)
        echo "  $file: $changes line changes"
    fi
done

echo ""
echo "✅ Automated migration complete!"
echo ""
echo "⚠️  MANUAL FIXES STILL REQUIRED:"
echo "   1. Add 'stream: Arc<CudaStream>' field to structs"
echo "   2. Initialize stream in constructors: Arc::new(context.default_stream())"
echo "   3. Add stream parameters to function signatures"
echo "   4. Review all '// TODO:' comments for variable reassignment"
echo "   5. Fix module.get_function() calls (need to store module first)"
echo ""
echo "📝 Next steps:"
echo "   1. Review changes: git diff foundation/quantum/src/"
echo "   2. Run: cargo check --all-features"
echo "   3. Fix any remaining compilation errors"
echo "   4. Run: cargo test --all-features"
