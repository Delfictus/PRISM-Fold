#!/usr/bin/env python3
"""
Automated cudarc 0.18.1 migration for foundation/quantum/src files
Fixes ALL remaining API incompatibilities systematically
"""

import re
import sys
from pathlib import Path

def fix_gpu_coloring(content):
    """Fix gpu_coloring.rs"""

    # Fix jones_plassmann_gpu signature
    content = re.sub(
        r'fn jones_plassmann_gpu\(\s*context: &Arc<CudaContext>,',
        r'fn jones_plassmann_gpu(\n        context: &Arc<CudaContext>,\n        stream: &Arc<CudaStream>,',
        content
    )

    # Fix all calls to download_adjacency to include stream
    content = re.sub(
        r'Self::download_adjacency\(context, &gpu_adjacency, n\)',
        r'Self::download_adjacency(context, stream, &gpu_adjacency, n)',
        content
    )

    # Fix all remaining context.alloc_zeros -> stream.alloc_zeros
    content = re.sub(
        r'context\s*\.\s*alloc_zeros::<([^>]+)>\(([^)]+)\)',
        r'stream.alloc_zeros::<\1>(\2)',
        content
    )

    # Fix all htod_sync_copy_into patterns
    content = re.sub(
        r'context\s*\.\s*htod_sync_copy_into\(&([^,]+), &mut ([^)]+)\)',
        r'let \2 = stream.clone_htod(&\1)',
        content
    )

    # Fix simpler htod patterns
    content = re.sub(
        r'context\.htod_sync_copy_into\(\[([^\]]+)\], &mut ([^)]+)\)',
        r'stream.memset_zeros(&mut \2)',  # For single-element inits, use memset
        content
    )

    # Fix all dtoh_sync_copy_into patterns
    content = re.sub(
        r'context\s*\.\s*dtoh_sync_copy_into\(&([^,]+), &mut ([^)]+)\)',
        r'let \2 = stream.clone_dtoh(&\1)',
        content
    )

    # Remove fork_default_stream() calls - use existing stream
    content = re.sub(
        r'let stream = (?:self\.)?context\.fork_default_stream\(\)\??;',
        r'// Using existing stream',
        content
    )

    # Fix load_ptx calls (old API)
    content = re.sub(
        r'context\s*\.\s*load_ptx\(\s*([^,]+),\s*"([^"]+)",\s*&\[([^\]]+)\]\s*\)',
        r'let module = context.load_module(&\1, &[\3])',
        content
    )

    # Fix get_func calls (old API)
    content = re.sub(
        r'context\s*\.\s*get_func\("([^"]+)",\s*"([^"]+)"\)',
        r'module.get_function("\2")',  # Module name no longer needed
        content
    )

    # Fix find_optimal_threshold_gpu signature
    content = re.sub(
        r'fn find_optimal_threshold_gpu\(\s*context: &Arc<CudaContext>,',
        r'fn find_optimal_threshold_gpu(\n        context: &Arc<CudaContext>,\n        stream: &Arc<CudaStream>,',
        content
    )

    # Fix build_adjacency_gpu calls in find_optimal_threshold_gpu
    content = re.sub(
        r'Self::build_adjacency_gpu\(context, coupling_matrix, mid_threshold\)',
        r'Self::build_adjacency_gpu(context, stream, coupling_matrix, mid_threshold)',
        content
    )

    return content

def fix_gpu_tsp(content):
    """Fix gpu_tsp.rs"""

    # Add CudaStream to imports
    content = re.sub(
        r'use cudarc::driver::\{CudaContext, LaunchConfig\};',
        r'use cudarc::driver::{CudaContext, CudaStream, LaunchConfig};',
        content
    )

    # Replace CudaDevice with CudaContext
    content = re.sub(r'CudaDevice', r'CudaContext', content)

    # Add stream field to GpuTspSolver struct
    content = re.sub(
        r'(pub struct GpuTspSolver \{[^}]*context: Arc<CudaContext>,)',
        r'\1\n    /// CUDA stream\n    stream: Arc<CudaStream>,',
        content
    )

    # Fix constructor to create stream
    content = re.sub(
        r'let device = CudaContext::new\(0\)\.context\(',
        r'let context = Arc::new(CudaContext::new(0).context(',
        content
    )

    # Add stream creation after context
    content = re.sub(
        r'(\s+let context = Arc::new\(CudaContext::new[^;]+;)',
        r'\1\n        let stream = Arc::new(context.default_stream());',
        content
    )

    # Fix all device references to context
    content = re.sub(r'\bdevice\b', r'context', content)

    # Add stream to struct initialization
    content = re.sub(
        r'(Ok\(Self \{[^}]*context,)',
        r'\1\n            stream,',
        content
    )

    # Fix memory operations
    content = re.sub(
        r'context\.alloc_zeros::<([^>]+)>\(([^)]+)\)',
        r'stream.alloc_zeros::<\1>(\2)',
        content
    )

    content = re.sub(
        r'context\.htod_sync_copy_into\(&([^,]+), &mut ([^)]+)\)',
        r'let \2 = stream.clone_htod(&\1)',
        content
    )

    content = re.sub(
        r'context\.dtoh_sync_copy_into\(&([^,]+), &mut ([^)]+)\)',
        r'let \2 = stream.clone_dtoh(&\1)',
        content
    )

    # Remove fork_default_stream
    content = re.sub(
        r'let stream = context\.fork_default_stream\(\)\??;',
        r'// Using existing stream',
        content
    )

    # Fix load_ptx
    content = re.sub(
        r'context\s*\.\s*load_ptx\(\s*([^,]+),\s*"([^"]+)",\s*&\[([^\]]+)\]\s*\)',
        r'let module = context.load_module(&\1, &[\3])',
        content
    )

    # Fix get_func
    content = re.sub(
        r'context\s*\.\s*get_func\("([^"]+)",\s*"([^"]+)"\)',
        r'module.get_function("\2")',
        content
    )

    # Fix compute_distance_matrix_gpu signature
    content = re.sub(
        r'fn compute_distance_matrix_gpu\(\s*device: &Arc<CudaContext>,',
        r'fn compute_distance_matrix_gpu(\n        context: &Arc<CudaContext>,\n        stream: &Arc<CudaStream>,',
        content
    )

    return content

def main():
    project_root = Path(__file__).parent.parent
    quantum_src = project_root / "foundation" / "quantum" / "src"

    files_to_fix = [
        ("gpu_coloring.rs", fix_gpu_coloring),
        ("gpu_tsp.rs", fix_gpu_tsp),
    ]

    for filename, fix_func in files_to_fix:
        filepath = quantum_src / filename
        if not filepath.exists():
            print(f"❌ {filename} not found")
            continue

        print(f"🔧 Fixing {filename}...")
        content = filepath.read_text()
        fixed_content = fix_func(content)

        if content != fixed_content:
            filepath.write_text(fixed_content)
            print(f"✅ {filename} fixed")
        else:
            print(f"ℹ️  {filename} unchanged")

    print("\n✅ Migration complete!")
    print("Run: cargo check --all-features")

if __name__ == "__main__":
    main()
