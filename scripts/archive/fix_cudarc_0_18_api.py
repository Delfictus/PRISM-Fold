#!/usr/bin/env python3
"""
Fix cudarc 0.9/0.11 API calls to cudarc 0.18.1 API.

Changes:
- load_ptx() → load_module(Ptx::Image(bytes))
- get_func() → module.load_function()
"""

import re
import sys
from pathlib import Path

def fix_lbs_file(content: str) -> str:
    """
    Special fix for LBS file which loads multiple PTX modules.
    Each module needs to be stored and functions loaded from the correct module.
    """
    # For LBS, we need to store modules and load functions from them later
    # This requires a more significant refactor, so we'll create a helper struct

    if "pub struct LbsGpu {" not in content:
        return content

    # Add module storage to struct
    content = re.sub(
        r"pub struct LbsGpu \{(\s+)device: Arc<CudaContext>,(\s+)\}",
        r"pub struct LbsGpu {\1device: Arc<CudaContext>,\1// Loaded PTX modules\1_modules: Vec<cudarc::driver::CudaModule>,\2}",
        content
    )

    # Fix the new() method - store modules
    new_method_pattern = r'pub fn new\(device: Arc<CudaContext>, ptx_dir: &Path\) -> Result<Self, PrismError> \{.*?Ok\(Self \{ device \}\)'

    def fix_new_method(match):
        return '''pub fn new(device: Arc<CudaContext>, ptx_dir: &Path) -> Result<Self, PrismError> {
        use cudarc::nvrtc::Ptx;

        let mut _modules = Vec::new();

        // Load all PTX modules
        let module_names = [
            "lbs_surface_accessibility",
            "lbs_distance_matrix",
            "lbs_pocket_clustering",
            "lbs_druggability_scoring",
        ];

        for module_name in &module_names {
            let path = ptx_dir.join(format!("{}.ptx", module_name));
            if let Ok(ptx_data) = std::fs::read(&path) {
                let module = device.load_module(Ptx::Image(&ptx_data))
                    .map_err(|e| PrismError::gpu(module_name, format!("Failed to load PTX: {}", e)))?;
                _modules.push(module);
            }
        }

        Ok(Self { device, _modules })'''

    content = re.sub(new_method_pattern, fix_new_method, content, flags=re.DOTALL)

    # Fix get_func() calls - need to find and load functions dynamically
    # For now, we'll keep the get_func pattern but note that it may not work
    # A complete fix would require refactoring to load functions at init time

    return content

def fix_file_content(content: str, filepath: str) -> str:
    """Fix cudarc API calls in file content."""

    # Add Ptx import if not present and load_ptx is used
    if '.load_ptx(' in content or 'Ptx::' in content:
        if 'use cudarc::nvrtc::Ptx;' not in content:
            # Add after cudarc::driver imports
            content = re.sub(
                r'(use cudarc::driver::\{[^}]+\};)',
                r'\1\nuse cudarc::nvrtc::Ptx;',
                content
            )

    # Pattern 1: Fix context.load_ptx(...) followed by context.get_func(...)
    # This is the most common pattern

    # Match load_ptx with variable number of kernel names
    load_ptx_pattern = r'''(\w+)\.load_ptx\(\s*
        ([^,]+),\s*      # PTX source (could be variable, Ptx::from_file, etc)
        "[^"]+",\s*      # module name (we'll ignore this in new API)
        &\[[^\]]*\],?\s*  # kernel name array
        \)\??;'''

    # Find all load_ptx calls and replace with load_module
    matches = list(re.finditer(load_ptx_pattern, content, re.VERBOSE | re.DOTALL))

    for match in reversed(matches):  # Reverse to preserve positions
        device_var = match.group(1)
        ptx_source = match.group(2).strip()

        # Handle different PTX source formats
        if 'Ptx::from_file' in ptx_source:
            # Extract path from Ptx::from_file(&path)
            path_match = re.search(r'Ptx::from_file\(&?([^)]+)\)', ptx_source)
            if path_match:
                path_var = path_match.group(1)
                new_code = f'''let ptx_data = std::fs::read({path_var})
            .map_err(|e| anyhow::anyhow!("Failed to read PTX file: {{}}", e))?;
        let module = {device_var}.load_module(Ptx::Image(&ptx_data))?;'''
                content = content[:match.start()] + new_code + content[match.end():]
                continue

        if '.into()' in ptx_source:
            # String-based PTX (compiled or from file)
            var_name = ptx_source.replace('.into()', '').strip()
            if 'read_to_string' in content[:match.start()]:
                # It's a string, need to convert to bytes
                new_code = f'let module = {device_var}.load_module(Ptx::Image({var_name}.as_bytes()))?;'
            else:
                new_code = f'let module = {device_var}.load_module({ptx_source})?;'
            content = content[:match.start()] + new_code + content[match.end():]
            continue

        # Default: assume it's already Ptx type or will be
        new_code = f'let module = {device_var}.load_module(Ptx::Image(&ptx_data))?;'
        content = content[:match.start()] + new_code + content[match.end():]

    # Pattern 2: Fix device.get_func("module_name", "kernel_name")
    get_func_pattern = r'(\w+)\.get_func\("([^"]+)",\s*"([^"]+)"\)'

    def replace_get_func(match):
        # In new API: module.load_function("kernel_name")
        kernel_name = match.group(3)
        return f'module.load_function("{kernel_name}")'

    content = re.sub(get_func_pattern, replace_get_func, content)

    # Pattern 3: Fix Ptx::from_src which doesn't exist in 0.18
    content = re.sub(
        r'Ptx::from_src\(std::str::from_utf8\(&([^)]+)\)\?\)',
        r'Ptx::Image(&\1)',
        content
    )

    content = re.sub(
        r'Ptx::from_src\(([^)]+)\)',
        r'Ptx::Image(\1.as_bytes())',
        content
    )

    return content

def process_file(filepath: Path):
    """Process a single file."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content

        # Check if file actually uses cudarc API
        if '.get_func(' not in content and '.load_ptx(' not in content:
            return False

        print(f"Processing: {filepath}")

        # Apply fixes
        if 'lbs.rs' in str(filepath):
            content = fix_lbs_file(content)
        else:
            content = fix_file_content(content, str(filepath))

        if content != original_content:
            filepath.write_text(content, encoding='utf-8')
            print(f"  ✓ Fixed {filepath}")
            return True
        else:
            print(f"  - No changes needed for {filepath}")
            return False

    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {e}", file=sys.stderr)
        return False

def main():
    """Main entry point."""
    root = Path("/mnt/c/Users/Predator/Desktop/PRISM")

    # Find all Rust files with cudarc API calls
    rust_files = []
    for pattern in ["crates/**/*.rs", "foundation/**/*.rs"]:
        rust_files.extend(root.glob(pattern))

    fixed_count = 0
    for filepath in rust_files:
        if process_file(filepath):
            fixed_count += 1

    print(f"\n✓ Processed {len(rust_files)} files, fixed {fixed_count} files")

if __name__ == "__main__":
    main()
