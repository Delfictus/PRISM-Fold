# PRISM WHCR Parameter Struct Refactor Agent

You are a specialized agent for refactoring WHCR kernel parameters from tuple-based to struct-based launching.

## Mission
Eliminate the 12-parameter limit in WHCR kernels by implementing a GPU-side parameter struct pattern, enabling:
- Unlimited kernel parameters
- Restored `hotspot_mask` functionality
- New `reservoir_priorities` and `wavelet_coeffs` parameters
- Cleaner, more maintainable kernel launches

## Current Problem
File: `crates/prism-gpu/src/whcr.rs:807-820`
```rust
// CURRENT: Limited to 12 params, had to REMOVE hotspot_mask
let params = (
    &self.d_coloring,             // 1
    &self.d_adjacency_row_ptr,    // 2
    &self.d_adjacency_col_idx,    // 3
    &d_conflict_vertices,         // 4
    num_conflict_vertices as i32, // 5
    num_colors as i32,            // 6
    stress_buffer,                // 7
    persistence_buffer,           // 8
    belief_buffer,                // 9
    self.num_vertices as i32,     // 10
    move_deltas_f64,              // 11
    &self.d_best_colors,          // 12
);
```

## Solution: Parameter Struct Pattern

### 1. Define Rust-side Struct
File: `crates/prism-gpu/src/whcr.rs`
```rust
/// GPU-compatible parameter struct for WHCR kernels
/// Must match CUDA struct exactly (repr(C) for ABI compatibility)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WhcrKernelParams {
    // Graph topology
    pub coloring: u64,           // Device pointer
    pub row_ptr: u64,            // Device pointer
    pub col_idx: u64,            // Device pointer

    // Conflict vertices
    pub conflict_vertices: u64,  // Device pointer
    pub num_conflict_vertices: i32,
    pub num_colors: i32,

    // Geometry coupling
    pub stress_scores: u64,      // Device pointer
    pub persistence_scores: u64, // Device pointer
    pub belief_distribution: u64,// Device pointer
    pub hotspot_mask: u64,       // Device pointer - RESTORED!

    // Output
    pub move_deltas: u64,        // Device pointer
    pub best_colors: u64,        // Device pointer

    // Metadata
    pub num_vertices: i32,
    pub belief_num_colors: i32,

    // NEW: Advanced features
    pub reservoir_priorities: u64, // Device pointer
    pub wavelet_coeffs: u64,       // Device pointer
    pub wavelet_level: i32,

    // Padding for alignment
    pub _pad: i32,
}

impl WhcrKernelParams {
    pub fn new() -> Self {
        Self {
            coloring: 0,
            row_ptr: 0,
            col_idx: 0,
            conflict_vertices: 0,
            num_conflict_vertices: 0,
            num_colors: 0,
            stress_scores: 0,
            persistence_scores: 0,
            belief_distribution: 0,
            hotspot_mask: 0,
            move_deltas: 0,
            best_colors: 0,
            num_vertices: 0,
            belief_num_colors: 0,
            reservoir_priorities: 0,
            wavelet_coeffs: 0,
            wavelet_level: 0,
            _pad: 0,
        }
    }
}
```

### 2. Update CUDA Kernel
File: `kernels/ptx/whcr.cu` (source)
```cuda
struct WhcrKernelParams {
    // Graph topology
    int* coloring;
    int* row_ptr;
    int* col_idx;

    // Conflict vertices
    int* conflict_vertices;
    int num_conflict_vertices;
    int num_colors;

    // Geometry coupling
    double* stress_scores;
    double* persistence_scores;
    double* belief_distribution;
    int* hotspot_mask;  // RESTORED!

    // Output
    double* move_deltas;
    int* best_colors;

    // Metadata
    int num_vertices;
    int belief_num_colors;

    // Advanced features
    float* reservoir_priorities;
    float* wavelet_coeffs;
    int wavelet_level;
    int _pad;
};

__global__ void evaluate_moves_f64(WhcrKernelParams* params) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params->num_conflict_vertices) return;

    int v = params->conflict_vertices[tid];

    // Access all parameters via params->field
    int current_color = params->coloring[v];
    double stress = params->stress_scores[v];
    double persistence = params->persistence_scores[v];
    int is_hotspot = params->hotspot_mask ? params->hotspot_mask[v] : 0;

    // ... rest of kernel logic
}
```

### 3. Update Rust Launch Code
```rust
// Build params struct
let mut params = WhcrKernelParams::new();
params.coloring = self.d_coloring.device_ptr() as u64;
params.row_ptr = self.d_adjacency_row_ptr.device_ptr() as u64;
params.col_idx = self.d_adjacency_col_idx.device_ptr() as u64;
params.conflict_vertices = d_conflict_vertices.device_ptr() as u64;
params.num_conflict_vertices = num_conflict_vertices as i32;
params.num_colors = num_colors as i32;
params.stress_scores = stress_buffer.device_ptr() as u64;
params.persistence_scores = persistence_buffer.device_ptr() as u64;
params.belief_distribution = belief_buffer.device_ptr() as u64;
params.hotspot_mask = self.d_hotspot_mask.as_ref()
    .map(|h| h.device_ptr() as u64)
    .unwrap_or(0);
params.move_deltas = move_deltas_f64.device_ptr() as u64;
params.best_colors = self.d_best_colors.device_ptr() as u64;
params.num_vertices = self.num_vertices as i32;
params.reservoir_priorities = self.d_reservoir_priorities.as_ref()
    .map(|r| r.device_ptr() as u64)
    .unwrap_or(0);

// Upload params struct to GPU
let d_params = self.device.htod_copy(vec![params])?;

// Single-pointer launch!
let cfg = LaunchConfig::for_num_elems(num_conflict_vertices as u32);
unsafe { self.evaluate_moves_f64.clone().launch(cfg, (&d_params,))? };
```

## Files to Modify
1. `crates/prism-gpu/src/whcr.rs` - Add struct, update launches
2. `prism-gpu/src/kernels/whcr.cu` - Add struct, update kernels
3. `kernels/ptx/whcr.ptx` - Recompile after .cu changes
4. `crates/prism-whcr/src/lib.rs` - Update any direct calls

## Validation
```bash
# Recompile PTX
CUDA_HOME=/usr/local/cuda-12.6 nvcc --ptx -o kernels/ptx/whcr.ptx \
    prism-gpu/src/kernels/whcr.cu -arch=sm_86 --std=c++14

# Build and test
cargo build --release --features cuda
cargo test --features cuda -- whcr
```

## Output Format
Report:
1. Struct definition added (Rust + CUDA)
2. Launch sites updated (count)
3. PTX recompiled successfully
4. Tests passing
5. New parameters enabled (hotspot_mask, reservoir_priorities)
