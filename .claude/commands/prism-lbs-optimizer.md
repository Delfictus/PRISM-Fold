# PRISM LBS Kernel Optimizer Agent

You are a specialized agent for optimizing LBS (Ligand Binding Site) GPU kernels.

## Mission
Optimize three critical LBS kernels to achieve production-grade performance:
1. **SASA Kernel**: O(N²) → O(N×27×neighbors) via spatial grid
2. **Pocket Clustering**: Race-free Jones-Plassmann algorithm
3. **Distance Matrix**: Batched/tiled computation

## Target Performance
| Kernel | Current | Target | Improvement |
|--------|---------|--------|-------------|
| SASA (10K atoms) | ~2000ms | <20ms | 100× |
| Pocket Clustering | Race conditions | Race-free | Correctness |
| Distance Matrix | O(N²) memory | Batched | <4GB VRAM |

---

## 1. SASA Kernel: Spatial Grid Optimization

### Current Problem
File: `crates/prism-gpu/src/kernels/lbs/surface_accessibility.cu`
- Every atom checks every other atom: O(N² × samples)
- For 10,000 atoms, 100M distance calculations per sample

### Solution: Uniform Spatial Grid
File: `crates/prism-gpu/src/kernels/lbs/surface_accessibility_v2.cu`

```cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#define GRID_RESOLUTION 8.0f  // Angstroms per cell (probe radius + max atom radius)
#define MAX_NEIGHBORS_PER_CELL 64
#define FIBONACCI_SAMPLES 92  // Golden angle sampling

// Spatial grid structure
struct SpatialGrid {
    int* cell_start;      // [num_cells] - Start index in sorted array
    int* cell_end;        // [num_cells] - End index in sorted array
    int* sorted_indices;  // [num_atoms] - Atom indices sorted by cell
    int3 dims;            // Grid dimensions
    float3 min_bounds;    // Minimum corner of grid
};

// Fibonacci sphere point generation (uniform distribution)
__device__ float3 fibonacci_sphere_point(
    float3 center, float radius, int sample_idx, int total_samples
) {
    float phi = CUDART_PI_F * (3.0f - sqrtf(5.0f));  // Golden angle
    float y = 1.0f - (2.0f * sample_idx + 1.0f) / total_samples;
    float r_xz = sqrtf(1.0f - y * y);
    float theta = phi * sample_idx;

    return make_float3(
        center.x + radius * r_xz * cosf(theta),
        center.y + radius * y,
        center.z + radius * r_xz * sinf(theta)
    );
}

// Convert position to cell index
__device__ int3 pos_to_cell(float3 pos, float3 min_bounds, int3 dims) {
    return make_int3(
        min((int)((pos.x - min_bounds.x) / GRID_RESOLUTION), dims.x - 1),
        min((int)((pos.y - min_bounds.y) / GRID_RESOLUTION), dims.y - 1),
        min((int)((pos.z - min_bounds.z) / GRID_RESOLUTION), dims.z - 1)
    );
}

__device__ int cell_to_idx(int3 cell, int3 dims) {
    return cell.x + cell.y * dims.x + cell.z * dims.x * dims.y;
}

__device__ bool valid_cell(int3 cell, int3 dims) {
    return cell.x >= 0 && cell.x < dims.x &&
           cell.y >= 0 && cell.y < dims.y &&
           cell.z >= 0 && cell.z < dims.z;
}

// Main SASA kernel with spatial grid
__global__ void compute_sasa_with_grid(
    const float4* __restrict__ atoms,      // x, y, z, radius
    const SpatialGrid grid,
    float* __restrict__ sasa_out,
    const int num_atoms,
    const int num_samples,
    const float probe_radius
) {
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_idx >= num_atoms) return;

    float4 atom = atoms[atom_idx];
    float3 center = make_float3(atom.x, atom.y, atom.z);
    float total_radius = atom.w + probe_radius;

    // Determine which cell this atom is in
    int3 my_cell = pos_to_cell(center, grid.min_bounds, grid.dims);

    int exposed_samples = 0;

    // Monte Carlo sampling with Fibonacci sphere
    for (int s = 0; s < num_samples; s++) {
        float3 sample_point = fibonacci_sphere_point(center, total_radius, s, num_samples);

        bool occluded = false;

        // Only search 27 neighboring cells (3x3x3 cube)
        for (int dx = -1; dx <= 1 && !occluded; dx++) {
            for (int dy = -1; dy <= 1 && !occluded; dy++) {
                for (int dz = -1; dz <= 1 && !occluded; dz++) {
                    int3 neighbor_cell = make_int3(
                        my_cell.x + dx,
                        my_cell.y + dy,
                        my_cell.z + dz
                    );

                    if (!valid_cell(neighbor_cell, grid.dims)) continue;

                    int cell_idx = cell_to_idx(neighbor_cell, grid.dims);
                    int start = grid.cell_start[cell_idx];
                    int end = grid.cell_end[cell_idx];

                    // Check atoms in this cell
                    for (int j = start; j < end && !occluded; j++) {
                        int other_idx = grid.sorted_indices[j];
                        if (other_idx == atom_idx) continue;

                        float4 other = atoms[other_idx];
                        float3 other_pos = make_float3(other.x, other.y, other.z);

                        float dx = sample_point.x - other_pos.x;
                        float dy = sample_point.y - other_pos.y;
                        float dz = sample_point.z - other_pos.z;
                        float dist_sq = dx*dx + dy*dy + dz*dz;

                        float occlusion_radius = other.w + probe_radius;
                        if (dist_sq < occlusion_radius * occlusion_radius) {
                            occluded = true;
                        }
                    }
                }
            }
        }

        if (!occluded) exposed_samples++;
    }

    // SASA = 4πr² × (exposed/total)
    float surface_area = 4.0f * CUDART_PI_F * total_radius * total_radius;
    sasa_out[atom_idx] = surface_area * (float)exposed_samples / (float)num_samples;
}

// Grid construction kernel
__global__ void compute_cell_assignments(
    const float4* __restrict__ atoms,
    int* __restrict__ cell_ids,
    const int num_atoms,
    const float3 min_bounds,
    const int3 dims
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_atoms) return;

    float4 atom = atoms[idx];
    int3 cell = pos_to_cell(make_float3(atom.x, atom.y, atom.z), min_bounds, dims);
    cell_ids[idx] = cell_to_idx(cell, dims);
}
```

---

## 2. Pocket Clustering: Jones-Plassmann Algorithm

### Current Problem
File: `crates/prism-gpu/src/kernels/lbs/pocket_clustering.cu`
- Greedy coloring with race conditions
- Non-deterministic results

### Solution: Jones-Plassmann with Random Priorities
File: `crates/prism-gpu/src/kernels/lbs/pocket_clustering_v2.cu`

```cuda
#define MAX_COLORS 256

// Jones-Plassmann parallel graph coloring
__global__ void jones_plassmann_iteration(
    const int* __restrict__ adj_list,
    const int* __restrict__ adj_offsets,
    const float* __restrict__ random_priorities,
    int* __restrict__ colors,
    int* __restrict__ active,  // 1 = uncolored, 0 = colored
    const int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices || active[v] == 0) return;

    float my_priority = random_priorities[v];

    // Check if I have highest priority among uncolored neighbors
    int start = adj_offsets[v];
    int end = adj_offsets[v + 1];

    bool is_local_max = true;
    bool neighbor_colors[MAX_COLORS];
    for (int i = 0; i < MAX_COLORS; i++) neighbor_colors[i] = false;

    for (int i = start; i < end; i++) {
        int neighbor = adj_list[i];

        if (colors[neighbor] >= 0) {
            // Neighbor already colored - record its color
            if (colors[neighbor] < MAX_COLORS) {
                neighbor_colors[colors[neighbor]] = true;
            }
        } else if (active[neighbor] && random_priorities[neighbor] > my_priority) {
            // Higher priority uncolored neighbor exists
            is_local_max = false;
        }
    }

    if (is_local_max) {
        // Find smallest available color (greedy)
        int my_color = 0;
        while (my_color < MAX_COLORS && neighbor_colors[my_color]) {
            my_color++;
        }

        colors[v] = my_color;
        active[v] = 0;  // Mark as colored
    }
}

// Count remaining uncolored vertices
__global__ void count_active(
    const int* __restrict__ active,
    int* __restrict__ count,
    const int num_vertices
) {
    __shared__ int block_count;
    if (threadIdx.x == 0) block_count = 0;
    __syncthreads();

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices && active[v]) {
        atomicAdd(&block_count, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, block_count);
    }
}
```

### Rust Driver
File: `crates/prism-gpu/src/lbs/clustering.rs`
```rust
pub fn parallel_coloring(&self, graph: &CsrGraph) -> Result<Vec<i32>> {
    let num_vertices = graph.num_vertices;

    // Generate random priorities (once)
    let priorities: Vec<f32> = (0..num_vertices)
        .map(|_| rand::random::<f32>())
        .collect();
    let d_priorities = self.device.htod_copy(priorities)?;

    // Initialize colors to -1 (uncolored)
    let mut d_colors = self.device.htod_copy(vec![-1i32; num_vertices])?;

    // Initialize active flags to 1
    let mut d_active = self.device.htod_copy(vec![1i32; num_vertices])?;

    // Iterate until all vertices colored
    let mut remaining = num_vertices;
    let mut iterations = 0;

    while remaining > 0 {
        iterations += 1;

        // Run Jones-Plassmann iteration
        let cfg = LaunchConfig::for_num_elems(num_vertices as u32);
        unsafe {
            self.jones_plassmann_kernel.clone().launch(cfg, (
                &graph.d_adj_list,
                &graph.d_adj_offsets,
                &d_priorities,
                &mut d_colors,
                &mut d_active,
                num_vertices as i32,
            ))?;
        }
        self.device.synchronize()?;

        // Count remaining
        remaining = self.count_active(&d_active)?;

        log::trace!("Jones-Plassmann iteration {}: {} remaining", iterations, remaining);
    }

    log::info!("Parallel coloring completed in {} iterations", iterations);

    self.device.dtoh_sync_copy(&d_colors)
}
```

---

## 3. Distance Matrix: Batched Tiled Computation

### Current Problem
- Full N×N matrix requires O(N²) memory
- For 10K atoms: 800MB just for distances

### Solution: Batched Tiles with Streaming
File: `crates/prism-gpu/src/kernels/lbs/distance_matrix_batched.cu`

```cuda
#define TILE_SIZE 32
#define BATCH_SIZE 4096

__global__ void distance_matrix_tiled(
    const float3* __restrict__ coords,
    float* __restrict__ distances,
    const int batch_offset_i,
    const int batch_offset_j,
    const int batch_size_i,
    const int batch_size_j,
    const int total_atoms
) {
    __shared__ float3 tile_i[TILE_SIZE];
    __shared__ float3 tile_j[TILE_SIZE];

    int local_i = threadIdx.y;
    int local_j = threadIdx.x;
    int global_i = batch_offset_i + blockIdx.y * TILE_SIZE + local_i;
    int global_j = batch_offset_j + blockIdx.x * TILE_SIZE + local_j;

    // Collaborative loading into shared memory
    if (global_i < total_atoms && local_j == 0) {
        tile_i[local_i] = coords[global_i];
    }
    if (global_j < total_atoms && local_i == 0) {
        tile_j[local_j] = coords[global_j];
    }
    __syncthreads();

    if (global_i < batch_offset_i + batch_size_i &&
        global_j < batch_offset_j + batch_size_j &&
        global_i < total_atoms && global_j < total_atoms)
    {
        float3 a = tile_i[local_i];
        float3 b = tile_j[local_j];

        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float dz = a.z - b.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);

        // Store in output batch matrix
        int out_i = global_i - batch_offset_i;
        int out_j = global_j - batch_offset_j;
        distances[out_i * batch_size_j + out_j] = dist;
    }
}

// For contact detection (sparse output)
__global__ void contact_detection_tiled(
    const float3* __restrict__ coords,
    const float* __restrict__ radii,
    int2* __restrict__ contacts,
    int* __restrict__ contact_count,
    const int max_contacts,
    const float contact_threshold,
    const int batch_offset_i,
    const int batch_offset_j,
    const int total_atoms
) {
    __shared__ float3 tile_i[TILE_SIZE];
    __shared__ float3 tile_j[TILE_SIZE];
    __shared__ float radii_i[TILE_SIZE];
    __shared__ float radii_j[TILE_SIZE];

    int local_i = threadIdx.y;
    int local_j = threadIdx.x;
    int global_i = batch_offset_i + blockIdx.y * TILE_SIZE + local_i;
    int global_j = batch_offset_j + blockIdx.x * TILE_SIZE + local_j;

    // Load tiles
    if (global_i < total_atoms && local_j == 0) {
        tile_i[local_i] = coords[global_i];
        radii_i[local_i] = radii[global_i];
    }
    if (global_j < total_atoms && local_i == 0) {
        tile_j[local_j] = coords[global_j];
        radii_j[local_j] = radii[global_j];
    }
    __syncthreads();

    if (global_i < total_atoms && global_j < total_atoms && global_i < global_j) {
        float3 a = tile_i[local_i];
        float3 b = tile_j[local_j];

        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float dz = a.z - b.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);

        float threshold = radii_i[local_i] + radii_j[local_j] + contact_threshold;

        if (dist < threshold) {
            int idx = atomicAdd(contact_count, 1);
            if (idx < max_contacts) {
                contacts[idx] = make_int2(global_i, global_j);
            }
        }
    }
}
```

---

## Files to Create/Modify
1. `crates/prism-gpu/src/kernels/lbs/surface_accessibility_v2.cu` - New
2. `crates/prism-gpu/src/kernels/lbs/pocket_clustering_v2.cu` - New
3. `crates/prism-gpu/src/kernels/lbs/distance_matrix_batched.cu` - New
4. `crates/prism-gpu/src/lbs/mod.rs` - Rust wrappers
5. `crates/prism-gpu/src/lbs/sasa.rs` - SASA Rust driver
6. `crates/prism-gpu/src/lbs/clustering.rs` - Clustering driver

## Validation
```bash
# Compile new PTX
for kernel in surface_accessibility_v2 pocket_clustering_v2 distance_matrix_batched; do
    nvcc --ptx -o kernels/ptx/lbs_${kernel}.ptx \
        crates/prism-gpu/src/kernels/lbs/${kernel}.cu \
        -arch=sm_86 --std=c++14
done

# Benchmark
cargo bench --package prism-lbs -- sasa
cargo bench --package prism-lbs -- clustering
```

## Output Format
Report:
1. Kernels implemented (3)
2. PTX compiled successfully
3. Performance improvement per kernel
4. Memory reduction achieved
5. Correctness verified (no race conditions)
