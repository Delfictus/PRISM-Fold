//! GPU-Accelerated Chromatic Graph Coloring
//!
//! Implements high-performance graph coloring on NVIDIA GPUs using cudarc.
//! Accelerates adjacency matrix construction, conflict detection, and DSATUR heuristic.

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashSet;
use std::sync::Arc;

/// GPU-accelerated chromatic coloring
pub struct GpuChromaticColoring {
    /// CUDA context
    context: Arc<CudaContext>,
    /// CUDA stream
    stream: Arc<CudaStream>,
    /// Number of colors used
    num_colors: usize,
    /// Color assignment (CPU)
    coloring: Vec<usize>,
    /// Adjacency matrix on GPU (packed as u8 for memory efficiency)
    gpu_adjacency: CudaSlice<u8>,
    /// Graph size
    num_vertices: usize,
    /// Conflict count
    conflict_count: usize,
}

impl GpuChromaticColoring {
    /// Create new GPU-accelerated chromatic coloring
    pub fn new_adaptive(coupling_matrix: &Array2<Complex64>, target_colors: usize) -> Result<Self> {
        let context = CudaContext::new(0).context("Failed to initialize CUDA device")?;
        let stream = context.default_stream();

        let threshold = Self::find_optimal_threshold_gpu(&context, &stream, coupling_matrix, target_colors)?;

        Self::new(coupling_matrix, target_colors, threshold, context, stream)
    }

    /// Create with explicit threshold
    pub fn new(
        coupling_matrix: &Array2<Complex64>,
        target_colors: usize,
        threshold: f64,
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        let n = coupling_matrix.nrows();

        if n == 0 {
            return Err(anyhow!("Empty coupling matrix"));
        }
        if target_colors == 0 {
            return Err(anyhow!("Target colors must be > 0"));
        }

        // Build adjacency matrix on GPU
        let gpu_adjacency = Self::build_adjacency_gpu(&context, &stream, coupling_matrix, threshold)?;

        // Compute coloring using Jones-Plassmann parallel algorithm on GPU
        let coloring = Self::jones_plassmann_gpu(&context, &stream, &gpu_adjacency, n, target_colors)?;

        let mut instance = Self {
            context: context.clone(),
            stream,
            num_colors: target_colors,
            coloring,
            gpu_adjacency,
            num_vertices: n,
            conflict_count: 0,
        };

        // Calculate conflicts on GPU
        instance.conflict_count = instance.count_conflicts_gpu()?;

        Ok(instance)
    }

    /// Build adjacency matrix on GPU (parallel)
    /// PRODUCTION-GRADE: Comprehensive error handling and validation
    fn build_adjacency_gpu(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        coupling_matrix: &Array2<Complex64>,
        threshold: f64,
    ) -> Result<CudaSlice<u8>> {
        let n = coupling_matrix.nrows();

        // Validate inputs
        if n != coupling_matrix.ncols() {
            return Err(anyhow!(
                "Coupling matrix must be square: got {}x{}",
                n,
                coupling_matrix.ncols()
            ));
        }

        if !(0.0..=1e10).contains(&threshold) {
            return Err(anyhow!(
                "Invalid threshold {}: must be in range [0, 1e10]",
                threshold
            ));
        }

        // Flatten coupling matrix and upload to GPU
        let mut coupling_flat = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let strength = coupling_matrix[[i, j]].norm();
                if !strength.is_finite() {
                    return Err(anyhow!(
                        "Invalid coupling strength at ({}, {}): {}",
                        i,
                        j,
                        strength
                    ));
                }
                coupling_flat.push(strength as f32);
            }
        }

        // Upload coupling matrix to GPU using stream-based API
        let gpu_coupling = stream
            .clone_htod(&coupling_flat)
            .context("Failed to upload coupling matrix to GPU")?;

        // Allocate adjacency matrix on GPU (packed as bits in u8 for efficiency)
        let adjacency_bytes = (n * n).div_ceil(8);
        let mut gpu_adjacency = stream
            .alloc_zeros::<u8>(adjacency_bytes)
            .context("Failed to allocate GPU adjacency matrix")?;

        // Load PTX kernel - try OUT_DIR first, then fallback to target/ptx/
        let ptx = if let Ok(out_dir) = std::env::var("OUT_DIR") {
            let ptx_path = std::path::Path::new(&out_dir).join("graph_coloring.ptx");
            if ptx_path.exists() {
                std::fs::read_to_string(&ptx_path)
                    .map_err(|e| anyhow!("Failed to load PTX from {:?}: {}", ptx_path, e))?
            } else {
                // Fallback to runtime location
                let runtime_path = std::path::Path::new("target/ptx/graph_coloring.ptx");
                std::fs::read_to_string(runtime_path).map_err(|e| {
                    anyhow!(
                        "Failed to load PTX from {:?}: {}. Run: cargo build --release",
                        runtime_path,
                        e
                    )
                })?
            }
        } else {
            // Runtime: use known location
            let runtime_path = std::path::Path::new("target/ptx/graph_coloring.ptx");
            std::fs::read_to_string(runtime_path).map_err(|e| {
                anyhow!(
                    "Failed to load PTX from {:?}: {}. Run: cargo build --release",
                    runtime_path,
                    e
                )
            })?
        };

        if ptx.is_empty() || !ptx.contains("build_adjacency") {
            return Err(anyhow!(
                "Invalid PTX file: missing build_adjacency kernel. Rebuild with: cargo clean && cargo build"
            ));
        }

        // Load PTX module and get kernel
        let ptx_parsed = Ptx::from_src(&ptx);
        let module = context
            .load_module(ptx_parsed)
            .context("Failed to load graph_coloring PTX module - check PTX and CUDA driver")?;

        let build_adjacency = module
            .load_function("build_adjacency")
            .context("Failed to get build_adjacency kernel function")?;

        let cfg = LaunchConfig::for_num_elems((n * n) as u32);

        let threshold_f32 = threshold as f32;
        let n_u32 = n as u32;

        // Launch kernel using stream
        unsafe {
            stream
                .launch_builder(&build_adjacency)
                .arg(&gpu_coupling)
                .arg(&threshold_f32)
                .arg(&mut gpu_adjacency)
                .arg(&n_u32)
                .launch(cfg)
                .context("GPU kernel launch failed - check CUDA runtime")?;
        }

        context
            .synchronize()
            .context("GPU synchronization failed after adjacency construction")?;

        Ok(gpu_adjacency)
    }

    /// Download adjacency matrix from GPU to CPU
    /// PRODUCTION-GRADE: Validation and symmetry enforcement
    fn download_adjacency(
        _context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        gpu_adjacency: &CudaSlice<u8>,
        n: usize,
    ) -> Result<Array2<bool>> {
        let adjacency_bytes = (n * n).div_ceil(8);

        let packed = stream
            .clone_dtoh(gpu_adjacency)
            .context("Failed to download adjacency matrix from GPU")?;

        if packed.len() != adjacency_bytes {
            return Err(anyhow!(
                "Adjacency buffer size mismatch: expected {} bytes, got {}",
                adjacency_bytes,
                packed.len()
            ));
        }

        let mut adjacency = Array2::from_elem((n, n), false);

        // PRODUCTION: Unpack bits from GPU byte array
        // Note: CUDA kernel writes to 32-bit words, so we need to read accordingly
        // Convert byte array to 32-bit words for correct bit extraction
        let words: Vec<u32> = packed
            .chunks(4)
            .map(|chunk| {
                let mut bytes = [0u8; 4];
                for (i, &b) in chunk.iter().enumerate() {
                    bytes[i] = b;
                }
                u32::from_le_bytes(bytes)
            })
            .collect();

        for i in 0..n {
            for j in 0..n {
                let bit_position = i * n + j;
                let word_idx = bit_position / 32;
                let bit_in_word = bit_position % 32;

                if word_idx < words.len() {
                    adjacency[[i, j]] = (words[word_idx] & (1u32 << bit_in_word)) != 0;
                }
            }
        }

        // PRODUCTION: Enforce symmetry for undirected graphs
        // GPU kernel should set both (i,j) and (j,i), but ensure consistency
        for i in 0..n {
            for j in (i + 1)..n {
                if adjacency[[i, j]] != adjacency[[j, i]] {
                    // Make symmetric by taking logical OR
                    let has_edge = adjacency[[i, j]] || adjacency[[j, i]];
                    adjacency[[i, j]] = has_edge;
                    adjacency[[j, i]] = has_edge;
                }
            }
            // Ensure no self-loops
            adjacency[[i, i]] = false;
        }

        Ok(adjacency)
    }

    /// Jones-Plassmann parallel graph coloring algorithm on GPU
    ///
    /// Iteratively finds independent sets and colors them in parallel
    fn jones_plassmann_gpu(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        gpu_adjacency: &CudaSlice<u8>,
        n: usize,
        max_colors: usize,
    ) -> Result<Vec<usize>> {
        // Load parallel coloring kernels
        let ptx = if let Ok(out_dir) = std::env::var("OUT_DIR") {
            let ptx_path = std::path::Path::new(&out_dir).join("parallel_coloring.ptx");
            if ptx_path.exists() {
                std::fs::read_to_string(&ptx_path).map_err(|e| {
                    anyhow!(
                        "Failed to load parallel_coloring PTX from {:?}: {}",
                        ptx_path,
                        e
                    )
                })?
            } else {
                let runtime_path = std::path::Path::new("target/ptx/parallel_coloring.ptx");
                std::fs::read_to_string(runtime_path)
                    .map_err(|e| anyhow!("Failed to load parallel_coloring PTX from {:?}: {}. Run: cargo build --release", runtime_path, e))?
            }
        } else {
            let runtime_path = std::path::Path::new("target/ptx/parallel_coloring.ptx");
            std::fs::read_to_string(runtime_path)
                .map_err(|e| anyhow!("Failed to load parallel_coloring PTX from {:?}: {}. Run: cargo build --release", runtime_path, e))?
        };

        // Load PTX module using cudarc 0.18.1 API
        let ptx_parsed = Ptx::from_src(&ptx);
        let module = context
            .load_module(ptx_parsed)
            .context("Failed to load parallel coloring PTX module")?;

        // Allocate GPU buffers
        let mut gpu_priorities = stream
            .alloc_zeros::<f32>(n)
            .context("Failed to allocate priorities buffer")?;
        let mut gpu_colors = stream
            .alloc_zeros::<u32>(n)
            .context("Failed to allocate colors buffer")?;
        let mut gpu_can_color = stream
            .alloc_zeros::<u32>(n)
            .context("Failed to allocate can_color buffer")?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Initialize priorities and colors
        let init_priorities = module
            .load_function("init_priorities")
            .context("Failed to get init_priorities kernel")?;

        let n_u32 = n as u32;
        let seed_u64 = seed;

        unsafe {
            stream
                .launch_builder(&init_priorities)
                .arg(&mut gpu_priorities)
                .arg(&mut gpu_colors)
                .arg(&n_u32)
                .arg(&seed_u64)
                .launch(cfg)
                .context("Failed to launch init_priorities kernel")?;
        }
        context.synchronize()?;

        // Load kernel functions once before loop
        let find_independent_set = module
            .load_function("find_independent_set")
            .context("Failed to get find_independent_set kernel")?;
        let color_independent_set = module
            .load_function("color_independent_set")
            .context("Failed to get color_independent_set kernel")?;
        let count_uncolored = module
            .load_function("count_uncolored")
            .context("Failed to get count_uncolored kernel")?;

        // Jones-Plassmann algorithm: iteratively color independent sets
        let max_iterations = n; // At most n iterations needed
        for _iteration in 0..max_iterations {
            // Find independent set (vertices with highest priority among uncolored neighbors)
            let n_u32 = n as u32;

            unsafe {
                stream
                    .launch_builder(&find_independent_set)
                    .arg(gpu_adjacency)
                    .arg(&gpu_priorities)
                    .arg(&gpu_colors)
                    .arg(&mut gpu_can_color)
                    .arg(&n_u32)
                    .launch(cfg)
                    .context("Failed to launch find_independent_set kernel")?;
            }
            context.synchronize()?;

            // Color the independent set with smallest available colors
            // Need shared memory for used_colors bit vector: (max_colors + 31) / 32 * 4 bytes
            let shared_mem_bytes = (max_colors.div_ceil(32) * 4) as u32;
            let cfg_with_shared = LaunchConfig {
                grid_dim: cfg.grid_dim,
                block_dim: cfg.block_dim,
                shared_mem_bytes,
            };

            let max_colors_u32 = max_colors as u32;

            unsafe {
                stream
                    .launch_builder(&color_independent_set)
                    .arg(gpu_adjacency)
                    .arg(&gpu_can_color)
                    .arg(&mut gpu_colors)
                    .arg(&n_u32)
                    .arg(&max_colors_u32)
                    .launch(cfg_with_shared)
                    .context("Failed to launch color_independent_set kernel")?;
            }
            context.synchronize()?;

            // Count how many vertices are still uncolored
            // Allocate fresh zero buffer each iteration
            let mut gpu_uncolored_count_iter = stream.alloc_zeros::<u32>(1)?;

            unsafe {
                stream
                    .launch_builder(&count_uncolored)
                    .arg(&gpu_colors)
                    .arg(&mut gpu_uncolored_count_iter)
                    .arg(&n_u32)
                    .launch(cfg)
                    .context("Failed to launch count_uncolored kernel")?;
            }
            context.synchronize()?;

            // Check if done
            let uncolored_count = stream
                .clone_dtoh(&gpu_uncolored_count_iter)
                .context("Failed to download uncolored count")?;
            if uncolored_count[0] == 0 {
                break; // All vertices colored
            }
        }

        // Download coloring from GPU
        let gpu_coloring = stream
            .clone_dtoh(&gpu_colors)
            .context("Failed to download coloring from GPU")?;

        // Convert to usize and validate
        let coloring: Vec<usize> = gpu_coloring
            .iter()
            .map(|&c| {
                if c == 0xFFFFFFFF {
                    return Err(anyhow!("Some vertices remain uncolored"));
                }
                if c >= max_colors as u32 {
                    return Err(anyhow!("Color {} exceeds max_colors {}", c, max_colors));
                }
                Ok(c as usize)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(coloring)
    }

    /// CPU-based greedy DSATUR coloring
    /// PRODUCTION-GRADE: Correct DSATUR implementation with validation
    fn greedy_coloring_cpu(adjacency: &Array2<bool>, max_colors: usize) -> Result<Vec<usize>> {
        let n = adjacency.nrows();

        // Edge cases
        if n == 0 {
            return Ok(Vec::new());
        }

        if max_colors == 0 {
            return Err(anyhow!("max_colors must be at least 1"));
        }

        let mut coloring = vec![usize::MAX; n];
        let mut uncolored: HashSet<usize> = (0..n).collect();

        // DSATUR Algorithm: Color vertex with highest saturation degree
        // Initialize by coloring first vertex (standard DSATUR)
        if !uncolored.is_empty() {
            coloring[0] = 0;
            uncolored.remove(&0);
        }

        while !uncolored.is_empty() {
            let v = Self::find_max_saturation_vertex(&uncolored, &coloring, adjacency);

            // Find smallest available color by checking neighbors
            let used_colors: HashSet<usize> = (0..n)
                .filter(|&u| {
                    // Only consider colored neighbors (exclude self)
                    u != v && coloring[u] != usize::MAX && adjacency[[v, u]]
                })
                .map(|u| coloring[u])
                .collect();

            let color = (0..max_colors)
                .find(|c| !used_colors.contains(c))
                .context("Not enough colors for valid coloring")?;

            coloring[v] = color;
            uncolored.remove(&v);
        }

        // PRODUCTION VALIDATION: Verify coloring is correct
        Self::validate_coloring(&coloring, adjacency)?;

        Ok(coloring)
    }

    /// Find vertex with maximum saturation degree (DSATUR heuristic)
    /// PRODUCTION-GRADE: Correct saturation and degree calculation
    fn find_max_saturation_vertex(
        uncolored: &HashSet<usize>,
        coloring: &[usize],
        adjacency: &Array2<bool>,
    ) -> usize {
        let n = coloring.len();
        let mut max_saturation = 0;
        let mut max_degree = 0;
        let mut best_vertex = *uncolored.iter().next().unwrap();

        for &v in uncolored {
            // Saturation degree: count distinct colors in neighborhood
            let saturation = (0..n)
                .filter(|&u| u != v && coloring[u] != usize::MAX && adjacency[[v, u]])
                .map(|u| coloring[u])
                .collect::<HashSet<_>>()
                .len();

            // Degree: count total neighbors (excluding self)
            let degree = (0..n).filter(|&u| u != v && adjacency[[v, u]]).count();

            // DSATUR: Select vertex with highest saturation
            // Break ties by highest degree
            if saturation > max_saturation || (saturation == max_saturation && degree > max_degree)
            {
                max_saturation = saturation;
                max_degree = degree;
                best_vertex = v;
            }
        }

        best_vertex
    }

    /// Validate that a coloring is correct (no adjacent vertices with same color)
    /// PRODUCTION-GRADE: Comprehensive validation with detailed error messages
    fn validate_coloring(coloring: &[usize], adjacency: &Array2<bool>) -> Result<()> {
        let n = coloring.len();

        if n != adjacency.nrows() || n != adjacency.ncols() {
            return Err(anyhow!(
                "Dimension mismatch: coloring has {} vertices but adjacency is {}x{}",
                n,
                adjacency.nrows(),
                adjacency.ncols()
            ));
        }

        // Check for uncolored vertices
        for (i, &color) in coloring.iter().enumerate() {
            if color == usize::MAX {
                return Err(anyhow!("Vertex {} is uncolored", i));
            }
        }

        // Check for conflicts (adjacent vertices with same color)
        let mut conflicts = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if adjacency[[i, j]] && coloring[i] == coloring[j] {
                    conflicts.push((i, j, coloring[i]));
                }
            }
        }

        if !conflicts.is_empty() {
            let conflict_list: Vec<String> = conflicts
                .iter()
                .take(5) // Show first 5 conflicts
                .map(|(i, j, c)| format!("({}-{}: color {})", i, j, c))
                .collect();

            return Err(anyhow!(
                "Invalid coloring: {} conflict(s) found. Examples: {}{}",
                conflicts.len(),
                conflict_list.join(", "),
                if conflicts.len() > 5 { ", ..." } else { "" }
            ));
        }

        // Verify adjacency matrix is symmetric (for undirected graphs)
        for i in 0..n {
            for j in (i + 1)..n {
                if adjacency[[i, j]] != adjacency[[j, i]] {
                    return Err(anyhow!(
                        "Adjacency matrix is not symmetric at ({}, {})",
                        i,
                        j
                    ));
                }
            }
        }

        // Verify no self-loops
        for i in 0..n {
            if adjacency[[i, i]] {
                return Err(anyhow!("Adjacency matrix has self-loop at vertex {}", i));
            }
        }

        Ok(())
    }

    /// Count conflicts on GPU (parallel)
    fn count_conflicts_gpu(&self) -> Result<usize> {
        let n = self.num_vertices;

        // Upload coloring to GPU
        let coloring_u32: Vec<u32> = self.coloring.iter().map(|&c| c as u32).collect();
        let gpu_coloring = self.stream.clone_htod(&coloring_u32)?;

        // Allocate output buffer for conflicts
        let mut gpu_conflicts = self.stream.alloc_zeros::<u32>(1)?;

        // Load PTX module for count_conflicts kernel
        let ptx = if let Ok(out_dir) = std::env::var("OUT_DIR") {
            let ptx_path = std::path::Path::new(&out_dir).join("graph_coloring.ptx");
            if ptx_path.exists() {
                std::fs::read_to_string(&ptx_path)
                    .map_err(|e| anyhow!("Failed to load PTX from {:?}: {}", ptx_path, e))?
            } else {
                let runtime_path = std::path::Path::new("target/ptx/graph_coloring.ptx");
                std::fs::read_to_string(runtime_path)
                    .map_err(|e| anyhow!("Failed to load PTX from {:?}: {}", runtime_path, e))?
            }
        } else {
            let runtime_path = std::path::Path::new("target/ptx/graph_coloring.ptx");
            std::fs::read_to_string(runtime_path)
                .map_err(|e| anyhow!("Failed to load PTX from {:?}: {}", runtime_path, e))?
        };

        let ptx_parsed = Ptx::from_src(&ptx);
        let module = self.context
            .load_module(ptx_parsed)
            .context("Failed to load graph_coloring PTX for count_conflicts")?;

        let count_conflicts = module
            .load_function("count_conflicts")
            .context("Failed to get count_conflicts kernel")?;

        let cfg = LaunchConfig::for_num_elems((n * n) as u32);
        let n_u32 = n as u32;

        unsafe {
            self.stream
                .launch_builder(&count_conflicts)
                .arg(&self.gpu_adjacency)
                .arg(&gpu_coloring)
                .arg(&mut gpu_conflicts)
                .arg(&n_u32)
                .launch(cfg)?;
        }

        self.context.synchronize()?;

        // Download result
        let conflicts = self.stream.clone_dtoh(&gpu_conflicts)?;
        Ok(conflicts[0] as usize)
    }

    /// Find optimal threshold using GPU-accelerated binary search
    fn find_optimal_threshold_gpu(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        coupling_matrix: &Array2<Complex64>,
        target_colors: usize,
    ) -> Result<f64> {
        let n = coupling_matrix.nrows();
        if n == 0 {
            return Ok(0.0);
        }

        // Collect coupling strengths
        let mut strengths: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let strength = coupling_matrix[[i, j]].norm();
                if strength > 1e-9 {
                    strengths.push(strength);
                }
            }
        }

        if strengths.is_empty() {
            // Empty graph (no edges) - any threshold works, use 1.0
            // This will result in a graph with 0 edges, which is 1-colorable
            return Ok(1.0);
        }

        strengths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        strengths.dedup();

        // Binary search for optimal threshold
        let mut low_idx = 0;
        let mut high_idx = strengths.len() - 1;
        let mut best_threshold = strengths[high_idx];

        while low_idx <= high_idx {
            let mid_idx = low_idx + (high_idx - low_idx) / 2;
            let mid_threshold = strengths[mid_idx];

            // Build graph with this threshold (on GPU)
            let gpu_adjacency = Self::build_adjacency_gpu(context, stream, coupling_matrix, mid_threshold)?;
            let cpu_adjacency = Self::download_adjacency(context, stream, &gpu_adjacency, n)?;

            // Test k-colorability
            if let Ok(_coloring) = Self::greedy_coloring_cpu(&cpu_adjacency, target_colors) {
                best_threshold = mid_threshold;
                if mid_idx == 0 {
                    break;
                }
                high_idx = mid_idx - 1;
            } else {
                low_idx = mid_idx + 1;
            }
        }

        Ok(best_threshold)
    }

    /// Verify coloring is valid
    pub fn verify_coloring(&self) -> bool {
        self.conflict_count == 0
    }

    /// Get color assignment
    pub fn get_coloring(&self) -> &[usize] {
        &self.coloring
    }

    /// Get conflict count
    pub fn get_conflict_count(&self) -> usize {
        self.conflict_count
    }

    /// Get number of colors
    pub fn num_colors(&self) -> usize {
        self.num_colors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_coloring_basic() {
        // Skip if no GPU available
        if CudaContext::new(0).is_err() {
            return;
        }

        let mut coupling = Array2::zeros((4, 4));
        coupling[[0, 1]] = Complex64::new(1.0, 0.0);
        coupling[[1, 0]] = Complex64::new(1.0, 0.0);
        coupling[[1, 2]] = Complex64::new(1.0, 0.0);
        coupling[[2, 1]] = Complex64::new(1.0, 0.0);
        coupling[[2, 3]] = Complex64::new(1.0, 0.0);
        coupling[[3, 2]] = Complex64::new(1.0, 0.0);

        let coloring = GpuChromaticColoring::new_adaptive(&coupling, 2).unwrap();
        assert!(coloring.verify_coloring());
    }
}
