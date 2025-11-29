//! GPU helpers for PRISM-LBS geometry and clustering.

use cudarc::driver::{CudaContext, CudaStream, CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use prism_core::PrismError;
use std::path::Path;
use std::sync::Arc;

/// GPU executor for LBS kernels (surface accessibility, distance matrix, clustering, scoring).
pub struct LbsGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    surface_func: CudaFunction,
    distance_func: CudaFunction,
    clustering_func: CudaFunction,
    scoring_func: CudaFunction,
}

impl LbsGpu {
    /// Load LBS PTX modules from `ptx_dir`. Expects:
    /// - lbs_surface_accessibility.ptx with `surface_accessibility_kernel`
    /// - lbs_distance_matrix.ptx with `distance_matrix_kernel`
    /// - lbs_pocket_clustering.ptx with `pocket_clustering_kernel`
    /// - lbs_druggability_scoring.ptx with `druggability_score_kernel`
    pub fn new(context: Arc<CudaContext>, ptx_dir: &Path) -> Result<Self, PrismError> {
        let stream = context.default_stream();

        // Load surface accessibility module
        let path = ptx_dir.join("lbs_surface_accessibility.ptx");
        let ptx_src = std::fs::read_to_string(&path)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", format!("Failed to read PTX: {}", e)))?;
        let surface_module = context.load_module(Ptx::from_src(ptx_src))
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", format!("Failed to load PTX: {}", e)))?;
        let surface_func = surface_module.load_function("surface_accessibility_kernel")
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", format!("Failed to load kernel: {}", e)))?;

        // Load distance matrix module
        let path = ptx_dir.join("lbs_distance_matrix.ptx");
        let ptx_src = std::fs::read_to_string(&path)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", format!("Failed to read PTX: {}", e)))?;
        let distance_module = context.load_module(Ptx::from_src(ptx_src))
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", format!("Failed to load PTX: {}", e)))?;
        let distance_func = distance_module.load_function("distance_matrix_kernel")
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", format!("Failed to load kernel: {}", e)))?;

        // Load pocket clustering module
        let path = ptx_dir.join("lbs_pocket_clustering.ptx");
        let ptx_src = std::fs::read_to_string(&path)
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", format!("Failed to read PTX: {}", e)))?;
        let clustering_module = context.load_module(Ptx::from_src(ptx_src))
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", format!("Failed to load PTX: {}", e)))?;
        let clustering_func = clustering_module.load_function("pocket_clustering_kernel")
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", format!("Failed to load kernel: {}", e)))?;

        // Load druggability scoring module
        let path = ptx_dir.join("lbs_druggability_scoring.ptx");
        let ptx_src = std::fs::read_to_string(&path)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", format!("Failed to read PTX: {}", e)))?;
        let scoring_module = context.load_module(Ptx::from_src(ptx_src))
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", format!("Failed to load PTX: {}", e)))?;
        let scoring_func = scoring_module.load_function("druggability_score_kernel")
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", format!("Failed to load kernel: {}", e)))?;

        Ok(Self {
            context,
            stream,
            surface_func,
            distance_func,
            clustering_func,
            scoring_func,
        })
    }

    /// Compute SASA and surface flags for atoms.
    pub fn surface_accessibility(
        &self,
        coords: &[[f32; 3]],
        radii: &[f32],
        samples: i32,
        probe_radius: f32,
    ) -> Result<(Vec<f32>, Vec<u8>), PrismError> {
        let n = coords.len();
        if radii.len() != n {
            return Err(PrismError::gpu(
                "lbs_surface_accessibility",
                "radii length mismatch",
            ));
        }
        let x: Vec<f32> = coords.iter().map(|c| c[0]).collect();
        let y: Vec<f32> = coords.iter().map(|c| c[1]).collect();
        let z: Vec<f32> = coords.iter().map(|c| c[2]).collect();

        let d_x = self.stream.clone_htod(&x)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let d_y = self.stream.clone_htod(&y)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let d_z = self.stream.clone_htod(&z)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let d_r = self.stream.clone_htod(radii)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let mut d_sasa = self.stream.alloc_zeros::<f32>(n)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let mut d_surface = self.stream.alloc_zeros::<u8>(n)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            self.stream
                .launch_builder(&self.surface_func)
                .arg(&d_x)
                .arg(&d_y)
                .arg(&d_z)
                .arg(&d_r)
                .arg(&(n as i32))
                .arg(&samples)
                .arg(&probe_radius)
                .arg(&mut d_sasa)
                .arg(&mut d_surface)
                .launch(cfg)
                .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        }

        let sasa = self.stream.clone_dtoh(&d_sasa)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let surface = self.stream.clone_dtoh(&d_surface)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        Ok((sasa, surface))
    }

    /// Compute full pairwise distance matrix (n x n).
    pub fn distance_matrix(&self, coords: &[[f32; 3]]) -> Result<Vec<f32>, PrismError> {
        let n = coords.len();
        let x: Vec<f32> = coords.iter().map(|c| c[0]).collect();
        let y: Vec<f32> = coords.iter().map(|c| c[1]).collect();
        let z: Vec<f32> = coords.iter().map(|c| c[2]).collect();

        let d_x = self.stream.clone_htod(&x)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;
        let d_y = self.stream.clone_htod(&y)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;
        let d_z = self.stream.clone_htod(&z)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;
        let mut d_out = self.stream.alloc_zeros::<f32>(n * n)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;

        let block = (16, 16, 1);
        let grid = (
            (n as u32 + block.0 - 1) / block.0,
            (n as u32 + block.1 - 1) / block.1,
            1,
        );
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.distance_func)
                .arg(&d_x)
                .arg(&d_y)
                .arg(&d_z)
                .arg(&(n as i32))
                .arg(&mut d_out)
                .launch(cfg)
                .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;
        }

        Ok(self.stream.clone_dtoh(&d_out)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?)
    }

    /// Greedy pocket clustering (graph coloring) on GPU.
    pub fn pocket_clustering(
        &self,
        row_ptr: &[i32],
        col_idx: &[i32],
        max_colors: i32,
    ) -> Result<Vec<i32>, PrismError> {
        let n = row_ptr.len().saturating_sub(1);
        let d_row = self.stream.clone_htod(row_ptr)
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?;
        let d_col = self.stream.clone_htod(col_idx)
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?;
        let mut d_colors = self.stream.alloc_zeros::<i32>(n)
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            self.stream
                .launch_builder(&self.clustering_func)
                .arg(&d_row)
                .arg(&d_col)
                .arg(&(n as i32))
                .arg(&max_colors)
                .arg(&mut d_colors)
                .launch(cfg)
                .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?;
        }

        Ok(self.stream.clone_dtoh(&d_colors)
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?)
    }

    /// GPU aggregation of druggability scores.
    pub fn druggability_score(
        &self,
        volume: &[f32],
        hydrophobicity: &[f32],
        enclosure: &[f32],
        depth: &[f32],
        hbond: &[f32],
        flexibility: &[f32],
        conservation: &[f32],
        topology: &[f32],
        weights: [f32; 8],
    ) -> Result<Vec<f32>, PrismError> {
        let n = volume.len();
        let inputs = [
            hydrophobicity.len(),
            enclosure.len(),
            depth.len(),
            hbond.len(),
            flexibility.len(),
            conservation.len(),
            topology.len(),
        ];
        if inputs.iter().any(|&l| l != n) {
            return Err(PrismError::gpu(
                "lbs_druggability_scoring",
                "input length mismatch",
            ));
        }

        let d_volume = self.stream.clone_htod(volume)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_hydro = self.stream.clone_htod(hydrophobicity)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_enclosure = self.stream.clone_htod(enclosure)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_depth = self.stream.clone_htod(depth)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_hbond = self.stream.clone_htod(hbond)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_flex = self.stream.clone_htod(flexibility)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_cons = self.stream.clone_htod(conservation)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_topo = self.stream.clone_htod(topology)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_weights = self.stream.clone_htod(&weights)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let mut d_out = self.stream.alloc_zeros::<f32>(n)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            self.stream
                .launch_builder(&self.scoring_func)
                .arg(&d_volume)
                .arg(&d_hydro)
                .arg(&d_enclosure)
                .arg(&d_depth)
                .arg(&d_hbond)
                .arg(&d_flex)
                .arg(&d_cons)
                .arg(&d_topo)
                .arg(&d_weights)
                .arg(&(n as i32))
                .arg(&mut d_out)
                .launch(cfg)
                .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        }

        Ok(self.stream.clone_dtoh(&d_out)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?)
    }
}
