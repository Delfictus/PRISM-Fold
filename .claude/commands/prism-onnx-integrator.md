# PRISM ONNX GNN Integrator Agent

You are a specialized agent for wiring the trained GNN model (gnn_model.onnx) into PRISM via ONNX Runtime.

## Mission
Replace the placeholder GNN implementation in `prism-gnn/src/models.rs` with real ONNX Runtime inference using:
- CUDA Execution Provider for GPU acceleration
- TensorRT optimization where available
- Proper input/output tensor handling
- Integration with LBS pocket detection

## Current State
- Trained model: `models/gnn/gnn_model.onnx` (440 KB)
- Model weights: `models/gnn/gnn_model.onnx.data` (5.4 MB)
- Placeholder: `crates/prism-gnn/src/models.rs:381-436` (OnnxGnn struct - not functional)

## Solution: ONNX Runtime Integration

### 1. Add Dependencies
File: `crates/prism-gnn/Cargo.toml`
```toml
[dependencies]
ort = { version = "2.0", features = ["cuda", "tensorrt"] }
ndarray = "0.16"
half = "2.4"  # For FP16 support

[features]
default = []
cuda = ["ort/cuda"]
tensorrt = ["ort/tensorrt"]
```

### 2. Implement ONNX Runtime Wrapper
File: `crates/prism-gnn/src/onnx_runtime.rs`
```rust
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, ArrayView2};
use ort::{
    execution_providers::{CUDAExecutionProvider, CPUExecutionProvider},
    session::{Session, SessionBuilder},
    tensor::OrtOwnedTensor,
    Environment, GraphOptimizationLevel, LoggingLevel,
};
use std::path::Path;
use std::sync::Arc;

/// ONNX Runtime environment (singleton)
static ORT_ENV: std::sync::OnceLock<Arc<Environment>> = std::sync::OnceLock::new();

fn get_ort_env() -> Result<Arc<Environment>> {
    if let Some(env) = ORT_ENV.get() {
        return Ok(env.clone());
    }

    let env = Environment::builder()
        .with_name("PRISM-GNN")
        .with_log_level(LoggingLevel::Warning)
        .build()?;

    let env = Arc::new(env);
    let _ = ORT_ENV.set(env.clone());
    Ok(env)
}

/// GNN inference via ONNX Runtime with CUDA acceleration
pub struct OnnxGnnRuntime {
    session: Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
    node_feature_dim: usize,
    max_nodes: usize,
    use_gpu: bool,
}

impl OnnxGnnRuntime {
    /// Load ONNX model with automatic GPU detection
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let env = get_ort_env()?;
        let model_path = model_path.as_ref();

        log::info!("Loading ONNX GNN model from: {}", model_path.display());

        // Try CUDA first, fall back to CPU
        let (session, use_gpu) = match Self::create_cuda_session(&env, model_path) {
            Ok(session) => {
                log::info!("ONNX Runtime: Using CUDA Execution Provider");
                (session, true)
            }
            Err(e) => {
                log::warn!("CUDA EP unavailable ({}), falling back to CPU", e);
                let session = Self::create_cpu_session(&env, model_path)?;
                (session, false)
            }
        };

        // Extract model metadata
        let input_names: Vec<String> = session
            .inputs
            .iter()
            .map(|i| i.name.clone())
            .collect();
        let output_names: Vec<String> = session
            .outputs
            .iter()
            .map(|o| o.name.clone())
            .collect();

        log::info!("ONNX inputs: {:?}", input_names);
        log::info!("ONNX outputs: {:?}", output_names);

        Ok(Self {
            session,
            input_names,
            output_names,
            node_feature_dim: 16, // From model architecture
            max_nodes: 10_000,
            use_gpu,
        })
    }

    fn create_cuda_session(env: &Environment, path: &Path) -> Result<Session> {
        SessionBuilder::new(env)?
            .with_execution_providers([
                CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .with_arena_extend_strategy(ort::ArenaExtendStrategy::NextPowerOfTwo)
                    .with_memory_limit(4 * 1024 * 1024 * 1024) // 4GB
                    .build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(path)
            .context("Failed to load ONNX model with CUDA EP")
    }

    fn create_cpu_session(env: &Environment, path: &Path) -> Result<Session> {
        SessionBuilder::new(env)?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(path)
            .context("Failed to load ONNX model with CPU EP")
    }

    /// Run GNN inference
    pub fn predict(
        &self,
        node_features: ArrayView2<f32>,
        edge_index: ArrayView2<i64>,
    ) -> Result<GnnPrediction> {
        let num_nodes = node_features.nrows();
        let num_edges = edge_index.ncols();

        log::debug!(
            "GNN inference: {} nodes, {} edges, GPU={}",
            num_nodes, num_edges, self.use_gpu
        );

        // Prepare inputs
        let node_features_tensor = node_features.to_owned();
        let edge_index_tensor = edge_index.to_owned();

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "node_features" => node_features_tensor,
            "edge_index" => edge_index_tensor,
        ]?)?;

        // Parse outputs based on model architecture
        self.parse_outputs(&outputs, num_nodes)
    }

    fn parse_outputs(
        &self,
        outputs: &ort::SessionOutputs,
        num_nodes: usize,
    ) -> Result<GnnPrediction> {
        // Color logits: [num_nodes, max_colors]
        let color_logits: OrtOwnedTensor<f32, _> = outputs
            .get("color_logits")
            .context("Missing 'color_logits' output")?
            .try_extract_tensor()?;

        // Chromatic number prediction: scalar
        let chromatic_pred: OrtOwnedTensor<f32, _> = outputs
            .get("chromatic_number")
            .context("Missing 'chromatic_number' output")?
            .try_extract_tensor()?;

        // Convert to prediction struct
        let color_probs = self.softmax_2d(color_logits.view());
        let chromatic_number = chromatic_pred[[0]].round() as usize;
        let confidence = self.compute_confidence(&color_probs);

        // Extract per-node predictions
        let node_colors: Vec<usize> = color_probs
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();

        Ok(GnnPrediction {
            chromatic_number,
            node_colors,
            color_probabilities: color_probs,
            confidence,
            gpu_accelerated: self.use_gpu,
        })
    }

    fn softmax_2d(&self, logits: ArrayView2<f32>) -> Array2<f32> {
        let mut result = logits.to_owned();
        for mut row in result.rows_mut() {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            row.mapv_inplace(|x| (x - max).exp() / exp_sum);
        }
        result
    }

    fn compute_confidence(&self, probs: &Array2<f32>) -> f32 {
        // Average max probability across all nodes
        probs
            .rows()
            .into_iter()
            .map(|row| row.iter().cloned().fold(0.0f32, f32::max))
            .sum::<f32>()
            / probs.nrows() as f32
    }

    pub fn is_gpu_enabled(&self) -> bool {
        self.use_gpu
    }
}

/// GNN prediction results
#[derive(Debug, Clone)]
pub struct GnnPrediction {
    /// Predicted chromatic number
    pub chromatic_number: usize,

    /// Per-node color assignments (argmax of probabilities)
    pub node_colors: Vec<usize>,

    /// Full probability distribution [num_nodes, num_colors]
    pub color_probabilities: Array2<f32>,

    /// Overall prediction confidence (0.0-1.0)
    pub confidence: f32,

    /// Whether GPU was used for inference
    pub gpu_accelerated: bool,
}
```

### 3. Wire into LBS Pipeline
File: `crates/prism-lbs/src/pocket/detector.rs`
```rust
use prism_gnn::{OnnxGnnRuntime, GnnPrediction};

pub struct PocketDetector {
    gnn: OnnxGnnRuntime,
    config: LbsConfig,
}

impl PocketDetector {
    pub fn new(config: &LbsConfig) -> Result<Self> {
        let model_path = config.gnn_model_path.as_ref()
            .ok_or_else(|| anyhow!("GNN model path not specified"))?;

        let gnn = OnnxGnnRuntime::load(model_path)?;

        Ok(Self {
            gnn,
            config: config.clone(),
        })
    }

    pub fn detect_pockets(&self, protein: &Protein) -> Result<Vec<Pocket>> {
        // Build graph representation from protein
        let (node_features, edge_index) = self.build_protein_graph(protein)?;

        // GNN inference (GPU-accelerated via ONNX CUDA EP)
        let prediction = self.gnn.predict(
            node_features.view(),
            edge_index.view(),
        )?;

        log::info!(
            "GNN prediction: chromatic={}, confidence={:.2}%, GPU={}",
            prediction.chromatic_number,
            prediction.confidence * 100.0,
            prediction.gpu_accelerated
        );

        // Convert GNN coloring to pocket regions
        self.colorings_to_pockets(protein, &prediction)
    }

    fn build_protein_graph(&self, protein: &Protein) -> Result<(Array2<f32>, Array2<i64>)> {
        // Build node features from residue properties
        let num_residues = protein.residues.len();
        let mut node_features = Array2::zeros((num_residues, 16));

        for (i, residue) in protein.residues.iter().enumerate() {
            node_features[[i, 0]] = residue.hydrophobicity();
            node_features[[i, 1]] = residue.charge();
            node_features[[i, 2]] = residue.polar();
            node_features[[i, 3]] = residue.aromatic();
            node_features[[i, 4]] = residue.sasa();
            // ... additional features
        }

        // Build edge index from contact map
        let mut edges = Vec::new();
        for (i, res_i) in protein.residues.iter().enumerate() {
            for (j, res_j) in protein.residues.iter().enumerate().skip(i + 1) {
                if res_i.distance_to(res_j) < 8.0 { // Contact threshold
                    edges.push((i as i64, j as i64));
                    edges.push((j as i64, i as i64)); // Undirected
                }
            }
        }

        let num_edges = edges.len();
        let mut edge_index = Array2::zeros((2, num_edges));
        for (idx, (src, dst)) in edges.iter().enumerate() {
            edge_index[[0, idx]] = *src;
            edge_index[[1, idx]] = *dst;
        }

        Ok((node_features, edge_index))
    }

    fn colorings_to_pockets(
        &self,
        protein: &Protein,
        prediction: &GnnPrediction,
    ) -> Result<Vec<Pocket>> {
        // Group residues by color assignment
        let mut color_groups: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();

        for (residue_idx, &color) in prediction.node_colors.iter().enumerate() {
            color_groups.entry(color).or_default().push(residue_idx);
        }

        // Convert groups to pockets
        let mut pockets = Vec::new();
        for (color_id, residue_indices) in color_groups {
            if residue_indices.len() >= self.config.min_pocket_size {
                let pocket = Pocket::from_residues(
                    protein,
                    &residue_indices,
                    color_id,
                    prediction.confidence,
                );
                pockets.push(pocket);
            }
        }

        // Sort by druggability score
        pockets.sort_by(|a, b| {
            b.druggability_score
                .partial_cmp(&a.druggability_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(pockets)
    }
}
```

## Files to Modify
1. `crates/prism-gnn/Cargo.toml` - Add ort, ndarray
2. `crates/prism-gnn/src/onnx_runtime.rs` - New file
3. `crates/prism-gnn/src/lib.rs` - Export module
4. `crates/prism-gnn/src/models.rs` - Replace placeholder OnnxGnn
5. `crates/prism-lbs/src/pocket/detector.rs` - Wire GNN

## Validation
```bash
# Test ONNX loading
cargo test --package prism-gnn -- onnx

# Test with actual model
RUST_LOG=debug cargo run --release --features cuda -- \
    --mode biomolecular \
    --input test_protein.pdb

# Verify GPU usage
nvidia-smi  # Should show ONNX Runtime process
```

## Output Format
Report:
1. ONNX Runtime version integrated
2. CUDA/TensorRT EP status
3. Model loading successful
4. Inference latency (ms)
5. LBS pipeline integration complete
