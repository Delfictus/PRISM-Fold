//! ONNX Runtime inference with CUDA acceleration

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "ort")]
use ort::{
    execution_providers::{CUDAExecutionProvider, CPUExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};

/// GNN inference via ONNX Runtime
pub struct OnnxGnnRuntime {
    #[cfg(feature = "ort")]
    session: Session,
    #[cfg(not(feature = "ort"))]
    _phantom: std::marker::PhantomData<()>,
    use_gpu: bool,
    node_feature_dim: usize,
}

impl OnnxGnnRuntime {
    /// Load ONNX model with automatic GPU detection
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        #[cfg(feature = "ort")]
        {
            let model_path = model_path.as_ref();
            log::info!("Loading ONNX GNN from: {}", model_path.display());

            // Try CUDA first
            let (session, use_gpu) = match Self::create_cuda_session(model_path) {
                Ok(s) => {
                    log::info!("ONNX Runtime: CUDA EP enabled");
                    (s, true)
                }
                Err(e) => {
                    log::warn!("CUDA EP unavailable: {}. Using CPU.", e);
                    (Self::create_cpu_session(model_path)?, false)
                }
            };

            Ok(Self {
                session,
                use_gpu,
                node_feature_dim: 16,
            })
        }

        #[cfg(not(feature = "ort"))]
        {
            anyhow::bail!(
                "ONNX Runtime not available. Enable 'ort' feature to use ONNX inference."
            );
        }
    }

    #[cfg(feature = "ort")]
    fn create_cuda_session(path: &Path) -> Result<Session> {
        Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default()
                .with_device_id(0)
                .build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(path)
            .context("CUDA session creation failed")
    }

    #[cfg(feature = "ort")]
    fn create_cpu_session(path: &Path) -> Result<Session> {
        Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(path)
            .context("CPU session creation failed")
    }

    /// Run GNN inference
    pub fn predict(
        &self,
        node_features: ArrayView2<f32>,
        edge_index: ArrayView2<i64>,
    ) -> Result<GnnPrediction> {
        #[cfg(feature = "ort")]
        {
            let num_nodes = node_features.nrows();
            let num_edges = edge_index.ncols();

            log::debug!(
                "GNN inference: {} nodes, {} edges, GPU={}",
                num_nodes,
                num_edges,
                self.use_gpu
            );

            // Create input tensors
            let node_tensor = Value::from_array(node_features.to_owned())?;
            let edge_tensor = Value::from_array(edge_index.to_owned())?;

            // Run inference
            let outputs = self.session.run(ort::inputs![
                "node_features" => node_tensor,
                "edge_index" => edge_tensor,
            ]?)?;

            self.parse_outputs(&outputs, num_nodes)
        }

        #[cfg(not(feature = "ort"))]
        {
            anyhow::bail!("ONNX Runtime not available");
        }
    }

    #[cfg(feature = "ort")]
    fn parse_outputs(
        &self,
        outputs: &ort::SessionOutputs,
        num_nodes: usize,
    ) -> Result<GnnPrediction> {
        // Extract color logits
        let color_logits = outputs
            .get("color_logits")
            .context("Missing color_logits output")?
            .try_extract_tensor::<f32>()?;

        // Extract chromatic prediction
        let chromatic_pred = outputs
            .get("chromatic_number")
            .context("Missing chromatic_number output")?
            .try_extract_tensor::<f32>()?;

        // Softmax on color logits
        let color_probs = self.softmax_2d(color_logits.view());

        // Argmax for node colors
        let node_colors: Vec<usize> = color_probs
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();

        let chromatic_number = chromatic_pred[[0]].round() as usize;
        let confidence = self.compute_confidence(&color_probs);

        Ok(GnnPrediction {
            chromatic_number,
            node_colors,
            color_probabilities: color_probs,
            confidence,
            gpu_accelerated: self.use_gpu,
        })
    }

    #[cfg(feature = "ort")]
    fn softmax_2d(&self, logits: ArrayView2<f32>) -> Array2<f32> {
        let mut result = logits.to_owned();
        for mut row in result.rows_mut() {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            row.mapv_inplace(|x| (x - max).exp() / exp_sum);
        }
        result
    }

    #[cfg(feature = "ort")]
    fn compute_confidence(&self, probs: &Array2<f32>) -> f32 {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnnPrediction {
    pub chromatic_number: usize,
    pub node_colors: Vec<usize>,
    pub color_probabilities: Array2<f32>,
    pub confidence: f32,
    pub gpu_accelerated: bool,
}

#[cfg(all(test, feature = "ort"))]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_onnx_runtime_loading() {
        // Test that the module compiles when ort feature is enabled
        // Actual model loading requires a valid ONNX file
    }
}
