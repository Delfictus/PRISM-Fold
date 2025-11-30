//! PRISM-LBS: Ligand Binding Site Prediction System
//!
//! Reframes binding site prediction as a graph coloring optimization problem
//! leveraging PRISM's quantum-neuromorphic-GPU architecture.
//!
//! ## Features
//! - GPU-accelerated pocket detection (4 CUDA kernels)
//! - FluxNet RL for druggability weight optimization
//! - GNN embeddings for enhanced pocket features
//! - Ensemble prediction with multi-method voting
//! - PDBBind training integration
//! - World-class allosteric site detection (4-stage pipeline)

pub mod allosteric;
pub mod features;
pub mod graph;
pub mod output;
pub mod phases;
pub mod pipeline_integration;
pub mod pocket;
pub mod scoring;
pub mod softspot;
pub mod structure;
pub mod training;
pub mod unified;
pub mod validation;

// Re-exports
pub use allosteric::{
    AllostericDetectionConfig, AllostericDetectionOutput, AllostericDetector, AllostericPocket,
    AllostericDetectionType, ConfidenceAssessment, CoverageGap, Domain, DomainInterface,
    HingeRegion, MultiModuleEvidence,
};
pub use graph::{GraphConfig, ProteinGraph, ProteinGraphBuilder};
pub use pocket::{Pocket, PocketDetector, PocketProperties};
pub use scoring::{DrugabilityClass, DruggabilityScore, DruggabilityScorer};
pub use softspot::{CrypticCandidate, CrypticConfidence, SoftSpotDetector};
pub use structure::{Atom, PdbParseOptions, ProteinStructure, Residue};
pub use unified::{
    Confidence, DetectionType, Evidence, UnifiedDetector, UnifiedOutput, UnifiedPocket,
};

use anyhow::Result;
#[cfg(feature = "cuda")]
use prism_gpu::context::{GpuContext, GpuSecurityConfig};
use serde::{Deserialize, Serialize};
use std::path::Path;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Main configuration for PRISM-LBS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LbsConfig {
    /// Graph construction parameters
    pub graph: GraphConfig,
    /// GPU acceleration toggle for LBS-specific kernels
    pub use_gpu: bool,
    /// Pocket geometry parameters
    pub geometry: pocket::GeometryConfig,

    /// Phase-specific configurations
    pub phase0: phases::SurfaceReservoirConfig,
    pub phase1: phases::PocketBeliefConfig,
    pub phase2: phases::PocketSamplingConfig,
    pub phase4: phases::CavityAnalysisConfig,
    pub phase6: phases::TopologicalPocketConfig,

    /// Scoring weights
    pub scoring: scoring::ScoringWeights,

    /// Output options
    pub output: OutputConfig,

    /// Maximum number of pockets to return
    pub top_n: usize,
}

impl Default for LbsConfig {
    fn default() -> Self {
        Self {
            graph: GraphConfig::default(),
            use_gpu: true,
            geometry: pocket::GeometryConfig::default(),
            phase0: phases::SurfaceReservoirConfig::default(),
            phase1: phases::PocketBeliefConfig::default(),
            phase2: phases::PocketSamplingConfig::default(),
            phase4: phases::CavityAnalysisConfig::default(),
            phase6: phases::TopologicalPocketConfig::default(),
            scoring: scoring::ScoringWeights::default(),
            output: OutputConfig::default(),
            top_n: 10,
        }
    }
}

impl LbsConfig {
    /// Load configuration from TOML file
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let config_str = std::fs::read_to_string(path)?;
        let config: LbsConfig = toml::from_str(&config_str)?;
        Ok(config)
    }
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub formats: Vec<OutputFormat>,
    pub include_pymol_script: bool,
    pub include_json: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            formats: vec![OutputFormat::Pdb, OutputFormat::Json],
            include_pymol_script: true,
            include_json: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Pdb,
    Json,
    Csv,
}

/// Main PRISM-LBS predictor
pub struct PrismLbs {
    config: LbsConfig,
    detector: PocketDetector,
    scorer: DruggabilityScorer,
    #[cfg(feature = "cuda")]
    gpu_ctx: Option<Arc<GpuContext>>,
}

impl PrismLbs {
    /// Create new predictor with given configuration
    pub fn new(config: LbsConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            return Self::new_with_gpu(config, None);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let detector = PocketDetector::new(config.clone())?;
            let scorer = DruggabilityScorer::new(config.scoring.clone());

            Ok(Self {
                config,
                detector,
                scorer,
            })
        }
    }

    /// Create predictor while reusing an existing GPU context (when CUDA is available).
    #[cfg(feature = "cuda")]
    pub fn new_with_gpu(config: LbsConfig, gpu_ctx: Option<Arc<GpuContext>>) -> Result<Self> {
        let detector = PocketDetector::new(config.clone())?;
        let scorer = DruggabilityScorer::new(config.scoring.clone());

        let gpu_ctx = if config.use_gpu {
            gpu_ctx.or_else(|| Self::init_gpu_context_from_env().ok())
        } else {
            None
        };

        Ok(Self {
            config,
            detector,
            scorer,
            gpu_ctx,
        })
    }

    #[cfg(feature = "cuda")]
    fn init_gpu_context_from_env() -> Result<Arc<GpuContext>, LbsError> {
        let device_id = std::env::var("PRISM_GPU_DEVICE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let ptx_dir = std::env::var("PRISM_PTX_DIR").unwrap_or_else(|_| "target/ptx".to_string());
        let ptx_path = Path::new(&ptx_dir);
        let security = GpuSecurityConfig::default();
        let ctx = GpuContext::new(device_id, security, ptx_path)
            .map_err(|e| LbsError::Gpu(format!("Failed to init GPU context: {}", e)))?;
        Ok(Arc::new(ctx))
    }

    /// Load configuration from TOML file
    pub fn from_config_file(path: &Path) -> Result<Self> {
        let config_str = std::fs::read_to_string(path)?;
        let config: LbsConfig = toml::from_str(&config_str)?;
        Self::new(config)
    }

    /// Predict binding sites for a protein structure
    pub fn predict(&self, structure: &ProteinStructure) -> Result<Vec<Pocket>> {
        log::info!("Starting PRISM-LBS prediction for {}", structure.title);

        // 1. Compute surface accessibility (GPU when available/configured)
        let mut structure = structure.clone();
        #[cfg(feature = "cuda")]
        {
            if self.config.use_gpu {
                if let Some(ctx) = &self.gpu_ctx {
                    let computer = structure::SurfaceComputer::default();
                    computer.compute_gpu(&mut structure, ctx)?;
                } else {
                    log::warn!(
                        "GPU requested for surface computation but no GPU context available; falling back to CPU"
                    );
                    structure.compute_surface_accessibility()?;
                }
            } else {
                structure.compute_surface_accessibility()?;
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            structure.compute_surface_accessibility()?;
        }

        // 2. Build protein graph
        let graph_builder = ProteinGraphBuilder::new(self.config.graph.clone());
        #[cfg(feature = "cuda")]
        let graph = if self.config.graph.use_gpu {
            graph_builder.build_with_gpu(&structure, self.gpu_ctx.as_deref())?
        } else {
            graph_builder.build(&structure)?
        };
        #[cfg(not(feature = "cuda"))]
        let graph = graph_builder.build(&structure)?;

        // 3. Run pocket detection through phases
        let mut pockets = self.detector.detect(&graph)?;

        // 4. Score pockets (GPU when available)
        #[cfg(feature = "cuda")]
        {
            if let Some(ctx) = &self.gpu_ctx {
                match self.scorer.score_batch_gpu(&pockets, ctx) {
                    Ok(scores) => {
                        for (pocket, score) in pockets.iter_mut().zip(scores) {
                            pocket.druggability_score = score;
                        }
                    }
                    Err(e) => {
                        log::warn!("GPU batch scoring failed, falling back to CPU: {}", e);
                        for pocket in &mut pockets {
                            pocket.druggability_score = self.scorer.score(pocket);
                        }
                    }
                }
            } else {
                for pocket in &mut pockets {
                    pocket.druggability_score = self.scorer.score(pocket);
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            for pocket in &mut pockets {
                pocket.druggability_score = self.scorer.score(pocket);
            }
        }

        // 5. Sort by druggability score
        pockets.sort_by(|a, b| {
            b.druggability_score
                .total
                .partial_cmp(&a.druggability_score.total)
                .unwrap()
        });

        // 6. Return top N
        pockets.truncate(self.config.top_n);

        log::info!("Found {} pockets", pockets.len());
        Ok(pockets)
    }

    /// Batch prediction for multiple structures
    pub async fn predict_batch(
        &self,
        structures: Vec<ProteinStructure>,
    ) -> Result<Vec<Vec<Pocket>>> {
        use rayon::prelude::*;

        structures.par_iter().map(|s| self.predict(s)).collect()
    }
}

/// Error types for PRISM-LBS
#[derive(Debug, thiserror::Error)]
pub enum LbsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("PDB parsing error: {0}")]
    PdbParse(String),

    #[error("Graph construction error: {0}")]
    GraphConstruction(String),

    #[error("Phase execution error: {0}")]
    PhaseExecution(String),

    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Training error: {0}")]
    Training(String),
}

impl From<anyhow::Error> for LbsError {
    fn from(err: anyhow::Error) -> Self {
        LbsError::Training(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LbsConfig::default();
        assert!(config.use_gpu);
        assert_eq!(config.top_n, 10);
    }
}
