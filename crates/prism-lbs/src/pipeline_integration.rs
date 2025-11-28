//! Minimal adapter hooks so prism-pipeline can call PRISM-LBS without tight coupling.
//!
//! The pipeline crate can import `prism_lbs::pipeline_integration::run_lbs` to execute
//! predictions inside existing orchestration.

use crate::{LbsConfig, Pocket, PrismLbs, ProteinStructure};
use anyhow::Result;
#[cfg(feature = "cuda")]
use prism_gpu::context::GpuContext;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Execute LBS prediction for a structure and return pockets.
pub fn run_lbs(structure: &ProteinStructure, config: &LbsConfig) -> Result<Vec<Pocket>> {
    #[cfg(feature = "cuda")]
    {
        let lbs = PrismLbs::new_with_gpu(config.clone(), None)?;
        return lbs.predict(structure);
    }

    #[cfg(not(feature = "cuda"))]
    {
        let lbs = PrismLbs::new(config.clone())?;
        lbs.predict(structure)
    }
}

/// Execute LBS prediction using an existing GPU context (preferred in orchestrated runs).
#[cfg(feature = "cuda")]
pub fn run_lbs_with_gpu(
    structure: &ProteinStructure,
    config: &LbsConfig,
    gpu_ctx: Option<Arc<GpuContext>>,
) -> Result<Vec<Pocket>> {
    let lbs = PrismLbs::new_with_gpu(config.clone(), gpu_ctx)?;
    lbs.predict(structure)
}
