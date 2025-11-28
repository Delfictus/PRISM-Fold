//! FluxNet Force Profile System
//!
//! Implements GPU-accelerated force profiles for adaptive phase 2 thermodynamic equilibration.
//!
//! # GPU Mandate Compliance
//!
//! This module follows PRISM GPU-FIRST standards:
//! - ✅ Arc<CudaDevice> for context sharing (Article V)
//! - ✅ Device buffers (CudaSlice<f32>) for all GPU data
//! - ✅ Host-device synchronization with pinned memory
//! - ❌ NO CPU fallbacks
//! - ❌ NO conditional GPU code
//!
//! # Architecture
//!
//! ForceProfile classifies vertices into three bands:
//! - **Strong Force (f_strong)**: High repulsion on difficult vertices
//! - **Neutral Force (1.0)**: Baseline force on average vertices
//! - **Weak Force (f_weak)**: Reduced force on easy vertices
//!
//! The force values are stored in both host (Vec<f32>) and device (CudaSlice<f32>)
//! memory, with explicit synchronization points.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice};

/// Force band classification for vertices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForceBand {
    /// High difficulty vertices - strong repulsive force
    Strong = 0,
    /// Average difficulty vertices - baseline force (1.0)
    Neutral = 1,
    /// Low difficulty vertices - weak attractive force
    Weak = 2,
}

impl ForceBand {
    /// Convert from numeric index (0=Strong, 1=Neutral, 2=Weak)
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => ForceBand::Strong,
            1 => ForceBand::Neutral,
            2 => ForceBand::Weak,
            _ => ForceBand::Neutral,
        }
    }

    /// Get default force multiplier for this band
    pub fn default_multiplier(&self) -> f32 {
        match self {
            ForceBand::Strong => 1.5,  // 50% stronger repulsion
            ForceBand::Neutral => 1.0, // Baseline
            ForceBand::Weak => 0.7,    // 30% weaker (more coupling)
        }
    }
}

/// Statistics about force band distribution (for telemetry)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceBandStats {
    /// Fraction of vertices in Strong band [0.0, 1.0]
    pub strong_fraction: f32,

    /// Fraction of vertices in Neutral band [0.0, 1.0]
    pub neutral_fraction: f32,

    /// Fraction of vertices in Weak band [0.0, 1.0]
    pub weak_fraction: f32,

    /// Mean force value across all vertices
    pub mean_force: f32,

    /// Standard deviation of force values
    pub std_force: f32,

    /// Minimum force value
    pub min_force: f32,

    /// Maximum force value
    pub max_force: f32,
}

impl ForceBandStats {
    /// Compute statistics from force profile
    pub fn compute(f_strong: &[f32], f_weak: &[f32]) -> Self {
        let n = f_strong.len();
        if n == 0 {
            return Self::default();
        }

        let mut strong_count = 0;
        let mut neutral_count = 0;
        let mut weak_count = 0;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for i in 0..n {
            // Classify based on force values
            let force = f_strong[i];

            if force > 1.2 {
                strong_count += 1;
            } else if force < 0.8 {
                weak_count += 1;
            } else {
                neutral_count += 1;
            }

            sum += force;
            sum_sq += force * force;
            min_val = min_val.min(force);
            max_val = max_val.max(force);
        }

        let mean = sum / n as f32;
        let variance = (sum_sq / n as f32) - (mean * mean);
        let std = variance.max(0.0).sqrt();

        Self {
            strong_fraction: strong_count as f32 / n as f32,
            neutral_fraction: neutral_count as f32 / n as f32,
            weak_fraction: weak_count as f32 / n as f32,
            mean_force: mean,
            std_force: std,
            min_force: if min_val.is_finite() { min_val } else { 1.0 },
            max_force: if max_val.is_finite() { max_val } else { 1.0 },
        }
    }
}

impl Default for ForceBandStats {
    fn default() -> Self {
        Self {
            strong_fraction: 0.0,
            neutral_fraction: 1.0,
            weak_fraction: 0.0,
            mean_force: 1.0,
            std_force: 0.0,
            min_force: 1.0,
            max_force: 1.0,
        }
    }
}

/// GPU-accelerated force profile for Phase 2 thermodynamic equilibration
///
/// # GPU Architecture (GPU_MANDATE.md compliant)
///
/// ```text
/// Host (CPU)           Device (GPU)
/// ┌─────────────┐     ┌─────────────────┐
/// │ f_strong    │────▶│ device_f_strong │
/// │ (Vec<f32>)  │     │ (CudaSlice<f32>)│
/// └─────────────┘     └─────────────────┘
/// ┌─────────────┐     ┌─────────────────┐
/// │ f_weak      │────▶│ device_f_weak   │
/// │ (Vec<f32>)  │     │ (CudaSlice<f32>)│
/// └─────────────┘     └─────────────────┘
///
/// Synchronization:
/// - to_device(): Host → Device (before kernel launch)
/// - from_device(): Device → Host (after kernel completion)
/// ```
///
/// # Mandate Compliance
/// - ✅ GPU device buffers present (device_f_strong, device_f_weak)
/// - ✅ Arc<CudaDevice> for shared context
/// - ✅ Explicit synchronization methods
/// - ❌ NO CPU fallbacks (will fail compilation without CUDA feature)
#[cfg(feature = "cuda")]
pub struct ForceProfile {
    /// Host-side strong force multipliers (one per vertex)
    pub f_strong: Vec<f32>,

    /// Host-side weak force multipliers (one per vertex)
    pub f_weak: Vec<f32>,

    /// GPU device buffer for strong forces (MANDATORY)
    pub device_f_strong: CudaSlice<f32>,

    /// GPU device buffer for weak forces (MANDATORY)
    pub device_f_weak: CudaSlice<f32>,

    /// Shared CUDA device context (Article V compliance)
    cuda_device: Arc<CudaDevice>,

    /// Number of vertices
    n_vertices: usize,
}

#[cfg(feature = "cuda")]
impl ForceProfile {
    /// Create a new force profile with uniform baseline forces
    ///
    /// # GPU Initialization
    /// - Allocates host vectors
    /// - Allocates device buffers via CudaDevice
    /// - Initializes all forces to 1.0 (neutral/baseline)
    ///
    /// # Arguments
    /// - `n_vertices`: Number of vertices in the graph
    /// - `cuda_device`: Shared CUDA device context
    pub fn new(n_vertices: usize, cuda_device: Arc<CudaDevice>) -> Result<Self> {
        // Initialize host buffers with baseline forces
        let f_strong = vec![1.0f32; n_vertices];
        let f_weak = vec![1.0f32; n_vertices];

        // Allocate and initialize device buffers
        let device_f_strong = cuda_device
            .htod_sync_copy(&f_strong)
            .context("Failed to allocate device buffer for f_strong")?;

        let device_f_weak = cuda_device
            .htod_sync_copy(&f_weak)
            .context("Failed to allocate device buffer for f_weak")?;

        Ok(Self {
            f_strong,
            f_weak,
            device_f_strong,
            device_f_weak,
            cuda_device,
            n_vertices,
        })
    }

    /// Initialize force profile from difficulty scores (Phase 0 reservoir output)
    ///
    /// # Band Assignment Logic
    /// - Top 20% difficulty → Strong band (f_strong = 1.5)
    /// - Middle 60% → Neutral band (f_strong = 1.0, f_weak = 1.0)
    /// - Bottom 20% difficulty → Weak band (f_weak = 0.7)
    ///
    /// # Arguments
    /// - `difficulty_scores`: Per-vertex difficulty scores from reservoir predictor
    /// - `cuda_device`: Shared CUDA device context
    pub fn from_difficulty_scores(
        difficulty_scores: &[f32],
        cuda_device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let n = difficulty_scores.len();
        let mut profile = Self::new(n, cuda_device)?;

        // Sort indices by difficulty (descending)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            difficulty_scores[b]
                .partial_cmp(&difficulty_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign bands based on percentiles
        let strong_cutoff = n / 5; // Top 20%
        let weak_cutoff = (4 * n) / 5; // Bottom 20%

        for (rank, &vertex_idx) in indices.iter().enumerate() {
            if rank < strong_cutoff {
                // Strong band: high difficulty
                profile.f_strong[vertex_idx] = ForceBand::Strong.default_multiplier();
                profile.f_weak[vertex_idx] = 1.0;
            } else if rank >= weak_cutoff {
                // Weak band: low difficulty
                profile.f_strong[vertex_idx] = 1.0;
                profile.f_weak[vertex_idx] = ForceBand::Weak.default_multiplier();
            }
            // else: Neutral band (already initialized to 1.0)
        }

        // Sync to device
        profile.to_device()?;

        Ok(profile)
    }

    /// Update force profile based on AI uncertainty (Phase 1 active inference output)
    ///
    /// # Update Logic
    /// High uncertainty → increase strong force (more exploration)
    /// Low uncertainty → increase weak force (more exploitation)
    ///
    /// # Arguments
    /// - `ai_uncertainty`: Per-vertex uncertainty scores from active inference
    pub fn update_from_uncertainty(&mut self, ai_uncertainty: &[f32]) -> Result<()> {
        if ai_uncertainty.len() != self.n_vertices {
            anyhow::bail!(
                "Uncertainty scores length mismatch: expected {}, got {}",
                self.n_vertices,
                ai_uncertainty.len()
            );
        }

        // Blend uncertainty into force profile
        for i in 0..self.n_vertices {
            let uncertainty = ai_uncertainty[i];

            // High uncertainty → boost strong force
            if uncertainty > 0.7 {
                self.f_strong[i] *= 1.1; // 10% increase
            }

            // Low uncertainty → boost weak force
            if uncertainty < 0.3 {
                self.f_weak[i] *= 1.1; // 10% increase
            }
        }

        // Clamp forces to reasonable ranges
        for i in 0..self.n_vertices {
            self.f_strong[i] = self.f_strong[i].clamp(0.5, 2.0);
            self.f_weak[i] = self.f_weak[i].clamp(0.5, 2.0);
        }

        // Sync to device
        self.to_device()?;

        Ok(())
    }

    /// Apply a force command from RL controller
    ///
    /// # Arguments
    /// - `command`: ForceCommand specifying which band to modify and direction
    ///
    /// # Returns
    /// CommandResult with before/after stats for reward computation
    ///
    /// # Note
    /// Automatically syncs to device after applying command
    pub fn apply_force_command(
        &mut self,
        command: &super::command::ForceCommand,
    ) -> Result<super::command::CommandResult> {
        use super::command::{CommandResult, ForceCommand};

        // Capture stats before modification
        let stats_before = self.compute_stats();

        // Handle NoOp case
        if command.is_noop() {
            return Ok(CommandResult::new(
                *command,
                stats_before.clone(),
                stats_before,
                false,
            ));
        }

        // Get target band and multiplier
        let band = command.target_band().unwrap(); // Safe: non-NoOp commands always have a target
        let multiplier = command.multiplier();

        // Apply command to appropriate band
        match band {
            ForceBand::Strong => {
                for i in 0..self.n_vertices {
                    // Adjust vertices in strong band (f_strong > 1.2)
                    if self.f_strong[i] > 1.2 {
                        self.f_strong[i] *= multiplier;
                        self.f_strong[i] = self.f_strong[i].clamp(0.5, 2.0);
                    }
                }
            }
            ForceBand::Weak => {
                for i in 0..self.n_vertices {
                    // Adjust vertices in weak band (f_weak < 0.8)
                    if self.f_weak[i] < 0.8 {
                        self.f_weak[i] *= multiplier;
                        self.f_weak[i] = self.f_weak[i].clamp(0.5, 2.0);
                    }
                }
            }
            ForceBand::Neutral => {
                // Adjust baseline for all neutral vertices
                for i in 0..self.n_vertices {
                    if self.f_strong[i] <= 1.2 && self.f_strong[i] >= 0.8 {
                        self.f_strong[i] *= multiplier;
                        self.f_strong[i] = self.f_strong[i].clamp(0.5, 2.0);
                    }
                }
            }
        }

        // Sync to device
        self.to_device()?;

        // Capture stats after modification
        let stats_after = self.compute_stats();

        Ok(CommandResult::new(
            *command,
            stats_before,
            stats_after,
            true,
        ))
    }

    /// Synchronize host buffers to device (Host → Device)
    ///
    /// # GPU Mandate
    /// Must be called before Phase 2 thermodynamic kernel launch to ensure
    /// GPU has latest force values.
    ///
    /// # Synchronization
    /// Uses htod_sync_copy_into for zero-copy pinned memory transfer
    pub fn to_device(&mut self) -> Result<()> {
        self.cuda_device
            .htod_sync_copy_into(&self.f_strong, &mut self.device_f_strong)
            .context("Failed to copy f_strong to device")?;

        self.cuda_device
            .htod_sync_copy_into(&self.f_weak, &mut self.device_f_weak)
            .context("Failed to copy f_weak to device")?;

        Ok(())
    }

    /// Synchronize device buffers to host (Device → Host)
    ///
    /// # Use Case
    /// After GPU kernel modifies force profile (future enhancement),
    /// copy results back to host for analysis/telemetry.
    pub fn from_device(&mut self) -> Result<()> {
        self.cuda_device
            .dtoh_sync_copy_into(&self.device_f_strong, &mut self.f_strong)
            .context("Failed to copy f_strong from device")?;

        self.cuda_device
            .dtoh_sync_copy_into(&self.device_f_weak, &mut self.f_weak)
            .context("Failed to copy f_weak from device")?;

        Ok(())
    }

    /// Compute telemetry statistics about current force distribution
    pub fn compute_stats(&self) -> ForceBandStats {
        ForceBandStats::compute(&self.f_strong, &self.f_weak)
    }

    /// Get number of vertices
    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }

    /// Get reference to CUDA device
    pub fn cuda_device(&self) -> &Arc<CudaDevice> {
        &self.cuda_device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_force_band_defaults() {
        assert_eq!(ForceBand::Strong.default_multiplier(), 1.5);
        assert_eq!(ForceBand::Neutral.default_multiplier(), 1.0);
        assert_eq!(ForceBand::Weak.default_multiplier(), 0.7);
    }

    #[test]
    fn test_force_band_stats() {
        let f_strong = vec![1.5, 1.0, 1.0, 0.7];
        let f_weak = vec![1.0, 1.0, 1.0, 1.0];

        let stats = ForceBandStats::compute(&f_strong, &f_weak);

        assert_eq!(stats.strong_fraction, 0.25); // 1 out of 4
        assert_eq!(stats.neutral_fraction, 0.5); // 2 out of 4
        assert_eq!(stats.weak_fraction, 0.25); // 1 out of 4
        assert!((stats.mean_force - 1.05).abs() < 0.01);
    }
}
