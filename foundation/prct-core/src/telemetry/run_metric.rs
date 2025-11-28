//! Runtime Telemetry Metrics
//!
//! Captures fine-grained performance and progress data during pipeline execution.
//! Metrics are streamed to JSONL for real-time monitoring and post-run analysis.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Pipeline phase identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseName {
    Reservoir,
    TransferEntropy,
    ActiveInference,
    Thermodynamic,
    Quantum,
    Memetic,
    Ensemble,
    Validation,
}

impl fmt::Display for PhaseName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhaseName::Reservoir => write!(f, "RESERVOIR"),
            PhaseName::TransferEntropy => write!(f, "TE"),
            PhaseName::ActiveInference => write!(f, "AI"),
            PhaseName::Thermodynamic => write!(f, "THERMO"),
            PhaseName::Quantum => write!(f, "QUANTUM"),
            PhaseName::Memetic => write!(f, "MEMETIC"),
            PhaseName::Ensemble => write!(f, "ENSEMBLE"),
            PhaseName::Validation => write!(f, "VALID"),
        }
    }
}

/// Execution mode for phase
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum PhaseExecMode {
    /// GPU execution succeeded
    GpuSuccess {
        #[serde(skip_serializing_if = "Option::is_none")]
        stream_id: Option<usize>,
    },

    /// CPU fallback (with reason)
    CpuFallback { reason: String },

    /// CPU-only (GPU disabled in config)
    CpuDisabled,
}

impl PhaseExecMode {
    pub fn gpu_success(stream_id: Option<usize>) -> Self {
        PhaseExecMode::GpuSuccess { stream_id }
    }

    pub fn cpu_fallback(reason: impl Into<String>) -> Self {
        PhaseExecMode::CpuFallback {
            reason: reason.into(),
        }
    }

    pub fn cpu_disabled() -> Self {
        PhaseExecMode::CpuDisabled
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, PhaseExecMode::GpuSuccess { .. })
    }
}

impl fmt::Display for PhaseExecMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhaseExecMode::GpuSuccess {
                stream_id: Some(id),
            } => write!(f, "GPU[stream={}]", id),
            PhaseExecMode::GpuSuccess { stream_id: None } => write!(f, "GPU"),
            PhaseExecMode::CpuFallback { reason } => write!(f, "CPU[fallback: {}]", reason),
            PhaseExecMode::CpuDisabled => write!(f, "CPU[disabled]"),
        }
    }
}

/// Optimization guidance for hypertuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationGuidance {
    /// Status: "on_track", "need_tuning", "excellent", "stagnant", "critical"
    pub status: String,

    /// Specific actionable recommendations
    pub recommendations: Vec<String>,

    /// Estimated final chromatic number if current trend continues
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_final_colors: Option<usize>,

    /// Confidence in guidance (0.0-1.0)
    pub confidence: f64,

    /// Gap to world record (83 colors for DSJC1000.5)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gap_to_world_record: Option<i32>,
}

impl OptimizationGuidance {
    pub fn on_track() -> Self {
        Self {
            status: "on_track".to_string(),
            recommendations: vec!["Continue with current parameters".to_string()],
            estimated_final_colors: None,
            confidence: 0.8,
            gap_to_world_record: None,
        }
    }

    pub fn excellent() -> Self {
        Self {
            status: "excellent".to_string(),
            recommendations: vec!["Outstanding progress, maintain settings".to_string()],
            estimated_final_colors: None,
            confidence: 0.95,
            gap_to_world_record: None,
        }
    }

    pub fn need_tuning(recommendations: Vec<String>) -> Self {
        Self {
            status: "need_tuning".to_string(),
            recommendations,
            estimated_final_colors: None,
            confidence: 0.7,
            gap_to_world_record: None,
        }
    }

    pub fn critical(recommendations: Vec<String>) -> Self {
        Self {
            status: "critical".to_string(),
            recommendations,
            estimated_final_colors: None,
            confidence: 0.9,
            gap_to_world_record: None,
        }
    }

    pub fn with_estimate(mut self, estimated_colors: usize) -> Self {
        self.estimated_final_colors = Some(estimated_colors);
        self
    }

    pub fn with_wr_gap(mut self, current_colors: usize, world_record: usize) -> Self {
        self.gap_to_world_record = Some((current_colors as i32) - (world_record as i32));
        self
    }
}

/// Single telemetry metric for a phase step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetric {
    /// ISO8601 timestamp
    pub timestamp: String,

    /// Pipeline phase
    pub phase: PhaseName,

    /// Step description (e.g., "temp_5", "replica_swap", "qubo_iteration_100")
    pub step: String,

    /// Current chromatic number
    pub chromatic_number: usize,

    /// Current conflict count
    pub conflicts: usize,

    /// Step duration in milliseconds
    pub duration_ms: f64,

    /// Execution mode (GPU/CPU)
    pub gpu_mode: PhaseExecMode,

    /// Phase-specific parameters (JSON object)
    pub parameters: serde_json::Value,

    /// Optional notes/warnings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,

    /// Optimization guidance for hypertuning
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimization_guidance: Option<OptimizationGuidance>,
}

impl RunMetric {
    /// Create new metric with current timestamp
    pub fn new(
        phase: PhaseName,
        step: impl Into<String>,
        chromatic_number: usize,
        conflicts: usize,
        duration_ms: f64,
        gpu_mode: PhaseExecMode,
    ) -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            phase,
            step: step.into(),
            chromatic_number,
            conflicts,
            duration_ms,
            gpu_mode,
            parameters: serde_json::Value::Null,
            notes: None,
            optimization_guidance: None,
        }
    }

    /// Add parameters as JSON
    pub fn with_parameters(mut self, params: serde_json::Value) -> Self {
        self.parameters = params;
        self
    }

    /// Add notes
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Add optimization guidance
    pub fn with_guidance(mut self, guidance: OptimizationGuidance) -> Self {
        self.optimization_guidance = Some(guidance);
        self
    }

    /// Format for terminal display
    pub fn format_terminal(&self) -> String {
        format!(
            "[{}][{}] {} | colors={} conflicts={} | {:.2}ms",
            self.phase,
            self.gpu_mode,
            self.step,
            self.chromatic_number,
            self.conflicts,
            self.duration_ms
        )
    }
}

/// Summary of entire run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    /// Run identifier
    pub run_id: String,

    /// Graph name
    pub graph_name: String,

    /// Total runtime in seconds
    pub total_runtime_sec: f64,

    /// Final chromatic number
    pub final_chromatic: usize,

    /// Final conflicts
    pub final_conflicts: usize,

    /// Total metrics recorded
    pub metric_count: usize,

    /// Phase breakdown
    pub phase_stats: Vec<PhaseStats>,

    /// GPU usage summary
    pub gpu_summary: GpuUsageSummary,
}

/// Statistics for single phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseStats {
    pub phase: PhaseName,
    pub total_time_ms: f64,
    pub step_count: usize,
    pub gpu_steps: usize,
    pub cpu_steps: usize,
    pub best_chromatic: usize,
}

/// GPU usage summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuUsageSummary {
    pub total_gpu_time_ms: f64,
    pub total_cpu_time_ms: f64,
    pub gpu_percentage: f64,
    pub streams_used: Vec<usize>,
    pub stream_mode: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_serialization() {
        let metric = RunMetric::new(
            PhaseName::Thermodynamic,
            "temp_5",
            115,
            0,
            123.45,
            PhaseExecMode::gpu_success(Some(2)),
        )
        .with_parameters(serde_json::json!({"temp": 0.5, "replicas": 56}));

        let json = serde_json::to_string(&metric).expect("Failed to serialize");
        let _deserialized: RunMetric = serde_json::from_str(&json).expect("Failed to deserialize");
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", PhaseName::Thermodynamic), "THERMO");
        assert_eq!(format!("{}", PhaseName::ActiveInference), "AI");
    }
}
