//! Core types for the neuromorphic computing engine

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Generic input data for the neuromorphic engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputData {
    /// Raw data values
    pub values: Vec<f64>,
    /// Timestamp of the data point
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Data source identifier
    pub source: String,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

impl InputData {
    /// Create new input data
    pub fn new(source: String, values: Vec<f64>) -> Self {
        Self {
            values,
            timestamp: chrono::Utc::now(),
            source,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: f64) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).copied()
    }
}

/// Individual spike in a neural network
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Spike {
    /// Neuron ID that fired
    pub neuron_id: usize,
    /// Time of the spike (milliseconds)
    pub time_ms: f64,
    /// Spike amplitude (optional)
    pub amplitude: Option<f32>,
}

impl Spike {
    /// Create a new spike
    pub fn new(neuron_id: usize, time_ms: f64) -> Self {
        Self {
            neuron_id,
            time_ms,
            amplitude: None,
        }
    }

    /// Create a spike with amplitude
    pub fn with_amplitude(neuron_id: usize, time_ms: f64, amplitude: f32) -> Self {
        Self {
            neuron_id,
            time_ms,
            amplitude: Some(amplitude),
        }
    }
}

/// Collection of spikes forming a pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikePattern {
    /// Individual spikes in the pattern
    pub spikes: Vec<Spike>,
    /// Duration of the pattern (milliseconds)
    pub duration_ms: f64,
    /// Pattern metadata
    pub metadata: PatternMetadata,
}

impl SpikePattern {
    /// Create a new spike pattern
    pub fn new(spikes: Vec<Spike>, duration_ms: f64) -> Self {
        Self {
            spikes,
            duration_ms,
            metadata: PatternMetadata::default(),
        }
    }

    /// Get the number of spikes in the pattern
    pub fn spike_count(&self) -> usize {
        self.spikes.len()
    }

    /// Get spikes within a time window
    pub fn spikes_in_window(&self, start_ms: f64, end_ms: f64) -> Vec<&Spike> {
        self.spikes
            .iter()
            .filter(|spike| spike.time_ms >= start_ms && spike.time_ms <= end_ms)
            .collect()
    }

    /// Calculate spike rate (spikes per second)
    pub fn spike_rate(&self) -> f64 {
        if self.duration_ms == 0.0 {
            0.0
        } else {
            (self.spikes.len() as f64) / (self.duration_ms / 1000.0)
        }
    }
}

/// Metadata associated with a spike pattern
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternMetadata {
    /// Pattern strength/confidence
    pub strength: f32,
    /// Pattern classification
    pub pattern_type: Option<String>,
    /// Source data identifier
    pub source: Option<String>,
    /// Additional custom metadata
    pub custom: HashMap<String, f64>,
}

/// Recognized pattern types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Pattern {
    /// Trend patterns
    Increasing,
    Decreasing,
    Stable,

    /// Change patterns
    Reversal,
    Breakout,

    /// Variation patterns
    HighVariability,
    LowVariability,

    /// Custom pattern with name
    Custom(String),
}

impl Pattern {
    /// Get pattern as string
    pub fn as_str(&self) -> &str {
        match self {
            Pattern::Increasing => "increasing",
            Pattern::Decreasing => "decreasing",
            Pattern::Stable => "stable",
            Pattern::Reversal => "reversal",
            Pattern::Breakout => "breakout",
            Pattern::HighVariability => "high_variability",
            Pattern::LowVariability => "low_variability",
            Pattern::Custom(name) => name,
        }
    }
}

/// Prediction output from the neuromorphic engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Predicted direction/action
    pub direction: PredictionDirection,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Predicted magnitude of change
    pub magnitude: Option<f32>,
    /// Time horizon for the prediction
    pub time_horizon_ms: f64,
    /// Detected patterns that led to this prediction
    pub patterns: Vec<Pattern>,
    /// Prediction metadata
    pub metadata: PredictionMetadata,
}

impl Prediction {
    /// Create a new prediction
    pub fn new(direction: PredictionDirection, confidence: f32, time_horizon_ms: f64) -> Self {
        Self {
            direction,
            confidence: confidence.clamp(0.0, 1.0),
            magnitude: None,
            time_horizon_ms,
            patterns: Vec::new(),
            metadata: PredictionMetadata::default(),
        }
    }

    /// Add a detected pattern
    pub fn with_pattern(mut self, pattern: Pattern) -> Self {
        self.patterns.push(pattern);
        self
    }

    /// Set magnitude
    pub fn with_magnitude(mut self, magnitude: f32) -> Self {
        self.magnitude = Some(magnitude);
        self
    }

    /// Check if prediction is positive
    pub fn is_positive(&self) -> bool {
        matches!(self.direction, PredictionDirection::Up)
    }

    /// Check if prediction is negative
    pub fn is_negative(&self) -> bool {
        matches!(self.direction, PredictionDirection::Down)
    }

    /// Check if prediction is neutral
    pub fn is_neutral(&self) -> bool {
        matches!(self.direction, PredictionDirection::Hold)
    }
}

/// Direction of a prediction
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionDirection {
    /// Upward movement expected
    Up,
    /// Downward movement expected
    Down,
    /// No significant movement expected
    Hold,
}

/// Metadata for predictions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictionMetadata {
    /// Model version used
    pub model_version: Option<String>,
    /// Processing latency in microseconds
    pub latency_us: Option<f64>,
    /// Number of spikes processed
    pub spike_count: Option<usize>,
    /// Reservoir state information
    pub reservoir_state: Option<HashMap<String, f64>>,
}

/// Spike encoding methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingMethod {
    /// Rate coding (spike frequency represents value)
    Rate,
    /// Temporal coding (spike timing represents value)
    Temporal,
    /// Population coding (multiple neurons represent value)
    Population,
    /// Phase coding (spike phase represents value)
    Phase,
}

/// Configuration for spike encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingConfig {
    /// Encoding method to use
    pub method: EncodingMethod,
    /// Time window for encoding (milliseconds)
    pub window_ms: f64,
    /// Number of neurons to use for encoding
    pub neuron_count: usize,
    /// Encoding parameters
    pub parameters: HashMap<String, f64>,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            method: EncodingMethod::Rate,
            window_ms: 1000.0,
            neuron_count: 1000,
            parameters: HashMap::new(),
        }
    }
}
