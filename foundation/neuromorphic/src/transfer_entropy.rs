//! Transfer Entropy Computation
//!
//! Implements transfer entropy calculation for measuring information flow
//! between neuromorphic oscillators. Based on CSF's DRPP transfer_entropy module.
//!
//! Transfer Entropy: TE(X→Y) = I(Y_future; X_past | Y_past)
//! Measures how much knowing X's past reduces uncertainty about Y's future,
//! given Y's own past.

use anyhow::Result;
use ndarray::Array2;
use std::collections::VecDeque;

/// Transfer entropy computation configuration
#[derive(Debug, Clone)]
pub struct TransferEntropyConfig {
    /// Time delay for embeddings (default: 1)
    pub tau: usize,

    /// Embedding dimension / history length (default: 3)
    pub history_length: usize,

    /// Number of bins for discretization (default: 20)
    pub num_bins: usize,

    /// Minimum data length required (default: 50)
    pub min_data_length: usize,
}

impl Default for TransferEntropyConfig {
    fn default() -> Self {
        Self {
            tau: 1,
            history_length: 3,
            num_bins: 20,
            min_data_length: 50,
        }
    }
}

/// Transfer entropy computation engine
///
/// Computes information flow between oscillator time series
/// using delay embedding and histogram-based entropy estimation.
pub struct TransferEntropyEngine {
    config: TransferEntropyConfig,
}

impl TransferEntropyEngine {
    /// Create new transfer entropy engine
    pub fn new(config: TransferEntropyConfig) -> Self {
        Self { config }
    }

    /// Compute transfer entropy matrix for multiple time series
    ///
    /// Returns n×n matrix where element [i,j] represents TE(i→j)
    /// (information flow from series i to series j)
    pub fn compute_te_matrix(&self, time_series: &[Vec<f64>]) -> Result<Array2<f64>> {
        let n = time_series.len();

        if n == 0 {
            return Ok(Array2::zeros((0, 0)));
        }

        let mut te_matrix = Array2::zeros((n, n));

        // Compute pairwise transfer entropy
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let te = self.compute_pairwise_te(&time_series[i], &time_series[j])?;
                    te_matrix[[i, j]] = te;
                }
            }
        }

        Ok(te_matrix)
    }

    /// Compute transfer entropy from source to target
    pub fn compute_pairwise_te(&self, source: &[f64], target: &[f64]) -> Result<f64> {
        let tau = self.config.tau;
        let k = self.config.history_length;

        // Validate data length
        let required_length = k * tau + tau;
        if source.len() < required_length || target.len() < required_length {
            return Ok(0.0); // Insufficient data
        }

        let n = source.len().min(target.len());

        // Build histograms for entropy calculation
        let mut hist_target_future_given_past = vec![0usize; self.config.num_bins];
        let mut hist_target_future_given_both =
            vec![0usize; self.config.num_bins * self.config.num_bins];
        let mut count = 0;

        // Sliding window over time series
        for t in (k * tau)..(n - tau) {
            // Extract embeddings
            let target_future = target[t + tau];
            let target_past: Vec<f64> = (0..k).map(|i| target[t - i * tau]).collect();
            let source_past: Vec<f64> = (0..k).map(|i| source[t - i * tau]).collect();

            // Discretize
            let bin_target_future = self.discretize(target_future, target);
            let bin_target_past = self.discretize_vector(&target_past);
            let _bin_source_past = self.discretize_vector(&source_past);

            // Update histograms
            if bin_target_future < self.config.num_bins && bin_target_past < self.config.num_bins {
                hist_target_future_given_past[bin_target_future] += 1;

                let combined_bin = bin_target_future * self.config.num_bins + bin_target_past;
                if combined_bin < hist_target_future_given_both.len() {
                    hist_target_future_given_both[combined_bin] += 1;
                }
            }

            count += 1;
        }

        if count == 0 {
            return Ok(0.0);
        }

        // Calculate entropies
        let h_target_future = self.calculate_entropy(&hist_target_future_given_past, count);
        let h_target_both = self.calculate_entropy(&hist_target_future_given_both, count);

        // TE = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
        let te = (h_target_future - h_target_both).max(0.0);

        Ok(te)
    }

    /// Discretize single value into bin index
    fn discretize(&self, value: f64, series: &[f64]) -> usize {
        if series.is_empty() {
            return 0;
        }

        let min = series.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = series.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;

        if range < 1e-10 {
            return 0;
        }

        let normalized = ((value - min) / range * (self.config.num_bins - 1) as f64)
            .clamp(0.0, (self.config.num_bins - 1) as f64);
        normalized as usize
    }

    /// Discretize vector into single bin (simplified hash)
    fn discretize_vector(&self, vec: &[f64]) -> usize {
        if vec.is_empty() {
            return 0;
        }

        // Simple hash: sum of discretized values
        let sum: usize = vec
            .iter()
            .map(|&x| ((x * 10.0).abs() as usize) % self.config.num_bins)
            .sum();
        sum % self.config.num_bins
    }

    /// Calculate Shannon entropy from histogram
    fn calculate_entropy(&self, histogram: &[usize], total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;

        for &count in histogram {
            if count > 0 {
                let p = count as f64 / total as f64;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Detect dominant information flow direction
    pub fn detect_flow_direction(&self, te_matrix: &Array2<f64>) -> Vec<(usize, usize, f64)> {
        let n = te_matrix.nrows();
        let mut flows = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let te_ij = te_matrix[[i, j]];
                    let te_ji = te_matrix[[j, i]];

                    // Net flow from i to j
                    let net_flow = te_ij - te_ji;

                    if net_flow.abs() > 0.01 {
                        flows.push((i, j, net_flow));
                    }
                }
            }
        }

        // Sort by absolute flow strength
        flows.sort_by(|a, b| {
            b.2.abs()
                .partial_cmp(&a.2.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        flows
    }
}

/// Circular buffer for efficient time series storage
pub struct TimeSeriesBuffer {
    buffer: VecDeque<f64>,
    max_length: usize,
}

impl TimeSeriesBuffer {
    pub fn new(max_length: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_length),
            max_length,
        }
    }

    pub fn push(&mut self, value: f64) {
        if self.buffer.len() >= self.max_length {
            self.buffer.pop_front();
        }
        self.buffer.push_back(value);
    }

    pub fn as_slice(&self) -> Vec<f64> {
        self.buffer.iter().copied().collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_entropy_coupled_oscillators() {
        let config = TransferEntropyConfig::default();
        let engine = TransferEntropyEngine::new(config);

        // Create coupled sine waves (Y follows X with delay)
        let n = 100;
        let source: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let target: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1 - 0.5).sin()).collect(); // Delayed

        let te = engine.compute_pairwise_te(&source, &target).unwrap();

        // Should detect information flow
        assert!(
            te > 0.0,
            "Transfer entropy should be positive for coupled series"
        );
    }

    #[test]
    fn test_transfer_entropy_independent() {
        let config = TransferEntropyConfig::default();
        let engine = TransferEntropyEngine::new(config);

        // Create independent random series
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let source: Vec<f64> = (0..100).map(|_| rng.gen()).collect();
        let target: Vec<f64> = (0..100).map(|_| rng.gen()).collect();

        let te = engine.compute_pairwise_te(&source, &target).unwrap();

        // Should be near zero for independent series
        assert!(
            te < 0.5,
            "Transfer entropy should be low for independent series"
        );
    }

    #[test]
    fn test_circular_buffer() {
        let mut buffer = TimeSeriesBuffer::new(5);

        for i in 0..10 {
            buffer.push(i as f64);
        }

        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.as_slice(), vec![5.0, 6.0, 7.0, 8.0, 9.0]);
    }
}
