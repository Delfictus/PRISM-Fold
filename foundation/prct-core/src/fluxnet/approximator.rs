use serde::{Deserialize, Serialize};

/// Lightweight linear approximator for Q-values.
///
/// Uses a per-action weight vector over RLState features. Trained via
/// simple stochastic gradient descent toward TD targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QApproximator {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    feature_dim: usize,
    learning_rate: f32,
}

impl QApproximator {
    pub fn new(num_actions: usize, feature_dim: usize, learning_rate: f32) -> Self {
        Self {
            weights: vec![vec![0.0; feature_dim]; num_actions],
            bias: vec![0.0; num_actions],
            feature_dim,
            learning_rate,
        }
    }

    pub fn predict(&self, features: &[f32]) -> Vec<f32> {
        debug_assert_eq!(features.len(), self.feature_dim);
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(w, b)| {
                let dot = w
                    .iter()
                    .zip(features.iter())
                    .map(|(wi, fi)| wi * fi)
                    .sum::<f32>();
                dot + b
            })
            .collect()
    }

    pub fn update(&mut self, features: &[f32], action_idx: usize, target: f32) {
        if action_idx >= self.weights.len() || features.len() != self.feature_dim {
            return;
        }

        let prediction = self.predict(features)[action_idx];
        let error = target - prediction;
        for (w, &f) in self.weights[action_idx].iter_mut().zip(features.iter()) {
            *w += self.learning_rate * error * f;
        }
        self.bias[action_idx] += self.learning_rate * error;
    }
}
