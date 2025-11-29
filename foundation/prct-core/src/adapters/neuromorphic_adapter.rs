//! Neuromorphic Adapter - GPU-Accelerated Implementation
//!
//! Connects PRCT domain logic to GPU-accelerated neuromorphic processing.
//! Uses foundation/neuromorphic with RTX 5070 CUDA acceleration.

use crate::errors::{PRCTError, Result};
use crate::ports::{NeuromorphicEncodingParams, NeuromorphicPort};
use shared_types::*;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
#[cfg(feature = "cuda")]
use neuromorphic_engine::gpu_reservoir::GpuReservoirComputer;
#[cfg(feature = "cuda")]
use neuromorphic_engine::reservoir::ReservoirConfig;
#[cfg(feature = "cuda")]
use neuromorphic_engine::{EncodingParameters, SpikeEncoder};

#[cfg(not(feature = "cuda"))]
use neuromorphic_engine::reservoir::ReservoirConfig;
#[cfg(not(feature = "cuda"))]
use neuromorphic_engine::{EncodingParameters, ReservoirComputer, SpikeEncoder};

/// Neuromorphic adapter using GPU-accelerated reservoir computing
pub struct NeuromorphicAdapter {
    #[cfg(feature = "cuda")]
    gpu_reservoir: GpuReservoirComputer,
    #[cfg(not(feature = "cuda"))]
    cpu_reservoir: ReservoirComputer,

    // Don't store encoder - create on demand to avoid ThreadRng Send/Sync issues
    config: NeuromorphicEncodingParams,
}

impl NeuromorphicAdapter {
    /// Create new neuromorphic adapter with GPU acceleration
    #[cfg(feature = "cuda")]
    pub fn new(cuda_device: Arc<CudaContext>) -> Result<Self> {
        // Configure reservoir for PRCT (optimized for pattern detection)
        let reservoir_config = ReservoirConfig {
            size: 1000,               // 1000 neurons for rich dynamics
            input_size: 100,          // Input dimensionality
            spectral_radius: 0.95,    // Edge of chaos (critical dynamics)
            connection_prob: 0.1,     // 10% sparsity (biological realism)
            leak_rate: 0.3,           // Moderate temporal memory
            input_scaling: 1.0,       // Unity input scaling
            noise_level: 0.01,        // Small noise for robustness
            enable_plasticity: false, // Disable plasticity for consistency
            stdp_profile: neuromorphic_engine::STDPProfile::Balanced,
        };

        // Create GPU reservoir using shared CUDA context
        let gpu_reservoir = GpuReservoirComputer::new_shared(reservoir_config, cuda_device)
            .map_err(|e| {
                PRCTError::NeuromorphicFailed(format!("Failed to create GPU reservoir: {}", e))
            })?;

        let config = NeuromorphicEncodingParams::default();

        Ok(Self {
            gpu_reservoir,
            config,
        })
    }

    /// Create new neuromorphic adapter with CPU fallback
    #[cfg(not(feature = "cuda"))]
    pub fn new() -> Result<Self> {
        // Configure reservoir for PRCT
        let cpu_reservoir = ReservoirComputer::new(
            1000, // reservoir_size
            100,  // input_size
            0.95, // spectral_radius
            0.1,  // connection_prob
            0.3,  // leak_rate
        )
        .map_err(|e| anyhow!("Failed to create CPU reservoir: {}", e))?;

        let config = NeuromorphicEncodingParams::default();

        Ok(Self {
            cpu_reservoir,
            config,
        })
    }

    /// Encode graph structure as spike pattern
    ///
    /// Converts graph topology (degrees, clustering) into temporal spike pattern
    fn graph_to_spike_pattern(
        &mut self,
        graph: &Graph,
        params: &NeuromorphicEncodingParams,
    ) -> Result<neuromorphic_engine::SpikePattern> {
        // Build input values from graph structure
        let mut values = Vec::new();

        // Encode vertex degrees as input values
        for i in 0..graph.num_vertices.min(params.num_neurons) {
            let degree = graph.adjacency[i * graph.num_vertices..(i + 1) * graph.num_vertices]
                .iter()
                .filter(|&&edge| edge)
                .count();

            // Normalize degree to [0,1] range
            let normalized_degree = degree as f64 / graph.num_vertices as f64;
            values.push(normalized_degree);
        }

        // Add clustering coefficient if we have multiple vertices
        if graph.num_vertices > 1 {
            let clustering = self.compute_clustering_coefficient(graph);
            values.push(clustering);
        }

        // Add graph density
        let density = (2.0 * graph.num_edges as f64)
            / (graph.num_vertices as f64 * (graph.num_vertices as f64 - 1.0)).max(1.0);
        values.push(density);

        // Create InputData with all values
        let input_data =
            neuromorphic_engine::types::InputData::new("graph_topology".to_string(), values);

        // Encode using rate coding (Poisson process)
        let encoding_params = EncodingParameters {
            max_rate: params.base_frequency * 2.0, // Max spike rate
            min_rate: params.base_frequency / 4.0, // Min spike rate
            delay_range_ms: params.time_window,
            neurons_per_feature: params.num_neurons / (graph.num_vertices.min(10)),
            base_frequency: params.base_frequency,
            phase_range: 2.0 * std::f64::consts::PI,
        };

        // Create encoder on demand (avoids ThreadRng Send/Sync issues)
        let mut encoder = SpikeEncoder::new(100, 100.0).map_err(|e| {
            PRCTError::NeuromorphicFailed(format!("Failed to create spike encoder: {}", e))
        })?;

        let spike_pattern = encoder
            .with_parameters(encoding_params)
            .with_encoding_method(neuromorphic_engine::types::EncodingMethod::Rate)
            .encode(&input_data)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("Spike encoding failed: {}", e)))?;

        Ok(spike_pattern)
    }

    /// Compute clustering coefficient for graph
    fn compute_clustering_coefficient(&self, graph: &Graph) -> f64 {
        if graph.num_vertices < 3 {
            return 0.0;
        }

        let mut total_clustering = 0.0;
        let mut valid_vertices = 0;

        for i in 0..graph.num_vertices {
            // Get neighbors of vertex i
            let neighbors: Vec<usize> = (0..graph.num_vertices)
                .filter(|&j| j != i && graph.adjacency[i * graph.num_vertices + j])
                .collect();

            let k = neighbors.len();
            if k < 2 {
                continue;
            }

            // Count triangles
            let mut triangles = 0;
            for &j in &neighbors {
                for &m in &neighbors {
                    if j < m && graph.adjacency[j * graph.num_vertices + m] {
                        triangles += 1;
                    }
                }
            }

            // Clustering coefficient for this vertex
            let c_i = (2.0 * triangles as f64) / (k as f64 * (k as f64 - 1.0));
            total_clustering += c_i;
            valid_vertices += 1;
        }

        if valid_vertices > 0 {
            total_clustering / valid_vertices as f64
        } else {
            0.0
        }
    }
}

impl NeuromorphicPort for NeuromorphicAdapter {
    fn encode_graph_as_spikes(
        &self,
        graph: &Graph,
        encoding_params: &NeuromorphicEncodingParams,
    ) -> Result<SpikePattern> {
        // Build input values from graph structure
        let mut values = Vec::new();

        // Encode vertex degrees as input values
        for i in 0..graph.num_vertices.min(encoding_params.num_neurons) {
            let degree = graph.adjacency[i * graph.num_vertices..(i + 1) * graph.num_vertices]
                .iter()
                .filter(|&&edge| edge)
                .count();

            let normalized_degree = degree as f64 / graph.num_vertices as f64;
            values.push(normalized_degree);
        }

        // Add clustering coefficient
        if graph.num_vertices > 1 {
            let clustering = self.compute_clustering_coefficient(graph);
            values.push(clustering);
        }

        // Add graph density
        let density = (2.0 * graph.num_edges as f64)
            / (graph.num_vertices as f64 * (graph.num_vertices as f64 - 1.0)).max(1.0);
        values.push(density);

        // Create InputData with all values
        let input_data =
            neuromorphic_engine::types::InputData::new("graph_topology".to_string(), values);

        // Encode using rate coding
        let encoding_params_struct = EncodingParameters {
            max_rate: encoding_params.base_frequency * 2.0,
            min_rate: encoding_params.base_frequency / 4.0,
            delay_range_ms: encoding_params.time_window,
            neurons_per_feature: encoding_params.num_neurons / (graph.num_vertices.min(10)),
            base_frequency: encoding_params.base_frequency,
            phase_range: 2.0 * std::f64::consts::PI,
        };

        // Create encoder on demand (avoids ThreadRng Send/Sync issues)
        let mut encoder = SpikeEncoder::new(100, 100.0).map_err(|e| {
            PRCTError::NeuromorphicFailed(format!("Failed to create spike encoder: {}", e))
        })?;

        let neuro_spike_pattern = encoder
            .with_parameters(encoding_params_struct)
            .with_encoding_method(neuromorphic_engine::types::EncodingMethod::Rate)
            .encode(&input_data)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("Spike encoding failed: {}", e)))?;

        // Convert to shared_types::SpikePattern
        let spikes: Vec<Spike> = neuro_spike_pattern
            .spikes
            .iter()
            .map(|s| Spike {
                neuron_id: s.neuron_id,
                time_ms: s.time_ms,
                amplitude: s.amplitude.unwrap_or(1.0) as f64, // Convert Option<f32> to f64
            })
            .collect();

        // Calculate num_neurons from max neuron_id + 1
        let num_neurons = neuro_spike_pattern
            .spikes
            .iter()
            .map(|s| s.neuron_id)
            .max()
            .map(|max_id| max_id + 1)
            .unwrap_or(encoding_params.num_neurons);

        Ok(SpikePattern {
            spikes,
            duration_ms: neuro_spike_pattern.duration_ms,
            num_neurons,
        })
    }

    fn process_and_detect_patterns(&self, spikes: &SpikePattern) -> Result<NeuroState> {
        // Convert shared_types::SpikePattern to neuromorphic_engine::types::SpikePattern
        let neuro_spikes: Vec<neuromorphic_engine::types::Spike> = spikes
            .spikes
            .iter()
            .map(|s| neuromorphic_engine::types::Spike {
                neuron_id: s.neuron_id,
                time_ms: s.time_ms,
                amplitude: Some(s.amplitude as f32), // Convert f64 to Option<f32>
            })
            .collect();

        let neuro_spike_pattern = neuromorphic_engine::types::SpikePattern {
            spikes: neuro_spikes,
            duration_ms: spikes.duration_ms,
            metadata: neuromorphic_engine::types::PatternMetadata::default(),
        };

        // Process through reservoir - need to make reservoir mutable
        // For now, create simplified state from pattern
        // Full implementation would require mutable reference or interior mutability

        // Calculate simplified metrics from spike pattern
        let num_spikes = spikes.spikes.len();
        let spike_density =
            num_spikes as f64 / (spikes.duration_ms * spikes.num_neurons as f64).max(1.0);

        // Create neuron states based on spike times
        let mut neuron_states = vec![0.0; spikes.num_neurons];
        for spike in &spikes.spikes {
            if spike.neuron_id < neuron_states.len() {
                // Accumulate spike contributions (exponential decay)
                let time_factor = (-spike.time_ms / 10.0).exp(); // 10ms time constant
                neuron_states[spike.neuron_id] += spike.amplitude * time_factor;
            }
        }

        // Normalize neuron states
        let max_state = neuron_states.iter().cloned().fold(0.0f64, f64::max);
        if max_state > 1e-10 {
            for state in &mut neuron_states {
                *state /= max_state;
            }
        }

        // Create spike pattern (binary: 1 if spiked, 0 otherwise)
        let mut spike_pattern = vec![0u8; spikes.num_neurons];
        for spike in &spikes.spikes {
            if spike.neuron_id < spike_pattern.len() {
                spike_pattern[spike.neuron_id] = 1;
            }
        }

        // Calculate coherence from spike synchrony
        let coherence = (spike_density * 10.0).min(1.0);

        // Pattern strength from spike density
        let pattern_strength = spike_density;

        Ok(NeuroState {
            neuron_states,
            spike_pattern,
            coherence,
            pattern_strength,
            timestamp_ns: 0,
        })
    }

    fn get_detected_patterns(&self) -> Result<Vec<DetectedPattern>> {
        // For now, return empty pattern list
        // Full implementation would use pattern detector
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_neuromorphic_adapter_gpu() {
        let device = CudaDevice::new(0).expect("CUDA device");
        let adapter = NeuromorphicAdapter::new(device).expect("adapter creation");

        // Create simple test graph
        let graph = Graph {
            num_vertices: 4,
            num_edges: 4,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
            adjacency: vec![
                false, true, false, true, true, false, true, false, false, true, false, true, true,
                false, true, false,
            ],
            coordinates: None,
        };

        let params = NeuromorphicEncodingParams::default();
        let spikes = adapter
            .encode_graph_as_spikes(&graph, &params)
            .expect("encoding");

        assert!(spikes.spikes.len() > 0);
        assert_eq!(spikes.num_neurons, 100);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_neuromorphic_adapter_cpu() {
        let adapter = NeuromorphicAdapter::new().expect("adapter creation");

        // Create simple test graph
        let graph = Graph {
            num_vertices: 4,
            num_edges: 4,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
            adjacency: vec![
                false, true, false, true, true, false, true, false, false, true, false, true, true,
                false, true, false,
            ],
            coordinates: None,
        };

        let params = NeuromorphicEncodingParams::default();
        let spikes = adapter
            .encode_graph_as_spikes(&graph, &params)
            .expect("encoding");

        assert!(spikes.spikes.len() > 0);
    }
}
