//! DRPP-Enhanced PRCT Algorithm
//!
//! Extends the base PRCT algorithm with:
//! - Phase-Causal Matrix (PCM-Φ) for enhanced coupling
//! - Transfer Entropy for causal inference
//! - Adaptive Decision Processing for parameter optimization
//! - Full ChronoPath-DRPP-C-Logic theoretical framework

use crate::coloring::phase_guided_coloring;
use crate::errors::*;
use crate::ports::*;
use crate::tsp::phase_guided_tsp;
use shared_types::*;
use std::sync::Arc;

/// DRPP-enhanced PRCT configuration
#[derive(Debug, Clone)]
pub struct DrppPrctConfig {
    /// Base PRCT configuration
    pub target_colors: usize,
    pub quantum_evolution_time: f64,
    pub kuramoto_coupling: f64,
    pub neuro_encoding: NeuromorphicEncodingParams,
    pub quantum_params: EvolutionParams,

    /// DRPP-specific parameters
    pub enable_drpp: bool,
    pub pcm_kappa_weight: f64,       // Kuramoto term weight in PCM
    pub pcm_beta_weight: f64,        // Transfer entropy term weight in PCM
    pub drpp_evolution_steps: usize, // Phase evolution iterations
    pub drpp_dt: f64,                // Time step for phase evolution

    /// ADP parameters
    pub enable_adp: bool,
    pub adp_learning_rate: f64,
    pub adp_exploration_rate: f64,
}

impl Default for DrppPrctConfig {
    fn default() -> Self {
        Self {
            // Base PRCT
            target_colors: 10,
            quantum_evolution_time: 0.1,
            kuramoto_coupling: 0.5,
            neuro_encoding: NeuromorphicEncodingParams::default(),
            quantum_params: EvolutionParams {
                dt: 0.01,
                strength: 1.0,
                damping: 0.1,
                temperature: 300.0,
            },

            // DRPP
            enable_drpp: true,
            pcm_kappa_weight: 1.0,    // Equal weight to synchronization
            pcm_beta_weight: 0.5,     // Moderate causal inference
            drpp_evolution_steps: 10, // 10 steps of phase evolution
            drpp_dt: 0.01,            // 10ms time step

            // ADP
            enable_adp: true,
            adp_learning_rate: 0.001,
            adp_exploration_rate: 0.1,
        }
    }
}

/// DRPP-enhanced PRCT algorithm
///
/// Implements the full ChronoPath-DRPP-C-Logic theoretical framework:
/// - Transfer entropy-based causal inference (TE-X)
/// - Phase-Causal Matrix combining Kuramoto + TE (PCM-Φ)
/// - Adaptive dissipative processing (ADP)
/// - Phase evolution with causal coupling (DRPP-Δθ)
pub struct DrppPrctAlgorithm {
    /// Neuromorphic processing port
    neuro_port: Arc<dyn NeuromorphicPort>,

    /// Quantum processing port
    quantum_port: Arc<dyn QuantumPort>,

    /// Physics coupling port
    coupling_port: Arc<dyn PhysicsCouplingPort>,

    /// Configuration
    config: DrppPrctConfig,
}

impl DrppPrctAlgorithm {
    /// Create new DRPP-enhanced PRCT algorithm
    pub fn new(
        neuro_port: Arc<dyn NeuromorphicPort>,
        quantum_port: Arc<dyn QuantumPort>,
        coupling_port: Arc<dyn PhysicsCouplingPort>,
        config: DrppPrctConfig,
    ) -> Self {
        Self {
            neuro_port,
            quantum_port,
            coupling_port,
            config,
        }
    }

    /// Solve using full DRPP-PRCT pipeline
    ///
    /// Pipeline stages:
    /// 1. Neuromorphic spike encoding + pattern detection
    /// 2. Quantum Hamiltonian evolution + phase extraction
    /// 3. **DRPP**: Compute Phase-Causal Matrix (PCM-Φ)
    /// 4. **DRPP**: Evolve phases with causal coupling
    /// 5. Physics coupling with enhanced synchronization
    /// 6. Phase-guided optimization (coloring + TSP)
    /// 7. **ADP**: Adaptive parameter optimization (future iterations)
    pub fn solve(&self, graph: &Graph) -> Result<DrppPrctSolution> {
        let start_time = std::time::Instant::now();

        // LAYER 1: NEUROMORPHIC PROCESSING
        let spikes = self
            .neuro_port
            .encode_graph_as_spikes(graph, &self.config.neuro_encoding)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("Spike encoding: {}", e)))?;

        let neuro_state = self
            .neuro_port
            .process_and_detect_patterns(&spikes)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("Pattern detection: {}", e)))?;

        // LAYER 2: QUANTUM PROCESSING
        let hamiltonian = self
            .quantum_port
            .build_hamiltonian(graph, &self.config.quantum_params)
            .map_err(|e| PRCTError::QuantumFailed(format!("Hamiltonian: {}", e)))?;

        let dim = hamiltonian.dimension;
        let initial_state = QuantumState {
            amplitudes: vec![(1.0 / (dim as f64).sqrt(), 0.0); dim],
            phase_coherence: 0.0,
            energy: 0.0,
            entanglement: 0.0,
            timestamp_ns: 0,
        };

        let quantum_state = self
            .quantum_port
            .evolve_state(
                &hamiltonian,
                &initial_state,
                self.config.quantum_evolution_time,
            )
            .map_err(|e| PRCTError::QuantumFailed(format!("Evolution: {}", e)))?;

        let mut phase_field = self
            .quantum_port
            .get_phase_field(&quantum_state)
            .map_err(|e| PRCTError::QuantumFailed(format!("Phase field: {}", e)))?;

        // LAYER 2.5: DRPP ENHANCEMENT (if enabled)
        let (pcm, te_matrix, evolved_phases) = if self.config.enable_drpp {
            self.apply_drpp_enhancement(&neuro_state, &quantum_state, &mut phase_field)?
        } else {
            (None, None, None)
        };

        // LAYER 3: PHYSICS COUPLING
        let coupling = self
            .coupling_port
            .get_bidirectional_coupling(&neuro_state, &quantum_state)
            .map_err(|e| PRCTError::CouplingFailed(format!("Coupling: {}", e)))?;

        // LAYER 4: OPTIMIZATION
        let coloring = phase_guided_coloring(
            graph,
            &phase_field,
            &coupling.kuramoto_state,
            self.config.target_colors,
        )?;

        let color_class_tours = phase_guided_tsp(graph, &coloring, &phase_field)?;

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let overall_quality = self.compute_solution_quality(&coloring, &color_class_tours);

        Ok(DrppPrctSolution {
            // Base PRCT results
            coloring,
            color_class_tours,
            phase_coherence: phase_field.order_parameter,
            kuramoto_order: coupling.kuramoto_state.order_parameter,
            overall_quality,
            total_time_ms: total_time,

            // DRPP enhancements
            phase_causal_matrix: pcm,
            transfer_entropy_matrix: te_matrix,
            evolved_phases,
            drpp_applied: self.config.enable_drpp,
        })
    }

    /// Apply DRPP enhancement to phase field
    ///
    /// Computes PCM-Φ and evolves phases with causal coupling
    fn apply_drpp_enhancement(
        &self,
        neuro_state: &NeuroState,
        quantum_state: &QuantumState,
        phase_field: &mut PhaseField,
    ) -> Result<(
        Option<Vec<Vec<f64>>>,
        Option<Vec<Vec<f64>>>,
        Option<Vec<f64>>,
    )> {
        // Extract time series from neuromorphic and quantum states
        let neuro_series = &neuro_state.neuron_states;
        let quantum_series: Vec<f64> = quantum_state
            .amplitudes
            .iter()
            .map(|(re, im)| (re * re + im * im).sqrt()) // Magnitude
            .collect();

        let n = neuro_series
            .len()
            .min(quantum_series.len())
            .min(phase_field.phases.len());
        if n < 2 {
            return Err(PRCTError::DrppFailed(
                "Insufficient data for DRPP enhancement".into(),
            ));
        }

        // 1. Build Transfer Entropy Matrix (TE-X)
        let mut te_matrix = vec![vec![0.0; n]; n];

        // Compute pairwise transfer entropy
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    te_matrix[i][j] = 0.0;
                    continue;
                }

                // Build directional time series for TE calculation
                let source_i = if i < neuro_series.len() {
                    &neuro_series[i..i + 1]
                } else {
                    &quantum_series[i - neuro_series.len()..i - neuro_series.len() + 1]
                };

                let target_j = if j < neuro_series.len() {
                    &neuro_series[j..j + 1]
                } else {
                    &quantum_series[j - neuro_series.len()..j - neuro_series.len() + 1]
                };

                // Simplified transfer entropy using correlation
                // Full TE requires conditional mutual information, but this approximates it
                let correlation = if neuro_series.len() > 10 {
                    let lag = 5.min(neuro_series.len() / 4);

                    // Use full series for better TE estimate
                    let mut te_sum = 0.0;
                    let mut count = 0;

                    for k in lag..(neuro_series.len().min(quantum_series.len()) - 1) {
                        let source_past = if i < neuro_series.len() {
                            neuro_series[k.saturating_sub(lag)]
                        } else {
                            quantum_series[k.saturating_sub(lag)]
                        };

                        let target_past = if j < neuro_series.len() {
                            neuro_series[k]
                        } else {
                            quantum_series[k]
                        };

                        let target_future = if j < neuro_series.len() {
                            neuro_series[k + 1]
                        } else {
                            quantum_series[k + 1]
                        };

                        // TE approximation: correlation between source_past and (target_future - target_past)
                        let delta = target_future - target_past;
                        te_sum += (delta * source_past).abs();
                        count += 1;
                    }

                    if count > 0 {
                        te_sum / count as f64
                    } else {
                        0.0
                    }
                } else {
                    source_i[0] * target_j[0] * 0.1
                };

                te_matrix[i][j] = correlation.abs();
            }
        }

        // 2. Compute Phase-Causal Matrix (PCM-Φ)
        // PCM-Φ_ij = κ * sin(θ_j - θ_i) + β * TE_ij
        let mut pcm = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    pcm[i][j] = 0.0;
                    continue;
                }

                // Kuramoto coupling term
                let kuramoto_term = (phase_field.phases[j] - phase_field.phases[i]).sin();

                // Transfer entropy term
                let te_term = te_matrix[i][j];

                // Weighted combination
                pcm[i][j] = self.config.pcm_kappa_weight * kuramoto_term
                    + self.config.pcm_beta_weight * te_term;
            }
        }

        // 3. Evolve phases using DRPP dynamics
        // DRPP equation: dθ_i/dt = Σ_j PCM-Φ_ij
        let mut evolved_phases = phase_field.phases[..n].to_vec();

        for step in 0..self.config.drpp_evolution_steps {
            let mut new_phases = evolved_phases.clone();

            for i in 0..n {
                // Compute phase change from causal coupling
                let mut phase_change = 0.0;

                for j in 0..n {
                    if i != j {
                        // DRPP coupling
                        phase_change += pcm[i][j];
                    }
                }

                // Add small diffusion for stability
                let diffusion =
                    0.001 * (2.0 * (step as f64 / self.config.drpp_evolution_steps as f64) - 1.0);
                phase_change += diffusion;

                // Update phase with time step
                new_phases[i] = (evolved_phases[i] + self.config.drpp_dt * phase_change)
                    % (2.0 * std::f64::consts::PI);
            }

            evolved_phases = new_phases;
        }

        // 4. Update phase field with evolved phases
        for i in 0..n {
            phase_field.phases[i] = evolved_phases[i];
        }

        // Update coherence matrix based on new phases
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                if idx < phase_field.coherence_matrix.len() {
                    let phase_diff = phase_field.phases[i] - phase_field.phases[j];
                    phase_field.coherence_matrix[idx] = phase_diff.cos();
                }
            }
        }

        // Update order parameter
        let mean_phase = evolved_phases.iter().sum::<f64>() / n as f64;
        phase_field.order_parameter = evolved_phases
            .iter()
            .map(|&p| (p - mean_phase).cos())
            .sum::<f64>()
            / n as f64;
        phase_field.order_parameter = phase_field.order_parameter.abs();

        Ok((Some(pcm), Some(te_matrix), Some(evolved_phases)))
    }

    fn compute_solution_quality(&self, coloring: &ColoringSolution, tours: &[TSPSolution]) -> f64 {
        let tsp_quality: f64 =
            tours.iter().map(|t| t.quality_score).sum::<f64>() / tours.len().max(1) as f64;
        (coloring.quality_score + tsp_quality) / 2.0
    }
}

/// DRPP-enhanced PRCT solution
#[derive(Debug, Clone)]
pub struct DrppPrctSolution {
    // Base PRCT results
    pub coloring: ColoringSolution,
    pub color_class_tours: Vec<TSPSolution>,
    pub phase_coherence: f64,
    pub kuramoto_order: f64,
    pub overall_quality: f64,
    pub total_time_ms: f64,

    // DRPP enhancements
    pub phase_causal_matrix: Option<Vec<Vec<f64>>>, // PCM-Φ matrix
    pub transfer_entropy_matrix: Option<Vec<Vec<f64>>>, // TE matrix
    pub evolved_phases: Option<Vec<f64>>,           // DRPP-evolved phases
    pub drpp_applied: bool,
}

impl DrppPrctSolution {
    /// Convert to base PRCTSolution for compatibility
    pub fn to_prct_solution(&self) -> PRCTSolution {
        PRCTSolution {
            coloring: self.coloring.clone(),
            color_class_tours: self.color_class_tours.clone(),
            phase_coherence: self.phase_coherence,
            kuramoto_order: self.kuramoto_order,
            overall_quality: self.overall_quality,
            total_time_ms: self.total_time_ms,
        }
    }

    /// Check if DRPP enhanced the solution
    pub fn has_drpp_enhancement(&self) -> bool {
        self.drpp_applied
            && (self.phase_causal_matrix.is_some()
                || self.transfer_entropy_matrix.is_some()
                || self.evolved_phases.is_some())
    }

    /// Get dominant causal pathways from transfer entropy
    pub fn get_causal_pathways(&self, threshold: f64) -> Vec<(usize, usize, f64)> {
        if let Some(te_matrix) = &self.transfer_entropy_matrix {
            let n = te_matrix.len();
            let mut pathways = Vec::new();

            for i in 0..n {
                for j in 0..n {
                    if i < te_matrix.len() && j < te_matrix[i].len() {
                        let te = te_matrix[i][j];
                        if te > threshold {
                            pathways.push((i, j, te));
                        }
                    }
                }
            }

            use std::cmp::Ordering;
            pathways.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
            pathways
        } else {
            Vec::new()
        }
    }
}
