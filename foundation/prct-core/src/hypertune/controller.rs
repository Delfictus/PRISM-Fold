//! Hypertuning Controller Implementation
//!
//! Monitors real-time telemetry and adjusts pipeline parameters
//! to optimize performance.

use crate::errors::*;
use crate::hypertune::action::{AdpControl, TelemetryEvent};
use crate::telemetry::{PhaseName, RunMetric};
use crossbeam_channel::Receiver;
use std::collections::VecDeque;
use std::time::Duration;

/// Configuration for hypertuning controller
#[derive(Debug, Clone)]
pub struct HypertuneConfig {
    /// Window size for detecting stalls
    pub stall_window: usize,

    /// Threshold for no improvement (iterations)
    pub stall_threshold: usize,

    /// Minimum efficiency threshold
    pub efficiency_threshold: f64,

    /// High conflict threshold
    pub conflict_threshold: usize,
}

impl Default for HypertuneConfig {
    fn default() -> Self {
        Self {
            stall_window: 50,
            stall_threshold: 100,
            efficiency_threshold: 0.5,
            conflict_threshold: 1000,
        }
    }
}

/// Hypertuning controller
pub struct HypertuneController {
    /// Telemetry receiver
    telemetry_rx: Receiver<RunMetric>,

    /// Recent metrics buffer
    metric_buffer: VecDeque<RunMetric>,

    /// Controller configuration
    config: HypertuneConfig,

    /// Detected events
    events: Vec<TelemetryEvent>,
}

impl HypertuneController {
    /// Create new controller
    pub fn new(telemetry_rx: Receiver<RunMetric>, config: HypertuneConfig) -> Self {
        Self {
            telemetry_rx,
            metric_buffer: VecDeque::with_capacity(config.stall_window),
            config,
            events: Vec::new(),
        }
    }

    /// Run controller loop (blocking)
    ///
    /// Continuously monitors metrics and generates control actions
    pub fn run(&mut self) -> Result<()> {
        loop {
            match self.telemetry_rx.recv_timeout(Duration::from_secs(1)) {
                Ok(metric) => {
                    self.process_metric(metric)?;
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // Continue waiting
                    continue;
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // Telemetry stream closed, exit
                    break;
                }
            }
        }

        Ok(())
    }

    /// Process single metric
    fn process_metric(&mut self, metric: RunMetric) -> Result<()> {
        // Add to buffer
        self.metric_buffer.push_back(metric.clone());

        if self.metric_buffer.len() > self.config.stall_window {
            self.metric_buffer.pop_front();
        }

        // Detect events
        let events = self.detect_events();

        // Generate actions
        for event in events {
            if let Some(action) = self.choose_action(&event) {
                eprintln!(
                    "[HYPERTUNE] Event: {:?} -> Action: {}",
                    event,
                    action.description()
                );
                // In a full implementation, send action to pipeline via channel
            }

            self.events.push(event);
        }

        Ok(())
    }

    /// Detect telemetry events
    fn detect_events(&self) -> Vec<TelemetryEvent> {
        let mut events = Vec::new();

        if self.metric_buffer.is_empty() {
            return events;
        }

        // Check for stalled chromatic number
        if self.metric_buffer.len() >= self.config.stall_threshold {
            let recent = &self.metric_buffer;
            let first_chromatic = recent.front().map(|m| m.chromatic_number).unwrap_or(0);
            let last_chromatic = recent.back().map(|m| m.chromatic_number).unwrap_or(0);

            if first_chromatic == last_chromatic && first_chromatic > 0 {
                events.push(TelemetryEvent::NoImprovement {
                    iterations: recent.len(),
                    chromatic_stuck_at: first_chromatic,
                });
            }
        }

        // Check for high conflicts
        if let Some(last) = self.metric_buffer.back() {
            if last.conflicts > self.config.conflict_threshold {
                events.push(TelemetryEvent::HighConflicts {
                    phase: last.phase,
                    conflicts: last.conflicts,
                    threshold: self.config.conflict_threshold,
                });
            }
        }

        // Check for phase-specific stalls
        for phase in [
            PhaseName::Thermodynamic,
            PhaseName::Quantum,
            PhaseName::Memetic,
        ] {
            let phase_metrics: Vec<_> = self
                .metric_buffer
                .iter()
                .filter(|m| m.phase == phase)
                .collect();

            if phase_metrics.len() >= 10 {
                let total_duration: f64 = phase_metrics.iter().map(|m| m.duration_ms).sum();
                let avg_duration = total_duration / phase_metrics.len() as f64;

                // If recent iterations are taking much longer
                if let Some(last) = phase_metrics.last() {
                    if last.duration_ms > avg_duration * 2.0 {
                        events.push(TelemetryEvent::LowEfficiency {
                            phase,
                            metric: "duration".to_string(),
                            current_value: last.duration_ms,
                            threshold: avg_duration * 2.0,
                        });
                    }
                }
            }
        }

        events
    }

    /// Choose control action for event
    fn choose_action(&self, event: &TelemetryEvent) -> Option<AdpControl> {
        match event {
            TelemetryEvent::PhaseStalled { phase, .. } => match phase {
                PhaseName::Thermodynamic => {
                    Some(AdpControl::AdjustThermoTemps { delta_percent: 10 })
                }
                PhaseName::Quantum => Some(AdpControl::IncreaseQuantumIterations {
                    additional_iters: 100,
                }),
                PhaseName::Memetic => Some(AdpControl::AdjustMemeticPopulation { delta: 10 }),
                _ => None,
            },
            TelemetryEvent::NoImprovement { iterations, .. } => {
                if *iterations > 200 {
                    Some(AdpControl::ResetToBaseline)
                } else {
                    Some(AdpControl::SetTeWeight { weight: 2.0 })
                }
            }
            TelemetryEvent::HighConflicts { phase, .. } => match phase {
                PhaseName::Thermodynamic => {
                    Some(AdpControl::AdjustThermoTemps { delta_percent: -10 })
                }
                _ => None,
            },
            TelemetryEvent::LowEfficiency { phase, .. } => match phase {
                PhaseName::Memetic => Some(AdpControl::AdjustMemeticPopulation { delta: -10 }),
                _ => None,
            },
        }
    }

    /// Get detected events
    pub fn events(&self) -> &[TelemetryEvent] {
        &self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::PhaseExecMode;
    use crossbeam_channel::bounded;

    #[test]
    fn test_controller_creation() {
        let (tx, rx) = bounded(100);
        let config = HypertuneConfig::default();
        let _controller = HypertuneController::new(rx, config);
    }

    #[test]
    fn test_event_detection() {
        let (tx, rx) = bounded(100);
        let config = HypertuneConfig {
            stall_window: 10,
            stall_threshold: 5,
            efficiency_threshold: 0.5,
            conflict_threshold: 50,
        };

        let mut controller = HypertuneController::new(rx, config);

        // Send high conflict metric
        let metric = RunMetric::new(
            PhaseName::Thermodynamic,
            "test",
            115,
            100,
            100.0,
            PhaseExecMode::gpu_success(Some(0)),
        );

        tx.send(metric).ok();
        // Would need to call run() or process_metric() to test event detection
    }
}
