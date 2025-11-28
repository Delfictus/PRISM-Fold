//! Telemetry Handle for Real-Time Metric Collection
//!
//! Provides thread-safe metric collection with:
//! - Buffered in-memory storage for monitoring
//! - Continuous JSONL file writing
//! - Lock-free recording via channels

use crate::errors::*;
use crate::telemetry::run_metric::{RunMetric, RunSummary};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::collections::VecDeque;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

/// Telemetry handle for metric collection
pub struct TelemetryHandle {
    /// Channel for sending metrics
    sender: Sender<TelemetryMessage>,

    /// In-memory circular buffer for live monitoring
    buffer: Arc<RwLock<VecDeque<RunMetric>>>,

    /// Writer thread handle
    writer_thread: Option<thread::JoinHandle<Result<()>>>,

    /// Run ID for this session
    run_id: String,

    /// Output file path
    output_path: PathBuf,
}

/// Internal message types
enum TelemetryMessage {
    Metric(RunMetric),
    Finalize(RunSummary),
    Shutdown,
}

impl TelemetryHandle {
    /// Create new telemetry handle
    ///
    /// # Arguments
    /// - `run_id`: Unique identifier for this run
    /// - `buffer_len`: Size of in-memory circular buffer (default: 1000)
    ///
    /// # Returns
    /// Handle with background writer thread
    pub fn new(run_id: &str, buffer_len: usize) -> Result<Self> {
        let output_dir = Path::new("target/run_artifacts");
        create_dir_all(output_dir).map_err(|e| {
            PRCTError::ConfigError(format!("Failed to create artifacts dir: {}", e))
        })?;

        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("live_metrics_{}_{}.jsonl", run_id, timestamp);
        let output_path = output_dir.join(filename);

        let file = File::create(&output_path)
            .map_err(|e| PRCTError::ConfigError(format!("Failed to create metrics file: {}", e)))?;
        let writer = Arc::new(Mutex::new(BufWriter::new(file)));

        let (sender, receiver) = bounded(1000);
        let buffer = Arc::new(RwLock::new(VecDeque::with_capacity(buffer_len)));

        let writer_clone = writer.clone();
        let buffer_clone = buffer.clone();
        let buffer_capacity = buffer_len;

        let writer_thread = thread::spawn(move || {
            Self::writer_loop(receiver, writer_clone, buffer_clone, buffer_capacity)
        });

        Ok(Self {
            sender,
            buffer,
            writer_thread: Some(writer_thread),
            run_id: run_id.to_string(),
            output_path,
        })
    }

    /// Record metric (non-blocking)
    pub fn record(&self, metric: RunMetric) {
        if let Err(e) = self.sender.send(TelemetryMessage::Metric(metric)) {
            eprintln!("Warning: Failed to send metric: {}", e);
        }
    }

    /// Get snapshot of recent metrics
    pub fn snapshot(&self) -> Vec<RunMetric> {
        self.buffer
            .read()
            .map(|buf| buf.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Finalize run with summary
    pub fn finalize(&self, summary: RunSummary) {
        if let Err(e) = self.sender.send(TelemetryMessage::Finalize(summary)) {
            eprintln!("Warning: Failed to send summary: {}", e);
        }
    }

    /// Get run ID
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Get output file path
    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    /// Writer thread loop
    fn writer_loop(
        receiver: Receiver<TelemetryMessage>,
        writer: Arc<Mutex<BufWriter<File>>>,
        buffer: Arc<RwLock<VecDeque<RunMetric>>>,
        buffer_capacity: usize,
    ) -> Result<()> {
        loop {
            match receiver.recv() {
                Ok(TelemetryMessage::Metric(metric)) => {
                    // Write to file
                    if let Ok(mut w) = writer.lock() {
                        let json = serde_json::to_string(&metric)
                            .map_err(|e| PRCTError::ConfigError(format!("JSON error: {}", e)))?;
                        writeln!(w, "{}", json)
                            .map_err(|e| PRCTError::ConfigError(format!("Write error: {}", e)))?;
                        w.flush()
                            .map_err(|e| PRCTError::ConfigError(format!("Flush error: {}", e)))?;
                    }

                    // Add to circular buffer
                    if let Ok(mut buf) = buffer.write() {
                        if buf.len() >= buffer_capacity {
                            buf.pop_front();
                        }
                        buf.push_back(metric);
                    }
                }
                Ok(TelemetryMessage::Finalize(summary)) => {
                    if let Ok(mut w) = writer.lock() {
                        writeln!(w, "--- RUN SUMMARY ---")
                            .map_err(|e| PRCTError::ConfigError(format!("Write error: {}", e)))?;
                        let json = serde_json::to_string_pretty(&summary)
                            .map_err(|e| PRCTError::ConfigError(format!("JSON error: {}", e)))?;
                        writeln!(w, "{}", json)
                            .map_err(|e| PRCTError::ConfigError(format!("Write error: {}", e)))?;
                        w.flush()
                            .map_err(|e| PRCTError::ConfigError(format!("Flush error: {}", e)))?;
                    }
                }
                Ok(TelemetryMessage::Shutdown) | Err(_) => {
                    break;
                }
            }
        }
        Ok(())
    }
}

impl Drop for TelemetryHandle {
    fn drop(&mut self) {
        // Send shutdown signal
        let _ = self.sender.send(TelemetryMessage::Shutdown);

        // Wait for writer thread
        if let Some(handle) = self.writer_thread.take() {
            if let Err(e) = handle.join() {
                eprintln!("Warning: Writer thread panicked: {:?}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::run_metric::{PhaseExecMode, PhaseName};

    #[test]
    fn test_telemetry_handle() {
        let handle = TelemetryHandle::new("test_run", 100).expect("Failed to create handle");

        let metric = RunMetric::new(
            PhaseName::Thermodynamic,
            "test_step",
            115,
            0,
            100.0,
            PhaseExecMode::gpu_success(Some(2)),
        );

        handle.record(metric.clone());

        // Give writer thread time to process
        std::thread::sleep(std::time::Duration::from_millis(100));

        let snapshot = handle.snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].step, "test_step");
    }
}
