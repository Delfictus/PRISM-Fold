//! Hypertuning Controller
//!
//! Monitors telemetry metrics and dynamically adjusts pipeline parameters
//! to improve performance and chromatic number reduction.

pub mod action;
pub mod controller;

pub use action::{AdpControl, TelemetryEvent};
pub use controller::HypertuneController;
