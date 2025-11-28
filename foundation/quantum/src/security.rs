//! Security validation for quantum operations
//! Simplified from PRCT engine security framework

use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Security error types
#[derive(Error, Debug)]
pub enum SecurityError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error(
        "Resource exhaustion: {resource_type} has {current_count} but max allowed is {max_allowed}"
    )]
    ResourceExhaustion {
        resource_type: String,
        current_count: usize,
        max_allowed: usize,
    },

    #[error("Operation timeout after {elapsed:?}")]
    OperationTimeout { elapsed: Duration },

    #[error("Security violation: {0}")]
    SecurityViolation(String),
}

/// Security validator for quantum operations
#[derive(Debug, Clone)]
pub struct SecurityValidator {
    /// Operation start times for timeout checking
    operation_starts: HashMap<String, Instant>,
    /// Maximum operation duration
    max_operation_duration: Duration,
}

impl SecurityValidator {
    /// Create new security validator
    pub fn new() -> Result<Self, SecurityError> {
        Ok(Self {
            operation_starts: HashMap::new(),
            max_operation_duration: Duration::from_secs(60), // 1 minute timeout
        })
    }

    /// Start monitoring an operation
    pub fn start_operation(&mut self, operation_name: &str) {
        self.operation_starts
            .insert(operation_name.to_string(), Instant::now());
    }

    /// Check if an operation has timed out
    pub fn check_timeout(&self, operation_name: &str) -> Result<(), SecurityError> {
        if let Some(start_time) = self.operation_starts.get(operation_name) {
            let elapsed = start_time.elapsed();
            if elapsed > self.max_operation_duration {
                return Err(SecurityError::OperationTimeout { elapsed });
            }
        }
        Ok(())
    }

    /// Validate numerical input
    pub fn validate_numerical_input(&self, value: f64, name: &str) -> Result<(), SecurityError> {
        if !value.is_finite() {
            return Err(SecurityError::InvalidInput(format!(
                "{} must be finite, got: {}",
                name, value
            )));
        }
        Ok(())
    }

    /// Validate array dimensions
    pub fn validate_array_dimensions(
        &self,
        rows: usize,
        cols: usize,
        max_size: usize,
    ) -> Result<(), SecurityError> {
        if rows * cols > max_size {
            return Err(SecurityError::ResourceExhaustion {
                resource_type: "array elements".to_string(),
                current_count: rows * cols,
                max_allowed: max_size,
            });
        }
        Ok(())
    }
}
