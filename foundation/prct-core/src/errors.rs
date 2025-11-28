//! Domain Errors

use thiserror::Error;

#[derive(Error, Debug)]
pub enum PRCTError {
    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),

    #[error("Coloring failed: {0}")]
    ColoringFailed(String),

    #[error("TSP solver failed: {0}")]
    TSPFailed(String),

    #[error("DRPP algorithm failed: {0}")]
    DrppFailed(String),

    #[error("Physics coupling failed: {0}")]
    CouplingFailed(String),

    #[error("Neuromorphic processing failed: {0}")]
    NeuromorphicFailed(String),

    #[error("Quantum processing failed: {0}")]
    QuantumFailed(String),

    #[error("Port operation failed: {0}")]
    PortError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("GPU error: {0}")]
    GpuError(String),

    #[error("Transfer entropy computation failed: {0}")]
    TransferEntropyFailed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, PRCTError>;
