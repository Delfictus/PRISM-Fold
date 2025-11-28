//! Shared Types - Zero Dependency Foundation
//!
//! Pure data types shared across neuromorphic, quantum, and PRCT domains.
//! This crate has ZERO dependencies to prevent circular dependency issues.
//!
//! Design Principle: Data structures ONLY, no behavior, no logic.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod coupling_types;
pub mod graph_types;
pub mod neuro_types;
pub mod quantum_types;

#[cfg(feature = "std")]
pub mod config_registry;

pub use coupling_types::*;
pub use graph_types::*;
pub use neuro_types::*;
pub use quantum_types::*;
