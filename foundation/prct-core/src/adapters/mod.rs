//! PRCT Adapter Implementations
//!
//! Concrete implementations of PRCT port interfaces using existing foundation components.
//! These adapters connect the domain logic to GPU-accelerated infrastructure.

pub mod coupling_adapter;
#[cfg(feature = "cuda")]
pub mod neuromorphic_adapter;
pub mod quantum_adapter;

pub use coupling_adapter::CouplingAdapter;
#[cfg(feature = "cuda")]
pub use neuromorphic_adapter::NeuromorphicAdapter;
pub use quantum_adapter::QuantumAdapter;
