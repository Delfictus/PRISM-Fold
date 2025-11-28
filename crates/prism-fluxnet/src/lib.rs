//! # prism-fluxnet
//!
//! Universal reinforcement learning controller for PRISM v2.
//!
//! FluxNet provides a unified RL framework that integrates across all 7 phases,
//! learning optimal parameter adjustments and phase transitions.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │        UniversalRLController                │
//! │  ┌────────────────────────────────────────┐ │
//! │  │  Phase-specific Q-tables (HashMap)     │ │
//! │  │  - "Phase0": [4096 states × 64 actions]│ │
//! │  │  - "Phase1": [4096 states × 64 actions]│ │
//! │  │  - ...                                  │ │
//! │  └────────────────────────────────────────┘ │
//! │  ┌────────────────────────────────────────┐ │
//! │  │  Shared Replay Buffer (VecDeque)       │ │
//! │  │  Transitions across all phases         │ │
//! │  └────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────┘
//!          ▲                          │
//!          │  State observation       │  Action selection
//!          │                          ▼
//! ┌─────────────────────────────────────────────┐
//! │       Phase Controllers (prism-phases)      │
//! │  Phase0, Phase1, ..., Phase7                │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! Implements PRISM GPU Plan §3: Universal FluxNet RL.

pub mod core;
pub mod curriculum;

// Re-export commonly used items
pub use core::actions::UniversalAction;
pub use core::controller::{RLConfig, UniversalRLController};
pub use core::state::{DiscretizationMode, UniversalRLState};

// Re-export curriculum types
pub use curriculum::{
    CurriculumBank, CurriculumEntry, CurriculumMetadata, DifficultyProfile, GraphStats,
};
