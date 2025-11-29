//! PRISM TUI Framework
//!
//! World-class interactive terminal interface with real-time visualization.

mod app;
pub mod event;
mod render;
mod theme;

pub use app::App;
pub use event::{Event, PipelineEvent, GpuEvent};
pub use theme::Theme;
