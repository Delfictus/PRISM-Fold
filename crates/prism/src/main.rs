//! PRISM-Fold: World-class interactive computational research interface
//!
//! Phase Resonance Integrated Solver Machine for Molecular Folding
//! GPU-accelerated graph coloring and ligand binding site prediction
//! with AI-native conversational interface and real-time visualization.

use anyhow::Result;
use clap::Parser;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use std::io::{self, stdout};

mod ai;
mod streaming;
mod ui;
mod widgets;

pub use ui::App;

/// PRISM-Fold version
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(name = "prism")]
#[command(version = VERSION)]
#[command(about = "PRISM-Fold: World-class GPU-accelerated graph coloring and molecular analysis")]
#[command(long_about = r#"
╔══════════════════════════════════════════════════════════════════════════╗
║  ◆ PRISM-Fold                                                            ║
║                                                                          ║
║  Phase Resonance Integrated Solver Machine for Molecular Folding         ║
║                                                                          ║
║  A bleeding-edge computational research platform featuring:              ║
║  • GPU-accelerated 7-phase optimization pipeline                         ║
║  • Real-time visualization of optimization dynamics                      ║
║  • AI-native conversational interface                                    ║
║  • Ligand binding site prediction with GNN                               ║
║  • Quantum-classical hybrid algorithms                                   ║
║  • Dendritic reservoir computing                                         ║
║                                                                          ║
║  Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.            ║
╚══════════════════════════════════════════════════════════════════════════╝
"#)]
struct Args {
    /// Input file (graph .col or protein .pdb)
    #[arg(short, long)]
    input: Option<String>,

    /// Mode: auto, coloring, biomolecular, materials
    #[arg(short, long, default_value = "auto")]
    mode: String,

    /// Start in non-interactive batch mode
    #[arg(long)]
    batch: bool,

    /// Enable verbose logging to file
    #[arg(short, long)]
    verbose: bool,

    /// GPU device ID
    #[arg(long, default_value = "0")]
    gpu: usize,

    /// Skip TUI, run with minimal output
    #[arg(long)]
    headless: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging (to file in TUI mode)
    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("prism=debug,prism_gpu=debug,prism_pipeline=info")
            .with_writer(std::fs::File::create("prism.log")?)
            .init();
    }

    if args.headless {
        // Headless mode - minimal output, no TUI
        run_headless(args)
    } else {
        // Full TUI mode
        run_tui(args)
    }
}

fn run_tui(args: Args) -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and run
    let mut app = App::new(args.input, args.mode, args.gpu)?;
    let result = app.run(&mut terminal);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    // Handle any errors from the app
    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }

    Ok(())
}

fn run_headless(args: Args) -> Result<()> {
    println!("◆ PRISM-Fold {} - Headless Mode", VERSION);

    if let Some(input) = args.input {
        println!("Processing: {}", input);
        // TODO: Run pipeline in headless mode
        println!("Headless mode not yet implemented. Use interactive mode.");
    } else {
        println!("No input file specified. Use --input <file>");
    }

    Ok(())
}
