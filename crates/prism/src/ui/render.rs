//! PRISM TUI Rendering
//!
//! World-class visualization rendering for the terminal interface.

use ratatui::{
    prelude::*,
    widgets::*,
};

use super::app::{App, AppMode, PhaseState, Focus};
use super::theme::Theme;
use crate::widgets;

/// Main render function
pub fn render(app: &App, frame: &mut Frame) {
    let area = frame.area();

    // Clear background
    frame.render_widget(
        Block::default().style(Style::default().bg(Theme::BG_PRIMARY)),
        area,
    );

    // Main layout: header, content, footer
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(20),    // Content
            Constraint::Length(3),  // Dialogue input
        ])
        .split(area);

    // Render header
    render_header(app, frame, main_layout[0]);

    // Render main content based on mode
    match app.mode {
        AppMode::GraphColoring => render_graph_mode(app, frame, main_layout[1]),
        AppMode::Biomolecular => render_biomolecular_mode(app, frame, main_layout[1]),
        AppMode::Welcome => render_welcome(app, frame, main_layout[1]),
        _ => render_welcome(app, frame, main_layout[1]),
    }

    // Render dialogue input
    render_dialogue_input(app, frame, main_layout[2]);

    // Render help overlay if active
    if app.show_help {
        render_help_overlay(frame, area);
    }
}

/// Render the header bar
fn render_header(app: &App, frame: &mut Frame, area: Rect) {
    let header_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(30),  // Title
            Constraint::Min(20),     // Mode/file
            Constraint::Length(35),  // GPU status
        ])
        .split(area);

    // Title
    let title = Paragraph::new(format!(" ◆ PRISM-Fold v{}", env!("CARGO_PKG_VERSION")))
        .style(Theme::title())
        .block(Block::default().borders(Borders::ALL).border_style(Theme::panel_border()));
    frame.render_widget(title, header_layout[0]);

    // Mode and file
    let mode_text = match app.mode {
        AppMode::GraphColoring => "Graph Coloring",
        AppMode::Biomolecular => "Biomolecular",
        AppMode::Materials => "Materials",
        AppMode::Welcome => "Welcome",
    };
    let file_text = app.input_path.as_deref().unwrap_or("No file loaded");
    let mode = Paragraph::new(format!(" {} │ {}", mode_text, file_text))
        .style(Theme::normal())
        .block(Block::default().borders(Borders::ALL).border_style(Theme::panel_border()));
    frame.render_widget(mode, header_layout[1]);

    // GPU status
    let gpu_color = Theme::gpu_util_color(app.gpu.utilization);
    let gpu_bar = format!(
        "{}",
        "█".repeat((app.gpu.utilization / 10.0) as usize)
    );
    let gpu_text = format!(
        " {} {} {:.0}% {}°C",
        app.gpu.name,
        gpu_bar,
        app.gpu.utilization,
        app.gpu.temperature
    );
    let gpu = Paragraph::new(gpu_text)
        .style(Style::default().fg(gpu_color))
        .block(Block::default().borders(Borders::ALL).border_style(Theme::panel_border()));
    frame.render_widget(gpu, header_layout[2]);
}

/// Render graph coloring mode
fn render_graph_mode(app: &App, frame: &mut Frame, area: Rect) {
    // Split into main content and dialogue
    let content_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(65),  // Visualizations
            Constraint::Percentage(35),  // Dialogue + metrics
        ])
        .split(area);

    // Left side: visualizations
    let viz_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50),  // Graph + energy landscape
            Constraint::Percentage(25),  // Replica swarm
            Constraint::Percentage(25),  // Quantum + dendritic + kernels
        ])
        .split(content_layout[0]);

    // Top row: graph and energy landscape
    let top_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(viz_layout[0]);

    render_graph_visualization(app, frame, top_row[0]);
    render_energy_landscape(app, frame, top_row[1]);

    // Middle: replica swarm
    render_replica_swarm(app, frame, viz_layout[1]);

    // Bottom row: quantum, dendritic, kernels
    let bottom_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(viz_layout[2]);

    render_quantum_state(app, frame, bottom_row[0]);
    render_dendritic_activity(app, frame, bottom_row[1]);
    render_gpu_kernels(app, frame, bottom_row[2]);

    // Right side: dialogue and metrics
    let right_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),     // Dialogue
            Constraint::Length(10),  // Pipeline
            Constraint::Length(8),   // Convergence
        ])
        .split(content_layout[1]);

    render_dialogue_history(app, frame, right_layout[0]);
    render_pipeline_flow(app, frame, right_layout[1]);
    render_convergence_chart(app, frame, right_layout[2]);
}

/// Render biomolecular mode
fn render_biomolecular_mode(app: &App, frame: &mut Frame, area: Rect) {
    let content_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(55),  // Protein visualization
            Constraint::Percentage(45),  // Analysis panels
        ])
        .split(area);

    // Left: protein structure
    render_protein_structure(app, frame, content_layout[0]);

    // Right: analysis panels
    let analysis_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),  // Pocket analysis
            Constraint::Length(8),   // GNN attention
            Constraint::Length(6),   // Pharmacophore
            Constraint::Min(8),      // Dialogue
        ])
        .split(content_layout[1]);

    render_pocket_analysis(app, frame, analysis_layout[0]);
    render_gnn_attention(app, frame, analysis_layout[1]);
    render_pharmacophore(app, frame, analysis_layout[2]);
    render_dialogue_history(app, frame, analysis_layout[3]);
}

/// Render welcome screen
fn render_welcome(app: &App, frame: &mut Frame, area: Rect) {
    let welcome_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(60),
            Constraint::Percentage(40),
        ])
        .split(area);

    // Welcome text
    let welcome_text = vec![
        Line::from(""),
        Line::from(Span::styled("  Welcome to PRISM-Fold", Theme::title())),
        Line::from(""),
        Line::from(Span::styled("  Phase Resonance Integrated Solver Machine", Theme::dim())),
        Line::from(Span::styled("  for Molecular Folding", Theme::dim())),
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled("  Get Started:", Theme::header())),
        Line::from(""),
        Line::from("  • load <file.col>    Load a DIMACS graph"),
        Line::from("  • load <file.pdb>    Load a protein structure"),
        Line::from("  • run                Start optimization"),
        Line::from("  • help               Show all commands"),
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled("  Features:", Theme::header())),
        Line::from(""),
        Line::from("  • GPU-accelerated 7-phase optimization"),
        Line::from("  • Quantum-classical hybrid algorithms"),
        Line::from("  • GNN-based binding site prediction"),
        Line::from("  • Real-time visualization"),
        Line::from("  • AI research assistant"),
    ];

    let welcome = Paragraph::new(welcome_text)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Theme::panel_border())
            .title(Span::styled(" ◆ PRISM-Fold ", Theme::panel_title())));
    frame.render_widget(welcome, welcome_layout[0]);

    // Dialogue on right
    render_dialogue_history(app, frame, welcome_layout[1]);
}

/// Render graph visualization
fn render_graph_visualization(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Live Graph Coloring ", Theme::panel_title()));

    // Sample graph visualization (would be real data in production)
    let graph_text = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("        "),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[0])),
            Span::raw("───"),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[1])),
            Span::raw("───"),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[2])),
        ]),
        Line::from(vec![
            Span::raw("       ╱│╲   │   ╱│╲"),
        ]),
        Line::from(vec![
            Span::raw("     "),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[3])),
            Span::raw(" │ "),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[4])),
            Span::raw("─┼─"),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[3])),
            Span::raw(" │ "),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[1])),
        ]),
        Line::from(vec![
            Span::raw("      ╲│╱   │   ╲│╱"),
        ]),
        Line::from(vec![
            Span::raw("       "),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[2])),
            Span::raw("───"),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[0])),
            Span::raw("───"),
            Span::styled("●", Style::default().fg(Theme::GRAPH_COLORS[4])),
        ]),
        Line::from(""),
        Line::from(format!(
            "  Colors: {} │ Conflicts: {} │ Vertices: 500",
            app.optimization.colors,
            app.optimization.conflicts
        )),
    ];

    let graph = Paragraph::new(graph_text).block(block);
    frame.render_widget(graph, area);
}

/// Render energy landscape
fn render_energy_landscape(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Energy Landscape ", Theme::panel_title()));

    let landscape = vec![
        Line::from(""),
        Line::from("       ╭───╮"),
        Line::from("      ╱     ╲    ╭─╮"),
        Line::from("     ╱   ◆   ╲  ╱   ╲"),
        Line::from("    ╱  here   ╲╱     ╲  ← target"),
        Line::from("   ╱           ╲      ╲"),
        Line::from("  ╱             ╲      ╲"),
        Line::from("  ─────────────────────────"),
        Line::from("    Iterations ────────→"),
    ];

    let widget = Paragraph::new(landscape).block(block);
    frame.render_widget(widget, area);
}

/// Render replica swarm
fn render_replica_swarm(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Parallel Tempering Replicas ", Theme::panel_title()));

    let temps = [0.01, 0.15, 0.38, 0.72, 1.20, 1.89];
    let colors = [22, 24, 27, 29, 33, 38];

    let mut lines = vec![Line::from("")];
    for (i, (t, c)) in temps.iter().zip(colors.iter()).enumerate() {
        let bar_len = (40.0 * (1.0 - (*c as f64 - 22.0) / 20.0)) as usize;
        let bar = "━".repeat(bar_len);
        let temp_color = Theme::temperature_color(i as f64 / 5.0);

        let is_best = i == 0;
        let suffix = if is_best { "◆ BEST" } else { "" };

        lines.push(Line::from(vec![
            Span::styled(format!("  T={:.2} ", t), Style::default().fg(temp_color)),
            Span::styled(bar, Style::default().fg(temp_color)),
            Span::styled(format!(" {}c {}", c, suffix), Theme::normal()),
        ]));
    }

    let widget = Paragraph::new(lines).block(block);
    frame.render_widget(widget, area);
}

/// Render quantum state
fn render_quantum_state(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Quantum State ", Theme::panel_title()));

    let amps = [(22, 0.67), (23, 0.24), (24, 0.09)];
    let mut lines = vec![
        Line::from(Span::styled("  |ψ⟩ superposition:", Theme::dim())),
        Line::from(""),
    ];

    for (color, amp) in amps {
        let bar_len = (amp * 15.0) as usize;
        let bar = "█".repeat(bar_len) + &"░".repeat(15 - bar_len);
        lines.push(Line::from(format!("  |{}⟩ {} {:.2}", color, bar, amp)));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(format!("  coherence: {:.2}", app.optimization.quantum_coherence)));

    let widget = Paragraph::new(lines).block(block);
    frame.render_widget(widget, area);
}

/// Render dendritic activity
fn render_dendritic_activity(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Dendritic Activity ", Theme::panel_title()));

    let activity = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("   ⚡", Style::default().fg(Theme::ACCENT)),
            Span::raw("    "),
            Span::styled("⚡", Style::default().fg(Theme::ACCENT)),
        ]),
        Line::from("  ╱ ╲  ╱ ╲  ╭─⚡"),
        Line::from(vec![
            Span::raw(" "),
            Span::styled("⚡", Style::default().fg(Theme::ACCENT)),
            Span::raw("   "),
            Span::styled("⚡", Style::default().fg(Theme::ACCENT)),
            Span::raw("    "),
            Span::styled("⚡", Style::default().fg(Theme::ACCENT)),
            Span::raw("   ╲"),
        ]),
        Line::from("  ╲ ╱  ╲ ╱  ╲   ⚡"),
        Line::from("   ⚡    ⚡────⚡"),
        Line::from(""),
        Line::from("  firing: 847/2048"),
    ];

    let widget = Paragraph::new(activity).block(block);
    frame.render_widget(widget, area);
}

/// Render GPU kernels
fn render_gpu_kernels(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" GPU Kernels ", Theme::panel_title()));

    let kernels = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw(" thermodynamic.ptx"),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("████████", Style::default().fg(Theme::SUCCESS)),
            Span::styled("░░", Style::default().fg(Theme::TEXT_DIM)),
            Span::raw(" 82%"),
        ]),
        Line::from(""),
        Line::from(" quantum.ptx"),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("░░░░░░░░░░", Style::default().fg(Theme::TEXT_DIM)),
            Span::raw(" idle"),
        ]),
    ];

    let widget = Paragraph::new(kernels).block(block);
    frame.render_widget(widget, area);
}

/// Render dialogue history
fn render_dialogue_history(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(if app.focus == Focus::Dialogue {
            Style::default().fg(Theme::ACCENT)
        } else {
            Theme::panel_border()
        })
        .title(Span::styled(" AI Assistant ", Theme::panel_title()));

    let messages: Vec<Line> = app.dialogue.messages.iter()
        .flat_map(|msg| {
            let style = if msg.is_user {
                Style::default().fg(Theme::ACCENT_SECONDARY)
            } else {
                Theme::normal()
            };
            let prefix = if msg.is_user { "> " } else { "  " };

            msg.content.lines()
                .map(|line| Line::from(Span::styled(format!("{}{}", prefix, line), style)))
                .collect::<Vec<_>>()
        })
        .collect();

    let widget = Paragraph::new(messages)
        .block(block)
        .wrap(Wrap { trim: true })
        .scroll((app.dialogue.scroll_offset as u16, 0));
    frame.render_widget(widget, area);
}

/// Render pipeline flow
fn render_pipeline_flow(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Phase Pipeline ", Theme::panel_title()));

    let mut lines = vec![Line::from("")];

    // Phase flow line
    let mut phase_spans = vec![Span::raw("  ")];
    for (i, phase) in app.phases.iter().enumerate() {
        let (symbol, style) = match phase.status {
            PhaseState::Completed => ("✓", Theme::success()),
            PhaseState::Running => ("▶", Style::default().fg(Theme::ACCENT)),
            PhaseState::Failed => ("✗", Theme::error()),
            PhaseState::Pending => ("○", Theme::dim()),
        };

        phase_spans.push(Span::styled(symbol, style));
        if i < app.phases.len() - 1 {
            phase_spans.push(Span::styled("─", Theme::dim()));
        }
    }
    lines.push(Line::from(phase_spans));

    // Phase names
    let names: String = app.phases.iter()
        .map(|p| format!("{:^3}", &p.name[..2]))
        .collect::<Vec<_>>()
        .join("");
    lines.push(Line::from(Span::styled(format!("  {}", names), Theme::dim())));

    // Progress bar for running phase
    if let Some(phase) = app.phases.iter().find(|p| p.status == PhaseState::Running) {
        let bar_width = 30;
        let filled = ((phase.progress / 100.0) * bar_width as f64) as usize;
        let bar = format!(
            "  [{}{}] {:.0}%",
            "█".repeat(filled),
            "░".repeat(bar_width - filled),
            phase.progress
        );
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(bar, Style::default().fg(Theme::progress_color(phase.progress)))));
    }

    let widget = Paragraph::new(lines).block(block);
    frame.render_widget(widget, area);
}

/// Render convergence chart
fn render_convergence_chart(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Convergence ", Theme::panel_title()));

    // Simple ASCII chart
    let chart = vec![
        Line::from("  72 ┤••"),
        Line::from("     │  •••"),
        Line::from("  48 ┤     ••••"),
        Line::from("     │         •••••"),
        Line::from(vec![
            Span::raw("  24 ┤              "),
            Span::styled("••••◆", Style::default().fg(Theme::SUCCESS)),
        ]),
        Line::from("     └─────────────────"),
    ];

    let widget = Paragraph::new(chart).block(block);
    frame.render_widget(widget, area);
}

/// Render protein structure (biomolecular mode)
fn render_protein_structure(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Protein Structure ", Theme::panel_title()));

    let structure = vec![
        Line::from(""),
        Line::from("                    ┌───── Active Site ─────┐"),
        Line::from(vec![
            Span::raw("                   ╱    "),
            Span::styled("● High druggability", Style::default().fg(Theme::ERROR)),
            Span::raw(" ╲"),
        ]),
        Line::from(vec![
            Span::raw("       "),
            Span::styled("╭━━━━━━━╮", Style::default().fg(Theme::PROTEIN_HELIX)),
            Span::raw("  ╱                          ╲  "),
            Span::styled("╭━━━━━━━╮", Style::default().fg(Theme::PROTEIN_HELIX)),
        ]),
        Line::from(vec![
            Span::styled(" ░░░░░░┃██████┃", Style::default().fg(Theme::PROTEIN_HELIX)),
            Span::raw("━━        ╭─── Ligand ───╮      "),
            Span::styled("┃██████┃░░░░░░", Style::default().fg(Theme::PROTEIN_HELIX)),
        ]),
        Line::from(vec![
            Span::raw("       "),
            Span::styled("┃██████┃", Style::default().fg(Theme::PROTEIN_HELIX)),
            Span::raw("    ░░░░░░│   "),
            Span::styled("⬡     ⬡", Style::default().fg(Theme::PROTEIN_LIGAND)),
            Span::raw("    │░░░░░░"),
            Span::styled("┃██████┃", Style::default().fg(Theme::PROTEIN_HELIX)),
        ]),
        Line::from(vec![
            Span::raw("       "),
            Span::styled("╰━━━━━━━╯", Style::default().fg(Theme::PROTEIN_HELIX)),
            Span::raw("  ╲░░░░░░│  "),
            Span::styled("⬡─⬡───⬡─⬡", Style::default().fg(Theme::PROTEIN_LIGAND)),
            Span::raw("   │░░░░░╱"),
            Span::styled("╰━━━━━━━╯", Style::default().fg(Theme::PROTEIN_HELIX)),
        ]),
        Line::from("          │        ╲░░░░░╰──────────────╯░░░╱        │"),
        Line::from("     "),
        Line::from(vec![
            Span::raw("         "),
            Span::styled("┌─────────┐", Style::default().fg(Theme::PROTEIN_SHEET)),
            Span::raw("     ╲░░░░░ Pocket ░░░░░╱     "),
            Span::styled("┌─────────┐", Style::default().fg(Theme::PROTEIN_SHEET)),
        ]),
        Line::from(vec![
            Span::raw("         "),
            Span::styled("│▓▓▓▓▓▓▓▓▓│", Style::default().fg(Theme::PROTEIN_SHEET)),
            Span::raw("         β-sheet         "),
            Span::styled("│▓▓▓▓▓▓▓▓▓│", Style::default().fg(Theme::PROTEIN_SHEET)),
        ]),
        Line::from(vec![
            Span::raw("         "),
            Span::styled("└─────────┘", Style::default().fg(Theme::PROTEIN_SHEET)),
            Span::raw("                          "),
            Span::styled("└─────────┘", Style::default().fg(Theme::PROTEIN_SHEET)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("━━━", Style::default().fg(Theme::PROTEIN_HELIX)),
            Span::raw(" α-helix  "),
            Span::styled("▓▓▓", Style::default().fg(Theme::PROTEIN_SHEET)),
            Span::raw(" β-sheet  "),
            Span::styled("░░░", Style::default().fg(Theme::PROTEIN_POCKET)),
            Span::raw(" pocket  "),
            Span::styled("⬡", Style::default().fg(Theme::PROTEIN_LIGAND)),
            Span::raw(" ligand"),
        ]),
    ];

    let widget = Paragraph::new(structure).block(block);
    frame.render_widget(widget, area);
}

/// Render pocket analysis
fn render_pocket_analysis(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Binding Pocket Analysis ", Theme::panel_title()));

    let analysis = vec![
        Line::from(""),
        Line::from("  Volume:    486 Å³     Depth: 12.4 Å"),
        Line::from("  Enclosure: 0.73"),
        Line::from(""),
        Line::from(vec![
            Span::raw("  Hydrophobic: "),
            Span::styled("████████", Style::default().fg(Theme::PHARM_HYDROPHOBIC)),
            Span::styled("░░", Style::default().fg(Theme::TEXT_DIM)),
            Span::raw(" 76%"),
        ]),
        Line::from("  H-bond sites: 8      Metal: Mg²⁺"),
        Line::from(vec![
            Span::raw("  Druggability: "),
            Span::styled("██████████", Style::default().fg(Theme::SUCCESS)),
            Span::raw(" 94%"),
        ]),
    ];

    let widget = Paragraph::new(analysis).block(block);
    frame.render_widget(widget, area);
}

/// Render GNN attention
fn render_gnn_attention(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" GNN Attention Map ", Theme::panel_title()));

    let residues = [
        ("ASP168", 0.94),
        ("LYS52", 0.71),
        ("GLU71", 0.52),
        ("VAL123", 0.41),
    ];

    let mut lines = vec![Line::from("")];
    for (res, attn) in residues {
        let bar_len = (attn * 20.0) as usize;
        let bar = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
        lines.push(Line::from(format!("  {} {} {:.2}", res, bar, attn)));
    }

    let widget = Paragraph::new(lines).block(block);
    frame.render_widget(widget, area);
}

/// Render pharmacophore features
fn render_pharmacophore(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::panel_border())
        .title(Span::styled(" Pharmacophore Features ", Theme::panel_title()));

    let features = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("●", Style::default().fg(Theme::PHARM_DONOR)),
            Span::raw(" H-bond Donor    ×4  "),
            Span::styled("●", Style::default().fg(Theme::PHARM_ACCEPTOR)),
            Span::raw(" H-bond Acceptor ×6"),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("●", Style::default().fg(Theme::PHARM_HYDROPHOBIC)),
            Span::raw(" Hydrophobic     ×3  "),
            Span::styled("●", Style::default().fg(Theme::PHARM_AROMATIC)),
            Span::raw(" Aromatic        ×2"),
        ]),
    ];

    let widget = Paragraph::new(features).block(block);
    frame.render_widget(widget, area);
}

/// Render dialogue input bar
fn render_dialogue_input(app: &App, frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(if app.focus == Focus::Dialogue {
            Style::default().fg(Theme::ACCENT)
        } else {
            Theme::panel_border()
        });

    let input_text = format!(" > {}_", app.input_buffer);
    let input = Paragraph::new(input_text)
        .style(Theme::normal())
        .block(block);
    frame.render_widget(input, area);
}

/// Render help overlay
fn render_help_overlay(frame: &mut Frame, area: Rect) {
    let help_area = centered_rect(60, 70, area);

    // Clear background
    frame.render_widget(Clear, help_area);

    let help_text = vec![
        Line::from(Span::styled(" Keyboard Shortcuts ", Theme::title())),
        Line::from(""),
        Line::from(" Navigation:"),
        Line::from("   Tab        Cycle focus between panels"),
        Line::from("   Esc        Close overlays / cancel"),
        Line::from("   F1         Toggle this help"),
        Line::from(""),
        Line::from(" Commands (type in dialogue):"),
        Line::from("   load <file>    Load graph or protein"),
        Line::from("   run / go       Start optimization"),
        Line::from("   stop / pause   Pause optimization"),
        Line::from("   status         Show current status"),
        Line::from("   set <p> <v>    Set parameter"),
        Line::from("   help           Show command help"),
        Line::from(""),
        Line::from(" Quick Actions:"),
        Line::from("   Ctrl+C/Q   Quit"),
        Line::from("   g          Focus graph view"),
        Line::from("   p          Focus protein view"),
        Line::from("   m          Focus metrics"),
    ];

    let help = Paragraph::new(help_text)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Theme::ACCENT))
            .title(Span::styled(" Help ", Theme::panel_title()))
            .style(Style::default().bg(Theme::BG_PANEL)));

    frame.render_widget(help, help_area);
}

/// Helper to create a centered rect
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
