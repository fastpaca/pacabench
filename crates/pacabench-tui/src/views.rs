//! Clean, minimal views inspired by modern web design
//!
//! Typography-driven, generous whitespace, clear hierarchy.

use crate::state::{AppState, LogLevel, RunStatus, View};
use crate::theme::{self, Animation, Theme};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use std::time::Duration;

/// Render the entire UI based on current state
pub fn render(frame: &mut Frame, state: &AppState, theme: &Theme, anim: Animation) {
    let area = frame.area();

    // Clear with background
    frame.render_widget(
        ratatui::widgets::Block::default().style(theme.background),
        area,
    );

    match &state.view {
        View::Dashboard => render_dashboard(frame, state, theme, anim, area),
        View::AgentDetail { agent } => render_agent_detail(frame, state, theme, agent, area),
        View::CaseDetail { agent, case_id } => {
            render_case_detail(frame, state, theme, agent, case_id, area)
        }
        View::Failures => render_failures(frame, state, theme, area),
        View::Completed => render_completed(frame, state, theme, area),
    }
}

/// Main dashboard view
fn render_dashboard(
    frame: &mut Frame,
    state: &AppState,
    theme: &Theme,
    anim: Animation,
    area: Rect,
) {
    let chunks = Layout::vertical([
        Constraint::Length(1), // Header
        Constraint::Length(1), // Blank
        Constraint::Length(1), // Progress bar
        Constraint::Length(1), // Progress text
        Constraint::Length(2), // Blank
        Constraint::Min(10),   // Content (agents + events)
        Constraint::Length(2), // Blank
        Constraint::Length(1), // Help
    ])
    .split(area);

    // Header
    render_header(frame, state, theme, anim, chunks[0]);

    // Progress bar
    render_progress_bar(frame, state, theme, chunks[2]);

    // Progress text
    render_progress_text(frame, state, theme, chunks[3]);

    // Content: agents on left, events on right
    let content_chunks =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[5]);

    render_agents_section(frame, state, theme, content_chunks[0]);
    render_events_section(frame, state, theme, content_chunks[1]);

    // Help line
    render_help(frame, theme, chunks[7]);
}

fn render_header(frame: &mut Frame, state: &AppState, theme: &Theme, anim: Animation, area: Rect) {
    let mut spans = vec![
        Span::styled("  ", theme.text),
        Span::styled(
            theme::chars::LIGHTNING,
            Style::default().fg(theme::NEON_YELLOW),
        ),
        Span::styled(" PACABENCH", theme.title),
    ];

    // Run ID
    if let Some(run_id) = &state.run_id {
        let short_id: String = run_id.chars().take(12).collect();
        spans.push(Span::styled("   ", theme.text));
        spans.push(Span::styled(short_id, theme.text_muted));
    }

    // Status
    spans.push(Span::styled("   ", theme.text));
    let (status_text, status_style) = match state.status {
        RunStatus::Idle => ("IDLE", theme.text_muted),
        RunStatus::Preparing => {
            let spinner = anim.spinner();
            return frame.render_widget(
                Paragraph::new(Line::from({
                    let mut s = spans;
                    s.push(Span::styled(
                        format!("PREPARING {}", spinner),
                        theme.progress,
                    ));
                    s
                })),
                area,
            );
        }
        RunStatus::Running => {
            let spinner = anim.spinner();
            return frame.render_widget(
                Paragraph::new(Line::from({
                    let mut s = spans;
                    s.push(Span::styled(format!("RUNNING {}", spinner), theme.progress));
                    s
                })),
                area,
            );
        }
        RunStatus::Completed => ("COMPLETED", theme.success),
        RunStatus::Aborted => ("ABORTED", theme.error),
    };
    spans.push(Span::styled(status_text, status_style));

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_progress_bar(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let pct = state.progress_pct();
    let width = area.width.saturating_sub(12) as usize; // Leave room for percentage
    let filled = (pct * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);

    let filled_bar = "━".repeat(filled);
    let empty_bar = "─".repeat(empty);
    let pct_str = format!(" {:.1}%", pct * 100.0);

    let line = Line::from(vec![
        Span::styled("  ", theme.text),
        Span::styled(filled_bar, Style::default().fg(theme::NEON_CYAN)),
        Span::styled(empty_bar, theme.text_dim),
        Span::styled(pct_str, theme.text_muted),
    ]);

    frame.render_widget(Paragraph::new(line), area);
}

fn render_progress_text(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let completed = state.total_completed();
    let total = state.total_cases;

    let mut spans = vec![
        Span::styled("  ", theme.text),
        Span::styled(format!("{} of {} cases", completed, total), theme.text),
    ];

    // ETA
    if let Some(eta) = state.eta() {
        let eta_str = format_duration(eta);
        spans.push(Span::styled(
            format!("   ETA {}", eta_str),
            theme.text_muted,
        ));
    }

    // Throughput
    let throughput = state.throughput_per_min();
    if throughput > 0.0 {
        spans.push(Span::styled(
            format!("   {:.1}/min", throughput),
            theme.text_muted,
        ));
    }

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_agents_section(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let mut lines = vec![
        Line::from(Span::styled("  AGENTS", theme.text_muted)),
        Line::from(""),
    ];

    // Calculate available height for agents (minus header and blank line)
    let max_agents = (area.height as usize).saturating_sub(3) / 3; // 3 lines per agent

    for (i, name) in state.agent_order.iter().enumerate().take(max_agents) {
        if let Some(agent) = state.agents.get(name) {
            let a = agent;
            let is_selected = state.selected_index == i;

            // Progress bar (16 chars to fit better)
            let pct = a.progress_pct();
            let bar_width = 16;
            let filled = (pct * bar_width as f64).round() as usize;

            let bar: String = (0..bar_width)
                .map(|j| if j < filled { '█' } else { '░' })
                .collect();

            let prefix = if is_selected { "▸ " } else { "  " };

            // Truncate name to fit (max 20 chars)
            let display_name: String = agent.name.chars().take(20).collect();
            let padded_name = format!("{:<20}", display_name);

            // Name line
            let name_style = if is_selected {
                theme.text.add_modifier(Modifier::BOLD)
            } else {
                theme.text
            };

            lines.push(Line::from(vec![
                Span::styled(prefix, theme.text_muted),
                Span::styled(padded_name, name_style),
                Span::styled(bar, Style::default().fg(theme::NEON_CYAN)),
                Span::styled(format!(" {:>3.0}%", pct * 100.0), theme.text_muted),
            ]));

            // Stats line - show completed/total as well
            let acc = a.accuracy() * 100.0;
            let acc_style = theme.accuracy_style(a.accuracy());

            lines.push(Line::from(vec![
                Span::styled("    ", theme.text),
                Span::styled(
                    format!("{}/{}", a.completed(), a.total_cases),
                    theme.text_muted,
                ),
                Span::styled(format!(" · {:.0}%", acc), acc_style),
                Span::styled(
                    format!(" · {} ✓ · {} ✗", a.passed(), a.failed()),
                    theme.text_muted,
                ),
            ]));

            lines.push(Line::from(""));
        }
    }

    let para = Paragraph::new(lines);
    frame.render_widget(para, area);
}

fn render_events_section(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let failures = state.failures.len();

    let header = if failures > 0 {
        Line::from(vec![
            Span::styled("  RECENT", theme.text_muted),
            Span::styled(format!("   {} failures ↓", failures), theme.error),
        ])
    } else {
        Line::from(Span::styled("  RECENT", theme.text_muted))
    };

    let mut lines = vec![header, Line::from("")];

    let visible = (area.height as usize).saturating_sub(3);
    for entry in state.events.iter().take(visible) {
        let (icon, icon_style) = match entry.level {
            LogLevel::Info => ("·", theme.text_muted),
            LogLevel::Success => ("✓", theme.success),
            LogLevel::Error => ("✗", theme.error),
            LogLevel::Warning => ("⚠", theme.warning),
        };

        let mut spans = vec![
            Span::styled("  ", theme.text),
            Span::styled(&entry.time, theme.text_dim),
            Span::styled("   ", theme.text),
            Span::styled(icon, icon_style),
            Span::styled("   ", theme.text),
        ];

        if let Some(agent) = &entry.agent {
            spans.push(Span::styled(
                format!("{} ", agent),
                Style::default().fg(theme::NEON_CYAN),
            ));
        }

        spans.push(Span::styled(&entry.message, theme.text));

        lines.push(Line::from(spans));
    }

    let para = Paragraph::new(lines);
    frame.render_widget(para, area);
}

fn render_help(frame: &mut Frame, theme: &Theme, area: Rect) {
    let line = Line::from(vec![
        Span::styled("  ", theme.text),
        Span::styled("q", theme.highlight),
        Span::styled(" quit · ", theme.text_muted),
        Span::styled("↑↓", theme.highlight),
        Span::styled(" navigate · ", theme.text_muted),
        Span::styled("enter", theme.highlight),
        Span::styled(" expand · ", theme.text_muted),
        Span::styled("f", theme.highlight),
        Span::styled(" failures · ", theme.text_muted),
        Span::styled("/", theme.highlight),
        Span::styled(" search", theme.text_muted),
    ]);

    frame.render_widget(Paragraph::new(line), area);
}

/// Agent detail view
fn render_agent_detail(
    frame: &mut Frame,
    state: &AppState,
    theme: &Theme,
    agent_name: &str,
    area: Rect,
) {
    let chunks = Layout::vertical([
        Constraint::Length(1), // Back header
        Constraint::Length(1), // Blank
        Constraint::Length(1), // Stats line
        Constraint::Length(2), // Blank
        Constraint::Min(5),    // Cases list
        Constraint::Length(2), // Blank
        Constraint::Length(1), // Help
    ])
    .split(area);

    // Header with back
    if let Some(agent) = state.agents.get(agent_name) {
        let a = agent;
        let line = Line::from(vec![
            Span::styled("  ← ", theme.text_muted),
            Span::styled(agent.name.as_str(), theme.title),
            Span::styled(
                format!(
                    "   {} of {} · {:.0}%",
                    a.completed(),
                    a.total_cases,
                    a.accuracy() * 100.0
                ),
                theme.text_muted,
            ),
        ]);
        frame.render_widget(Paragraph::new(line), chunks[0]);

        // Stats
        let stats_line = Line::from(vec![
            Span::styled("  ", theme.text),
            Span::styled(format!("{} passed", a.passed()), theme.success),
            Span::styled(" · ", theme.text_muted),
            Span::styled(format!("{} failed", a.failed()), theme.error),
            Span::styled(" · ", theme.text_muted),
            Span::styled(format!("{} errors", a.errors()), theme.warning),
            Span::styled(
                format!(
                    " · avg {:.1}s · ${:.4}",
                    a.avg_duration_ms() / 1000.0,
                    a.cost_usd()
                ),
                theme.text_muted,
            ),
        ]);
        frame.render_widget(Paragraph::new(stats_line), chunks[2]);

        // Cases header
        let mut lines = vec![
            Line::from(Span::styled("  CASES", theme.text_muted)),
            Line::from(""),
        ];

        // Show cases
        for (case_id, record) in a.cases.iter().take(20) {
            let (icon, style) = match record.outcome {
                crate::state::CaseOutcome::Running => ("◐", theme.progress),
                crate::state::CaseOutcome::Passed => ("✓", theme.success),
                crate::state::CaseOutcome::Failed => ("✗", theme.error),
                crate::state::CaseOutcome::Error => ("⚠", theme.warning),
            };

            lines.push(Line::from(vec![
                Span::styled("  ", theme.text),
                Span::styled(icon, style),
                Span::styled(format!("   {}", case_id), theme.text),
            ]));
        }

        frame.render_widget(Paragraph::new(lines), chunks[4]);
    }

    // Help
    let help = Line::from(vec![
        Span::styled("  ", theme.text),
        Span::styled("esc", theme.highlight),
        Span::styled(" back · ", theme.text_muted),
        Span::styled("↑↓", theme.highlight),
        Span::styled(" navigate · ", theme.text_muted),
        Span::styled("enter", theme.highlight),
        Span::styled(" details", theme.text_muted),
    ]);
    frame.render_widget(Paragraph::new(help), chunks[6]);
}

/// Case detail view
fn render_case_detail(
    frame: &mut Frame,
    _state: &AppState,
    theme: &Theme,
    agent: &str,
    case_id: &str,
    area: Rect,
) {
    let chunks = Layout::vertical([
        Constraint::Length(1), // Back header
        Constraint::Length(2), // Blank
        Constraint::Length(1), // Info
        Constraint::Min(5),    // Details
        Constraint::Length(1), // Help
    ])
    .split(area);

    // Header
    let line = Line::from(vec![
        Span::styled("  ← ", theme.text_muted),
        Span::styled(case_id, theme.title),
    ]);
    frame.render_widget(Paragraph::new(line), chunks[0]);

    // Info
    let info = Line::from(vec![
        Span::styled("  agent          ", theme.text_muted),
        Span::styled(agent, theme.text),
    ]);
    frame.render_widget(Paragraph::new(info), chunks[2]);

    // Help
    let help = Line::from(vec![
        Span::styled("  ", theme.text),
        Span::styled("esc", theme.highlight),
        Span::styled(" back · ", theme.text_muted),
        Span::styled("c", theme.highlight),
        Span::styled(" copy", theme.text_muted),
    ]);
    frame.render_widget(Paragraph::new(help), chunks[4]);
}

/// Failures view
fn render_failures(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let chunks = Layout::vertical([
        Constraint::Length(1), // Header
        Constraint::Length(2), // Blank
        Constraint::Min(5),    // List
        Constraint::Length(1), // Help
    ])
    .split(area);

    // Header
    let line = Line::from(vec![
        Span::styled("  ← ", theme.text_muted),
        Span::styled("FAILURES", theme.title),
        Span::styled(format!("   {}", state.failures.len()), theme.error),
    ]);
    frame.render_widget(Paragraph::new(line), chunks[0]);

    // List
    let mut lines = Vec::new();
    for (i, f) in state.failures.iter().enumerate() {
        let prefix = if state.selected_index == i {
            "▸ "
        } else {
            "  "
        };
        lines.push(Line::from(vec![
            Span::styled(prefix, theme.text_muted),
            Span::styled("✗", theme.error),
            Span::styled(format!("   {}/{}", f.agent, f.case_id), theme.text),
            Span::styled(format!(" — {}", f.reason), theme.text_muted),
        ]));
    }
    frame.render_widget(Paragraph::new(lines), chunks[2]);

    // Help
    let help = Line::from(vec![
        Span::styled("  ", theme.text),
        Span::styled("esc", theme.highlight),
        Span::styled(" back · ", theme.text_muted),
        Span::styled("enter", theme.highlight),
        Span::styled(" details", theme.text_muted),
    ]);
    frame.render_widget(Paragraph::new(help), chunks[3]);
}

/// Completed view with aggregated metrics
fn render_completed(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let chunks = Layout::vertical([
        Constraint::Length(1), // Header
        Constraint::Length(1), // Blank
        Constraint::Length(3), // Summary stats
        Constraint::Length(1), // Blank
        Constraint::Min(6),    // Per-agent metrics
        Constraint::Length(1), // Help
    ])
    .split(area);

    // Header
    let status_span = match state.status {
        RunStatus::Completed => Span::styled("COMPLETED", theme.success),
        RunStatus::Aborted => Span::styled("ABORTED", theme.error),
        _ => Span::styled("DONE", theme.text_muted),
    };

    let mut header = vec![
        Span::styled("  ", theme.text),
        Span::styled(
            theme::chars::LIGHTNING,
            Style::default().fg(theme::NEON_YELLOW),
        ),
        Span::styled(" Summary", theme.title),
        Span::styled("   ", theme.text),
        status_span,
    ];

    if let Some(run_id) = &state.run_id {
        let short_id: String = run_id.chars().take(12).collect();
        header.push(Span::styled("   ", theme.text));
        header.push(Span::styled(short_id, theme.text_muted));
    }

    frame.render_widget(Paragraph::new(Line::from(header)), chunks[0]);

    // Summary stats
    let mut summary_lines: Vec<Line> = Vec::new();
    if let Some(m) = &state.metrics {
        let acc_pct = m.accuracy * 100.0;
        summary_lines.push(Line::from(vec![
            Span::styled("  Accuracy ", theme.text_muted),
            Span::styled(format!("{acc_pct:.1}%"), theme.accuracy_style(m.accuracy)),
            Span::styled("   Passed ", theme.text_muted),
            Span::styled(
                format!("{}/{}", m.total_cases - m.failed_cases, m.total_cases),
                theme.success,
            ),
            Span::styled("   Failed ", theme.text_muted),
            Span::styled(format!("{}", m.failed_cases), theme.error),
            Span::styled("   Attempts ", theme.text_muted),
            Span::styled(
                format!("{:.2} avg · {} max", m.avg_attempts, m.max_attempts),
                theme.text,
            ),
        ]));

        summary_lines.push(Line::from(vec![
            Span::styled("  Latency ", theme.text_muted),
            Span::styled(
                format!("p50 {:.1}s", m.p50_duration_ms / 1000.0),
                theme.text,
            ),
            Span::styled(" · ", theme.text_muted),
            Span::styled(
                format!("p95 {:.1}s", m.p95_duration_ms / 1000.0),
                theme.text,
            ),
            Span::styled("   LLM ", theme.text_muted),
            Span::styled(format!("avg {:.1}ms", m.avg_llm_latency_ms), theme.text),
            Span::styled(" · ", theme.text_muted),
            Span::styled(format!("p95 {:.1}ms", m.p95_llm_latency_ms), theme.text),
        ]));

        summary_lines.push(Line::from(vec![
            Span::styled("  Tokens ", theme.text_muted),
            Span::styled(
                format!(
                    "in {} · out {} · cached {}",
                    m.total_input_tokens, m.total_output_tokens, m.total_cached_tokens
                ),
                theme.text,
            ),
            Span::styled("   Judge ", theme.text_muted),
            Span::styled(
                format!(
                    "in {} · out {}",
                    m.total_judge_input_tokens, m.total_judge_output_tokens
                ),
                theme.text,
            ),
        ]));
    } else {
        summary_lines.push(Line::from(vec![Span::styled(
            "  Waiting for completion metrics...",
            theme.text_muted,
        )]));
    }
    frame.render_widget(Paragraph::new(summary_lines), chunks[2]);

    // Per-agent metrics
    let mut agent_lines: Vec<Line> = vec![
        Line::from(Span::styled("  AGENTS", theme.text_muted)),
        Line::from(""),
    ];
    for name in &state.agent_order {
        if let Some(m) = state.agent_metrics.get(name) {
            let acc_pct = m.accuracy * 100.0;
            agent_lines.push(Line::from(vec![
                Span::styled("  ", theme.text),
                Span::styled(format!("{:<16}", name), theme.text),
                Span::styled(
                    format!("{acc_pct:>5.1}% ",),
                    theme.accuracy_style(m.accuracy),
                ),
                Span::styled(
                    format!(
                        "p50 {:.1}s · p95 {:.1}s",
                        m.p50_duration_ms / 1000.0,
                        m.p95_duration_ms / 1000.0
                    ),
                    theme.text_muted,
                ),
                Span::styled(
                    format!("   fail {}", m.failed_cases),
                    if m.failed_cases > 0 {
                        theme.error
                    } else {
                        theme.success
                    },
                ),
            ]));
        }
    }
    frame.render_widget(Paragraph::new(agent_lines), chunks[4]);

    // Help
    let help = Line::from(vec![
        Span::styled("  ", theme.text),
        Span::styled("q", theme.highlight),
        Span::styled(" quit · ", theme.text_muted),
        Span::styled("f", theme.highlight),
        Span::styled(" failures · ", theme.text_muted),
        Span::styled("esc", theme.highlight),
        Span::styled(" dashboard", theme.text_muted),
    ]);
    frame.render_widget(Paragraph::new(help), chunks[5]);
}

/// Format duration as human readable
fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    let mins = secs / 60;
    let hours = mins / 60;

    if hours > 0 {
        format!("{}h {}m", hours, mins % 60)
    } else if mins > 0 {
        format!("{}m {}s", mins, secs % 60)
    } else {
        format!("{}s", secs)
    }
}
