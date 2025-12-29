//! Clean, minimal views inspired by modern web design
//!
//! Typography-driven, generous whitespace, clear hierarchy.

use crate::state::{AppState, LogLevel, RunStatus, View};
use crate::theme::{self, Animation, Theme};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
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
    let layout = Layout::vertical([
        Constraint::Length(1), // Header
        Constraint::Length(2), // Blank
        Constraint::Length(5), // Stats row
        Constraint::Length(1), // Blank
        Constraint::Min(10),   // Main content (agents + events)
        Constraint::Length(1), // Blank
        Constraint::Length(1), // Help
    ])
    .split(area);

    render_header(frame, state, theme, anim, layout[0]);
    render_stat_cards(frame, state, theme, layout[2]);

    let main_chunks = Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(layout[4]);

    render_agents_table(frame, state, theme, main_chunks[0]);
    render_events_panel(frame, state, theme, main_chunks[1]);

    render_help(frame, theme, layout[6]);
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

fn render_stat_cards(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let chunks = Layout::horizontal([
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
    ])
    .split(area);

    let completed = state.total_completed();
    let total = state.total_cases.max(1);
    let pct = state.progress_pct() * 100.0;
    render_stat_card(
        frame,
        theme,
        chunks[0],
        "Progress",
        format!("{pct:.1}%"),
        format!("{completed}/{total} cases"),
    );

    let accuracy = state.accuracy() * 100.0;
    render_stat_card(
        frame,
        theme,
        chunks[1],
        "Accuracy",
        format!("{accuracy:.1}%"),
        format!("{} passed", state.total_passed()),
    );

    let throughput = state.throughput_per_min();
    render_stat_card(
        frame,
        theme,
        chunks[2],
        "Throughput",
        format!("{throughput:.1}/min"),
        "cases per minute".to_string(),
    );

    let eta = state
        .eta()
        .map(format_duration)
        .unwrap_or_else(|| "—".into());
    render_stat_card(
        frame,
        theme,
        chunks[3],
        "ETA",
        eta,
        format!("elapsed {}", format_duration(state.elapsed())),
    );
}

fn render_stat_card(
    frame: &mut Frame,
    theme: &Theme,
    area: Rect,
    title: &str,
    value: String,
    detail: String,
) {
    let block = Block::default()
        .title(Span::styled(format!(" {title} "), theme.text_muted))
        .borders(Borders::ALL)
        .border_style(theme.border);

    let content = Paragraph::new(vec![
        Line::from(Span::styled(
            value,
            theme.title.add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(detail, theme.text_muted)),
    ])
    .block(block);

    frame.render_widget(content, area);
}

fn render_agents_table(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let header = ["Agent", "Progress", "Done", "Pass/Fail/Err"];
    let widths = [
        Constraint::Length(18),
        Constraint::Length(18),
        Constraint::Length(12),
        Constraint::Length(18),
    ];

    let rows: Vec<Row> = state
        .agent_order
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            let a = state.agents.get(name);
            let (progress, done, pfe, acc_style) = if let Some(agent) = a {
                let pct = agent.progress_pct();
                let bar = gauge_line(pct, 12);
                let done = format!("{}/{}", agent.completed(), agent.total_cases);
                let pfe = format!(
                    "{} ✓ / {} ✗ / {} ⚠",
                    agent.passed(),
                    agent.failed(),
                    agent.errors()
                );
                (bar, done, pfe, theme.accuracy_style(agent.accuracy()))
            } else {
                ("".into(), "".into(), "".into(), theme.text_muted)
            };

            let mut row = Row::new(vec![
                Cell::from(Span::styled(name.clone(), theme.text)),
                Cell::from(Span::styled(
                    progress,
                    Style::default().fg(theme::NEON_CYAN),
                )),
                Cell::from(Span::styled(done, theme.text_muted)),
                Cell::from(Span::styled(pfe, acc_style)),
            ]);

            if state.selected_index == idx {
                row = row.style(theme.highlight);
            }

            row
        })
        .collect();

    let table = Table::new(rows, widths)
        .header(
            Row::new(
                header
                    .iter()
                    .map(|h| Cell::from(Span::styled(*h, theme.text_muted)))
                    .collect::<Vec<Cell>>(),
            )
            .bottom_margin(1),
        )
        .column_spacing(2)
        .block(
            Block::default()
                .title(Span::styled(" Agents ", theme.text_muted))
                .borders(Borders::ALL)
                .border_style(theme.border),
        )
        .row_highlight_style(theme.highlight);

    frame.render_widget(table, area);
}

fn gauge_line(pct: f64, width: usize) -> String {
    let pct = pct.clamp(0.0, 1.0);
    let filled = (pct * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    let mut s = String::new();
    for _ in 0..filled {
        s.push('█');
    }
    for _ in 0..empty {
        s.push('░');
    }
    format!("{s} {:>4.0}%", pct * 100.0)
}

fn render_events_panel(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let layout = Layout::vertical([
        Constraint::Min(6),
        Constraint::Length(1),
        Constraint::Min(4),
    ])
    .split(area);

    render_events_list(frame, state, theme, layout[0]);
    render_failures_summary(frame, state, theme, layout[2]);
}

fn render_events_list(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let mut lines = vec![
        Line::from(Span::styled("  Recent events", theme.text_muted)),
        Line::from(""),
    ];

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
            Span::styled("  ", theme.text),
            Span::styled(icon, icon_style),
            Span::styled("  ", theme.text),
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

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border);

    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_failures_summary(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let mut lines = vec![Line::from(vec![
        Span::styled("  Failures ", theme.text_muted),
        Span::styled(format!("{}", state.failures.len()), theme.error),
    ])];

    for failure in state.failures.iter().take(3) {
        lines.push(Line::from(vec![
            Span::styled("    ✗ ", theme.error),
            Span::styled(format!("{}/{}", failure.agent, failure.case_id), theme.text),
            Span::styled(" — ", theme.text_muted),
            Span::styled(&failure.reason, theme.text_muted),
        ]));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border);

    frame.render_widget(Paragraph::new(lines).block(block), area);
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
        Span::styled("esc", theme.highlight),
        Span::styled(" back", theme.text_muted),
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
    let layout = Layout::vertical([
        Constraint::Length(1), // Header
        Constraint::Length(1), // Blank
        Constraint::Length(5), // Stat cards
        Constraint::Length(1), // Blank
        Constraint::Min(8),    // Main content
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

    frame.render_widget(Paragraph::new(Line::from(header)), layout[0]);

    render_completed_cards(frame, state, theme, layout[2]);

    let main_chunks = Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(layout[4]);

    let left = Layout::vertical([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(main_chunks[0]);

    render_agent_metrics_table(frame, state, theme, left[0]);
    render_failures_table(frame, state, theme, left[1]);
    render_events_panel(frame, state, theme, main_chunks[1]);

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
    frame.render_widget(Paragraph::new(help), layout[5]);
}

fn render_completed_cards(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let rows =
        Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).split(area);
    let top = Layout::horizontal([
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
    ])
    .split(rows[0]);
    let bottom = Layout::horizontal([
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
    ])
    .split(rows[1]);

    let metrics = state.metrics.as_ref();
    let total = metrics.map(|m| m.total_cases).unwrap_or(state.total_cases);
    let failed = metrics.map(|m| m.failed_cases).unwrap_or(0);
    let passed = total.saturating_sub(failed);
    let accuracy = metrics
        .map(|m| m.accuracy * 100.0)
        .unwrap_or_else(|| state.accuracy() * 100.0);

    render_stat_card(
        frame,
        theme,
        top[0],
        "Accuracy",
        format!("{accuracy:.1}%"),
        format!("{passed}/{total} passed"),
    );

    if let Some(m) = metrics {
        render_stat_card(
            frame,
            theme,
            top[1],
            "Latency (run)",
            format!("p50 {:.1}s", m.p50_duration_ms / 1000.0),
            format!("p95 {:.1}s", m.p95_duration_ms / 1000.0),
        );

        render_stat_card(
            frame,
            theme,
            top[2],
            "Attempts",
            format!("avg {:.2}", m.avg_attempts),
            format!("max {}", m.max_attempts),
        );

        render_stat_card(
            frame,
            theme,
            top[3],
            "Tokens",
            format!("in {}", m.total_input_tokens),
            format!("out {}", m.total_output_tokens),
        );

        render_stat_card(
            frame,
            theme,
            bottom[0],
            "LLM latency",
            format!("avg {:.1}ms", m.avg_llm_latency_ms),
            format!("p95 {:.1}ms", m.p95_llm_latency_ms),
        );

        render_stat_card(
            frame,
            theme,
            bottom[1],
            "Judge tokens",
            format!("in {}", m.total_judge_input_tokens),
            format!("out {}", m.total_judge_output_tokens),
        );

        render_stat_card(
            frame,
            theme,
            bottom[2],
            "Throughput",
            format!("{:.1}/min", state.throughput_per_min()),
            format!("elapsed {}", format_duration(state.elapsed())),
        );

        render_stat_card(
            frame,
            theme,
            bottom[3],
            "Failures",
            format!("{}", failed),
            "see table below".to_string(),
        );
    } else {
        for chunk in top.iter().skip(1).chain(bottom.iter()) {
            render_stat_card(
                frame,
                theme,
                *chunk,
                "Pending",
                "—".into(),
                "Waiting for metrics".into(),
            );
        }
    }
}

fn render_agent_metrics_table(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let header = ["Agent", "Accuracy", "Latency", "Failed"];
    let widths = [
        Constraint::Length(18),
        Constraint::Length(10),
        Constraint::Length(18),
        Constraint::Length(10),
    ];

    let rows: Vec<Row> = state
        .agent_order
        .iter()
        .map(|name| {
            if let Some(m) = state.agent_metrics.get(name) {
                let acc_pct = m.accuracy * 100.0;
                Row::new(vec![
                    Cell::from(Span::styled(name.clone(), theme.text)),
                    Cell::from(Span::styled(
                        format!("{acc_pct:>5.1}%"),
                        theme.accuracy_style(m.accuracy),
                    )),
                    Cell::from(Span::styled(
                        format!(
                            "p50 {:.1}s · p95 {:.1}s",
                            m.p50_duration_ms / 1000.0,
                            m.p95_duration_ms / 1000.0
                        ),
                        theme.text_muted,
                    )),
                    Cell::from(Span::styled(
                        format!("{}", m.failed_cases),
                        if m.failed_cases > 0 {
                            theme.error
                        } else {
                            theme.success
                        },
                    )),
                ])
            } else {
                Row::new(vec![
                    Cell::from(Span::styled(name.clone(), theme.text_muted)),
                    Cell::from(Span::styled("—", theme.text_muted)),
                    Cell::from(Span::styled("—", theme.text_muted)),
                    Cell::from(Span::styled("—", theme.text_muted)),
                ])
            }
        })
        .collect();

    let table = Table::new(rows, widths)
        .header(
            Row::new(
                header
                    .iter()
                    .map(|h| Cell::from(Span::styled(*h, theme.text_muted)))
                    .collect::<Vec<Cell>>(),
            )
            .bottom_margin(1),
        )
        .column_spacing(2)
        .block(
            Block::default()
                .title(Span::styled(" Agent metrics ", theme.text_muted))
                .borders(Borders::ALL)
                .border_style(theme.border),
        )
        .row_highlight_style(theme.highlight);

    frame.render_widget(table, area);
}

fn render_failures_table(frame: &mut Frame, state: &AppState, theme: &Theme, area: Rect) {
    let header = ["Agent", "Case", "Reason"];
    let widths = [
        Constraint::Length(16),
        Constraint::Length(20),
        Constraint::Fill(1),
    ];

    let rows: Vec<Row> = state
        .failures
        .iter()
        .take(10)
        .map(|f| {
            Row::new(vec![
                Cell::from(Span::styled(f.agent.clone(), theme.text)),
                Cell::from(Span::styled(f.case_id.clone(), theme.text_muted)),
                Cell::from(Span::styled(f.reason.clone(), theme.error)),
            ])
        })
        .collect();

    let table = Table::new(rows, widths)
        .header(
            Row::new(
                header
                    .iter()
                    .map(|h| Cell::from(Span::styled(*h, theme.text_muted)))
                    .collect::<Vec<Cell>>(),
            )
            .bottom_margin(1),
        )
        .column_spacing(2)
        .block(
            Block::default()
                .title(Span::styled(" Recent failures ", theme.text_muted))
                .borders(Borders::ALL)
                .border_style(theme.border),
        );

    frame.render_widget(table, area);
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
