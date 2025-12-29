//! PacaBench TUI - Minimal terminal interface for LLM agent benchmarking
//!
//! A clean, focused dashboard for monitoring long-running benchmarks.

mod state;
mod theme;
mod views;

use anyhow::Result;
use clap::Parser;
use crossterm::{
    event::{
        DisableMouseCapture, EnableMouseCapture, Event as CrosstermEvent, EventStream, KeyCode,
        KeyEventKind, KeyModifiers,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::StreamExt;
use pacabench_core::config::ConfigOverrides;
use pacabench_core::{Benchmark, Command, Config, Event as BenchEvent, RunResult};
use ratatui::prelude::*;
use state::{AppState, LogLevel, RunStatus, View};
use std::io::stdout;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use theme::{Animation, Theme};
use tokio::sync::broadcast::error::{RecvError, TryRecvError};
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

#[derive(Parser, Debug)]
#[command(name = "pacabench-tui")]
#[command(about = "Minimal terminal UI for PacaBench benchmarking")]
#[command(version)]
struct Args {
    /// Path to pacabench.yaml config file
    #[arg(short, long, default_value = "pacabench.yaml")]
    config: PathBuf,

    /// Number of concurrent workers
    #[arg(short = 'j', long)]
    concurrency: Option<usize>,

    /// Timeout per case in seconds
    #[arg(short, long)]
    timeout: Option<f64>,

    /// Maximum retries per case
    #[arg(short, long)]
    retries: Option<usize>,

    /// Resume a specific run ID
    #[arg(long)]
    resume: Option<String>,

    /// Retry failed cases from a run
    #[arg(long)]
    retry: Option<String>,

    /// Limit number of cases to run
    #[arg(short, long)]
    limit: Option<usize>,

    /// Demo mode - show UI without running benchmark
    #[arg(long)]
    demo: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run the app
    let result = run_app(&mut terminal, args).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    args: Args,
) -> Result<()> {
    let theme = Theme::default();
    let mut state = AppState::new();

    if args.demo {
        run_demo_loop(terminal, &theme, &mut state).await
    } else {
        // Load config
        let overrides = ConfigOverrides {
            concurrency: args.concurrency,
            timeout_seconds: args.timeout,
            max_retries: args.retries,
            runs_dir: None,
            cache_dir: None,
        };
        let config = Config::from_file(&args.config, overrides)?;
        let run_id = args.resume.or(args.retry);

        run_benchmark_loop(terminal, &theme, &mut state, config, run_id, args.limit).await
    }
}

/// Run the TUI with a live benchmark
async fn run_benchmark_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    theme: &Theme,
    state: &mut AppState,
    config: Config,
    run_id: Option<String>,
    limit: Option<usize>,
) -> Result<()> {
    state.add_event(
        LogLevel::Info,
        None,
        format!("Loading benchmark: {}", config.name),
    );
    state.status = RunStatus::Preparing;

    let benchmark = Benchmark::new(config);
    let event_rx = benchmark.subscribe();
    let cmd_tx = benchmark.command_sender();

    // Spawn benchmark task
    let run_handle = tokio::spawn(async move { benchmark.run(run_id, limit).await });

    let mut event_rx_opt = Some(event_rx);
    run_event_loop(
        terminal,
        theme,
        state,
        &mut event_rx_opt,
        Some(cmd_tx),
        Some(run_handle),
    )
    .await
}

/// Run the TUI in demo mode with simulated data
async fn run_demo_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    theme: &Theme,
    state: &mut AppState,
) -> Result<()> {
    // Initialize demo state
    state.run_id = Some("demo_run_2a4f8b".to_string());
    state.status = RunStatus::Running;
    state.start_time = Some(Instant::now());
    state.total_cases = 5000;

    // Add demo agents
    let demo_agents = vec![
        ("claude-3-opus", 2000u64, 1340u64, 1220u64, 120u64),
        ("gpt-4-turbo", 1500, 780, 663, 117),
        ("gemini-pro", 1500, 920, 736, 184),
    ];

    for (name, total, completed, passed, failed) in &demo_agents {
        state.agent_order.push(name.to_string());
        let mut agent = state::AgentState::new(name.to_string(), *total);
        agent.seed_progress(*completed, *passed, *failed, 0);
        state.agents.insert(name.to_string(), agent);
    }

    state.add_event(
        LogLevel::Info,
        None,
        "Demo mode - simulated data".to_string(),
    );
    state.add_event(
        LogLevel::Success,
        Some("claude-3-opus"),
        "case_0847 passed".to_string(),
    );
    state.add_event(
        LogLevel::Success,
        Some("gpt-4-turbo"),
        "case_0291 passed".to_string(),
    );
    state.add_event(
        LogLevel::Error,
        Some("gemini-pro"),
        "case_0563 failed".to_string(),
    );
    state.add_event(
        LogLevel::Success,
        Some("claude-3-opus"),
        "case_0848 passed".to_string(),
    );

    // Add a demo failure
    state.failures.push(state::FailureEntry {
        agent: "gemini-pro".to_string(),
        case_id: "case_0563".to_string(),
        reason: "assertion failed: expected 42, got 41".to_string(),
    });

    run_event_loop(terminal, theme, state, &mut None, None, None).await
}

/// Drain any buffered benchmark events without blocking to keep the UI from
/// falling behind during high-concurrency runs.
fn drain_bench_events(
    event_rx: Option<&mut broadcast::Receiver<BenchEvent>>,
    state: &mut AppState,
) -> bool {
    let mut updated = false;

    if let Some(rx) = event_rx {
        loop {
            match rx.try_recv() {
                Ok(event) => {
                    state.handle_event(event);
                    updated = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Lagged(skipped)) => {
                    state.add_event(
                        LogLevel::Warning,
                        None,
                        format!("Missed {skipped} events (receiver lagged)"),
                    );
                    updated = true;
                    // Keep draining to catch up to the most recent state.
                }
                Err(TryRecvError::Closed) => break,
            }
        }
    }

    updated
}

/// Main event loop
async fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    theme: &Theme,
    state: &mut AppState,
    event_rx: &mut Option<broadcast::Receiver<BenchEvent>>,
    cmd_tx: Option<mpsc::UnboundedSender<Command>>,
    mut run_handle: Option<JoinHandle<pacabench_core::error::Result<RunResult>>>,
) -> Result<()> {
    let mut tick = interval(Duration::from_millis(50));
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut frame_count: u64 = 0;
    let mut events = EventStream::new();

    loop {
        let mut should_render = false;
        tokio::select! {
            maybe_input = events.next() => {
                match maybe_input {
                    Some(Ok(CrosstermEvent::Key(key))) if key.kind == KeyEventKind::Press => {
                        match handle_key(key.code, key.modifiers, state, cmd_tx.as_ref()) {
                            KeyAction::Quit => break,
                            KeyAction::ExitAfterRun => {
                                state.exit_requested = true;
                                should_render = true;
                            }
                            KeyAction::Continue => {}
                        }
                    }
                    Some(Err(err)) => {
                        state.add_event(LogLevel::Error, None, format!("Input error: {err}"));
                        state.exit_requested = true;
                    }
                    _ => {}
                }
            }
            bench_event = async {
                if let Some(rx) = event_rx.as_mut() {
                    Some(rx.recv().await)
                } else {
                    None
                }
            }, if event_rx.is_some() => {
                match bench_event {
                    Some(Ok(event)) => {
                        state.handle_event(event);
                        should_render = true;
                        // Catch up on any buffered events to stay current.
                        should_render |= drain_bench_events(event_rx.as_mut(), state);
                    }
                    Some(Err(RecvError::Lagged(skipped))) => {
                        state.add_event(
                            LogLevel::Warning,
                            None,
                            format!("Missed {skipped} events (receiver lagged)"),
                        );
                        should_render = true;
                        should_render |= drain_bench_events(event_rx.as_mut(), state);
                    }
                    Some(Err(RecvError::Closed)) | None => {
                        *event_rx = None;
                    }
                }
            }
            run_result = async {
                if let Some(handle) = &mut run_handle {
                    Some(handle.await)
                } else {
                    None
                }
            }, if run_handle.is_some() => {
                should_render = true;
                run_handle = None;
                if let Some(result) = run_result {
                    match result {
                        Ok(Ok(run)) => {
                            state.status = if run.aborted {
                                RunStatus::Aborted
                            } else {
                                RunStatus::Completed
                            };
                            if state.total_cases == 0 {
                                state.total_cases = run.stats.planned_cases;
                            }
                            if state.run_stats.is_none() {
                                state.run_stats = Some(run.stats.clone());
                            }
                            state.view = View::Completed;
                            // Only exit if the user already requested to quit.
                            // Otherwise keep the UI up to show the summary.
                        }
                        Ok(Err(e)) => {
                            state.add_event(LogLevel::Error, None, format!("Benchmark error: {e}"));
                            state.status = RunStatus::Aborted;
                            state.exit_requested = true;
                        }
                        Err(join_err) => {
                            state.add_event(
                                LogLevel::Error,
                                None,
                                format!("Benchmark task panicked: {join_err}"),
                            );
                            state.status = RunStatus::Aborted;
                            state.exit_requested = true;
                        }
                    }
                }
            }
            _ = tick.tick() => {
                frame_count = frame_count.wrapping_add(1);
                should_render = true;
            }
        }

        // Opportunistically drain any buffered events to prevent lag under load.
        should_render |= drain_bench_events(event_rx.as_mut(), state);

        if should_render {
            let anim = Animation::new(frame_count);
            terminal.draw(|f| views::render(f, state, theme, anim))?;
        }

        if state.exit_requested
            && !matches!(state.status, RunStatus::Running | RunStatus::Preparing)
        {
            break;
        }
    }

    if let Some(handle) = run_handle {
        if state.status == RunStatus::Running && !handle.is_finished() {
            handle.abort();
        }
        let _ = handle.await;
    }

    Ok(())
}

enum KeyAction {
    Quit,
    ExitAfterRun,
    Continue,
}

fn handle_key(
    code: KeyCode,
    modifiers: KeyModifiers,
    state: &mut AppState,
    cmd_tx: Option<&mpsc::UnboundedSender<Command>>,
) -> KeyAction {
    // Global shortcuts
    match code {
        KeyCode::Char('q') | KeyCode::Char('Q') => {
            if matches!(state.status, RunStatus::Running | RunStatus::Preparing)
                && !modifiers.contains(KeyModifiers::CONTROL)
            {
                if let Some(tx) = cmd_tx {
                    let _ = tx.send(Command::Abort {
                        reason: "User requested quit".to_string(),
                    });
                    return KeyAction::ExitAfterRun;
                }
                return KeyAction::Quit;
            }
            return KeyAction::Quit;
        }
        KeyCode::Esc => {
            // Navigate back
            match &state.view {
                View::Dashboard => {}
                View::AgentDetail { .. } => {
                    state.view = View::Dashboard;
                }
                View::CaseDetail { agent, .. } => {
                    state.view = View::AgentDetail {
                        agent: agent.clone(),
                    };
                }
                View::Failures => {
                    state.view = View::Dashboard;
                }
                View::Completed => {
                    state.view = View::Dashboard;
                }
            }
        }
        KeyCode::Char('f') | KeyCode::Char('F') => {
            // Toggle failures view
            if state.view == View::Failures {
                state.view = View::Dashboard;
            } else {
                state.view = View::Failures;
                state.selected_index = 0;
            }
        }
        KeyCode::Up | KeyCode::Char('k') => {
            if state.selected_index > 0 {
                state.selected_index -= 1;
            }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            let max = match &state.view {
                View::Dashboard => state.agent_order.len().saturating_sub(1),
                View::AgentDetail { agent } => state
                    .agents
                    .get(agent)
                    .map(|a| a.cases.len().saturating_sub(1))
                    .unwrap_or(0),
                View::Failures => state.failures.len().saturating_sub(1),
                View::CaseDetail { .. } => 0,
                View::Completed => 0,
            };
            if state.selected_index < max {
                state.selected_index += 1;
            }
        }
        KeyCode::Enter => {
            // Drill down
            match &state.view {
                View::Dashboard => {
                    if let Some(agent_name) = state.agent_order.get(state.selected_index) {
                        state.view = View::AgentDetail {
                            agent: agent_name.clone(),
                        };
                        state.selected_index = 0;
                    }
                }
                View::AgentDetail { agent } => {
                    if let Some(agent_state) = state.agents.get(agent) {
                        let case_keys: Vec<_> = agent_state.cases.keys().cloned().collect();
                        if let Some(case_id) = case_keys.get(state.selected_index) {
                            state.view = View::CaseDetail {
                                agent: agent.clone(),
                                case_id: case_id.clone(),
                            };
                        }
                    }
                }
                View::Failures => {
                    if let Some(failure) = state.failures.get(state.selected_index) {
                        state.view = View::CaseDetail {
                            agent: failure.agent.clone(),
                            case_id: failure.case_id.clone(),
                        };
                    }
                }
                View::CaseDetail { .. } => {}
                View::Completed => {}
            }
        }
        _ => {}
    }

    KeyAction::Continue
}
