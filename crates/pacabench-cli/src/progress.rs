//! Indicatif-based progress display for the CLI.
//!
//! Provides rich terminal output using progress bars for each agent.
//! Receives events via tokio channel and updates the display.
//! Cost is computed here using pricing tables from the pricing module.

use crate::pricing::calculate_cost_from_metrics;
use console::style;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use pacabench_core::Event;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::broadcast;

const MICRODOLLARS_PER_USD: f64 = 1_000_000.0;

struct AgentState {
    bar: ProgressBar,
    passed: AtomicU64,
    failed: AtomicU64,
    errors: AtomicU64,
    cost_micros: AtomicU64,
}

impl AgentState {
    fn new(bar: ProgressBar) -> Self {
        Self {
            bar,
            passed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            cost_micros: AtomicU64::new(0),
        }
    }

    fn add_cost(&self, usd: f64) {
        let micros = (usd * MICRODOLLARS_PER_USD) as u64;
        self.cost_micros.fetch_add(micros, Ordering::Relaxed);
    }

    fn cost_usd(&self) -> f64 {
        self.cost_micros.load(Ordering::Relaxed) as f64 / MICRODOLLARS_PER_USD
    }

    fn update_message(&self, start_time: Option<Instant>) {
        let passed = self.passed.load(Ordering::Relaxed);
        let failed = self.failed.load(Ordering::Relaxed);
        let errors = self.errors.load(Ordering::Relaxed);
        let cost = self.cost_usd();

        let elapsed = start_time.map(|t| t.elapsed().as_secs()).unwrap_or(0);

        let elapsed_str = if elapsed >= 60 {
            format!("{}m{}s", elapsed / 60, elapsed % 60)
        } else {
            format!("{}s", elapsed)
        };

        let msg = format!(
            "{} {} {} {} {} {} {} {}",
            style("✓").green(),
            style(passed).green().bold(),
            style("✗").red(),
            style(failed).red().bold(),
            style("⚠").yellow(),
            style(errors).yellow(),
            style(format!("${:.3}", cost)).cyan(),
            style(elapsed_str).dim(),
        );
        self.bar.set_message(msg);
    }
}

/// Progress display using indicatif for rich terminal output.
pub struct ProgressDisplay {
    multi: MultiProgress,
    agents: Mutex<HashMap<String, AgentState>>,
    start_time: Mutex<Option<Instant>>,
}

impl Default for ProgressDisplay {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressDisplay {
    pub fn new() -> Self {
        Self {
            multi: MultiProgress::new(),
            agents: Mutex::new(HashMap::new()),
            start_time: Mutex::new(None),
        }
    }

    /// Process events until the channel closes.
    pub async fn run(self, mut rx: broadcast::Receiver<Event>) {
        loop {
            match rx.recv().await {
                Ok(event) => self.handle_event(event),
                Err(broadcast::error::RecvError::Closed) => break,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
            }
        }
    }

    fn handle_event(&self, event: Event) {
        match event {
            Event::RunStarted {
                run_id,
                total_cases,
                resuming,
                completed_cases,
                agents,
                datasets: _,
            } => {
                *self.start_time.lock() = Some(Instant::now());

                if resuming {
                    println!(
                        "{} Resuming {} ({} already done)",
                        style("→").cyan().bold(),
                        style(&run_id).bold(),
                        completed_cases
                    );
                } else {
                    println!(
                        "{} Starting {} ({} cases)",
                        style("→").cyan().bold(),
                        style(&run_id).bold(),
                        total_cases
                    );
                }

                let max_name_len = agents.iter().map(|n| n.len()).max().unwrap_or(10);
                let cases_per_agent = if agents.is_empty() {
                    total_cases
                } else {
                    total_cases / agents.len() as u64
                };
                let completed_per_agent = if agents.is_empty() {
                    completed_cases
                } else {
                    completed_cases / agents.len() as u64
                };

                let mut agents_guard = self.agents.lock();
                let mut agent_names = agents.clone();
                agent_names.sort();

                for agent_name in agent_names {
                    let bar_style = ProgressStyle::with_template(&format!(
                        "{{spinner:.green}} {{prefix:<{max_name_len}}} [{{bar:30.cyan/blue}}] {{pos}}/{{len}} {{msg}}"
                    ))
                    .unwrap()
                    .progress_chars("█▓▒░  ");

                    let bar = self.multi.add(ProgressBar::new(cases_per_agent));
                    bar.set_style(bar_style);
                    bar.set_prefix(agent_name.clone());
                    bar.set_position(completed_per_agent);

                    let state = AgentState::new(bar);
                    state.update_message(*self.start_time.lock());
                    state
                        .bar
                        .enable_steady_tick(std::time::Duration::from_millis(100));
                    agents_guard.insert(agent_name, state);
                }
            }

            Event::CaseStarted { .. } => {}

            Event::CaseCompleted {
                agent,
                passed,
                is_error,
                input_tokens,
                output_tokens,
                ..
            } => {
                let agents_guard = self.agents.lock();
                if let Some(state) = agents_guard.get(&agent) {
                    if is_error {
                        state.errors.fetch_add(1, Ordering::Relaxed);
                    } else if passed {
                        state.passed.fetch_add(1, Ordering::Relaxed);
                    } else {
                        state.failed.fetch_add(1, Ordering::Relaxed);
                    }

                    let cost_usd = calculate_cost_from_metrics(input_tokens, output_tokens, 0);
                    state.add_cost(cost_usd);
                    state.bar.inc(1);
                    state.update_message(*self.start_time.lock());
                }
            }

            Event::CircuitTripped { error_ratio } => {
                let agents_guard = self.agents.lock();
                if let Some((_, state)) = agents_guard.iter().next() {
                    state.bar.println(format!(
                        "{} Circuit breaker tripped at {:.1}% error rate",
                        style("⚠").yellow().bold(),
                        error_ratio * 100.0
                    ));
                }
            }

            Event::RunCompleted {
                run_id,
                total_cases,
                passed_cases,
                failed_cases,
                aborted,
                metrics,
                agent_metrics,
            } => {
                let total_cost_usd = calculate_cost_from_metrics(
                    metrics.total_input_tokens,
                    metrics.total_output_tokens,
                    metrics.total_cached_tokens,
                );
                let judge_cost_usd = calculate_cost_from_metrics(
                    metrics.total_judge_input_tokens,
                    metrics.total_judge_output_tokens,
                    0,
                );

                {
                    let mut agents_guard = self.agents.lock();
                    for (_, state) in agents_guard.drain() {
                        state.bar.finish_and_clear();
                    }
                }

                let elapsed = self
                    .start_time
                    .lock()
                    .map(|t| t.elapsed().as_secs_f64())
                    .unwrap_or(0.0);

                let status = if aborted {
                    style("ABORTED").red().bold()
                } else {
                    style("COMPLETED").green().bold()
                };

                println!();
                println!(
                    "{} Run {} {}",
                    style("✓").green().bold(),
                    style(&run_id).bold(),
                    status
                );
                println!();

                let accuracy = if total_cases > 0 {
                    passed_cases as f64 / total_cases as f64 * 100.0
                } else {
                    0.0
                };

                println!(
                    "  {} {}/{} ({:.1}%)",
                    style("Passed:").dim(),
                    style(passed_cases).green(),
                    total_cases,
                    accuracy
                );
                println!("  {} {}", style("Failed:").dim(), style(failed_cases).red());
                println!(
                    "  {} {}",
                    style("Cost:").dim(),
                    style(format!("${:.4}", total_cost_usd)).cyan()
                );
                println!("  {} {:.1}s", style("Duration:").dim(), elapsed);
                println!(
                    "  {} {:.1}% / {:.1}% / {:.1}% | {} {:.0} / {:.0} ms | {} {:.0}/{:.0} (judge {}/{})",
                    style("Acc/Prec/Rec:").dim(),
                    metrics.accuracy * 100.0,
                    metrics.precision * 100.0,
                    metrics.recall * 100.0,
                    style("Duration p50/p95:").dim(),
                    metrics.p50_duration_ms,
                    metrics.p95_duration_ms,
                    style("Tokens in/out:").dim(),
                    metrics.total_input_tokens,
                    metrics.total_output_tokens,
                    metrics.total_judge_input_tokens,
                    metrics.total_judge_output_tokens
                );
                println!(
                    "  {} {:.0}/{:.0}/{:.0} ms | {} ${:.4} (judge ${:.4}) | {} {:.1}/{}",
                    style("LLM latency avg/p50/p95:").dim(),
                    metrics.avg_llm_latency_ms,
                    metrics.p50_llm_latency_ms,
                    metrics.p95_llm_latency_ms,
                    style("Cost:").dim(),
                    total_cost_usd,
                    judge_cost_usd,
                    style("Attempts avg/max:").dim(),
                    metrics.avg_attempts,
                    metrics.max_attempts
                );

                if !agent_metrics.is_empty() {
                    println!();
                    println!("  Per agent:");
                    let mut names: Vec<_> = agent_metrics.keys().cloned().collect();
                    names.sort();
                    for name in names {
                        if let Some(m) = agent_metrics.get(&name) {
                            let agent_cost = calculate_cost_from_metrics(
                                m.total_input_tokens,
                                m.total_output_tokens,
                                m.total_cached_tokens,
                            );
                            let agent_judge_cost = calculate_cost_from_metrics(
                                m.total_judge_input_tokens,
                                m.total_judge_output_tokens,
                                0,
                            );
                            println!(
                                "    {} acc {:.1}% p50 {:.0}ms tokens {}/{} cost ${:.4} (judge ${:.4}) attempts {:.1}/{}",
                                style(&name).bold(),
                                m.accuracy * 100.0,
                                m.p50_duration_ms,
                                m.total_input_tokens,
                                m.total_output_tokens,
                                agent_cost,
                                agent_judge_cost,
                                m.avg_attempts,
                                m.max_attempts
                            );
                        }
                    }
                }
                println!();

                if aborted {
                    println!("{} Run aborted", style("⚠").yellow().bold());
                }
            }

            Event::Error { message } => {
                println!("{} {}", style("Error:").red().bold(), message);
            }
        }
    }
}
