//! Indicatif-based progress reporter for the CLI.
//!
//! Provides rich terminal output using progress bars for each agent.
//! All counters use atomics for lock-free concurrent updates.

use console::style;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use pacabench_core::reporter::{ProgressEvent, ProgressReporter};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Microdollars per USD (for storing cost as atomic integer).
const MICRODOLLARS_PER_USD: f64 = 1_000_000.0;

/// Per-agent progress state.
///
/// All counters are atomic for lock-free updates from concurrent tasks.
struct AgentState {
    bar: ProgressBar,
    passed: AtomicU64,
    failed: AtomicU64,
    errors: AtomicU64,
    /// Cost stored as microdollars (1 USD = 1,000,000 microdollars).
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

    /// Add cost in USD (converted to microdollars internally).
    fn add_cost(&self, usd: f64) {
        let micros = (usd * MICRODOLLARS_PER_USD) as u64;
        self.cost_micros.fetch_add(micros, Ordering::Relaxed);
    }

    /// Get cost in USD.
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

/// Progress reporter using indicatif for rich terminal output.
pub struct IndicatifReporter {
    multi: MultiProgress,
    agents: Mutex<HashMap<String, AgentState>>,
    start_time: Mutex<Option<Instant>>,
}

impl Default for IndicatifReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl IndicatifReporter {
    pub fn new() -> Self {
        Self {
            multi: MultiProgress::new(),
            agents: Mutex::new(HashMap::new()),
            start_time: Mutex::new(None),
        }
    }
}

impl ProgressReporter for IndicatifReporter {
    fn report(&self, event: ProgressEvent) {
        match event {
            ProgressEvent::RunStarted {
                run_id,
                total_cases,
                resuming,
                completed_cases,
                agents,
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

                // Find max agent name length for alignment
                let max_name_len = agents.keys().map(|n| n.len()).max().unwrap_or(10);

                // Create one progress bar per agent
                let mut agents_guard = self.agents.lock();
                let mut agent_names: Vec<_> = agents.keys().cloned().collect();
                agent_names.sort();

                for agent_name in agent_names {
                    let progress = agents.get(&agent_name).unwrap();
                    let bar_style = ProgressStyle::with_template(&format!(
                        "{{spinner:.green}} {{prefix:<{max_name_len}}} [{{bar:30.cyan/blue}}] {{pos}}/{{len}} {{msg}}"
                    ))
                    .unwrap()
                    .progress_chars("█▓▒░  ");

                    let bar = self.multi.add(ProgressBar::new(progress.total_cases));
                    bar.set_style(bar_style);
                    bar.set_prefix(agent_name.clone());
                    bar.set_position(progress.completed_cases);

                    let state = AgentState::new(bar);
                    state.update_message(*self.start_time.lock());
                    state
                        .bar
                        .enable_steady_tick(std::time::Duration::from_millis(100));
                    agents_guard.insert(agent_name, state);
                }
            }
            ProgressEvent::CaseCompleted {
                agent_name,
                passed,
                is_system_error,
                cost_usd,
                ..
            } => {
                let agents_guard = self.agents.lock();
                if let Some(state) = agents_guard.get(&agent_name) {
                    if is_system_error {
                        state.errors.fetch_add(1, Ordering::Relaxed);
                    } else if passed {
                        state.passed.fetch_add(1, Ordering::Relaxed);
                    } else {
                        state.failed.fetch_add(1, Ordering::Relaxed);
                    }

                    state.add_cost(cost_usd);
                    state.bar.inc(1);
                    state.update_message(*self.start_time.lock());
                }
            }
            ProgressEvent::CircuitTripped { error_ratio } => {
                let agents_guard = self.agents.lock();
                if let Some((_, state)) = agents_guard.iter().next() {
                    state.bar.println(format!(
                        "{} Circuit breaker tripped at {:.1}% error rate",
                        style("⚠").yellow().bold(),
                        error_ratio * 100.0
                    ));
                }
            }
            ProgressEvent::RunCompleted {
                run_id,
                total_cases,
                passed_cases,
                failed_cases,
                total_cost_usd,
                circuit_tripped,
                metrics,
                agents,
            } => {
                // Finish all agent bars
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

                let status = if circuit_tripped {
                    style("FAILED").red().bold()
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
                    metrics.total_cost_usd,
                    metrics.total_judge_cost_usd,
                    style("Attempts avg/max:").dim(),
                    metrics.avg_attempts,
                    metrics.max_attempts
                );
                if !agents.is_empty() {
                    println!();
                    println!("  Per agent:");
                    let mut names: Vec<_> = agents.keys().cloned().collect();
                    names.sort();
                    for name in names {
                        if let Some(m) = agents.get(&name) {
                            println!(
                                "    {} acc {:.1}% p50 {:.0}ms tokens {}/{} cost ${:.4} (judge ${:.4}) attempts {:.1}/{}",
                                style(&name).bold(),
                                m.accuracy * 100.0,
                                m.p50_duration_ms,
                                m.total_input_tokens,
                                m.total_output_tokens,
                                m.total_cost_usd,
                                m.total_judge_cost_usd,
                                m.avg_attempts,
                                m.max_attempts
                            );
                        }
                    }
                }
                println!();

                if circuit_tripped {
                    println!(
                        "{} Run aborted due to high error rate",
                        style("⚠").yellow().bold()
                    );
                }
            }
        }
    }
}
