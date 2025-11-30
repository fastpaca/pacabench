//! Indicatif-based progress reporter for the CLI.

use console::style;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use pacabench_core::reporter::{ProgressEvent, ProgressReporter};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Per-agent progress state.
struct AgentState {
    bar: ProgressBar,
    passed: AtomicU64,
    failed: AtomicU64,
    errors: AtomicU64,
    cost: Mutex<f64>,
}

impl AgentState {
    fn new(bar: ProgressBar) -> Self {
        Self {
            bar,
            passed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            cost: Mutex::new(0.0),
        }
    }

    fn update_message(&self, start_time: Option<Instant>) {
        let passed = self.passed.load(Ordering::Relaxed);
        let failed = self.failed.load(Ordering::Relaxed);
        let errors = self.errors.load(Ordering::Relaxed);
        let cost = *self.cost.lock().unwrap();

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
                *self.start_time.lock().unwrap() = Some(Instant::now());

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
                let mut agents_guard = self.agents.lock().unwrap();
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
                    state.update_message(*self.start_time.lock().unwrap());
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
                let agents_guard = self.agents.lock().unwrap();
                if let Some(state) = agents_guard.get(&agent_name) {
                    if is_system_error {
                        state.errors.fetch_add(1, Ordering::Relaxed);
                    } else if passed {
                        state.passed.fetch_add(1, Ordering::Relaxed);
                    } else {
                        state.failed.fetch_add(1, Ordering::Relaxed);
                    }

                    *state.cost.lock().unwrap() += cost_usd;
                    state.bar.inc(1);
                    state.update_message(*self.start_time.lock().unwrap());
                }
            }
            ProgressEvent::CircuitTripped { error_ratio } => {
                let agents_guard = self.agents.lock().unwrap();
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
            } => {
                // Finish all agent bars
                {
                    let mut agents_guard = self.agents.lock().unwrap();
                    for (_, state) in agents_guard.drain() {
                        state.bar.finish_and_clear();
                    }
                }

                let elapsed = self
                    .start_time
                    .lock()
                    .unwrap()
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
