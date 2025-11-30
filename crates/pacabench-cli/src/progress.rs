//! Indicatif-based progress reporter for the CLI.

use console::style;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use pacabench_core::reporter::{ProgressEvent, ProgressReporter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Progress reporter using indicatif for rich terminal output.
pub struct IndicatifReporter {
    multi: MultiProgress,
    main_bar: Mutex<Option<ProgressBar>>,
    start_time: Mutex<Option<Instant>>,
    passed: AtomicU64,
    failed: AtomicU64,
    errors: AtomicU64,
    total_cost: Mutex<f64>,
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
            main_bar: Mutex::new(None),
            start_time: Mutex::new(None),
            passed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            total_cost: Mutex::new(0.0),
        }
    }

    fn update_message(&self, bar: &ProgressBar) {
        let passed = self.passed.load(Ordering::Relaxed);
        let failed = self.failed.load(Ordering::Relaxed);
        let errors = self.errors.load(Ordering::Relaxed);
        let cost = *self.total_cost.lock().unwrap();

        let elapsed = self
            .start_time
            .lock()
            .unwrap()
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0);

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
        bar.set_message(msg);
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
            } => {
                *self.start_time.lock().unwrap() = Some(Instant::now());

                let bar_style = ProgressStyle::with_template(
                    "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .unwrap()
                .progress_chars("█▓▒░  ");

                let bar = self.multi.add(ProgressBar::new(total_cases));
                bar.set_style(bar_style);
                bar.set_position(completed_cases);

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

                self.update_message(&bar);
                bar.enable_steady_tick(std::time::Duration::from_millis(100));

                *self.main_bar.lock().unwrap() = Some(bar);
            }
            ProgressEvent::CaseCompleted {
                passed,
                is_system_error,
                cost_usd,
                ..
            } => {
                if is_system_error {
                    self.errors.fetch_add(1, Ordering::Relaxed);
                } else if passed {
                    self.passed.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.failed.fetch_add(1, Ordering::Relaxed);
                }

                *self.total_cost.lock().unwrap() += cost_usd;

                if let Some(bar) = self.main_bar.lock().unwrap().as_ref() {
                    bar.inc(1);
                    self.update_message(bar);
                }
            }
            ProgressEvent::CircuitTripped { error_ratio } => {
                if let Some(bar) = self.main_bar.lock().unwrap().as_ref() {
                    bar.println(format!(
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
                if let Some(bar) = self.main_bar.lock().unwrap().take() {
                    bar.finish_and_clear();
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
