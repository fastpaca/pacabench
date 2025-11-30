//! Progress reporting trait and types for benchmark execution.

use serde::{Deserialize, Serialize};

/// Events emitted during benchmark execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProgressEvent {
    /// Run started with total case count.
    RunStarted {
        run_id: String,
        total_cases: u64,
        resuming: bool,
        completed_cases: u64,
    },
    /// A case completed (pass or fail).
    CaseCompleted {
        case_id: String,
        agent_name: String,
        dataset_name: String,
        passed: bool,
        is_system_error: bool,
        duration_ms: f64,
        cost_usd: f64,
    },
    /// Run finished.
    RunCompleted {
        run_id: String,
        total_cases: u64,
        passed_cases: u64,
        failed_cases: u64,
        total_cost_usd: f64,
        circuit_tripped: bool,
    },
    /// Circuit breaker tripped.
    CircuitTripped { error_ratio: f64 },
}

/// Trait for progress reporters.
///
/// Implementors receive events during benchmark execution and can
/// display progress, log to file, etc.
pub trait ProgressReporter: Send + Sync {
    /// Called when a progress event occurs.
    fn report(&self, event: ProgressEvent);
}

/// A no-op reporter that discards all events.
#[derive(Debug, Default)]
pub struct NullReporter;

impl ProgressReporter for NullReporter {
    fn report(&self, _event: ProgressEvent) {}
}

/// A simple reporter that prints to stdout.
#[derive(Debug, Default)]
pub struct PrintReporter;

impl ProgressReporter for PrintReporter {
    fn report(&self, event: ProgressEvent) {
        match event {
            ProgressEvent::RunStarted {
                run_id,
                total_cases,
                resuming,
                completed_cases,
            } => {
                if resuming {
                    println!(
                        "Resuming run {run_id}: {completed_cases}/{total_cases} cases already done"
                    );
                } else {
                    println!("Starting run {run_id}: {total_cases} cases");
                }
            }
            ProgressEvent::CaseCompleted {
                case_id,
                agent_name,
                passed,
                is_system_error,
                ..
            } => {
                let status = if is_system_error {
                    "ERROR"
                } else if passed {
                    "PASS"
                } else {
                    "FAIL"
                };
                println!("[{status}] {agent_name}/{case_id}");
            }
            ProgressEvent::RunCompleted {
                run_id,
                total_cases,
                passed_cases,
                total_cost_usd,
                circuit_tripped,
                ..
            } => {
                let status = if circuit_tripped {
                    "FAILED"
                } else {
                    "COMPLETED"
                };
                println!(
                    "Run {run_id} {status}: {passed_cases}/{total_cases} passed, cost ${total_cost_usd:.4}"
                );
            }
            ProgressEvent::CircuitTripped { error_ratio } => {
                println!(
                    "Circuit breaker tripped: {:.1}% error rate",
                    error_ratio * 100.0
                );
            }
        }
    }
}
