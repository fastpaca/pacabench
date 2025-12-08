//! Application state management
//!
//! Single-owner state, updated from benchmark events and rendered immutably.

use pacabench_core::types::AggregatedMetrics;
use pacabench_core::Event;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Maximum events to keep in the log
const MAX_EVENTS: usize = 500;

/// Default per-token pricing used for live estimates (fallback only)
const DEFAULT_TOKEN_COST: f64 = 0.000_002;

/// Case outcome for retry tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseOutcome {
    Running,
    Passed,
    Failed,
    Error,
}

#[derive(Debug, Clone)]
pub struct CaseRecord {
    pub outcome: CaseOutcome,
    pub attempts: u32,
    /// Highest attempt we've observed for this case (guards against out-of-order events).
    pub last_attempt: u32,
    pub last_duration_ms: f64,
    pub total_duration_ms: f64,
    pub total_cost_usd: f64,
}

impl CaseRecord {
    fn new_running(attempt: u32) -> Self {
        Self {
            outcome: CaseOutcome::Running,
            attempts: attempt.max(1),
            last_attempt: attempt.max(1),
            last_duration_ms: 0.0,
            total_duration_ms: 0.0,
            total_cost_usd: 0.0,
        }
    }

    fn mark_running(&mut self, attempt: u32) {
        // Ignore stale events that arrive out-of-order.
        if attempt < self.last_attempt {
            return;
        }

        self.outcome = CaseOutcome::Running;
        self.attempts = self.attempts.max(attempt.max(1));
        self.last_attempt = attempt.max(self.last_attempt);
        self.last_duration_ms = 0.0;
    }

    fn apply_completion(
        &mut self,
        outcome: CaseOutcome,
        duration_ms: f64,
        cost_usd: f64,
        attempt: u32,
    ) {
        // Ignore stale completions; keep the newest attempt as source of truth.
        if attempt < self.last_attempt {
            return;
        }

        self.last_attempt = attempt;
        self.outcome = outcome;
        self.attempts = self.attempts.max(attempt.max(1));
        self.last_duration_ms = duration_ms;
        self.total_duration_ms += duration_ms;
        if cost_usd.is_finite() && cost_usd >= 0.0 {
            self.total_cost_usd += cost_usd;
        }
    }
}

/// Per-agent state with aggregate and per-case tracking
pub struct AgentState {
    pub name: String,
    pub total_cases: u64,
    pub cases: HashMap<String, CaseRecord>,
    baseline_completed: u64,
    baseline_passed: u64,
    baseline_failed: u64,
    baseline_errors: u64,
    baseline_cost_usd: f64,
    baseline_duration_ms: f64,
}

impl AgentState {
    pub fn new(name: String, total_cases: u64) -> Self {
        Self {
            name,
            total_cases,
            cases: HashMap::new(),
            baseline_completed: 0,
            baseline_passed: 0,
            baseline_failed: 0,
            baseline_errors: 0,
            baseline_cost_usd: 0.0,
            baseline_duration_ms: 0.0,
        }
    }

    pub fn seed_progress(&mut self, completed: u64, passed: u64, failed: u64, errors: u64) {
        self.baseline_completed = completed;
        self.baseline_passed = passed;
        self.baseline_failed = failed;
        self.baseline_errors = errors;
    }

    pub fn seed_completed_without_breakdown(&mut self, completed: u64) {
        // When we only know "completed" (resume), assume they were successful so totals stay consistent.
        self.baseline_completed = completed;
        self.baseline_passed = completed;
    }

    pub fn upsert_running(&mut self, key: String, attempt: u32) {
        self.cases
            .entry(key)
            .and_modify(|c| c.mark_running(attempt))
            .or_insert_with(|| CaseRecord::new_running(attempt));
    }

    pub fn record_case(
        &mut self,
        key: String,
        outcome: CaseOutcome,
        duration_ms: f64,
        cost_usd: f64,
        attempt: u32,
    ) {
        self.cases
            .entry(key)
            .and_modify(|c| c.apply_completion(outcome, duration_ms, cost_usd, attempt))
            .or_insert_with(|| {
                let mut record = CaseRecord::new_running(attempt);
                record.apply_completion(outcome, duration_ms, cost_usd, attempt);
                record
            });
    }

    pub fn completed(&self) -> u64 {
        self.baseline_completed + self.actual_completed()
    }

    pub fn passed(&self) -> u64 {
        self.baseline_passed + self.actual_count(CaseOutcome::Passed)
    }

    pub fn failed(&self) -> u64 {
        self.baseline_failed + self.actual_count(CaseOutcome::Failed)
    }

    pub fn errors(&self) -> u64 {
        self.baseline_errors + self.actual_count(CaseOutcome::Error)
    }

    pub fn cost_usd(&self) -> f64 {
        self.baseline_cost_usd + self.cases.values().map(|c| c.total_cost_usd).sum::<f64>()
    }

    pub fn avg_duration_ms(&self) -> f64 {
        let completed = self.completed();
        if completed == 0 {
            return 0.0;
        }

        let actual_sum: f64 = self
            .cases
            .values()
            .filter(|c| c.outcome != CaseOutcome::Running)
            .map(|c| c.last_duration_ms)
            .sum();

        (self.baseline_duration_ms + actual_sum) / completed as f64
    }

    pub fn progress_pct(&self) -> f64 {
        let denom = self.total_cases.max(self.cases.len() as u64).max(1);
        self.completed() as f64 / denom as f64
    }

    pub fn accuracy(&self) -> f64 {
        let completed = self.completed();
        if completed == 0 {
            0.0
        } else {
            self.passed() as f64 / completed as f64
        }
    }

    fn actual_completed(&self) -> u64 {
        self.cases
            .values()
            .filter(|c| c.outcome != CaseOutcome::Running)
            .count() as u64
    }

    fn actual_count(&self, outcome: CaseOutcome) -> u64 {
        self.cases.values().filter(|c| c.outcome == outcome).count() as u64
    }
}

/// Event log entry
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub time: String,
    pub level: LogLevel,
    pub agent: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Success,
    Error,
    Warning,
}

/// Current view mode
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum View {
    Dashboard,
    AgentDetail { agent: String },
    CaseDetail { agent: String, case_id: String },
    Failures,
    Completed,
}

/// Run status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RunStatus {
    #[default]
    Idle,
    Preparing,
    Running,
    Completed,
    Aborted,
}

/// Main application state
pub struct AppState {
    // Run info
    pub run_id: Option<String>,
    pub status: RunStatus,
    pub start_time: Option<Instant>,
    pub resuming: bool,
    pub retrying: bool,

    // Totals
    pub total_cases: u64,
    pub metrics: Option<AggregatedMetrics>,
    pub agent_metrics: HashMap<String, AggregatedMetrics>,

    // Agent state
    pub agents: HashMap<String, AgentState>,
    pub agent_order: Vec<String>,

    // Event log
    pub events: VecDeque<LogEntry>,

    // Failures for quick access
    pub failures: Vec<FailureEntry>,

    // UI state
    pub view: View,
    pub selected_index: usize,
    pub circuit_tripped: bool,
    pub exit_requested: bool,
}

#[derive(Debug, Clone)]
pub struct FailureEntry {
    pub agent: String,
    pub case_id: String,
    pub reason: String,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            run_id: None,
            status: RunStatus::Idle,
            start_time: None,
            resuming: false,
            retrying: false,
            total_cases: 0,
            metrics: None,
            agent_metrics: HashMap::new(),
            agents: HashMap::new(),
            agent_order: Vec::new(),
            events: VecDeque::with_capacity(MAX_EVENTS),
            failures: Vec::new(),
            view: View::Dashboard,
            selected_index: 0,
            circuit_tripped: false,
            exit_requested: false,
        }
    }

    pub fn total_completed(&self) -> u64 {
        self.agents.values().map(AgentState::completed).sum()
    }

    pub fn total_passed(&self) -> u64 {
        self.agents.values().map(AgentState::passed).sum()
    }

    pub fn progress_pct(&self) -> f64 {
        let denom = if self.total_cases > 0 {
            self.total_cases
        } else {
            self.agents.values().map(|a| a.total_cases).sum::<u64>()
        };

        if denom == 0 {
            0.0
        } else {
            self.total_completed() as f64 / denom as f64
        }
    }

    pub fn accuracy(&self) -> f64 {
        let completed = self.total_completed();
        if completed == 0 {
            0.0
        } else {
            self.total_passed() as f64 / completed as f64
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.map(|t| t.elapsed()).unwrap_or_default()
    }

    pub fn eta(&self) -> Option<Duration> {
        let completed = self.total_completed();
        let remaining = self.total_cases.saturating_sub(completed);

        if completed == 0 || remaining == 0 {
            return None;
        }

        let elapsed = self.elapsed();
        let rate = completed as f64 / elapsed.as_secs_f64();
        if rate <= 0.0 {
            return None;
        }

        let eta_secs = remaining as f64 / rate;
        Some(Duration::from_secs_f64(eta_secs))
    }

    pub fn throughput_per_min(&self) -> f64 {
        let elapsed_mins = self.elapsed().as_secs_f64() / 60.0;
        if elapsed_mins <= 0.0 {
            0.0
        } else {
            self.total_completed() as f64 / elapsed_mins
        }
    }

    pub fn add_event(&mut self, level: LogLevel, agent: Option<&str>, message: String) {
        let time = chrono::Local::now().format("%H:%M:%S").to_string();
        self.events.push_front(LogEntry {
            time,
            level,
            agent: agent.map(String::from),
            message,
        });
        while self.events.len() > MAX_EVENTS {
            self.events.pop_back();
        }
    }

    /// Process a benchmark event
    pub fn handle_event(&mut self, event: Event) {
        match event {
            Event::RunStarted {
                run_id,
                total_cases,
                resuming,
                retrying,
                completed_cases,
                agents,
                agent_totals,
                agent_completed,
                ..
            } => {
                self.run_id = Some(run_id);
                self.status = RunStatus::Running;
                self.start_time = Some(Instant::now());
                self.resuming = resuming;
                self.retrying = retrying;
                self.total_cases = total_cases;

                self.agent_order = agents.clone();
                self.agents.clear();
                for name in agents {
                    let total = agent_totals.get(&name).copied().unwrap_or(0);
                    let completed = agent_completed.get(&name).copied().unwrap_or(0);
                    let mut state = AgentState::new(name.clone(), total);
                    state.seed_completed_without_breakdown(completed);
                    self.agents.insert(name, state);
                }

                let msg = if retrying {
                    format!("Retrying {} failed cases", total_cases)
                } else if resuming {
                    format!("Resuming run ({} done)", completed_cases)
                } else {
                    format!("Starting run with {} cases", total_cases)
                };
                self.add_event(LogLevel::Info, None, msg);
            }

            Event::CaseStarted {
                agent,
                case_id,
                dataset,
                attempt,
                ..
            } => {
                if let Some(state) = self.agents.get_mut(&agent) {
                    let key = format!("{}:{}", dataset, case_id);
                    state.upsert_running(key, attempt);
                }

                if attempt > 1 {
                    self.add_event(
                        LogLevel::Warning,
                        Some(&agent),
                        format!("Retry attempt {} for {}", attempt, case_id),
                    );
                }
            }

            Event::CaseCompleted {
                agent,
                case_id,
                dataset,
                passed,
                is_error,
                duration_ms,
                input_tokens,
                output_tokens,
                cached_tokens,
                judge_input_tokens,
                judge_output_tokens,
                judge_cached_tokens,
                attempt,
                ..
            } => {
                let outcome = if passed {
                    CaseOutcome::Passed
                } else if is_error {
                    CaseOutcome::Error
                } else {
                    CaseOutcome::Failed
                };

                let effective_tokens = input_tokens
                    .saturating_add(output_tokens)
                    .saturating_add(judge_input_tokens)
                    .saturating_add(judge_output_tokens)
                    .saturating_sub(cached_tokens)
                    .saturating_sub(judge_cached_tokens);
                let cost = effective_tokens as f64 * DEFAULT_TOKEN_COST;

                if let Some(state) = self.agents.get_mut(&agent) {
                    let key = format!("{}:{}", dataset, case_id);
                    state.record_case(key, outcome, duration_ms, cost, attempt);
                }

                let level = if passed {
                    LogLevel::Success
                } else if is_error {
                    LogLevel::Warning
                } else {
                    LogLevel::Error
                };

                let status = if passed {
                    "passed"
                } else if is_error {
                    "error"
                } else {
                    "failed"
                };

                let msg = format!("{} {}", case_id, status);
                self.add_event(level, Some(&agent), msg.clone());

                // Track failures
                if !passed {
                    let reason = if is_error {
                        "system error"
                    } else {
                        "test failed"
                    };
                    self.failures.push(FailureEntry {
                        agent: agent.clone(),
                        case_id,
                        reason: reason.to_string(),
                    });
                }
            }

            Event::CircuitTripped { error_ratio } => {
                self.circuit_tripped = true;
                self.add_event(
                    LogLevel::Error,
                    None,
                    format!(
                        "Circuit breaker tripped ({:.0}% errors)",
                        error_ratio * 100.0
                    ),
                );
            }

            Event::Error { message } => {
                self.add_event(LogLevel::Error, None, message);
            }

            Event::RunCompleted {
                aborted,
                metrics,
                agent_metrics,
                total_cases,
                ..
            } => {
                self.status = if aborted {
                    RunStatus::Aborted
                } else {
                    RunStatus::Completed
                };
                self.metrics = Some(metrics);
                self.agent_metrics = agent_metrics;
                self.total_cases = total_cases;
                self.view = View::Completed;

                let msg = if aborted {
                    "Run aborted".to_string()
                } else {
                    format!("Run completed - {:.1}% accuracy", self.accuracy() * 100.0)
                };
                self.add_event(LogLevel::Info, None, msg);
            }
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
