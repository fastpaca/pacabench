//! Run state management.
//!
//! Provides explicit state tracking for benchmark runs with proper state transitions.

use crate::types::{CaseKey, CaseResult, RunStatus};
use std::collections::{HashMap, HashSet};

/// Tracks state for a benchmark run.
pub struct RunState {
    #[allow(dead_code)]
    run_id: String,
    pub status: RunStatus,
    pending: HashSet<CaseKey>,
    completed: HashMap<CaseKey, CaseResult>,
    attempt_counts: HashMap<CaseKey, u32>,
    max_retries: u32,
    total_cases: u64,
    agent_totals: HashMap<String, u64>,
    agent_completed: HashMap<String, u64>,
}

impl RunState {
    pub fn new(
        run_id: String,
        max_retries: u32,
        total_cases: u64,
        agent_totals: HashMap<String, u64>,
        existing_results: Vec<CaseResult>,
    ) -> Self {
        let mut completed = HashMap::new();
        let mut attempt_counts = HashMap::new();
        let mut agent_completed = HashMap::new();

        for result in existing_results {
            let key = result.key();
            attempt_counts.insert(key.clone(), result.attempt);
            if result.passed {
                completed.insert(key.clone(), result.clone());
                *agent_completed.entry(key.agent.clone()).or_insert(0) += 1;
            }
        }

        Self {
            run_id,
            status: RunStatus::Pending,
            pending: HashSet::new(),
            completed,
            attempt_counts,
            max_retries,
            total_cases,
            agent_totals,
            agent_completed,
        }
    }

    pub fn register_case(&mut self, key: CaseKey, attempt: u32) {
        self.pending.insert(key.clone());
        self.attempt_counts.insert(key, attempt);
    }

    pub fn total_cases(&self) -> u64 {
        self.total_cases
    }

    pub fn completed_cases(&self) -> u64 {
        self.completed.len() as u64
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Record a case completion. Returns true if a retry should be scheduled.
    pub fn mark_completed(&mut self, result: CaseResult) -> bool {
        let key = result.key();
        let attempt = result.attempt;
        self.attempt_counts.insert(key.clone(), attempt);

        let needs_retry =
            !result.passed && result.error_type.is_retryable() && attempt < self.max_retries;

        if !needs_retry {
            self.pending.remove(&key);
            self.completed.insert(key.clone(), result);
            *self.agent_completed.entry(key.agent.clone()).or_insert(0) += 1;
        }

        needs_retry
    }

    pub fn transition(&mut self, to: RunStatus) {
        self.status = to;
    }

    pub fn agent_totals(&self) -> HashMap<String, u64> {
        self.agent_totals.clone()
    }

    pub fn agent_completed(&self) -> HashMap<String, u64> {
        self.agent_completed.clone()
    }
}
