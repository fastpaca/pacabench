//! Run state management.
//!
//! Provides explicit state tracking for benchmark runs with proper state transitions.

use crate::protocol::WorkItem;
use crate::types::{CaseKey, CaseResult, RunStatus};
use std::collections::{HashMap, HashSet};

/// Tracks state for a benchmark run.
pub struct RunState {
    pub run_id: String,
    pub status: RunStatus,
    cases: HashMap<CaseKey, WorkItem>,
    pending: HashSet<CaseKey>,
    completed: HashMap<CaseKey, CaseResult>,
    attempt_counts: HashMap<CaseKey, u32>,
    max_retries: u32,
}

impl RunState {
    pub fn new(run_id: String, work_items: Vec<WorkItem>, max_retries: u32) -> Self {
        let mut cases = HashMap::new();
        let mut pending = HashSet::new();

        for item in work_items {
            let key = item.key();
            cases.insert(key.clone(), item);
            pending.insert(key);
        }

        Self {
            run_id,
            status: RunStatus::Pending,
            cases,
            pending,
            completed: HashMap::new(),
            attempt_counts: HashMap::new(),
            max_retries,
        }
    }

    /// Resume from existing results - mark completed cases.
    pub fn resume_from(&mut self, existing_results: Vec<CaseResult>) {
        for result in existing_results {
            let key = result.key();
            if result.passed {
                self.pending.remove(&key);
                self.completed.insert(key.clone(), result.clone());
            }
            self.attempt_counts.insert(key, result.attempt);
        }
    }

    pub fn total_cases(&self) -> u64 {
        self.cases.len() as u64
    }

    pub fn completed_cases(&self) -> u64 {
        self.completed.len() as u64
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn is_done(&self) -> bool {
        self.pending.is_empty()
    }

    /// Get initial work items (those not yet completed).
    pub fn initial_work_items(&self) -> Vec<WorkItem> {
        self.pending
            .iter()
            .filter_map(|key| self.cases.get(key).cloned())
            .map(|mut item| {
                // Set attempt based on previous attempts
                if let Some(&prev) = self.attempt_counts.get(&item.key()) {
                    item.attempt = prev + 1;
                }
                item
            })
            .collect()
    }

    /// Record a case completion. Returns true if a retry should be scheduled.
    pub fn mark_completed(&mut self, key: CaseKey, result: CaseResult) -> bool {
        let attempt = result.attempt;
        self.attempt_counts.insert(key.clone(), attempt);

        let needs_retry = !result.passed
            && result.error_type.is_retryable()
            && attempt < self.max_retries;

        if !needs_retry {
            self.pending.remove(&key);
            self.completed.insert(key, result);
        }

        needs_retry
    }

    /// Get the retry work item for a case.
    pub fn retry_item(&self, key: &CaseKey) -> Option<WorkItem> {
        let item = self.cases.get(key)?;
        let attempt = self.attempt_counts.get(key).copied().unwrap_or(0) + 1;
        Some(WorkItem {
            run_id: item.run_id.clone(),
            agent_name: item.agent_name.clone(),
            dataset_name: item.dataset_name.clone(),
            case_id: item.case_id.clone(),
            case: item.case.clone(),
            attempt,
        })
    }

    pub fn results(&self) -> Vec<CaseResult> {
        self.completed.values().cloned().collect()
    }

    pub fn transition(&mut self, to: RunStatus) {
        self.status = to;
    }
}

