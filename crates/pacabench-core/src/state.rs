use crate::types::{CaseKey, CaseResult};
use std::collections::{HashMap, HashSet};

pub struct RunState {
    pending: HashSet<CaseKey>,
    completed: HashSet<CaseKey>,
    max_retries: u32,
    total_cases: u64,
    agent_totals: HashMap<String, u64>,
    agent_completed: HashMap<String, u64>,
}

impl RunState {
    pub fn new(
        max_retries: u32,
        total_cases: u64,
        agent_totals: HashMap<String, u64>,
        existing_results: Vec<CaseResult>,
    ) -> Self {
        let mut completed = HashSet::new();
        let mut agent_completed = HashMap::new();

        for result in existing_results {
            let key = result.key();
            if result.passed {
                completed.insert(key.clone());
                *agent_completed.entry(key.agent.clone()).or_insert(0) += 1;
            }
        }

        Self {
            pending: HashSet::new(),
            completed,
            max_retries,
            total_cases,
            agent_totals,
            agent_completed,
        }
    }

    pub fn register_case(&mut self, key: CaseKey) {
        self.pending.insert(key);
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

        let needs_retry =
            !result.passed && result.error_type.is_retryable() && attempt < self.max_retries;

        if !needs_retry {
            self.pending.remove(&key);
            self.completed.insert(key.clone());
            *self.agent_completed.entry(key.agent).or_insert(0) += 1;
        }

        needs_retry
    }

    pub fn agent_totals(&self) -> HashMap<String, u64> {
        self.agent_totals.clone()
    }

    pub fn agent_completed(&self) -> HashMap<String, u64> {
        self.agent_completed.clone()
    }
}
