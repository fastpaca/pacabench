//! Protocol types for benchmark communication.
//!
//! Public API:
//! - [`Event`]: Events emitted during benchmark execution
//! - [`Command`]: Commands to control benchmark execution
//!
//! Internal types (used by scheduler/workers):
//! - [`WorkItem`]: Unit of work
//! - [`WorkResult`]: Result from processing a work item

use crate::types::{
    AggregatedMetrics, Case, CaseKey, CaseResult, ErrorType, EvaluationResult, LlmMetrics,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// PUBLIC API - Events
// ============================================================================

/// Events emitted during benchmark execution.
///
/// Subscribe to these events to observe progress, display UI, or log results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    /// Benchmark run started.
    RunStarted {
        run_id: String,
        total_cases: u64,
        resuming: bool,
        completed_cases: u64,
        agents: Vec<String>,
        datasets: Vec<String>,
    },

    /// A case started processing.
    CaseStarted {
        run_id: String,
        case_id: String,
        agent: String,
        dataset: String,
        attempt: u32,
    },

    /// A case completed (pass or fail).
    CaseCompleted {
        run_id: String,
        case_id: String,
        agent: String,
        dataset: String,
        passed: bool,
        is_error: bool,
        attempt: u32,
        duration_ms: f64,
        input_tokens: u64,
        output_tokens: u64,
    },

    /// Benchmark run finished.
    RunCompleted {
        run_id: String,
        total_cases: u64,
        passed_cases: u64,
        failed_cases: u64,
        aborted: bool,
        metrics: AggregatedMetrics,
        agent_metrics: HashMap<String, AggregatedMetrics>,
    },

    /// Circuit breaker tripped due to high error rate.
    CircuitTripped { error_ratio: f64 },

    /// System error occurred.
    Error { message: String },
}

// ============================================================================
// PUBLIC API - Commands
// ============================================================================

/// Commands to control benchmark execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// Graceful stop: finish current cases, then exit.
    Stop { reason: String },
    /// Immediate abort: cancel in-flight work.
    Abort { reason: String },
}

// ============================================================================
// INTERNAL - Work Items
// ============================================================================

/// A single unit of work: run a case through an agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkItem {
    pub run_id: String,
    pub agent_name: String,
    pub dataset_name: String,
    pub case_id: String,
    pub case: Case,
    pub attempt: u32,
}

impl WorkItem {
    pub fn new(run_id: String, agent_name: String, case: Case) -> Self {
        Self {
            run_id,
            agent_name,
            dataset_name: case.dataset_name.clone(),
            case_id: case.case_id.clone(),
            case,
            attempt: 1,
        }
    }

    pub fn retry(&self) -> Self {
        Self {
            run_id: self.run_id.clone(),
            agent_name: self.agent_name.clone(),
            dataset_name: self.dataset_name.clone(),
            case_id: self.case_id.clone(),
            case: self.case.clone(),
            attempt: self.attempt + 1,
        }
    }

    pub fn key(&self) -> CaseKey {
        CaseKey::new(&self.agent_name, &self.dataset_name, &self.case_id)
    }
}

// ============================================================================
// INTERNAL - Work Results
// ============================================================================

/// Result from processing a work item.
#[derive(Clone, Debug)]
pub struct WorkResult {
    pub item: WorkItem,
    pub passed: bool,
    pub output: Option<String>,
    pub error: Option<String>,
    pub error_type: ErrorType,
    pub duration_ms: f64,
    pub llm_metrics: LlmMetrics,
    pub evaluation: Option<EvaluationResult>,
}

/// Internal command from benchmark to workers.
#[derive(Clone, Debug)]
pub enum WorkerCommand {
    Stop,
    Abort,
}

impl WorkResult {
    pub fn is_error(&self) -> bool {
        self.error_type.is_error()
    }

    pub fn to_case_result(&self, timestamp: String) -> CaseResult {
        CaseResult {
            case_id: self.item.case_id.clone(),
            dataset_name: self.item.dataset_name.clone(),
            agent_name: self.item.agent_name.clone(),
            passed: self.passed,
            attempt: self.item.attempt,
            output: self.output.clone(),
            error: self.error.clone(),
            error_type: self.error_type,
            runner_duration_ms: self.duration_ms,
            llm_metrics: self.llm_metrics.clone(),
            timestamp: Some(timestamp),
            f1_score: self.evaluation.as_ref().and_then(|e| e.f1_score),
            f1_passed: self.evaluation.as_ref().map(|e| e.passed),
            judge_passed: self.evaluation.as_ref().map(|e| e.passed),
            judge_reason: self.evaluation.as_ref().and_then(|e| e.reason.clone()),
            judge_metrics: self
                .evaluation
                .as_ref()
                .and_then(|e| e.judge_metrics.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_work_item_retry() {
        let case = Case {
            case_id: "test-1".into(),
            dataset_name: "test-ds".into(),
            input: "question".into(),
            expected: Some("answer".into()),
            history: vec![],
            metadata: HashMap::new(),
        };

        let item = WorkItem::new("run-123".into(), "agent-1".into(), case);
        assert_eq!(item.attempt, 1);

        let retry = item.retry();
        assert_eq!(retry.attempt, 2);
        assert_eq!(retry.case_id, "test-1");
    }
}
