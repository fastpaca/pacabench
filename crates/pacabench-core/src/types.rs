//! Core domain types for PacaBench.
//!
//! This module defines the data structures used throughout the benchmark:
//!
//! - [`Case`]: A single benchmark case to evaluate
//! - [`EvaluationResult`]: Result of evaluating a runner output
//! - [`CaseResult`]: Combined result of running and evaluating a case
//! - [`LlmMetrics`]: Metrics from LLM API calls (tokens, latency)
//! - [`JudgeMetrics`]: Metrics from evaluator LLM calls
//! - [`AggregatedMetrics`]: Aggregated metrics across all cases
//! - [`RunStatus`]: Explicit state for benchmark runs
//! - [`ErrorType`]: Classification of errors
//! - [`Event`]: Events emitted during benchmark execution
//! - [`Command`]: Commands to control benchmark execution

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Metadata map for extensible case data (e.g., multiple choice options).
pub type Metadata = HashMap<String, Value>;

/// Explicit state for benchmark runs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    #[default]
    Pending,
    Loading,
    Running,
    Finalizing,
    Completed,
    Failed,
    Aborted,
}

impl RunStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Aborted)
    }
}

/// Classification of errors during case execution.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ErrorType {
    #[default]
    None,
    /// Task failed but can be retried (e.g., wrong answer).
    TaskFailure,
    /// System error that may be transient (e.g., timeout, API error).
    SystemFailure,
    /// Fatal error that should stop execution.
    FatalError,
}

impl ErrorType {
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::SystemFailure)
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Self::SystemFailure | Self::FatalError)
    }
}

/// Metrics from LLM API calls collected by the proxy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct LlmMetrics {
    pub call_count: u64,
    pub latency_ms: Vec<f64>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl LlmMetrics {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    pub fn total_latency_ms(&self) -> f64 {
        self.latency_ms.iter().sum()
    }

    pub fn avg_latency_ms(&self) -> f64 {
        if self.latency_ms.is_empty() {
            0.0
        } else {
            self.total_latency_ms() / self.latency_ms.len() as f64
        }
    }
}

/// Metrics from evaluator LLM calls (e.g., LLM-as-judge).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct JudgeMetrics {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    pub latency_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl JudgeMetrics {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// A single benchmark case to evaluate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Case {
    pub case_id: String,
    pub dataset_name: String,
    pub input: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<Value>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: Metadata,
}

/// Result of evaluating a runner output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct EvaluationResult {
    pub passed: bool,
    pub score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default)]
    pub latency_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f1_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_metrics: Option<JudgeMetrics>,
}

impl EvaluationResult {
    pub fn pass(score: f64) -> Self {
        Self {
            passed: true,
            score,
            ..Default::default()
        }
    }

    pub fn fail(score: f64, reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            score,
            reason: Some(reason.into()),
            ..Default::default()
        }
    }
}

/// Combined result of running and evaluating a case.
///
/// This struct captures both the runner output and evaluation results.
/// It is persisted to JSONL files and used for aggregating metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaseResult {
    /// Unique identifier for the case within its dataset.
    pub case_id: String,
    /// Name of the dataset this case belongs to.
    pub dataset_name: String,
    /// Name of the agent that processed this case.
    pub agent_name: String,
    /// Whether the case passed evaluation.
    pub passed: bool,
    /// Attempt number (1-based, increments on retry).
    pub attempt: u32,
    /// Raw output from the runner, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Error message if the runner failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Classification of the error (none, task failure, system failure, fatal).
    #[serde(default)]
    pub error_type: ErrorType,
    /// Total wall-clock time for the runner, in milliseconds.
    #[serde(default)]
    pub runner_duration_ms: f64,
    /// LLM metrics collected by the proxy during this case.
    #[serde(default)]
    pub llm_metrics: LlmMetrics,
    /// ISO 8601 timestamp when this result was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// F1 score from F1 evaluator, if used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f1_score: Option<f64>,
    /// Whether F1 score met the threshold, if F1 evaluator was used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f1_passed: Option<bool>,
    /// Whether LLM judge passed, if judge evaluator was used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_passed: Option<bool>,
    /// Reason from LLM judge, if judge evaluator was used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_reason: Option<String>,
    /// Metrics from judge LLM call (tokens, latency), if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_metrics: Option<JudgeMetrics>,
}

impl CaseResult {
    pub fn key(&self) -> CaseKey {
        CaseKey {
            agent: self.agent_name.clone(),
            dataset: self.dataset_name.clone(),
            case_id: self.case_id.clone(),
        }
    }
}

/// Unique identifier for a case within a run (agent + dataset + case_id).
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct CaseKey {
    pub agent: String,
    pub dataset: String,
    pub case_id: String,
}

impl CaseKey {
    pub fn new(
        agent: impl Into<String>,
        dataset: impl Into<String>,
        case_id: impl Into<String>,
    ) -> Self {
        Self {
            agent: agent.into(),
            dataset: dataset.into(),
            case_id: case_id.into(),
        }
    }
}

/// Aggregated metrics across all cases in a run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AggregatedMetrics {
    pub accuracy: f64,
    pub total_cases: u64,
    pub failed_cases: u64,
    pub p50_duration_ms: f64,
    pub p95_duration_ms: f64,
    pub avg_llm_latency_ms: f64,
    pub p50_llm_latency_ms: f64,
    pub p95_llm_latency_ms: f64,
    pub total_llm_calls: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cached_tokens: u64,
    pub total_judge_input_tokens: u64,
    pub total_judge_output_tokens: u64,
    pub total_attempts: u64,
    pub avg_attempts: f64,
    pub max_attempts: u32,
}

/// Events emitted during benchmark execution.
///
/// Subscribe to these events to observe progress, display UI, or log results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    /// Benchmark run started.
    RunStarted {
        run_id: String,
        total_cases: u64,
        /// True if resuming an interrupted (non-terminal) run.
        resuming: bool,
        /// True if retrying failed cases from a completed run.
        #[serde(default)]
        retrying: bool,
        completed_cases: u64,
        agents: Vec<String>,
        datasets: Vec<String>,
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        agent_totals: HashMap<String, u64>,
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        agent_completed: HashMap<String, u64>,
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
        #[serde(default)]
        cached_tokens: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(default)]
        judge_input_tokens: u64,
        #[serde(default)]
        judge_output_tokens: u64,
        #[serde(default)]
        judge_cached_tokens: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        judge_model: Option<String>,
        /// Raw output from the runner, if any.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        /// Error message if the runner failed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        /// Reason from LLM judge, if judge evaluator was used.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        judge_reason: Option<String>,
    },

    /// Benchmark run finished.
    RunCompleted {
        /// Whether the run was aborted before completion.
        aborted: bool,
        /// Complete run statistics - the single source of truth (boxed to reduce enum size).
        stats: Box<crate::stats::RunStats>,
    },

    /// Circuit breaker tripped due to high error rate.
    CircuitTripped { error_ratio: f64 },

    /// System error occurred.
    Error { message: String },
}

/// Commands to control benchmark execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// Graceful stop: finish current cases, then exit.
    Stop { reason: String },
    /// Immediate abort: cancel in-flight work.
    Abort { reason: String },
}
