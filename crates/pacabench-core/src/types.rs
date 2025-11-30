//! Core domain types for PacaBench.
//!
//! This module defines the data structures used throughout the benchmark:
//!
//! - [`Case`]: A single benchmark case to evaluate
//! - [`RunnerOutput`]: Output from running a case through an agent
//! - [`EvaluationResult`]: Result of evaluating a runner output
//! - [`CaseResult`]: Combined result of running and evaluating a case
//! - [`LlmMetrics`]: Metrics from LLM API calls (tokens, latency)
//! - [`JudgeMetrics`]: Metrics from evaluator LLM calls
//! - [`AggregatedMetrics`]: Aggregated metrics across all cases
//! - [`RunStatus`]: Explicit state for benchmark runs
//! - [`ErrorType`]: Classification of errors

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Metadata map for extensible case data (e.g., multiple choice options).
pub type Metadata = HashMap<String, Value>;

// ============================================================================
// RUN STATUS
// ============================================================================

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

// ============================================================================
// ERROR TYPE
// ============================================================================

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

// ============================================================================
// LLM METRICS (typed, not HashMap)
// ============================================================================

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

// ============================================================================
// JUDGE METRICS (typed, not HashMap)
// ============================================================================

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

// ============================================================================
// CASE
// ============================================================================

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

// ============================================================================
// RUNNER OUTPUT
// ============================================================================

/// Output from running a case through an agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RunnerOutput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default)]
    pub error_type: ErrorType,
    #[serde(default)]
    pub duration_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_traceback: Option<String>,
}

impl RunnerOutput {
    pub fn success(output: String, duration_ms: f64) -> Self {
        Self {
            output: Some(output),
            error: None,
            error_type: ErrorType::None,
            duration_ms,
            error_traceback: None,
        }
    }

    pub fn failure(error: String, error_type: ErrorType, duration_ms: f64) -> Self {
        Self {
            output: None,
            error: Some(error),
            error_type,
            duration_ms,
            error_traceback: None,
        }
    }

    pub fn is_success(&self) -> bool {
        self.output.is_some() && self.error.is_none()
    }
}

// ============================================================================
// EVALUATION RESULT
// ============================================================================

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

// ============================================================================
// CASE RESULT
// ============================================================================

/// Combined result of running and evaluating a case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaseResult {
    pub case_id: String,
    pub dataset_name: String,
    pub agent_name: String,
    pub passed: bool,
    pub attempt: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default)]
    pub error_type: ErrorType,
    #[serde(default)]
    pub runner_duration_ms: f64,
    #[serde(default)]
    pub llm_metrics: LlmMetrics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f1_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f1_passed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_passed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_reason: Option<String>,
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

// ============================================================================
// CASE KEY
// ============================================================================

/// Unique identifier for a case within a run (agent + dataset + case_id).
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct CaseKey {
    pub agent: String,
    pub dataset: String,
    pub case_id: String,
}

impl CaseKey {
    pub fn new(agent: impl Into<String>, dataset: impl Into<String>, case_id: impl Into<String>) -> Self {
        Self {
            agent: agent.into(),
            dataset: dataset.into(),
            case_id: case_id.into(),
        }
    }
}

// ============================================================================
// AGGREGATED METRICS
// ============================================================================

/// Aggregated metrics across all cases in a run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AggregatedMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
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
