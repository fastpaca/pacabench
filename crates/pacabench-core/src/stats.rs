//! Single source of truth for all run statistics.
//!
//! All metrics, counts, and aggregates derive from persisted results.
//! Nothing is pre-computed or cached in metadata - this module computes
//! everything from `results.jsonl` and `system_errors.jsonl`.
//!
//! # Design Principle
//!
//! ```text
//! results.jsonl + system_errors.jsonl + metadata.json (immutable facts)
//!     ↓
//! RunStats::compute()  [ONE function, ONE code path]
//!     ↓
//! All display, export, events
//! ```

use crate::metrics::aggregate_results;
use crate::persistence::{ErrorEntry, RunMetadata};
use crate::types::{AggregatedMetrics, CaseResult, ErrorType, RunStatus};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Complete run statistics derived from persisted data.
///
/// This is THE canonical source for all run metrics. All display code,
/// export code, and events should use this structure rather than
/// recomputing from raw data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStats {
    pub run_id: String,
    pub status: RunStatus,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_total_cases: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_cases: Option<u64>,

    pub agents: Vec<String>,
    pub datasets: Vec<String>,

    /// Cases we planned to run (from metadata, set at start)
    pub planned_cases: u64,

    /// If this is a retry, the original run_id
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_of: Option<String>,

    /// Cases that completed (persisted to results.jsonl)
    pub completed_cases: u64,
    pub passed_cases: u64,
    pub failed_cases: u64,

    /// Accuracy = passed / completed (not passed / planned)
    pub accuracy: f64,

    // NOTE: These counts include ALL errors encountered during the run,
    // including transient errors that were later retried successfully.
    // They represent the total error occurrences, not final failures.
    /// Total system errors logged (includes retried errors)
    pub system_error_count: u64,
    /// Total fatal errors logged (includes retried errors)
    pub fatal_error_count: u64,

    /// Detailed metrics
    pub metrics: AggregatedMetrics,
    pub by_agent: HashMap<String, AgentStats>,
    pub by_dataset: HashMap<String, DatasetStats>,

    /// Token/cost breakdown
    pub tokens: TokenStats,

    /// Failure details (with actual reasons)
    pub failures: Vec<FailureDetail>,
}

/// Per-agent statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    pub agent_name: String,
    pub completed_cases: u64,
    pub passed_cases: u64,
    pub failed_cases: u64,
    pub accuracy: f64,
    pub metrics: AggregatedMetrics,
    pub tokens: TokenStats,
}

/// Per-dataset statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    pub dataset_name: String,
    pub completed_cases: u64,
    pub passed_cases: u64,
    pub failed_cases: u64,
    pub accuracy: f64,
    pub metrics: AggregatedMetrics,
}

/// Token counts for cost calculation.
///
/// Cost calculation is done at the CLI layer using these counts,
/// because pricing is presentation-layer logic that changes frequently.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenStats {
    // Agent LLM calls
    pub agent_input_tokens: u64,
    pub agent_output_tokens: u64,
    pub agent_cached_tokens: u64,
    pub agent_calls: u64,

    // Judge LLM calls
    pub judge_input_tokens: u64,
    pub judge_output_tokens: u64,
    pub judge_cached_tokens: u64,

    /// Models observed (for cost calculation and display)
    pub models_used: Vec<String>,

    /// Token usage broken down per model (includes judge where available)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub per_model: HashMap<String, ModelTokenUsage>,
}

impl TokenStats {
    pub fn total_agent_tokens(&self) -> u64 {
        self.agent_input_tokens + self.agent_output_tokens
    }

    pub fn total_judge_tokens(&self) -> u64 {
        self.judge_input_tokens + self.judge_output_tokens
    }
}

/// A single failure with its actual stored reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureDetail {
    pub case_id: String,
    pub dataset_name: String,
    pub agent_name: String,
    pub error_type: ErrorType,
    /// The actual reason from storage: judge_reason > error > "failed evaluation"
    pub reason: String,
    pub attempt: u32,
}

/// Per-model token usage for accurate pricing and display.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelTokenUsage {
    pub agent_input_tokens: u64,
    pub agent_output_tokens: u64,
    pub agent_cached_tokens: u64,
    pub agent_calls: u64,

    pub judge_input_tokens: u64,
    pub judge_output_tokens: u64,
    pub judge_cached_tokens: u64,
}

impl RunStats {
    /// Compute all statistics from persisted data.
    ///
    /// This is THE single computation function. All metrics derive from here.
    /// Call this once and use the result everywhere - do not recompute.
    pub fn compute(metadata: &RunMetadata, results: &[CaseResult], errors: &[ErrorEntry]) -> Self {
        let completed_cases = results.len() as u64;
        let passed_cases = results.iter().filter(|r| r.passed).count() as u64;
        let failed_cases = completed_cases.saturating_sub(passed_cases);

        let accuracy = if completed_cases > 0 {
            passed_cases as f64 / completed_cases as f64
        } else {
            0.0
        };

        // Aggregate metrics using existing function
        let metrics = aggregate_results(results);

        // Per-agent stats
        let by_agent = compute_by_agent(results);

        // Per-dataset stats
        let by_dataset = compute_by_dataset(results);

        // Token stats with actual model info
        let tokens = compute_tokens(results);

        // Failures with actual stored reasons
        let failures = collect_failures(results, errors);

        // Error counts from persisted error log
        let system_error_count = errors
            .iter()
            .filter(|e| e.error_type == ErrorType::SystemFailure)
            .count() as u64;
        let fatal_error_count = errors
            .iter()
            .filter(|e| e.error_type == ErrorType::FatalError)
            .count() as u64;

        Self {
            run_id: metadata.run_id.clone(),
            status: metadata.status,
            start_time: metadata.start_time.clone(),
            end_time: metadata.completed_time.clone(),
            original_total_cases: metadata.original_total_cases.or(Some(metadata.total_cases)),
            active_cases: metadata.active_cases,
            agents: metadata.agents.clone(),
            datasets: metadata.datasets.clone(),
            planned_cases: metadata.total_cases,
            retry_of: metadata.retry_of.clone(),
            completed_cases,
            passed_cases,
            failed_cases,
            accuracy,
            system_error_count,
            fatal_error_count,
            metrics,
            by_agent,
            by_dataset,
            tokens,
            failures,
        }
    }

    /// Create empty stats for a run with no results yet.
    pub fn empty(metadata: &RunMetadata) -> Self {
        Self {
            run_id: metadata.run_id.clone(),
            status: metadata.status,
            start_time: metadata.start_time.clone(),
            end_time: metadata.completed_time.clone(),
            original_total_cases: metadata.original_total_cases.or(Some(metadata.total_cases)),
            active_cases: metadata.active_cases,
            agents: metadata.agents.clone(),
            datasets: metadata.datasets.clone(),
            planned_cases: metadata.total_cases,
            retry_of: metadata.retry_of.clone(),
            completed_cases: 0,
            passed_cases: 0,
            failed_cases: 0,
            accuracy: 0.0,
            system_error_count: 0,
            fatal_error_count: 0,
            metrics: AggregatedMetrics::default(),
            by_agent: HashMap::new(),
            by_dataset: HashMap::new(),
            tokens: TokenStats::default(),
            failures: Vec::new(),
        }
    }
}

fn compute_by_agent(results: &[CaseResult]) -> HashMap<String, AgentStats> {
    let mut grouped: HashMap<String, Vec<&CaseResult>> = HashMap::new();
    for r in results {
        grouped.entry(r.agent_name.clone()).or_default().push(r);
    }

    grouped
        .into_iter()
        .map(|(agent_name, cases)| {
            let completed = cases.len() as u64;
            let passed = cases.iter().filter(|r| r.passed).count() as u64;
            let failed = completed.saturating_sub(passed);
            let accuracy = if completed > 0 {
                passed as f64 / completed as f64
            } else {
                0.0
            };

            let owned: Vec<CaseResult> = cases.into_iter().cloned().collect();
            let metrics = aggregate_results(&owned);
            let tokens = compute_tokens(&owned);

            (
                agent_name.clone(),
                AgentStats {
                    agent_name,
                    completed_cases: completed,
                    passed_cases: passed,
                    failed_cases: failed,
                    accuracy,
                    metrics,
                    tokens,
                },
            )
        })
        .collect()
}

fn compute_by_dataset(results: &[CaseResult]) -> HashMap<String, DatasetStats> {
    let mut grouped: HashMap<String, Vec<&CaseResult>> = HashMap::new();
    for r in results {
        grouped.entry(r.dataset_name.clone()).or_default().push(r);
    }

    grouped
        .into_iter()
        .map(|(dataset_name, cases)| {
            let completed = cases.len() as u64;
            let passed = cases.iter().filter(|r| r.passed).count() as u64;
            let failed = completed.saturating_sub(passed);
            let accuracy = if completed > 0 {
                passed as f64 / completed as f64
            } else {
                0.0
            };

            let owned: Vec<CaseResult> = cases.into_iter().cloned().collect();
            let metrics = aggregate_results(&owned);

            (
                dataset_name.clone(),
                DatasetStats {
                    dataset_name,
                    completed_cases: completed,
                    passed_cases: passed,
                    failed_cases: failed,
                    accuracy,
                    metrics,
                },
            )
        })
        .collect()
}

fn compute_tokens(results: &[CaseResult]) -> TokenStats {
    let mut stats = TokenStats::default();
    let mut models = HashSet::new();
    let mut per_model: HashMap<String, ModelTokenUsage> = HashMap::new();

    for r in results {
        stats.agent_input_tokens += r.llm_metrics.input_tokens;
        stats.agent_output_tokens += r.llm_metrics.output_tokens;
        stats.agent_cached_tokens += r.llm_metrics.cached_tokens;
        stats.agent_calls += r.llm_metrics.call_count;

        let agent_model = r
            .llm_metrics
            .model
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        models.insert(agent_model.clone());
        let entry = per_model.entry(agent_model).or_default();
        entry.agent_input_tokens += r.llm_metrics.input_tokens;
        entry.agent_output_tokens += r.llm_metrics.output_tokens;
        entry.agent_cached_tokens += r.llm_metrics.cached_tokens;
        entry.agent_calls += r.llm_metrics.call_count;

        if let Some(ref jm) = r.judge_metrics {
            let judge_model = jm.model.clone().unwrap_or_else(|| "unknown".to_string());
            models.insert(judge_model.clone());
            let judge_entry = per_model.entry(judge_model).or_default();
            judge_entry.judge_input_tokens += jm.input_tokens;
            judge_entry.judge_output_tokens += jm.output_tokens;
            judge_entry.judge_cached_tokens += jm.cached_tokens;

            stats.judge_input_tokens += jm.input_tokens;
            stats.judge_output_tokens += jm.output_tokens;
            stats.judge_cached_tokens += jm.cached_tokens;
        }
    }

    let mut models_used: Vec<String> = models.into_iter().collect();
    models_used.sort();

    stats.models_used = models_used;
    stats.per_model = per_model;

    stats
}

/// Collect failures with actual stored reasons.
///
/// Prefers: judge_reason > error > "failed evaluation"
fn collect_failures(results: &[CaseResult], errors: &[ErrorEntry]) -> Vec<FailureDetail> {
    let mut failures = Vec::new();
    let mut seen_keys: HashSet<(String, String, String)> = HashSet::new();

    // Failed case results
    for r in results.iter().filter(|r| !r.passed) {
        let reason = r
            .judge_reason
            .clone()
            .or_else(|| r.error.clone())
            .unwrap_or_else(|| "failed evaluation".to_string());

        seen_keys.insert((
            r.agent_name.clone(),
            r.dataset_name.clone(),
            r.case_id.clone(),
        ));

        failures.push(FailureDetail {
            case_id: r.case_id.clone(),
            dataset_name: r.dataset_name.clone(),
            agent_name: r.agent_name.clone(),
            error_type: r.error_type,
            reason,
            attempt: r.attempt,
        });
    }

    // System errors from error log
    for e in errors {
        let key = (
            e.agent_name.clone().unwrap_or_default(),
            e.dataset_name.clone().unwrap_or_default(),
            e.case_id.clone().unwrap_or_default(),
        );

        // Skip if we already recorded a failure for this case result
        if seen_keys.contains(&key) {
            continue;
        }

        failures.push(FailureDetail {
            case_id: e.case_id.clone().unwrap_or_else(|| "-".into()),
            dataset_name: e.dataset_name.clone().unwrap_or_else(|| "-".into()),
            agent_name: e.agent_name.clone().unwrap_or_else(|| "-".into()),
            error_type: e.error_type,
            reason: e.error.clone().unwrap_or_else(|| "system error".into()),
            attempt: 0,
        });
    }

    failures
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LlmMetrics;

    fn make_result(case_id: &str, agent: &str, passed: bool) -> CaseResult {
        CaseResult {
            case_id: case_id.into(),
            dataset_name: "test-ds".into(),
            agent_name: agent.into(),
            passed,
            attempt: 1,
            output: Some("output".into()),
            error: None,
            error_type: ErrorType::None,
            runner_duration_ms: 100.0,
            llm_metrics: LlmMetrics {
                call_count: 1,
                latency_ms: vec![50.0],
                input_tokens: 100,
                output_tokens: 50,
                cached_tokens: 10,
                model: Some("gpt-4o".into()),
            },
            timestamp: Some("2024-01-01T00:00:00Z".into()),
            f1_score: None,
            f1_passed: None,
            judge_passed: None,
            judge_reason: None,
            judge_metrics: None,
        }
    }

    fn make_metadata() -> RunMetadata {
        RunMetadata {
            run_id: "test-run".into(),
            status: RunStatus::Completed,
            config_fingerprint: "abc123".into(),
            agents: vec!["agent1".into()],
            datasets: vec!["test-ds".into()],
            total_cases: 3,
            active_cases: Some(3),
            original_total_cases: Some(3),
            retry_of: None,
            completed_cases: 3,
            start_time: Some("2024-01-01T00:00:00Z".into()),
            completed_time: Some("2024-01-01T00:01:00Z".into()),
            system_error_count: 0,
        }
    }

    #[test]
    fn test_compute_basic_stats() {
        let metadata = make_metadata();
        let results = vec![
            make_result("case1", "agent1", true),
            make_result("case2", "agent1", true),
            make_result("case3", "agent1", false),
        ];
        let errors = vec![];

        let stats = RunStats::compute(&metadata, &results, &errors);

        assert_eq!(stats.completed_cases, 3);
        assert_eq!(stats.passed_cases, 2);
        assert_eq!(stats.failed_cases, 1);
        assert!((stats.accuracy - 0.6666).abs() < 0.01);
    }

    #[test]
    fn test_compute_tokens() {
        let metadata = make_metadata();
        let results = vec![
            make_result("case1", "agent1", true),
            make_result("case2", "agent1", true),
        ];
        let errors = vec![];

        let stats = RunStats::compute(&metadata, &results, &errors);

        assert_eq!(stats.tokens.agent_input_tokens, 200);
        assert_eq!(stats.tokens.agent_output_tokens, 100);
        assert_eq!(stats.tokens.agent_cached_tokens, 20);
        assert_eq!(stats.tokens.models_used, vec!["gpt-4o".to_string()]);
        assert_eq!(
            stats
                .tokens
                .per_model
                .get("gpt-4o")
                .map(|m| m.agent_input_tokens),
            Some(200)
        );
    }

    #[test]
    fn test_compute_by_agent() {
        let mut metadata = make_metadata();
        metadata.agents = vec!["agent1".into(), "agent2".into()];

        let results = vec![
            make_result("case1", "agent1", true),
            make_result("case2", "agent1", false),
            make_result("case1", "agent2", true),
        ];
        let errors = vec![];

        let stats = RunStats::compute(&metadata, &results, &errors);

        assert_eq!(stats.by_agent.len(), 2);

        let agent1 = &stats.by_agent["agent1"];
        assert_eq!(agent1.completed_cases, 2);
        assert_eq!(agent1.passed_cases, 1);
        assert!((agent1.accuracy - 0.5).abs() < 0.01);

        let agent2 = &stats.by_agent["agent2"];
        assert_eq!(agent2.completed_cases, 1);
        assert_eq!(agent2.passed_cases, 1);
        assert!((agent2.accuracy - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_failure_reasons_prefer_judge() {
        let metadata = make_metadata();
        let mut result = make_result("case1", "agent1", false);
        result.error = Some("runner error".into());
        result.judge_reason = Some("judge says no".into());

        let results = vec![result];
        let errors = vec![];

        let stats = RunStats::compute(&metadata, &results, &errors);

        assert_eq!(stats.failures.len(), 1);
        assert_eq!(stats.failures[0].reason, "judge says no");
    }

    #[test]
    fn test_system_errors_counted() {
        let metadata = make_metadata();
        let results = vec![];
        let errors = vec![
            ErrorEntry {
                timestamp: "2024-01-01T00:00:00Z".into(),
                error_type: ErrorType::SystemFailure,
                agent_name: Some("agent1".into()),
                dataset_name: Some("test-ds".into()),
                case_id: Some("case1".into()),
                error: Some("timeout".into()),
            },
            ErrorEntry {
                timestamp: "2024-01-01T00:00:01Z".into(),
                error_type: ErrorType::FatalError,
                agent_name: None,
                dataset_name: None,
                case_id: None,
                error: Some("crash".into()),
            },
        ];

        let stats = RunStats::compute(&metadata, &results, &errors);

        assert_eq!(stats.system_error_count, 1);
        assert_eq!(stats.fatal_error_count, 1);
        assert_eq!(stats.failures.len(), 2);
    }
}
