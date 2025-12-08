//! Output formatting for run results (show, export commands).
//!
//! All formatting functions should take [`RunStats`] as input - the single
//! source of truth for all run metrics. Do not re-aggregate from raw results.

use crate::pricing::calculate_cost_from_tokens;
use pacabench_core::persistence::{ErrorEntry, RunSummary};
use pacabench_core::stats::RunStats;
use pacabench_core::types::ErrorType;
use pacabench_core::CaseResult;

// Show command formatting

pub fn print_run_list(runs: &[RunSummary], limit: usize) {
    if runs.is_empty() {
        println!("No runs found.");
        return;
    }

    println!("Runs (showing up to {limit}):");
    println!(
        "{:<28} {:<10} {:>14} {:>10}",
        "run_id", "status", "cases", "progress"
    );

    for r in runs.iter().take(limit.min(runs.len())) {
        let progress = format!("{:.0}%", r.progress * 100.0);
        let cases = if r.total_cases > 0 {
            format!("{}/{}", r.completed_cases, r.total_cases)
        } else {
            "-".into()
        };
        let status = format!("{:?}", r.status).to_lowercase();

        println!(
            "{:<28} {:<10} {:>14} {:>10}",
            r.run_id, status, cases, progress
        );
    }

    if runs.len() > limit {
        println!("... and {} more", runs.len() - limit);
    }
}

// RunStats-based formatting (single source of truth)

/// Print run details from RunStats - the single source of truth.
///
/// This function uses pre-computed stats rather than re-aggregating from results.
/// Cost is calculated using actual model info from the run.
pub fn print_run_stats(stats: &RunStats) {
    let status = format!("{:?}", stats.status).to_lowercase();
    let case_scope = if let Some(orig) = stats
        .original_total_cases
        .filter(|orig| *orig != stats.planned_cases)
    {
        format!(
            "{}/{} (originally {})",
            stats.completed_cases, stats.planned_cases, orig
        )
    } else {
        format!("{}/{}", stats.completed_cases, stats.planned_cases)
    };

    println!("Run {} [{}] cases {case_scope}", stats.run_id, status);

    if let Some(retry) = &stats.retry_of {
        println!("Retry of: {retry}");
    }
    if let Some(active) = stats
        .active_cases
        .filter(|active| *active != stats.planned_cases)
    {
        println!("Scheduled this run: {} case(s)", active);
    }

    if stats.completed_cases == 0 {
        println!("No results yet.");
        return;
    }

    // Use actual model for cost calculation
    let cost = calculate_cost_from_tokens(&stats.tokens);

    println!(
        "Accuracy {:.1}% | Failed {}",
        stats.accuracy * 100.0,
        stats.failed_cases
    );
    println!(
        "Duration p50={:.0}ms p95={:.0}ms | LLM latency avg/p50/p95 = {:.0}/{:.0}/{:.0} ms",
        stats.metrics.p50_duration_ms,
        stats.metrics.p95_duration_ms,
        stats.metrics.avg_llm_latency_ms,
        stats.metrics.p50_llm_latency_ms,
        stats.metrics.p95_llm_latency_ms
    );
    println!(
        "Tokens in/out: {}/{} (judge {}/{}) | LLM calls {} | Cost ${:.4} (judge ${:.4}) | Attempts avg/max {:.1}/{}",
        stats.tokens.agent_input_tokens,
        stats.tokens.agent_output_tokens,
        stats.tokens.judge_input_tokens,
        stats.tokens.judge_output_tokens,
        stats.tokens.agent_calls,
        cost.agent_cost_usd,
        cost.judge_cost_usd,
        stats.metrics.avg_attempts,
        stats.metrics.max_attempts
    );

    if !stats.tokens.models_used.is_empty() {
        println!("Models: {}", stats.tokens.models_used.join(", "));
    }

    if stats.system_error_count > 0 || stats.fatal_error_count > 0 {
        println!(
            "Errors logged: {} system, {} fatal (includes retried)",
            stats.system_error_count, stats.fatal_error_count
        );
    }

    // Per-agent breakdown
    if !stats.by_agent.is_empty() {
        println!("\nBy Agent:");
        let mut agents: Vec<_> = stats.by_agent.values().collect();
        agents.sort_by(|a, b| a.agent_name.cmp(&b.agent_name));

        for agent in agents {
            let agent_cost = calculate_cost_from_tokens(&agent.tokens);
            println!(
                "  {}: {}/{} ({:.1}%) p50={:.0}ms cost=${:.4}",
                agent.agent_name,
                agent.passed_cases,
                agent.completed_cases,
                agent.accuracy * 100.0,
                agent.metrics.p50_duration_ms,
                agent_cost.total_cost_usd
            );
        }
    }

    // Failures with actual reasons
    if !stats.failures.is_empty() {
        println!("\nFailures (showing up to 10):");
        for failure in stats.failures.iter().take(10) {
            println!(
                "  - {}/{} {}: {}",
                failure.dataset_name, failure.case_id, failure.agent_name, failure.reason
            );
        }
        if stats.failures.len() > 10 {
            println!("  ... and {} more", stats.failures.len() - 10);
        }
    }
}

/// Export schema version.
///
/// ## v2 (current)
/// - Structure derived from `RunStats` (single source of truth)
/// - Added `schema_version` field
/// - Added `tokens.per_model` for per-model token breakdown
/// - Added `original_total_cases`, `active_cases`, `retry_of` for retry tracking
/// - `by_agent` replaces `agents` with computed stats (no raw results per-agent)
/// - `failures` array includes `reason` from judge/error
/// - `cases` field (optional, via `--include-cases`) contains raw `CaseResult` array
/// - Error counts include transient errors (not just final failures)
///
/// ## v1 (legacy, no longer produced)
/// - `agents` contained per-agent raw results
/// - No per-model token breakdown
/// - No retry lineage tracking
/// - Error counts only included final failures
pub const EXPORT_SCHEMA_VERSION: &str = "v2";

/// Build JSON export from RunStats.
pub fn build_export_json_from_stats(
    stats: &RunStats,
    cases: Option<&[CaseResult]>,
) -> serde_json::Value {
    let cost = calculate_cost_from_tokens(&stats.tokens);

    serde_json::json!({
        "schema_version": EXPORT_SCHEMA_VERSION,
        "run_id": stats.run_id,
        "status": format!("{:?}", stats.status).to_lowercase(),
        "start_time": stats.start_time,
        "end_time": stats.end_time,
        "planned_cases": stats.planned_cases,
        "original_total_cases": stats.original_total_cases,
        "active_cases": stats.active_cases,
        "completed_cases": stats.completed_cases,
        "passed_cases": stats.passed_cases,
        "failed_cases": stats.failed_cases,
        "accuracy": stats.accuracy,
        "retry_of": stats.retry_of,
        "metrics": stats.metrics,
        "tokens": {
            "agent_input": stats.tokens.agent_input_tokens,
            "agent_output": stats.tokens.agent_output_tokens,
            "agent_cached": stats.tokens.agent_cached_tokens,
            "judge_input": stats.tokens.judge_input_tokens,
            "judge_output": stats.tokens.judge_output_tokens,
            "models_used": stats.tokens.models_used,
            "per_model": stats.tokens.per_model.iter().map(|(model, usage)| {
                (model.clone(), serde_json::json!({
                    "agent_input": usage.agent_input_tokens,
                    "agent_output": usage.agent_output_tokens,
                    "agent_cached": usage.agent_cached_tokens,
                    "agent_calls": usage.agent_calls,
                    "judge_input": usage.judge_input_tokens,
                    "judge_output": usage.judge_output_tokens,
                    "judge_cached": usage.judge_cached_tokens,
                }))
            }).collect::<serde_json::Map<String, serde_json::Value>>()
        },
        "cost": {
            "agent_usd": cost.agent_cost_usd,
            "judge_usd": cost.judge_cost_usd,
            "total_usd": cost.total_cost_usd,
        },
        "by_agent": stats.by_agent.iter().map(|(name, agent)| {
            let agent_cost = calculate_cost_from_tokens(&agent.tokens);
            (name.clone(), serde_json::json!({
                "completed_cases": agent.completed_cases,
                "passed_cases": agent.passed_cases,
                "failed_cases": agent.failed_cases,
                "accuracy": agent.accuracy,
                "metrics": agent.metrics,
                "cost_usd": agent_cost.total_cost_usd,
            }))
        }).collect::<serde_json::Map<String, serde_json::Value>>(),
        "failures": stats.failures.iter().map(|f| {
            serde_json::json!({
                "case_id": f.case_id,
                "dataset": f.dataset_name,
                "agent": f.agent_name,
                "error_type": format!("{:?}", f.error_type).to_lowercase(),
                "reason": f.reason,
                "attempt": f.attempt,
            })
        }).collect::<Vec<_>>(),
        "system_error_count": stats.system_error_count,
        "fatal_error_count": stats.fatal_error_count,
        "cases": cases.map(|c| serde_json::to_value(c).unwrap_or_default())
    })
}

/// Build Markdown export from RunStats.
pub fn build_export_markdown_from_stats(stats: &RunStats) -> String {
    let mut md = String::new();
    let cost = calculate_cost_from_tokens(&stats.tokens);

    md.push_str(&format!("# Run: {}\n\n", stats.run_id));
    md.push_str(&format!("_Schema {}_\n\n", EXPORT_SCHEMA_VERSION));
    md.push_str("## Summary\n\n");
    md.push_str(&format!(
        "- **Status**: {}\n",
        format!("{:?}", stats.status).to_lowercase()
    ));
    if let Some(retry) = &stats.retry_of {
        md.push_str(&format!("- **Retry of**: {}\n", retry));
    }
    md.push_str(&format!(
        "- **Cases**: {} / {}\n",
        stats.completed_cases, stats.planned_cases
    ));
    if let Some(active) = stats
        .active_cases
        .filter(|active| *active != stats.planned_cases)
    {
        md.push_str(&format!("- **Scheduled This Run**: {}\n", active));
    }
    if let Some(orig) = stats
        .original_total_cases
        .filter(|orig| *orig != stats.planned_cases)
    {
        md.push_str(&format!("- **Original Planned Cases**: {}\n", orig));
    }
    md.push_str(&format!("- **Accuracy**: {:.1}%\n", stats.accuracy * 100.0));
    md.push_str(&format!(
        "- **Duration (p50/p95)**: {:.0}ms / {:.0}ms\n",
        stats.metrics.p50_duration_ms, stats.metrics.p95_duration_ms
    ));
    md.push_str(&format!(
        "- **LLM Latency (avg/p50/p95)**: {:.0}ms / {:.0}ms / {:.0}ms\n",
        stats.metrics.avg_llm_latency_ms,
        stats.metrics.p50_llm_latency_ms,
        stats.metrics.p95_llm_latency_ms
    ));
    md.push_str(&format!(
        "- **Tokens (in/out)**: {} / {}\n",
        stats.tokens.agent_input_tokens, stats.tokens.agent_output_tokens
    ));
    md.push_str(&format!(
        "- **Judge Tokens (in/out)**: {} / {}\n",
        stats.tokens.judge_input_tokens, stats.tokens.judge_output_tokens
    ));
    md.push_str(&format!(
        "- **Cost**: ${:.4} (judge ${:.4})\n",
        cost.agent_cost_usd, cost.judge_cost_usd
    ));
    md.push_str(&format!(
        "- **Attempts (avg/max)**: {:.1} / {}\n",
        stats.metrics.avg_attempts, stats.metrics.max_attempts
    ));

    if !stats.tokens.models_used.is_empty() {
        md.push_str(&format!(
            "- **Models**: {}\n",
            stats.tokens.models_used.join(", ")
        ));
    }

    if !stats.tokens.per_model.is_empty() {
        md.push_str("\n### Tokens by Model\n\n");
        md.push_str("| Model | Agent In | Agent Out | Agent Cached | Judge In | Judge Out |\n");
        md.push_str("|-------|----------|-----------|--------------|----------|-----------|\n");
        let mut models: Vec<_> = stats.tokens.per_model.iter().collect();
        models.sort_by(|a, b| a.0.cmp(b.0));
        for (model, usage) in models {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                model,
                usage.agent_input_tokens,
                usage.agent_output_tokens,
                usage.agent_cached_tokens,
                usage.judge_input_tokens,
                usage.judge_output_tokens
            ));
        }
    }

    // Per-agent table
    if !stats.by_agent.is_empty() {
        md.push_str("\n## By Agent\n\n");
        md.push_str("| Agent | Passed/Total | Accuracy | p50 | Cost |\n");
        md.push_str("|-------|--------------|----------|-----|------|\n");

        let mut agents: Vec<_> = stats.by_agent.values().collect();
        agents.sort_by(|a, b| a.agent_name.cmp(&b.agent_name));

        for agent in agents {
            let agent_cost = calculate_cost_from_tokens(&agent.tokens);
            md.push_str(&format!(
                "| {} | {}/{} | {:.1}% | {:.0}ms | ${:.4} |\n",
                agent.agent_name,
                agent.passed_cases,
                agent.completed_cases,
                agent.accuracy * 100.0,
                agent.metrics.p50_duration_ms,
                agent_cost.total_cost_usd
            ));
        }
    }

    // Failures
    if !stats.failures.is_empty() {
        md.push_str("\n## Failures\n\n");
        for failure in &stats.failures {
            md.push_str(&format!(
                "- **{}/{}** ({}): {}\n",
                failure.dataset_name, failure.case_id, failure.agent_name, failure.reason
            ));
        }
    }

    md
}

// Case-level display (still needs raw results)
pub fn print_cases(
    run_id: &str,
    results: &[CaseResult],
    errors: &[ErrorEntry],
    failures_only: bool,
    limit: usize,
) {
    use std::collections::HashSet;

    println!("\nCases for {run_id}:");
    let mut rows: Vec<(String, String, String, String, String)> = Vec::new();
    let mut seen_keys: HashSet<(String, String, String)> = HashSet::new();

    for r in results {
        let status = if matches!(
            r.error_type,
            ErrorType::SystemFailure | ErrorType::FatalError
        ) {
            "error"
        } else if r.passed {
            "pass"
        } else {
            "fail"
        };
        if failures_only && status == "pass" {
            continue;
        }

        seen_keys.insert((
            r.agent_name.clone(),
            r.dataset_name.clone(),
            r.case_id.clone(),
        ));

        let summary = r
            .judge_reason
            .clone()
            .or_else(|| r.error.clone())
            .or_else(|| r.output.clone())
            .unwrap_or_else(|| "-".into());
        rows.push((
            r.case_id.clone(),
            r.agent_name.clone(),
            r.dataset_name.clone(),
            status.to_string(),
            summary,
        ));
    }

    // Only add error entries that don't have a corresponding case result
    for e in errors {
        let key = (
            e.agent_name.clone().unwrap_or_default(),
            e.dataset_name.clone().unwrap_or_default(),
            e.case_id.clone().unwrap_or_default(),
        );
        if seen_keys.contains(&key) {
            continue;
        }

        let status = match e.error_type {
            ErrorType::FatalError => "fatal",
            _ => "error",
        };
        let case_id = e.case_id.clone().unwrap_or_else(|| "-".into());
        let agent = e.agent_name.clone().unwrap_or_else(|| "-".into());
        let dataset = e.dataset_name.clone().unwrap_or_else(|| "-".into());
        let summary = e.error.clone().unwrap_or_else(|| "unknown error".into());
        rows.push((case_id, agent, dataset, status.into(), summary));
    }

    if rows.is_empty() {
        println!("No cases recorded.");
        return;
    }

    rows.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));

    println!(
        "{:<12} {:<16} {:<16} {:<8} output/error",
        "case_id", "agent", "dataset", "status"
    );
    let total = rows.len();
    for (case_id, agent, dataset, status, summary) in rows.into_iter().take(limit) {
        let truncated = if summary.len() > 80 {
            format!("{}...", &summary[..77])
        } else {
            summary
        };
        println!(
            "{:<12} {:<16} {:<16} {:<8} {}",
            case_id, agent, dataset, status, truncated
        );
    }

    if total > limit {
        println!("... showing {limit} of {total}");
    }
}
