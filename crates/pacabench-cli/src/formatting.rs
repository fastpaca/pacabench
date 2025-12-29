//! Output formatting for run results (show, export commands).
//!
//! All formatting functions should take [`RunStats`] as input - the single
//! source of truth for all run metrics. Do not re-aggregate from raw results.

use crate::pricing::calculate_cost_from_tokens;
use console::style;
use pacabench_core::persistence::{ErrorEntry, RunSummary};
use pacabench_core::stats::RunStats;
use pacabench_core::types::ErrorType;
use pacabench_core::CaseResult;

/// Print a k6-style metric line with dots as separator
fn print_metric(name: &str, value: &str) {
    let width: usize = 20;
    let dots = ".".repeat(width.saturating_sub(name.len()));
    println!("     {}{} {}", style(name).cyan(), style(dots).dim(), value);
}

/// Print a k6-style section header
fn print_section(name: &str) {
    println!();
    println!("     {}", style(name).magenta().bold());
}

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

/// Format duration in human-readable form
fn format_duration_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.1}s", ms / 1000.0)
    } else {
        format!("{:.0}ms", ms)
    }
}

/// Format large numbers with commas for readability
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

/// Print run details from RunStats - k6 style output.
pub fn print_run_stats(stats: &RunStats) {
    let cost = calculate_cost_from_tokens(&stats.tokens);

    // Header
    println!();
    let status_text = match stats.status {
        pacabench_core::RunStatus::Completed => "COMPLETED",
        pacabench_core::RunStatus::Aborted => "ABORTED",
        pacabench_core::RunStatus::Failed => "FAILED",
        pacabench_core::RunStatus::Running => "RUNNING",
        pacabench_core::RunStatus::Loading => "LOADING",
        pacabench_core::RunStatus::Pending => "PENDING",
        pacabench_core::RunStatus::Finalizing => "FINALIZING",
    };
    let status_styled = match stats.status {
        pacabench_core::RunStatus::Completed => style(status_text).green().bold(),
        pacabench_core::RunStatus::Aborted | pacabench_core::RunStatus::Failed => {
            style(status_text).red().bold()
        }
        _ => style(status_text).yellow(),
    };
    println!(
        "     {} {}",
        style(&stats.run_id).bold().cyan(),
        status_styled
    );
    if let Some(retry) = &stats.retry_of {
        println!("     {} {}", style("retry of:").dim(), retry);
    }

    if stats.completed_cases == 0 {
        println!();
        println!("     No results yet.");
        println!();
        return;
    }

    // Accuracy with visual bar
    let acc_pct = stats.accuracy * 100.0;
    let bar_width = 20;
    let filled = ((stats.accuracy * bar_width as f64).round() as usize).min(bar_width);
    let empty = bar_width - filled;
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
    let acc_colored = if acc_pct >= 80.0 {
        style(format!("{:.1}%", acc_pct)).green()
    } else if acc_pct >= 50.0 {
        style(format!("{:.1}%", acc_pct)).yellow()
    } else {
        style(format!("{:.1}%", acc_pct)).red()
    };

    print_section("results");
    print_metric(
        "accuracy",
        &format!(
            "{} {}  passed={}  failed={}",
            style(bar).cyan(),
            acc_colored,
            style(stats.passed_cases).green(),
            style(stats.failed_cases).red()
        ),
    );
    print_metric(
        "cases",
        &format!("{}/{}", stats.completed_cases, stats.planned_cases),
    );

    print_section("performance");
    print_metric(
        "duration",
        &format!(
            "p50={} p95={}",
            format_duration_ms(stats.metrics.p50_duration_ms),
            format_duration_ms(stats.metrics.p95_duration_ms)
        ),
    );
    print_metric(
        "llm_latency",
        &format!(
            "avg={} p50={} p95={}",
            format_duration_ms(stats.metrics.avg_llm_latency_ms),
            format_duration_ms(stats.metrics.p50_llm_latency_ms),
            format_duration_ms(stats.metrics.p95_llm_latency_ms)
        ),
    );
    print_metric(
        "attempts",
        &format!(
            "avg={:.1} max={}",
            stats.metrics.avg_attempts, stats.metrics.max_attempts
        ),
    );

    print_section("tokens");
    print_metric(
        "agent_input",
        &format_number(stats.tokens.agent_input_tokens),
    );
    print_metric(
        "agent_output",
        &format_number(stats.tokens.agent_output_tokens),
    );
    print_metric("llm_calls", &stats.tokens.agent_calls.to_string());

    if stats.tokens.judge_input_tokens > 0 || stats.tokens.judge_output_tokens > 0 {
        print_metric(
            "judge_input",
            &format_number(stats.tokens.judge_input_tokens),
        );
        print_metric(
            "judge_output",
            &format_number(stats.tokens.judge_output_tokens),
        );
    }

    print_section("cost");
    print_metric("total", &format!("${:.4}", cost.total_cost_usd));
    print_metric("agent", &format!("${:.4}", cost.agent_cost_usd));
    print_metric("judge", &format!("${:.4}", cost.judge_cost_usd));

    if !stats.tokens.models_used.is_empty() {
        print_metric("models", &stats.tokens.models_used.join(", "));
    }

    if stats.system_error_count > 0 || stats.fatal_error_count > 0 {
        print_section("errors");
        if stats.system_error_count > 0 {
            print_metric(
                "system_errors",
                &style(stats.system_error_count).yellow().to_string(),
            );
        }
        if stats.fatal_error_count > 0 {
            print_metric(
                "fatal_errors",
                &style(stats.fatal_error_count).red().to_string(),
            );
        }
    }

    // Agents
    if !stats.by_agent.is_empty() {
        print_section("agents");

        let mut agents: Vec<_> = stats.by_agent.values().collect();
        agents.sort_by(|a, b| {
            b.accuracy
                .partial_cmp(&a.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for agent in agents {
            let agent_cost = calculate_cost_from_tokens(&agent.tokens);
            let acc = agent.accuracy * 100.0;
            let acc_styled = if acc >= 80.0 {
                style(format!("{:.1}%", acc)).green()
            } else if acc >= 50.0 {
                style(format!("{:.1}%", acc)).yellow()
            } else {
                style(format!("{:.1}%", acc)).red()
            };
            print_metric(
                &agent.agent_name,
                &format!(
                    "{}  passed={}  failed={}  p50={}  ${:.4}",
                    acc_styled,
                    style(agent.passed_cases).green(),
                    style(agent.failed_cases).red(),
                    format_duration_ms(agent.metrics.p50_duration_ms),
                    agent_cost.total_cost_usd
                ),
            );
        }
    }

    // Failures
    if !stats.failures.is_empty() {
        print_section(&format!("failures ({})", stats.failures.len()));
        println!();

        for failure in stats.failures.iter().take(10) {
            let reason = if failure.reason.len() > 60 {
                format!("{}...", &failure.reason[..57])
            } else {
                failure.reason.clone()
            };
            println!(
                "     {} {}",
                style(format!("{}/{}", failure.agent_name, failure.case_id)).white(),
                style(reason).dim()
            );
        }
        if stats.failures.len() > 10 {
            println!(
                "     {} {}",
                style("...").dim(),
                style(format!("and {} more", stats.failures.len() - 10)).dim()
            );
        }
    }

    println!();
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
