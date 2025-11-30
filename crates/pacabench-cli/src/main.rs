//! Thin CLI wrapper for the Rust rewrite of PacaBench.

mod progress;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use pacabench_core::config::load_config;
use pacabench_core::metrics::aggregate_results;
use pacabench_core::orchestrator::Orchestrator;
use pacabench_core::persistence::{
    default_dataset_cache_dir, list_run_summaries, resolve_runs_dir, ErrorEntry, RunMetadata,
    RunStore, RunSummary,
};
use pacabench_core::types::{CaseResult, ErrorType};
use progress::IndicatifReporter;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Parser)]
#[command(name = "pacabench", about = "Rust rewrite of the PacaBench CLI")]
struct Cli {
    /// Path to the benchmark configuration file.
    #[arg(short, long, default_value = "pacabench.yaml")]
    config: String,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Load and print the parsed configuration.
    ShowConfig,

    /// Run the benchmark.
    Run {
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        run_id: Option<String>,
        #[arg(long)]
        runs_dir: Option<String>,
        #[arg(long)]
        cache_dir: Option<String>,
        /// Force a new run even if run_id exists with different config.
        #[arg(long)]
        force_new: bool,
        /// Comma-separated list of agents to run (e.g. --agents agent1,agent2).
        #[arg(long, short = 'a')]
        agents: Option<String>,
    },

    /// Show aggregated metrics for a run id.
    Show {
        /// Run ID (supports partial match).
        #[arg()]
        run_id: Option<String>,
        #[arg(long)]
        runs_dir: Option<String>,
        /// Show individual case rows.
        #[arg(long)]
        cases: bool,
        /// Show only failures in the case list.
        #[arg(long)]
        failures: bool,
        /// Limit number of cases shown.
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },

    /// Retry failed cases from a previous run.
    Retry {
        /// Run ID to retry from.
        #[arg()]
        run_id: String,
        #[arg(long)]
        runs_dir: Option<String>,
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Export results to JSON or Markdown.
    Export {
        /// Run ID to export.
        #[arg()]
        run_id: String,
        /// Output format: json or markdown.
        #[arg(long, default_value = "json")]
        format: String,
        /// Output file path (defaults to stdout).
        #[arg(short, long)]
        output: Option<String>,
        #[arg(long)]
        runs_dir: Option<String>,
    },

    /// Initialize a new pacabench.yaml configuration file.
    Init {
        /// Name for the benchmark.
        #[arg(long, default_value = "my-benchmark")]
        name: String,
        /// Output path.
        #[arg(short, long, default_value = "pacabench.yaml")]
        output: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = PathBuf::from(&cli.config);

    if let Some(Command::Init { name, output }) = cli.command {
        let template = format!(
            r#"name: {name}
version: "0.1.0"

config:
  concurrency: 4
  timeout_seconds: 60
  max_retries: 2
  proxy:
    enabled: true

agents:
  - name: my-agent
    command: python agent.py

datasets:
  - name: my-dataset
    source: ./data

output:
  directory: ./runs
"#
        );
        fs::write(&output, template)?;
        println!("Created {output}");
        return Ok(());
    }

    let config = load_config(&config_path)?;

    let root_dir = config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    match cli.command {
        Some(Command::ShowConfig) => {
            println!(
                "Loaded benchmark '{}': {} agent(s), {} dataset(s).",
                config.name,
                config.agents.len(),
                config.datasets.len()
            );
            let yaml = serde_yaml::to_string(&config)?;
            println!("{yaml}");
        }
        Some(Command::Run {
            limit,
            run_id,
            runs_dir,
            cache_dir,
            force_new,
            agents,
        }) => {
            let runs_dir = resolve_runs_dir(
                Some(&config),
                runs_dir.map(PathBuf::from),
                Some(&config_path),
            );
            let cache_dir = cache_dir
                .map(PathBuf::from)
                .unwrap_or_else(default_dataset_cache_dir);

            // Filter agents if specified
            let mut config = config;
            if let Some(agents_filter) = agents {
                let filter_set: std::collections::HashSet<&str> =
                    agents_filter.split(',').map(|s| s.trim()).collect();
                let available: Vec<String> = config.agents.iter().map(|a| a.name.clone()).collect();
                config
                    .agents
                    .retain(|a| filter_set.contains(a.name.as_str()));
                if config.agents.is_empty() {
                    return Err(anyhow!(
                        "No agents found matching filter: {}. Available: {}",
                        agents_filter,
                        available.join(", ")
                    ));
                }
                println!(
                    "Running {} agent(s): {}",
                    config.agents.len(),
                    config
                        .agents
                        .iter()
                        .map(|a| a.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            let reporter = Arc::new(IndicatifReporter::new());
            let orch = Orchestrator::new(config, root_dir, cache_dir, runs_dir)
                .with_reporter(reporter)
                .with_config_path(config_path.clone());
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(orch.run(run_id, limit, force_new))?;
        }
        Some(Command::Show {
            run_id,
            runs_dir,
            cases,
            failures,
            limit,
        }) => {
            let runs_dir = resolve_runs_dir(
                Some(&config),
                runs_dir.map(PathBuf::from),
                Some(&config_path),
            );
            if run_id.is_none() {
                let runs = list_run_summaries(&runs_dir)?;
                print_run_list(&runs, limit);
                return Ok(());
            }

            let partial = run_id.as_ref().expect("guarded above");
            let resolved_id = resolve_run_id(&runs_dir, partial)?;
            let store = RunStore::new(runs_dir.join(&resolved_id))?;
            let results = store.load_results()?;
            let errors = store.load_errors()?;
            let metadata = store.read_metadata()?;
            print_run_details(&resolved_id, metadata, &results, &errors);
            if cases {
                print_cases(&resolved_id, &results, &errors, failures, limit);
            }
        }
        Some(Command::Retry {
            run_id,
            runs_dir,
            limit,
        }) => {
            let runs_dir = resolve_runs_dir(
                Some(&config),
                runs_dir.map(PathBuf::from),
                Some(&config_path),
            );
            let resolved_id = resolve_run_id(&runs_dir, &run_id)?;

            // Load existing results and find failed cases
            let store = RunStore::new(runs_dir.join(&resolved_id))?;
            let results = store.load_results()?;
            let failed_count = results.iter().filter(|r| !r.passed).count();

            if failed_count == 0 {
                println!("No failed cases to retry in run {resolved_id}");
                return Ok(());
            }

            println!(
                "Retrying {} failed cases from run {resolved_id}",
                failed_count
            );

            let cache_dir = default_dataset_cache_dir();
            let reporter = Arc::new(IndicatifReporter::new());
            let orch = Orchestrator::new(config, root_dir, cache_dir, runs_dir)
                .with_reporter(reporter)
                .with_config_path(config_path.clone());
            let rt = tokio::runtime::Runtime::new()?;
            // Resume the run - it will re-attempt failed cases
            rt.block_on(orch.run(Some(resolved_id), limit, false))?;
        }
        Some(Command::Export {
            run_id,
            format,
            output,
            runs_dir,
        }) => {
            let runs_dir = resolve_runs_dir(
                Some(&config),
                runs_dir.map(PathBuf::from),
                Some(&config_path),
            );
            let resolved_id = resolve_run_id(&runs_dir, &run_id)?;
            let store = RunStore::new(runs_dir.join(&resolved_id))?;
            let results = store.load_results()?;
            let errors = store.load_errors()?;
            let metadata = store.read_metadata()?;

            let content = match format.as_str() {
                "json" => serde_json::to_string_pretty(&build_export_json(
                    &resolved_id,
                    metadata.as_ref(),
                    &results,
                    &errors,
                ))?,
                "markdown" | "md" => {
                    build_export_markdown(&resolved_id, metadata.as_ref(), &results, &errors)
                }
                _ => return Err(anyhow!("unsupported format: {format}")),
            };

            match output {
                Some(path) => {
                    fs::write(&path, &content)?;
                    println!("Exported to {path}");
                }
                None => println!("{content}"),
            }
        }
        Some(Command::Init { .. }) => unreachable!(),
        None => {
            println!(
                "Loaded benchmark '{}': {} agent(s), {} dataset(s).",
                config.name,
                config.agents.len(),
                config.datasets.len()
            );
            println!("\nUse --help to see available commands.");
        }
    }

    Ok(())
}

fn print_run_list(runs: &[RunSummary], limit: usize) {
    if runs.is_empty() {
        println!("No runs found.");
        return;
    }

    println!("Runs (showing up to {limit}):");
    println!(
        "{:<28} {:<10} {:>14} {:>10} {:>10}",
        "run_id", "status", "cases", "progress", "cost"
    );

    for r in runs.iter().take(limit.min(runs.len())) {
        let progress = r
            .progress
            .map(|p| format!("{:.0}%", p * 100.0))
            .unwrap_or_else(|| "-".into());
        let cases = if r.total_cases > 0 {
            format!("{}/{}", r.completed_cases, r.total_cases)
        } else {
            "-".into()
        };
        let cost = r
            .total_cost_usd
            .map(|c| format!("${:.3}", c))
            .unwrap_or_else(|| "-".into());

        println!(
            "{:<28} {:<10} {:>14} {:>10} {:>10}",
            r.run_id, r.status, cases, progress, cost
        );
    }

    if runs.len() > limit {
        println!("... and {} more", runs.len() - limit);
    }
}

fn print_run_details(
    run_id: &str,
    metadata: Option<RunMetadata>,
    results: &[CaseResult],
    errors: &[ErrorEntry],
) {
    let status = metadata
        .as_ref()
        .map(|m| m.status.as_str())
        .unwrap_or("unknown");
    let total_cases = metadata
        .as_ref()
        .map(|m| m.total_cases)
        .unwrap_or(results.len() as u64);
    let completed_cases = metadata
        .as_ref()
        .map(|m| m.completed_cases)
        .unwrap_or(results.len() as u64);

    println!(
        "Run {run_id} [{status}] cases {completed}/{total}",
        completed = completed_cases,
        total = total_cases
    );

    if results.is_empty() {
        println!("No results yet.");
        return;
    }

    let agg = aggregate_results(results);
    println!(
        "Accuracy {acc:.1}% | Precision {prec:.1}% | Recall {rec:.1}% | Failed {failed}",
        acc = agg.accuracy * 100.0,
        prec = agg.precision * 100.0,
        rec = agg.recall * 100.0,
        failed = agg.failed_cases
    );
    println!(
        "Duration p50={:.0}ms p95={:.0}ms | LLM latency avg/p50/p95 = {:.0}/{:.0}/{:.0} ms",
        agg.p50_duration_ms,
        agg.p95_duration_ms,
        agg.avg_llm_latency_ms,
        agg.p50_llm_latency_ms,
        agg.p95_llm_latency_ms
    );
    println!(
        "Tokens in/out: {}/{} (judge {}/{}) | LLM calls {} | Cost ${:.4} (judge ${:.4}) | Attempts avg/max {:.1}/{}",
        agg.total_input_tokens,
        agg.total_output_tokens,
        agg.total_judge_input_tokens,
        agg.total_judge_output_tokens,
        agg.total_llm_calls,
        agg.total_cost_usd,
        agg.total_judge_cost_usd,
        agg.avg_attempts,
        agg.max_attempts
    );
    if !errors.is_empty() {
        println!("System errors: {}", errors.len());
    }

    let mut grouped: BTreeMap<(String, String), Vec<CaseResult>> = BTreeMap::new();
    for r in results {
        grouped
            .entry((r.agent_name.clone(), r.dataset_name.clone()))
            .or_default()
            .push(r.clone());
    }

    if !grouped.is_empty() {
        println!("\nBy Agent/Dataset:");
        for ((agent, dataset), group) in grouped {
            let metrics = aggregate_results(&group);
            let passed = group.iter().filter(|r| r.passed).count();
            let total = group.len();
            let cost = metrics.total_cost_usd + metrics.total_judge_cost_usd;
            println!(
                "  {agent} on {dataset}: {passed}/{total} ({acc:.1}%) p50={p50:.0}ms cost=${cost:.4}",
                acc = metrics.accuracy * 100.0,
                p50 = metrics.p50_duration_ms,
                cost = cost
            );
        }
    }

    let failures = collect_failures(results, errors);
    if !failures.is_empty() {
        println!("\nFailures (showing up to 10):");
        for line in failures.iter().take(10) {
            println!("  - {line}");
        }
        if failures.len() > 10 {
            println!("  ... and {} more", failures.len() - 10);
        }
    }
}

fn print_cases(
    run_id: &str,
    results: &[CaseResult],
    errors: &[ErrorEntry],
    failures_only: bool,
    limit: usize,
) {
    println!("\nCases for {run_id}:");
    let mut rows: Vec<(String, String, String, String, String)> = Vec::new();

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
        let summary = r
            .output
            .clone()
            .or_else(|| r.error.clone())
            .unwrap_or_else(|| "-".into());
        rows.push((
            r.case_id.clone(),
            r.agent_name.clone(),
            r.dataset_name.clone(),
            status.to_string(),
            summary,
        ));
    }

    for e in errors {
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

fn collect_failures(results: &[CaseResult], errors: &[ErrorEntry]) -> Vec<String> {
    let mut failures = Vec::new();
    for r in results {
        if r.passed {
            continue;
        }
        let reason = r
            .error
            .clone()
            .or_else(|| r.judge_reason.clone())
            .unwrap_or_else(|| "failed evaluation".into());
        failures.push(format!(
            "{}/{} {}: {}",
            r.dataset_name, r.case_id, r.agent_name, reason
        ));
    }
    for e in errors {
        let cid = e.case_id.clone().unwrap_or_else(|| "-".into());
        let ds = e.dataset_name.clone().unwrap_or_else(|| "-".into());
        let agent = e.agent_name.clone().unwrap_or_else(|| "-".into());
        let reason = e.error.clone().unwrap_or_else(|| "system error".into());
        failures.push(format!("{ds}/{cid} {agent}: {reason}"));
    }
    failures
}

fn build_export_json(
    run_id: &str,
    metadata: Option<&RunMetadata>,
    results: &[CaseResult],
    errors: &[ErrorEntry],
) -> serde_json::Value {
    let mut agents_map = serde_json::Map::new();
    let mut agent_names: Vec<String> = results.iter().map(|r| r.agent_name.clone()).collect();
    agent_names.sort();
    agent_names.dedup();

    for agent in agent_names {
        let agent_results: Vec<CaseResult> = results
            .iter()
            .filter(|r| r.agent_name == agent)
            .cloned()
            .collect();
        let metrics = aggregate_results(&agent_results);
        let res_entries: Vec<serde_json::Value> = agent_results.iter().map(|r| json!(r)).collect();
        agents_map.insert(
            agent.clone(),
            json!({
                "metrics": metrics,
                "results": res_entries,
            }),
        );
    }

    json!({
        "run_id": run_id,
        "status": metadata.map(|m| m.status.clone()).unwrap_or_else(|| "unknown".into()),
        "start_time": metadata.and_then(|m| m.start_time.clone()),
        "completed_time": metadata.and_then(|m| m.completed_time.clone()),
        "total_cases": metadata.map(|m| m.total_cases).unwrap_or(results.len() as u64),
        "completed_cases": metadata.map(|m| m.completed_cases).unwrap_or(results.len() as u64),
        "agents": agents_map,
        "system_errors": errors,
    })
}

fn build_export_markdown(
    run_id: &str,
    metadata: Option<&RunMetadata>,
    results: &[CaseResult],
    errors: &[ErrorEntry],
) -> String {
    let mut md = String::new();
    let agg = aggregate_results(results);
    md.push_str(&format!("# Run: {run_id}\n\n"));
    md.push_str("## Summary\n\n");
    md.push_str(&format!(
        "- **Status**: {}\n",
        metadata.map(|m| m.status.as_str()).unwrap_or("unknown")
    ));
    md.push_str(&format!(
        "- **Cases**: {} / {}\n",
        metadata
            .map(|m| m.completed_cases)
            .unwrap_or(results.len() as u64),
        metadata
            .map(|m| m.total_cases)
            .unwrap_or(results.len() as u64)
    ));
    md.push_str(&format!(
        "- **Accuracy / Precision / Recall**: {:.1}% / {:.1}% / {:.1}%\n",
        agg.accuracy * 100.0,
        agg.precision * 100.0,
        agg.recall * 100.0
    ));
    md.push_str(&format!(
        "- **Duration (p50/p95)**: {:.0}ms / {:.0}ms\n",
        agg.p50_duration_ms, agg.p95_duration_ms
    ));
    md.push_str(&format!(
        "- **LLM Latency (avg/p50/p95)**: {:.0}ms / {:.0}ms / {:.0}ms\n",
        agg.avg_llm_latency_ms, agg.p50_llm_latency_ms, agg.p95_llm_latency_ms
    ));
    md.push_str(&format!(
        "- **Tokens (in/out)**: {} / {}\n",
        agg.total_input_tokens, agg.total_output_tokens
    ));
    md.push_str(&format!(
        "- **Judge Tokens (in/out)**: {} / {}\n",
        agg.total_judge_input_tokens, agg.total_judge_output_tokens
    ));
    md.push_str(&format!(
        "- **Cost**: ${:.4} (judge ${:.4})\n",
        agg.total_cost_usd, agg.total_judge_cost_usd
    ));
    md.push_str(&format!(
        "- **Attempts (avg/max)**: {:.1} / {}\n",
        agg.avg_attempts, agg.max_attempts
    ));

    let mut grouped: BTreeMap<(String, String), Vec<CaseResult>> = BTreeMap::new();
    for r in results {
        grouped
            .entry((r.agent_name.clone(), r.dataset_name.clone()))
            .or_default()
            .push(r.clone());
    }

    if !grouped.is_empty() {
        md.push_str("\n## Agent/Dataset\n\n");
        md.push_str("| Agent | Dataset | Passed/Total | Accuracy | Cost |\n");
        md.push_str("|-------|---------|--------------|----------|------|\n");
        for ((agent, dataset), group) in grouped {
            let metrics = aggregate_results(&group);
            let passed = group.iter().filter(|r| r.passed).count();
            let total = group.len();
            let cost = metrics.total_cost_usd + metrics.total_judge_cost_usd;
            md.push_str(&format!(
                "| {} | {} | {}/{} | {:.1}% | ${:.4} |\n",
                agent,
                dataset,
                passed,
                total,
                metrics.accuracy * 100.0,
                cost
            ));
        }
    }

    let failures = collect_failures(results, errors);
    if !failures.is_empty() {
        md.push_str("\n## Failures\n\n");
        for line in failures {
            md.push_str(&format!("- {line}\n"));
        }
    }

    md
}

/// Resolve a partial run ID to a full run ID by finding matching directories.
fn resolve_run_id(runs_dir: &std::path::Path, partial: &str) -> Result<String> {
    let summaries = list_run_summaries(runs_dir)?;
    if summaries.is_empty() {
        return Err(anyhow!("no runs found in {}", runs_dir.display()));
    }

    if let Some(exact) = summaries.iter().find(|s| s.run_id == partial) {
        return Ok(exact.run_id.clone());
    }

    let matches: Vec<&RunSummary> = summaries
        .iter()
        .filter(|s| s.run_id.starts_with(partial) || s.run_id.contains(partial))
        .collect();

    match matches.len() {
        0 => Err(anyhow!("no run found matching '{partial}'")),
        1 => Ok(matches[0].run_id.clone()),
        _ => {
            let options: Vec<_> = matches.iter().map(|s| s.run_id.clone()).collect();
            Err(anyhow!(
                "ambiguous run ID '{partial}', matches: {}",
                options.join(", ")
            ))
        }
    }
}
