//! CLI for PacaBench - a local-first benchmark harness for LLM agents.

mod formatting;
mod init;
mod pricing;
mod progress;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use formatting::{
    build_export_json, build_export_markdown, print_cases, print_run_details, print_run_list,
};
use pacabench_core::config::ConfigOverrides;
use pacabench_core::persistence::{list_run_summaries, RunStore, RunSummary};
use pacabench_core::{Benchmark, Config};
use progress::ProgressDisplay;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "pacabench",
    about = "A local-first benchmark harness for LLM agents"
)]
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
        /// Override concurrency (number of parallel workers).
        #[arg(long)]
        concurrency: Option<usize>,
        /// Override timeout in seconds.
        #[arg(long)]
        timeout: Option<f64>,
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

    /// Initialize a new pacabench project with example files.
    Init {
        /// Name for the benchmark. Defaults to project name from pyproject.toml if present.
        #[arg(long)]
        name: Option<String>,
        /// Overwrite existing files.
        #[arg(long)]
        force: bool,
        /// Skip updating pyproject.toml.
        #[arg(long)]
        no_pyproject: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = PathBuf::from(&cli.config);

    // Handle init separately (doesn't need config file)
    if let Some(Command::Init {
        name,
        force,
        no_pyproject,
    }) = cli.command
    {
        let project_name = name.unwrap_or_else(|| {
            init::read_pyproject_name().unwrap_or_else(|| "my-benchmark".to_string())
        });
        return init::execute(&project_name, force, !no_pyproject);
    }

    // Build overrides based on command
    let overrides = match &cli.command {
        Some(Command::Run {
            runs_dir,
            cache_dir,
            concurrency,
            timeout,
            ..
        }) => ConfigOverrides {
            concurrency: *concurrency,
            timeout_seconds: *timeout,
            max_retries: None,
            runs_dir: runs_dir.clone(),
            cache_dir: cache_dir.clone(),
        },
        _ => ConfigOverrides::default(),
    };

    // Load config
    let mut config = Config::from_file(&config_path, overrides)?;

    match cli.command {
        Some(Command::ShowConfig) => cmd_show_config(&config),
        Some(Command::Run {
            limit,
            run_id,
            agents,
            ..
        }) => cmd_run(&mut config, run_id, limit, agents)?,
        Some(Command::Show {
            run_id,
            runs_dir,
            cases,
            failures,
            limit,
        }) => cmd_show(&config, run_id, runs_dir, cases, failures, limit)?,
        Some(Command::Retry {
            run_id,
            runs_dir,
            limit,
        }) => cmd_retry(&mut config, &run_id, runs_dir, limit)?,
        Some(Command::Export {
            run_id,
            format,
            output,
            runs_dir,
        }) => cmd_export(&config, &run_id, &format, output, runs_dir)?,
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

// ============================================================================
// Command handlers
// ============================================================================

fn cmd_show_config(config: &Config) {
    println!(
        "Loaded benchmark '{}': {} agent(s), {} dataset(s).",
        config.name,
        config.agents.len(),
        config.datasets.len()
    );
    let yaml = serde_yaml::to_string(&config).expect("config serialization should succeed");
    println!("{yaml}");
}

fn cmd_run(
    config: &mut Config,
    run_id: Option<String>,
    limit: Option<usize>,
    agents: Option<String>,
) -> Result<()> {
    // Filter agents if specified
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

    let bench = Benchmark::new(config.clone());
    let events = bench.subscribe();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let display = ProgressDisplay::new();
        let display_handle = tokio::spawn(display.run(events));
        let result = bench.run(run_id, limit).await;
        let _ = display_handle.await;
        result
    })?;

    Ok(())
}

fn cmd_show(
    config: &Config,
    run_id: Option<String>,
    runs_dir: Option<String>,
    cases: bool,
    failures: bool,
    limit: usize,
) -> Result<()> {
    let runs_dir = runs_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| config.runs_dir.clone());

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

    Ok(())
}

fn cmd_retry(
    config: &mut Config,
    run_id: &str,
    runs_dir: Option<String>,
    limit: Option<usize>,
) -> Result<()> {
    if let Some(dir) = runs_dir {
        config.runs_dir = PathBuf::from(dir);
    }

    let resolved_id = resolve_run_id(&config.runs_dir, run_id)?;
    let store = RunStore::new(config.runs_dir.join(&resolved_id))?;
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

    let bench = Benchmark::new(config.clone());
    let events = bench.subscribe();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let display = ProgressDisplay::new();
        let display_handle = tokio::spawn(display.run(events));
        let result = bench.run(Some(resolved_id), limit).await;
        let _ = display_handle.await;
        result
    })?;

    Ok(())
}

fn cmd_export(
    config: &Config,
    run_id: &str,
    format: &str,
    output: Option<String>,
    runs_dir: Option<String>,
) -> Result<()> {
    let runs_dir = runs_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| config.runs_dir.clone());
    let resolved_id = resolve_run_id(&runs_dir, run_id)?;
    let store = RunStore::new(runs_dir.join(&resolved_id))?;
    let results = store.load_results()?;
    let errors = store.load_errors()?;
    let metadata = store.read_metadata()?;

    let content = match format {
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

    Ok(())
}

// ============================================================================
// Utilities
// ============================================================================

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
