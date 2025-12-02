//! Benchmark - the main entry point for running benchmarks.

use crate::config::Config;
use crate::datasets::{get_dataset_loader, DatasetContext};
use crate::error::{PacabenchError, Result};
use crate::metrics::aggregate_results;
use crate::persistence::{
    compute_config_fingerprint, generate_run_id, iso_timestamp_now, RunMetadata, RunStore,
};
use crate::retry::RetryPolicy;
use crate::state::RunState;
use crate::types::{AggregatedMetrics, Case, Command, Event, RunStatus};
use crate::worker::{WorkItem, WorkerPool};
use anyhow::anyhow;
use std::collections::HashMap;
use tokio::sync::{broadcast, mpsc};
use tracing::info;

/// Result of a benchmark run.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub run_id: String,
    pub metrics: AggregatedMetrics,
    pub agent_metrics: HashMap<String, AggregatedMetrics>,
    pub aborted: bool,
}

/// The main benchmark runner.
pub struct Benchmark {
    config: Config,
    event_tx: broadcast::Sender<Event>,
    cmd_tx: mpsc::UnboundedSender<Command>,
    cmd_rx: Option<mpsc::UnboundedReceiver<Command>>,
}

impl Benchmark {
    /// Create a new benchmark from a config.
    pub fn new(config: Config) -> Self {
        let (event_tx, _) = broadcast::channel(1024);
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();

        Self {
            config,
            event_tx,
            cmd_tx,
            cmd_rx: Some(cmd_rx),
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.event_tx.subscribe()
    }

    pub fn send(&self, cmd: Command) {
        let _ = self.cmd_tx.send(cmd);
    }

    pub fn command_sender(&self) -> mpsc::UnboundedSender<Command> {
        self.cmd_tx.clone()
    }

    /// Run the benchmark.
    pub async fn run(mut self, run_id: Option<String>, limit: Option<usize>) -> Result<RunResult> {
        let mut cmd_rx = self
            .cmd_rx
            .take()
            .ok_or_else(|| PacabenchError::Internal(anyhow!("run() can only be called once")))?;

        // Phase 1: Initialize
        let run_id = run_id.unwrap_or_else(|| generate_run_id(&self.config.name));
        let run_dir = self.config.runs_dir.join(&run_id);
        std::fs::create_dir_all(&run_dir)?;
        let store = RunStore::new(&run_dir)?;

        // Phase 2: Load datasets
        let datasets = self.load_datasets(limit).await?;
        let agent_names: Vec<String> = self.config.agents.iter().map(|a| a.name.clone()).collect();
        let dataset_names: Vec<String> = self
            .config
            .datasets
            .iter()
            .map(|d| d.name.clone())
            .collect();

        // Build work items
        let mut work_items = Vec::new();
        for agent in &self.config.agents {
            for (_ds_name, cases) in &datasets {
                for case in cases {
                    work_items.push(WorkItem::new(
                        run_id.clone(),
                        agent.name.clone(),
                        case.clone(),
                    ));
                }
            }
        }

        // Initialize state
        let retry_policy = RetryPolicy::new(self.config.global.max_retries as u32, 100);
        let mut state = RunState::new(run_id.clone(), work_items, retry_policy.max_retries);

        // Check for resume
        if let Some(meta) = store.read_metadata()? {
            if !meta.status.is_terminal() {
                let existing = store.load_results()?;
                state.resume_from(existing);
                info!(
                    "Resuming run with {} completed cases",
                    state.completed_cases()
                );
            }
        }

        // Initialize metadata
        let config_fingerprint = compute_config_fingerprint(&self.config)?;
        let mut metadata = RunMetadata::new(
            run_id.clone(),
            config_fingerprint,
            agent_names.clone(),
            dataset_names.clone(),
            state.total_cases(),
        );
        metadata.completed_cases = state.completed_cases();
        metadata.start_time = Some(iso_timestamp_now());
        metadata.status = RunStatus::Running;
        store.write_metadata(&metadata)?;

        // Copy config file
        if let Some(src) = &self.config.config_path {
            if src.exists() {
                let dest = run_dir.join("pacabench.yaml");
                if !dest.exists() {
                    std::fs::copy(src, &dest).ok();
                }
            }
        }

        // Emit run started
        self.emit(Event::RunStarted {
            run_id: run_id.clone(),
            total_cases: state.total_cases(),
            resuming: state.completed_cases() > 0,
            completed_cases: state.completed_cases(),
            agents: agent_names.clone(),
            datasets: dataset_names.clone(),
        });

        // Handle empty run
        if state.is_done() {
            return self.finalize_run(state, metadata, store, false).await;
        }

        // Phase 3: Execute
        state.transition(RunStatus::Running);

        let concurrency = self.config.global.concurrency.max(1);
        let mut pool = WorkerPool::start(concurrency, &self.config, self.event_tx.clone()).await?;

        // Push initial work
        pool.push_batch(state.initial_work_items()).await;

        let mut aborted = false;
        let mut pending_count = state.pending_count();

        // Process results
        while pending_count > 0 && !aborted {
            tokio::select! {
                Some(cmd) = cmd_rx.recv() => {
                    match cmd {
                        Command::Stop { reason } => {
                            info!("Stop requested: {}", reason);
                            aborted = true;
                        }
                        Command::Abort { reason } => {
                            info!("Abort requested: {}", reason);
                            aborted = true;
                        }
                    }
                }

                Some(result) = pool.recv() => {
                    let key = result.item.key();

                    self.emit(Event::CaseCompleted {
                        run_id: run_id.clone(),
                        case_id: result.item.case_id.clone(),
                        agent: result.item.agent_name.clone(),
                        dataset: result.item.dataset_name.clone(),
                        passed: result.passed,
                        is_error: result.error_type.is_error(),
                        attempt: result.item.attempt,
                        duration_ms: result.duration_ms,
                        input_tokens: result.llm_metrics.input_tokens,
                        output_tokens: result.llm_metrics.output_tokens,
                    });

                    let needs_retry = state.mark_completed(key.clone(), result.to_case_result(iso_timestamp_now()));

                    if needs_retry {
                        let backoff = retry_policy.backoff_duration(result.item.attempt);
                        tokio::time::sleep(backoff).await;

                        if let Some(retry_item) = state.retry_item(&key) {
                            pool.push(retry_item).await;
                        }
                    } else {
                        pending_count = state.pending_count();

                        if let Some(case_result) = state.results().iter().find(|r| r.key() == key) {
                            store.append_result(case_result)?;
                        }

                        metadata.completed_cases = state.completed_cases();
                        if result.error_type.is_error() {
                            metadata.system_error_count += 1;
                        }
                        store.write_metadata(&metadata)?;
                    }
                }

                else => break,
            }
        }

        pool.shutdown().await;
        self.finalize_run(state, metadata, store, aborted).await
    }

    async fn load_datasets(&self, limit: Option<usize>) -> Result<Vec<(String, Vec<Case>)>> {
        let mut out = Vec::new();
        for ds in &self.config.datasets {
            let ctx = DatasetContext {
                root_dir: self.config.root_dir.clone(),
                cache_dir: self.config.cache_dir.clone(),
            };
            let loader = get_dataset_loader(ds.clone(), ctx)
                .map_err(|e| PacabenchError::dataset(ds.name.clone(), e))?;
            let cases = loader
                .load(limit)
                .await
                .map_err(|e| PacabenchError::dataset(ds.name.clone(), e))?;
            out.push((ds.name.clone(), cases));
        }
        Ok(out)
    }

    async fn finalize_run(
        &self,
        state: RunState,
        mut metadata: RunMetadata,
        store: RunStore,
        aborted: bool,
    ) -> Result<RunResult> {
        metadata.status = if aborted {
            RunStatus::Aborted
        } else {
            RunStatus::Completed
        };
        metadata.completed_time = Some(iso_timestamp_now());
        metadata.completed_cases = state.completed_cases();
        store.write_metadata(&metadata)?;

        let results = store.load_results()?;
        let metrics = aggregate_results(&results);
        let agent_metrics = aggregate_by_agent(&results);

        let passed = results.iter().filter(|r| r.passed).count() as u64;
        let failed = state.total_cases().saturating_sub(passed);

        self.emit(Event::RunCompleted {
            run_id: state.run_id.clone(),
            total_cases: state.total_cases(),
            passed_cases: passed,
            failed_cases: failed,
            aborted,
            metrics: metrics.clone(),
            agent_metrics: agent_metrics.clone(),
        });

        Ok(RunResult {
            run_id: state.run_id,
            metrics,
            agent_metrics,
            aborted,
        })
    }

    fn emit(&self, event: Event) {
        let _ = self.event_tx.send(event);
    }
}

fn aggregate_by_agent(results: &[crate::types::CaseResult]) -> HashMap<String, AggregatedMetrics> {
    let mut grouped: HashMap<String, Vec<crate::types::CaseResult>> = HashMap::new();
    for r in results {
        grouped
            .entry(r.agent_name.clone())
            .or_default()
            .push(r.clone());
    }
    grouped
        .into_iter()
        .map(|(k, v)| (k, aggregate_results(&v)))
        .collect()
}
