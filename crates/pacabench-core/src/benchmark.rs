//! Benchmark - the main entry point for running benchmarks.

use crate::config::{AgentConfig, Config};
use crate::datasets::{get_dataset_loader, DatasetContext, DatasetLoader};
use crate::error::{PacabenchError, Result};
use crate::persistence::{
    compute_config_fingerprint, generate_run_id, iso_timestamp_now, ErrorEntry, RunMetadata,
    RunStore,
};
use crate::retry::RetryPolicy;
use crate::state::RunState;
use crate::types::{CaseKey, Command, Event, RunStatus};
use crate::worker::{WorkItem, WorkResult, WorkerPool};
use anyhow::anyhow;
use futures_util::StreamExt;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio_util::time::DelayQueue;
use tracing::info;

/// Result of a benchmark run.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Complete run statistics - the single source of truth.
    pub stats: crate::stats::RunStats,
    /// Whether the run was aborted before completion.
    pub aborted: bool,
}

/// The main benchmark runner.
pub struct Benchmark {
    config: Config,
    event_tx: broadcast::Sender<Event>,
    cmd_tx: mpsc::UnboundedSender<Command>,
    cmd_rx: Option<mpsc::UnboundedReceiver<Command>>,
}

/// Everything needed to execute a benchmark run, prepared from config.
struct PreparedRun {
    run_id: String,
    store: RunStore,
    state: RunState,
    metadata: RunMetadata,
    passed: HashSet<CaseKey>,
    attempt_counts: HashMap<CaseKey, u32>,
    retry_policy: RetryPolicy,
    /// Cases that were in the original run (for retry scenarios).
    /// If Some, only these cases should be processed. If None, process all cases.
    eligible_cases: Option<HashSet<CaseKey>>,
    /// True if this is a retry of a completed run (vs resume of interrupted run).
    is_retry: bool,
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

    /// Subscribe to events emitted during the benchmark run.
    ///
    /// Returns a broadcast receiver that will receive all [`Event`] variants
    /// as the benchmark progresses.
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.event_tx.subscribe()
    }

    /// Send a command to control the running benchmark.
    ///
    /// Use [`Command::Stop`] for graceful shutdown or [`Command::Abort`] for
    /// immediate termination.
    pub fn send(&self, cmd: Command) {
        let _ = self.cmd_tx.send(cmd);
    }

    /// Get a cloneable sender for commands.
    ///
    /// Useful for passing to signal handlers or other async tasks that need
    /// to control the benchmark.
    pub fn command_sender(&self) -> mpsc::UnboundedSender<Command> {
        self.cmd_tx.clone()
    }

    /// Run the benchmark.
    pub async fn run(mut self, run_id: Option<String>, limit: Option<usize>) -> Result<RunResult> {
        let cmd_rx = self
            .cmd_rx
            .take()
            .ok_or_else(|| PacabenchError::Internal(anyhow!("run() can only be called once")))?;

        // Phase 1: Prepare
        let (prepared, dataset_loaders) = self.prepare_run(run_id, limit).await?;
        self.emit_run_started(&prepared);

        if prepared.state.total_cases() == 0 {
            return self
                .finalize_run(prepared.state, prepared.metadata, prepared.store, false)
                .await;
        }

        // Phase 2: Start workers and producer
        let concurrency = self.config.global.concurrency.max(1);
        let queue_capacity = concurrency.saturating_mul(4).max(1);

        let pool = WorkerPool::start(
            concurrency,
            queue_capacity,
            &self.config,
            self.event_tx.clone(),
        )
        .await?;

        let (work_tx, work_rx) =
            mpsc::channel::<WorkItem>(queue_capacity * self.config.agents.len().max(1));

        let producer = spawn_case_producer(
            dataset_loaders,
            self.config.agents.clone(),
            prepared.run_id.clone(),
            Arc::new(prepared.passed.clone()),
            Arc::new(prepared.attempt_counts.clone()),
            prepared.eligible_cases.clone().map(Arc::new),
            work_tx,
            limit,
        );

        // Phase 3: Event loop (consumes prepared, returns final state)
        let (state, metadata, store, aborted) = self
            .run_event_loop(cmd_rx, pool, work_rx, prepared, producer)
            .await?;

        // Phase 4: Finalize
        self.finalize_run(state, metadata, store, aborted).await
    }

    /// Prepare all resources needed for a benchmark run.
    ///
    /// Returns the prepared run state and the dataset loaders (separate because
    /// loaders are only needed for producer setup, not during the event loop).
    async fn prepare_run(
        &self,
        run_id: Option<String>,
        limit: Option<usize>,
    ) -> Result<(PreparedRun, Vec<(String, Box<dyn DatasetLoader>)>)> {
        // Generate run ID and create storage
        let run_id = run_id.unwrap_or_else(|| generate_run_id(&self.config.name));
        let run_dir = self.config.runs_dir.join(&run_id);
        std::fs::create_dir_all(&run_dir).map_err(PacabenchError::Persistence)?;
        let store = RunStore::new(&run_dir)?;

        let agent_names: Vec<String> = self.config.agents.iter().map(|a| a.name.clone()).collect();
        let dataset_names: Vec<String> = self
            .config
            .datasets
            .iter()
            .map(|d| d.name.clone())
            .collect();

        // Build dataset loaders
        let dataset_loaders: Vec<(String, Box<dyn DatasetLoader>)> = self
            .config
            .datasets
            .iter()
            .map(|ds| {
                let ctx = DatasetContext {
                    root_dir: self.config.root_dir.clone(),
                    cache_dir: self.config.cache_dir.clone(),
                };
                get_dataset_loader(ds.clone(), ctx)
                    .map(|loader| (ds.name.clone(), loader))
                    .map_err(|e| PacabenchError::dataset(ds.name.clone(), e))
            })
            .collect::<Result<Vec<_>>>()?;

        let retry_policy = RetryPolicy::new(self.config.global.max_retries as u32, 100);

        // Load prior results if they exist.
        // For non-terminal runs (interrupted), we resume.
        // For terminal runs (completed/aborted), we retry failed cases only.
        let metadata_opt = store.read_metadata()?;
        let (existing_results, eligible_cases, is_retry) = if let Some(ref meta) = metadata_opt {
            let results = store.load_results()?;
            let completed = results.iter().filter(|r| r.passed).count();

            if meta.status.is_terminal() {
                // Retry scenario: only process cases that were in the original run
                let all_case_keys: HashSet<CaseKey> = results.iter().map(|r| r.key()).collect();
                info!(
                    "Retrying run with {} prior cases ({} passed)",
                    results.len(),
                    completed
                );
                (results, Some(all_case_keys), true)
            } else {
                // Resume scenario: continue with all cases (no restriction)
                info!("Resuming run with {} completed cases", completed);
                (results, None, false)
            }
        } else {
            // Fresh run
            (Vec::new(), None, false)
        };

        // Build attempt counts and passed set from existing results
        let attempt_counts: HashMap<CaseKey, u32> = existing_results
            .iter()
            .map(|r| (r.key(), r.attempt))
            .collect();
        let passed: HashSet<CaseKey> = existing_results
            .iter()
            .filter(|r| r.passed)
            .map(|r| r.key())
            .collect();

        // Calculate total cases and agent totals
        // For retry (eligible_cases is Some), use the eligible cases count
        // For fresh run or resume, count from dataset
        let (total_cases, agent_totals) = if let Some(ref eligible) = eligible_cases {
            // Retry scenario: count eligible cases (failed cases from original run)
            // Group by agent to get per-agent totals
            // Initialize all agents to 0 so progress display doesn't use incorrect fallbacks
            let mut per_agent: HashMap<String, u64> = agent_names
                .iter()
                .map(|name| (name.clone(), 0u64))
                .collect();
            for key in eligible.iter() {
                // Only count cases that haven't passed yet
                if !passed.contains(key) {
                    *per_agent.entry(key.agent.clone()).or_default() += 1;
                }
            }
            let total = per_agent.values().sum();
            (total, per_agent)
        } else {
            // Fresh run or resume: count from dataset
            let mut dataset_case_counts: HashMap<String, u64> = HashMap::new();
            for (name, loader) in dataset_loaders.iter() {
                let count = loader
                    .count_cases(limit)
                    .await
                    .map_err(|e| PacabenchError::dataset(name.clone(), e))?;
                dataset_case_counts.insert(name.clone(), count as u64);
            }

            let total_per_agent: u64 = dataset_case_counts.values().copied().sum();
            let total_cases = total_per_agent * agent_names.len() as u64;

            let agent_totals: HashMap<String, u64> = agent_names
                .iter()
                .map(|agent| (agent.clone(), total_per_agent))
                .collect();

            (total_cases, agent_totals)
        };

        // Initialize state
        let state = RunState::new(
            run_id.clone(),
            retry_policy.max_retries,
            total_cases,
            agent_totals,
            existing_results,
        );

        // Initialize metadata
        let config_fingerprint = compute_config_fingerprint(&self.config)?;
        let mut metadata = RunMetadata::new(
            run_id.clone(),
            config_fingerprint,
            agent_names,
            dataset_names,
            state.total_cases(),
        );

        // Preserve original totals and retry lineage for reporting
        if let Some(prev_meta) = metadata_opt {
            metadata.original_total_cases = Some(
                prev_meta
                    .original_total_cases
                    .unwrap_or(prev_meta.total_cases),
            );
            if is_retry {
                metadata.retry_of = Some(prev_meta.run_id);
                // planned cases for reporting should reflect the original run,
                // while active_cases captures what this retry will execute.
                metadata.total_cases = metadata
                    .original_total_cases
                    .unwrap_or(metadata.total_cases);
                metadata.active_cases = Some(state.total_cases());
            }
        }
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

        let prepared = PreparedRun {
            run_id,
            store,
            state,
            metadata,
            passed,
            attempt_counts,
            retry_policy,
            eligible_cases,
            is_retry,
        };

        Ok((prepared, dataset_loaders))
    }

    /// Emit the RunStarted event.
    fn emit_run_started(&self, prepared: &PreparedRun) {
        // For retry, completed_cases represents passed cases from original run (not relevant to show)
        // For resume, completed_cases represents progress so far
        let resuming = !prepared.is_retry && prepared.state.completed_cases() > 0;

        self.emit(Event::RunStarted {
            run_id: prepared.run_id.clone(),
            total_cases: prepared.state.total_cases(),
            resuming,
            retrying: prepared.is_retry,
            completed_cases: if prepared.is_retry {
                0 // Don't show "already done" for retry - it's misleading
            } else {
                prepared.state.completed_cases()
            },
            agents: prepared.metadata.agents.clone(),
            datasets: prepared.metadata.datasets.clone(),
            agent_totals: prepared.state.agent_totals(),
            agent_completed: if prepared.is_retry {
                HashMap::new() // Don't set initial progress for retry
            } else {
                prepared.state.agent_completed()
            },
        });
    }

    /// Main event loop: process commands, results, retries, and new work items.
    ///
    /// Consumes all inputs and returns the final state for finalization.
    async fn run_event_loop(
        &self,
        mut cmd_rx: mpsc::UnboundedReceiver<Command>,
        mut pool: WorkerPool,
        mut work_rx: mpsc::Receiver<WorkItem>,
        mut prepared: PreparedRun,
        producer: JoinHandle<Result<()>>,
    ) -> Result<(RunState, RunMetadata, RunStore, bool)> {
        prepared.state.transition(RunStatus::Running);

        let mut retry_queue: DelayQueue<WorkItem> = DelayQueue::new();
        let mut aborted = false;
        let mut production_done = false;
        let mut pending_count = prepared.state.pending_count();

        while (!production_done || pending_count > 0 || !retry_queue.is_empty()) && !aborted {
            tokio::select! {
                // Handle commands (stop/abort)
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

                // Handle completed work results
                Some(result) = pool.recv(), if pending_count > 0 => {
                    self.handle_result(result, &mut prepared, &mut retry_queue).await?;
                    pending_count = prepared.state.pending_count();
                }

                // Handle retries that are ready
                Some(expired) = retry_queue.next() => {
                    let item = expired.into_inner();
                    pool.push(item).await;
                }

                // Handle new work items from producer
                recv = work_rx.recv(), if !production_done => {
                    match recv {
                        Some(item) => {
                            let attempt = item.attempt;
                            let key = item.key();
                            prepared.state.register_case(key, attempt);
                            pending_count = prepared.state.pending_count();
                            pool.push(item).await;
                        }
                        None => {
                            production_done = true;
                        }
                    }
                }
            }
        }

        // Cleanup
        drop(work_rx);
        pool.shutdown().await;

        match producer.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(e) => {
                return Err(PacabenchError::Internal(anyhow!(
                    "case production task failed: {e}"
                )))
            }
        }

        Ok((prepared.state, prepared.metadata, prepared.store, aborted))
    }

    /// Handle a single completed work result.
    async fn handle_result(
        &self,
        result: WorkResult,
        prepared: &mut PreparedRun,
        retry_queue: &mut DelayQueue<WorkItem>,
    ) -> Result<()> {
        self.emit(Event::CaseCompleted {
            run_id: prepared.run_id.clone(),
            case_id: result.item.case_id.clone(),
            agent: result.item.agent_name.clone(),
            dataset: result.item.dataset_name.clone(),
            passed: result.passed,
            is_error: result.error_type.is_error(),
            attempt: result.item.attempt,
            duration_ms: result.duration_ms,
            input_tokens: result.llm_metrics.input_tokens,
            output_tokens: result.llm_metrics.output_tokens,
            cached_tokens: result.llm_metrics.cached_tokens,
            model: result.llm_metrics.model.clone(),
            judge_input_tokens: result
                .evaluation
                .as_ref()
                .and_then(|e| e.judge_metrics.as_ref())
                .map(|m| m.input_tokens)
                .unwrap_or(0),
            judge_output_tokens: result
                .evaluation
                .as_ref()
                .and_then(|e| e.judge_metrics.as_ref())
                .map(|m| m.output_tokens)
                .unwrap_or(0),
            judge_cached_tokens: result
                .evaluation
                .as_ref()
                .and_then(|e| e.judge_metrics.as_ref())
                .map(|m| m.cached_tokens)
                .unwrap_or(0),
            judge_model: result
                .evaluation
                .as_ref()
                .and_then(|e| e.judge_metrics.as_ref())
                .and_then(|m| m.model.clone()),
            output: result.output.as_ref().map(|o| o.to_string()),
            error: result.error.clone(),
            judge_reason: result.evaluation.as_ref().and_then(|e| e.reason.clone()),
        });

        let case_result = result.to_case_result(iso_timestamp_now());
        let needs_retry = prepared.state.mark_completed(case_result.clone());

        // Persist system/fatal errors for reporting, even if we will retry.
        if result.error_type.is_error() {
            let error_entry = ErrorEntry {
                timestamp: iso_timestamp_now(),
                error_type: result.error_type,
                agent_name: Some(result.item.agent_name.clone()),
                dataset_name: Some(result.item.dataset_name.clone()),
                case_id: Some(result.item.case_id.clone()),
                error: result.error.clone(),
            };
            prepared.store.append_error(&error_entry)?;
            prepared.metadata.system_error_count += 1;
        }

        if needs_retry {
            let backoff = prepared.retry_policy.backoff_duration(result.item.attempt);
            let retry_item = result.item.retry();
            // Persist updated metadata (e.g., system error counts) even when retrying
            prepared.store.write_metadata(&prepared.metadata)?;
            retry_queue.insert(retry_item, backoff);
        } else {
            // Persist final result (includes error info in CaseResult.error_type/error)
            prepared.store.append_result(&case_result)?;

            prepared.metadata.completed_cases = prepared.state.completed_cases();
            prepared.store.write_metadata(&prepared.metadata)?;
        }

        Ok(())
    }

    async fn finalize_run(
        &self,
        _state: RunState,
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
        metadata.completed_cases = metadata.active_cases.unwrap_or(metadata.total_cases);
        store.write_metadata(&metadata)?;

        // Load complete stats - single source of truth
        let stats = store.load_stats()?;

        self.emit(Event::RunCompleted {
            aborted,
            stats: Box::new(stats.clone()),
        });

        Ok(RunResult { stats, aborted })
    }

    fn emit(&self, event: Event) {
        let _ = self.event_tx.send(event);
    }
}

/// Spawn a task that streams cases from datasets and feeds work items to the pool.
///
/// If `eligible_cases` is Some, only cases in that set will be processed (retry scenario).
/// If None, all cases from the datasets will be processed (fresh run or resume).
#[allow(clippy::too_many_arguments)]
fn spawn_case_producer(
    dataset_loaders: Vec<(String, Box<dyn DatasetLoader>)>,
    agents: Vec<AgentConfig>,
    run_id: String,
    passed: Arc<HashSet<CaseKey>>,
    attempts: Arc<HashMap<CaseKey, u32>>,
    eligible_cases: Option<Arc<HashSet<CaseKey>>>,
    work_tx: mpsc::Sender<WorkItem>,
    limit: Option<usize>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        for (dataset_name, loader) in dataset_loaders {
            let mut stream = loader
                .stream_cases(limit)
                .await
                .map_err(|e| PacabenchError::dataset(dataset_name.clone(), e))?;

            while let Some(case) = stream.next().await {
                let case = case.map_err(|e| PacabenchError::dataset(dataset_name.clone(), e))?;

                for agent in &agents {
                    let key = CaseKey::new(&agent.name, &case.dataset_name, &case.case_id);

                    // Skip if case was not in the original run (retry scenario)
                    if let Some(ref eligible) = eligible_cases {
                        if !eligible.contains(&key) {
                            continue;
                        }
                    }

                    // Skip already passed cases
                    if passed.contains(&key) {
                        continue;
                    }

                    let attempt = attempts.get(&key).copied().unwrap_or(0) + 1;
                    let item = WorkItem {
                        run_id: run_id.clone(),
                        agent_name: agent.name.clone(),
                        dataset_name: case.dataset_name.clone(),
                        case_id: case.case_id.clone(),
                        case: case.clone(),
                        attempt,
                    };

                    if work_tx.send(item).await.is_err() {
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    })
}
