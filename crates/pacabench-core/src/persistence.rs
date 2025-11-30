//! Run directory resolution and basic persistence helpers.

use crate::config::BenchmarkConfig;
use crate::types::{CaseResult, ErrorType};
use anyhow::Result;
use chrono::Utc;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const RUNS_ENV_VAR: &str = "PACABENCH_RUNS_DIR";

static DEFAULT_RUNS_DIR: Lazy<PathBuf> = Lazy::new(|| {
    if let Ok(val) = env::var(RUNS_ENV_VAR) {
        return PathBuf::from(val);
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("runs")
});

pub fn default_runs_dir() -> PathBuf {
    DEFAULT_RUNS_DIR.clone()
}

pub fn default_dataset_cache_dir() -> PathBuf {
    let home = env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    home.join(".cache").join("pacabench").join("datasets")
}

/// Resolve the runs directory, applying override, config output dir, and config path for relative paths.
pub fn resolve_runs_dir(
    config: Option<&BenchmarkConfig>,
    override_dir: Option<PathBuf>,
    config_path: Option<&Path>,
) -> PathBuf {
    if let Some(ov) = override_dir {
        return absolute_from(ov, config_path);
    }

    if let Some(cfg) = config {
        return absolute_from(PathBuf::from(&cfg.output.directory), config_path);
    }

    default_runs_dir()
}

fn absolute_from(path: PathBuf, base: Option<&Path>) -> PathBuf {
    if path.is_absolute() {
        path
    } else if let Some(b) = base.and_then(|p| p.parent()) {
        b.join(path)
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    pub run_id: String,
    pub status: String,
    pub config_fingerprint: String,
    #[serde(default)]
    pub total_cases: u64,
    #[serde(default)]
    pub completed_cases: u64,
    #[serde(default)]
    pub start_time: Option<String>,
    #[serde(default)]
    pub completed_time: Option<String>,
    #[serde(default)]
    pub system_error_count: u64,
    #[serde(default)]
    pub extras: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEntry {
    pub timestamp: String,
    #[serde(default)]
    pub error_type: ErrorType,
    #[serde(default)]
    pub agent_name: Option<String>,
    #[serde(default)]
    pub dataset_name: Option<String>,
    #[serde(default)]
    pub case_id: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RunSummary {
    pub run_id: String,
    pub status: String,
    pub start_time: Option<String>,
    pub completed_time: Option<String>,
    pub completed_cases: u64,
    pub total_cases: u64,
    pub progress: Option<f64>,
    pub datasets: Vec<String>,
    pub agents: Vec<String>,
    pub total_cost_usd: Option<f64>,
}

#[derive(Debug)]
pub struct RunStore {
    run_dir: PathBuf,
    results_path: PathBuf,
    errors_path: PathBuf,
    metadata_path: PathBuf,
}

impl RunStore {
    pub fn new(run_dir: impl AsRef<Path>) -> Result<Self> {
        let run_dir = run_dir.as_ref().to_path_buf();
        fs::create_dir_all(&run_dir)?;
        Ok(Self {
            results_path: run_dir.join("results.jsonl"),
            errors_path: run_dir.join("system_errors.jsonl"),
            metadata_path: run_dir.join("metadata.json"),
            run_dir,
        })
    }

    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    pub fn write_metadata(&self, metadata: &RunMetadata) -> Result<()> {
        let mut file = File::create(&self.metadata_path)?;
        let json = serde_json::to_string_pretty(metadata)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn append_result(&self, result: &CaseResult) -> Result<()> {
        let mut file = File::options()
            .create(true)
            .append(true)
            .open(&self.results_path)?;
        let line = serde_json::to_string(result)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    pub fn append_error(&self, entry: &ErrorEntry) -> Result<()> {
        let mut file = File::options()
            .create(true)
            .append(true)
            .open(&self.errors_path)?;
        let line = serde_json::to_string(entry)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    pub fn read_metadata(&self) -> Result<Option<RunMetadata>> {
        if !self.metadata_path.exists() {
            return Ok(None);
        }
        let data = std::fs::read_to_string(&self.metadata_path)?;
        let meta: RunMetadata = serde_json::from_str(&data)?;
        Ok(Some(meta))
    }

    /// Load results, keeping only the most recent entry per (agent, dataset, case_id).
    pub fn load_results(&self) -> Result<Vec<CaseResult>> {
        if !self.results_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.results_path)?;
        let reader = BufReader::new(file);

        let mut dedup: HashMap<(String, String, String), CaseResult> = HashMap::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let value: CaseResult = serde_json::from_str(&line)?;
            let key = (
                value.agent_name.clone(),
                value.dataset_name.clone(),
                value.case_id.clone(),
            );
            dedup.insert(key, value);
        }

        Ok(dedup.into_values().collect())
    }

    pub fn load_errors(&self) -> Result<Vec<ErrorEntry>> {
        if !self.errors_path.exists() {
            return Ok(Vec::new());
        }
        let file = File::open(&self.errors_path)?;
        let reader = BufReader::new(file);

        let mut entries = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let value: ErrorEntry = serde_json::from_str(&line)?;
            entries.push(value);
        }
        Ok(entries)
    }
}

pub fn iso_timestamp_now() -> String {
    Utc::now().to_rfc3339()
}

pub fn compute_config_fingerprint(config: &BenchmarkConfig) -> Result<String> {
    let value = serde_json::to_value(config)?;
    let canonical = canonical_json_string(&value);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn canonical_json_string(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap_or_default(),
        Value::Array(arr) => {
            let inner: Vec<String> = arr.iter().map(canonical_json_string).collect();
            format!("[{}]", inner.join(","))
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let parts: Vec<String> = entries
                .iter()
                .map(|(k, v)| {
                    // String keys always serialize successfully to JSON
                    let key_json = serde_json::to_string(k).unwrap_or_else(|_| format!("\"{}\"", k));
                    format!("{}:{}", key_json, canonical_json_string(v))
                })
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

pub fn generate_run_id(config_name: &str) -> String {
    let ts = iso_timestamp_now().replace([':', '-'], "");
    format!("{config_name}-{ts}")
}

/// Return run summaries from a runs directory, sorted by start time descending.
pub fn list_run_summaries(runs_dir: &Path) -> Result<Vec<RunSummary>> {
    if !runs_dir.exists() {
        return Ok(Vec::new());
    }

    let mut summaries = Vec::new();
    for entry in fs::read_dir(runs_dir)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let store = RunStore::new(&path)?;
        if let Some(meta) = store.read_metadata()? {
            let progress = if meta.total_cases > 0 {
                Some(meta.completed_cases as f64 / meta.total_cases as f64)
            } else {
                None
            };
            let datasets = meta
                .extras
                .get("datasets")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let agents = meta
                .extras
                .get("agents")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            let total_cost_usd = path
                .join("status.json")
                .exists()
                .then(|| fs::read_to_string(path.join("status.json")).ok())
                .flatten()
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                .and_then(|v| v.get("total_cost").and_then(|c| c.as_f64()));

            summaries.push(RunSummary {
                run_id: path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default(),
                status: meta.status,
                start_time: meta.start_time,
                completed_time: meta.completed_time,
                completed_cases: meta.completed_cases,
                total_cases: meta.total_cases,
                progress,
                datasets,
                agents,
                total_cost_usd,
            });
        }
    }

    summaries.sort_by(|a, b| {
        let a_ts = parse_iso(&a.start_time).unwrap_or_else(chrono::Utc::now);
        let b_ts = parse_iso(&b.start_time).unwrap_or_else(chrono::Utc::now);
        b_ts.cmp(&a_ts)
    });
    Ok(summaries)
}

fn parse_iso(ts: &Option<String>) -> Option<chrono::DateTime<chrono::Utc>> {
    ts.as_ref()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc))
}

// ============================================================================
// Channel-based async persistence writer
// ============================================================================

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::warn;

/// Commands sent to the persistence writer task.
#[derive(Debug)]
pub enum PersistenceCommand {
    /// Append a case result to results.jsonl.
    AppendResult(CaseResult),
    /// Append an error to system_errors.jsonl.
    AppendError(ErrorEntry),
    /// Update the run metadata.
    UpdateMetadata(RunMetadata),
    /// Flush any buffered writes.
    Flush,
    /// Shutdown the writer task.
    Shutdown,
}

/// Handle for sending persistence commands.
///
/// This handle is cheap to clone and can be shared across tasks.
#[derive(Clone)]
pub struct PersistenceHandle {
    tx: mpsc::Sender<PersistenceCommand>,
}

impl PersistenceHandle {
    /// Spawn a background persistence writer task.
    ///
    /// Returns a handle for sending commands and a join handle for the task.
    pub fn spawn(store: RunStore) -> (Self, JoinHandle<()>) {
        let (tx, rx) = mpsc::channel(256);
        let handle = tokio::spawn(persistence_writer_task(store, rx));
        (Self { tx }, handle)
    }

    /// Write a case result (non-blocking).
    pub async fn write_result(&self, result: CaseResult) -> Result<()> {
        self.tx
            .send(PersistenceCommand::AppendResult(result))
            .await
            .map_err(|_| anyhow::anyhow!("persistence channel closed"))?;
        Ok(())
    }

    /// Write an error entry (non-blocking).
    pub async fn write_error(&self, error: ErrorEntry) -> Result<()> {
        self.tx
            .send(PersistenceCommand::AppendError(error))
            .await
            .map_err(|_| anyhow::anyhow!("persistence channel closed"))?;
        Ok(())
    }

    /// Update run metadata (non-blocking).
    pub async fn update_metadata(&self, metadata: RunMetadata) -> Result<()> {
        self.tx
            .send(PersistenceCommand::UpdateMetadata(metadata))
            .await
            .map_err(|_| anyhow::anyhow!("persistence channel closed"))?;
        Ok(())
    }

    /// Request a flush of buffered writes.
    pub async fn flush(&self) -> Result<()> {
        self.tx
            .send(PersistenceCommand::Flush)
            .await
            .map_err(|_| anyhow::anyhow!("persistence channel closed"))?;
        Ok(())
    }

    /// Shutdown the writer task gracefully.
    pub async fn shutdown(&self) -> Result<()> {
        self.tx
            .send(PersistenceCommand::Shutdown)
            .await
            .map_err(|_| anyhow::anyhow!("persistence channel closed"))?;
        Ok(())
    }

    /// Try to write a result without blocking (returns immediately if channel full).
    pub fn try_write_result(&self, result: CaseResult) -> bool {
        self.tx
            .try_send(PersistenceCommand::AppendResult(result))
            .is_ok()
    }
}

/// Background task that processes persistence commands.
async fn persistence_writer_task(store: RunStore, mut rx: mpsc::Receiver<PersistenceCommand>) {
    while let Some(cmd) = rx.recv().await {
        match cmd {
            PersistenceCommand::AppendResult(result) => {
                if let Err(e) = store.append_result(&result) {
                    warn!("failed to write result: {e}");
                }
            }
            PersistenceCommand::AppendError(error) => {
                if let Err(e) = store.append_error(&error) {
                    warn!("failed to write error: {e}");
                }
            }
            PersistenceCommand::UpdateMetadata(metadata) => {
                if let Err(e) = store.write_metadata(&metadata) {
                    warn!("failed to write metadata: {e}");
                }
            }
            PersistenceCommand::Flush => {
                // Currently all writes are synchronous, so nothing to flush
            }
            PersistenceCommand::Shutdown => {
                break;
            }
        }
    }
}
