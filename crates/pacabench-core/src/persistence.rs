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
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap(),
                        canonical_json_string(v)
                    )
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
