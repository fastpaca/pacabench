//! Run persistence: storing and loading benchmark results.

use crate::config::BenchmarkConfig;
use crate::types::{CaseKey, CaseResult, ErrorType, RunStatus};
use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

// ============================================================================
// PATH RESOLUTION (no defaults - all from config)
// ============================================================================

/// Resolve the runs directory from config or override.
pub fn resolve_runs_dir(
    config: &BenchmarkConfig,
    override_dir: Option<PathBuf>,
    config_path: Option<&Path>,
) -> PathBuf {
    let path = override_dir.unwrap_or_else(|| PathBuf::from(&config.output.directory));
    make_absolute(path, config_path)
}

fn make_absolute(path: PathBuf, base: Option<&Path>) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    if let Some(parent) = base.and_then(|p| p.parent()) {
        return parent.join(path);
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(path)
}

// ============================================================================
// RUN METADATA
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    pub run_id: String,
    pub status: RunStatus,
    pub config_fingerprint: String,
    pub agents: Vec<String>,
    pub datasets: Vec<String>,
    pub total_cases: u64,
    pub completed_cases: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_time: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_time: Option<String>,
    #[serde(default)]
    pub system_error_count: u64,
}

impl RunMetadata {
    pub fn new(
        run_id: String,
        config_fingerprint: String,
        agents: Vec<String>,
        datasets: Vec<String>,
        total_cases: u64,
    ) -> Self {
        Self {
            run_id,
            status: RunStatus::Pending,
            config_fingerprint,
            agents,
            datasets,
            total_cases,
            completed_cases: 0,
            start_time: None,
            completed_time: None,
            system_error_count: 0,
        }
    }

    pub fn progress(&self) -> f64 {
        if self.total_cases == 0 {
            0.0
        } else {
            self.completed_cases as f64 / self.total_cases as f64
        }
    }
}

// ============================================================================
// ERROR ENTRY
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEntry {
    pub timestamp: String,
    pub error_type: ErrorType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub case_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ============================================================================
// RUN SUMMARY (for listing runs)
// ============================================================================

#[derive(Debug, Clone)]
pub struct RunSummary {
    pub run_id: String,
    pub status: RunStatus,
    pub start_time: Option<String>,
    pub completed_time: Option<String>,
    pub completed_cases: u64,
    pub total_cases: u64,
    pub progress: f64,
    pub datasets: Vec<String>,
    pub agents: Vec<String>,
}

// ============================================================================
// RUN STORE
// ============================================================================

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
        let json = serde_json::to_string_pretty(metadata)?;
        fs::write(&self.metadata_path, json)?;
        Ok(())
    }

    pub fn read_metadata(&self) -> Result<Option<RunMetadata>> {
        if !self.metadata_path.exists() {
            return Ok(None);
        }
        let data = fs::read_to_string(&self.metadata_path)?;
        let meta: RunMetadata = serde_json::from_str(&data)?;
        Ok(Some(meta))
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

    pub fn load_results(&self) -> Result<Vec<CaseResult>> {
        if !self.results_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.results_path)?;
        let reader = BufReader::new(file);

        let mut dedup: HashMap<CaseKey, CaseResult> = HashMap::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let result: CaseResult = serde_json::from_str(&line)?;
            dedup.insert(result.key(), result);
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
            entries.push(serde_json::from_str(&line)?);
        }
        Ok(entries)
    }
}

// ============================================================================
// UTILITIES
// ============================================================================

pub fn iso_timestamp_now() -> String {
    Utc::now().to_rfc3339()
}

pub fn compute_config_fingerprint(config: &BenchmarkConfig) -> Result<String> {
    let json = serde_json::to_string(config)?;
    let hash = Sha256::digest(json.as_bytes());
    Ok(format!("{:x}", hash))
}

pub fn generate_run_id(config_name: &str) -> String {
    let ts = iso_timestamp_now().replace([':', '-'], "");
    format!("{config_name}-{ts}")
}

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
            let progress = meta.progress();
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
                datasets: meta.datasets,
                agents: meta.agents,
            });
        }
    }

    summaries.sort_by(|a, b| {
        let a_ts = parse_iso(&a.start_time);
        let b_ts = parse_iso(&b.start_time);
        b_ts.cmp(&a_ts)
    });
    Ok(summaries)
}

fn parse_iso(ts: &Option<String>) -> Option<chrono::DateTime<chrono::Utc>> {
    ts.as_ref()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc))
}
