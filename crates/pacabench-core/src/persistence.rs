//! Run persistence: storing and loading benchmark results.

use crate::config::Config;
use crate::error::{PacabenchError, Result};
use crate::stats::RunStats;
use crate::types::{CaseKey, CaseResult, ErrorType, RunStatus};
use anyhow::anyhow;
use chrono::Utc;
use nanoid::nanoid;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    pub run_id: String,
    pub status: RunStatus,
    pub config_fingerprint: String,
    pub agents: Vec<String>,
    pub datasets: Vec<String>,
    pub total_cases: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_cases: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_total_cases: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_of: Option<String>,
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
            active_cases: Some(total_cases),
            original_total_cases: Some(total_cases),
            retry_of: None,
            completed_cases: 0,
            start_time: None,
            completed_time: None,
            system_error_count: 0,
        }
    }

    pub fn progress(&self) -> f64 {
        let denom = self.active_cases.unwrap_or(self.total_cases);
        if denom == 0 {
            0.0
        } else {
            self.completed_cases as f64 / denom as f64
        }
    }
}

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
        fs::create_dir_all(&run_dir).map_err(PacabenchError::Persistence)?;
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
        fs::write(&self.metadata_path, json).map_err(PacabenchError::Persistence)?;
        Ok(())
    }

    pub fn read_metadata(&self) -> Result<Option<RunMetadata>> {
        if !self.metadata_path.exists() {
            return Ok(None);
        }
        let data = fs::read_to_string(&self.metadata_path).map_err(PacabenchError::Persistence)?;
        let meta: RunMetadata = serde_json::from_str(&data)?;
        Ok(Some(meta))
    }

    pub fn append_result(&self, result: &CaseResult) -> Result<()> {
        let mut file = File::options()
            .create(true)
            .append(true)
            .open(&self.results_path)
            .map_err(PacabenchError::Persistence)?;
        let line = serde_json::to_string(result)?;
        writeln!(file, "{line}").map_err(PacabenchError::Persistence)?;
        Ok(())
    }

    pub fn append_error(&self, entry: &ErrorEntry) -> Result<()> {
        let mut file = File::options()
            .create(true)
            .append(true)
            .open(&self.errors_path)
            .map_err(PacabenchError::Persistence)?;
        let line = serde_json::to_string(entry)?;
        writeln!(file, "{line}").map_err(PacabenchError::Persistence)?;
        Ok(())
    }

    pub fn load_results(&self) -> Result<Vec<CaseResult>> {
        if !self.results_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.results_path).map_err(PacabenchError::Persistence)?;
        let reader = BufReader::new(file);

        let mut dedup: HashMap<CaseKey, CaseResult> = HashMap::new();
        for line in reader.lines() {
            let line = line.map_err(PacabenchError::Persistence)?;
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
        let file = File::open(&self.errors_path).map_err(PacabenchError::Persistence)?;
        let reader = BufReader::new(file);

        let mut entries = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(PacabenchError::Persistence)?;
            if line.trim().is_empty() {
                continue;
            }
            entries.push(serde_json::from_str(&line)?);
        }
        Ok(entries)
    }

    /// Load complete run statistics from persisted files.
    ///
    /// This is the single source of truth for all run metrics.
    /// All display code should use this rather than loading individual
    /// files and recomputing.
    pub fn load_stats(&self) -> Result<RunStats> {
        let metadata = self.read_metadata()?;
        let results = self.load_results()?;
        let errors = self.load_errors()?;

        match metadata {
            Some(meta) => Ok(RunStats::compute(&meta, &results, &errors)),
            None => Err(PacabenchError::Internal(anyhow!(
                "no metadata found for run"
            ))),
        }
    }
}

pub fn iso_timestamp_now() -> String {
    Utc::now().to_rfc3339()
}

pub fn compute_config_fingerprint(config: &Config) -> Result<String> {
    let json = serde_json::to_string(config)?;
    let hash = Sha256::digest(json.as_bytes());
    Ok(format!("{:x}", hash))
}

pub fn generate_run_id(config_name: &str) -> String {
    let id = nanoid!(8);
    format!("{config_name}-{id}")
}

pub fn list_run_summaries(runs_dir: &Path) -> Result<Vec<RunSummary>> {
    if !runs_dir.exists() {
        return Ok(Vec::new());
    }

    let mut summaries = Vec::new();
    for entry in fs::read_dir(runs_dir).map_err(PacabenchError::Persistence)? {
        let path = entry.map_err(PacabenchError::Persistence)?.path();
        if !path.is_dir() {
            continue;
        }
        let store = RunStore::new(&path)?;
        if let Some(meta) = store.read_metadata()? {
            let progress = meta.progress();
            let display_total = meta.active_cases.unwrap_or(meta.total_cases);
            summaries.push(RunSummary {
                run_id: path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default(),
                status: meta.status,
                start_time: meta.start_time,
                completed_time: meta.completed_time,
                completed_cases: meta.completed_cases,
                total_cases: display_total,
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
