//! Run management: metadata, resume tracking, attempts.

use crate::config::BenchmarkConfig;
use crate::persistence::{compute_config_fingerprint, iso_timestamp_now, RunMetadata, RunStore};
use crate::types::{CaseResult, ErrorType};
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct RunManager {
    store: RunStore,
    pub run_id: String,
    pub resuming: bool,
    config_fingerprint: String,
    completed_entries: HashSet<(String, String, String)>,
    passed_entries: HashSet<(String, String, String)>,
    case_attempts: HashMap<(String, String, String), u32>,
    pub total_cases: u64,
    pub system_error_count: u64,
}

impl RunManager {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &BenchmarkConfig,
        runs_dir: PathBuf,
        run_id: Option<String>,
        force_new: bool,
    ) -> Result<Self> {
        let fingerprint = compute_config_fingerprint(config)?;
        let run_id = run_id.unwrap_or_else(|| super::persistence::generate_run_id(&config.name));
        let run_dir = resolve_run_dir(&runs_dir, &run_id);
        std::fs::create_dir_all(&run_dir)?;
        let store = RunStore::new(&run_dir)?;

        let mut resuming = false;
        let mut total_cases = 0;
        let mut system_error_count = 0;
        let mut completed_entries = HashSet::new();
        let mut passed_entries = HashSet::new();
        let mut case_attempts = HashMap::new();

        if let Some(meta) = store.read_metadata()? {
            if meta.config_fingerprint != fingerprint && !force_new {
                return Err(anyhow!(
                    "run directory {} exists with different config fingerprint",
                    run_dir.display()
                ));
            }
            resuming = true;
            total_cases = meta.total_cases;
            system_error_count = meta.system_error_count;
            let results = store.load_results()?;
            for r in results {
                let key = (
                    r.agent_name.clone(),
                    r.dataset_name.clone(),
                    r.case_id.clone(),
                );
                completed_entries.insert(key.clone());
                if r.passed {
                    passed_entries.insert(key.clone());
                }
                case_attempts.insert(key, r.attempt);
            }
        }

        Ok(Self {
            store,
            run_id,
            resuming,
            config_fingerprint: fingerprint,
            completed_entries,
            passed_entries,
            case_attempts,
            total_cases,
            system_error_count,
        })
    }

    pub fn initialize_metadata(&self) -> Result<()> {
        let meta = RunMetadata {
            run_id: self.run_id.clone(),
            status: "running".to_string(),
            config_fingerprint: self.config_fingerprint.clone(),
            total_cases: self.total_cases,
            completed_cases: self.completed_entries.len() as u64,
            start_time: Some(iso_timestamp_now()),
            completed_time: None,
            system_error_count: self.system_error_count,
            extras: Default::default(),
        };
        self.store.write_metadata(&meta)
    }

    pub fn set_total_cases(&mut self, total: u64) -> Result<()> {
        self.total_cases = total;
        self.update_metadata_status("running")
    }

    pub fn append_result(&mut self, result: &CaseResult) -> Result<()> {
        self.store.append_result(result)?;
        let key = (
            result.agent_name.clone(),
            result.dataset_name.clone(),
            result.case_id.clone(),
        );
        self.completed_entries.insert(key.clone());
        if result.passed {
            self.passed_entries.insert(key.clone());
        }
        self.case_attempts.insert(key, result.attempt.max(1));
        self.update_metadata_status("running")
    }

    pub fn append_error(&mut self, error: &crate::persistence::ErrorEntry) -> Result<()> {
        self.store.append_error(error)?;
        if matches!(error.error_type, ErrorType::SystemFailure) {
            self.system_error_count += 1;
        }
        self.update_metadata_status("running")
    }

    pub fn mark_completed(&self, failed: bool) -> Result<()> {
        let status = if failed { "failed" } else { "completed" };
        self.update_metadata_status(status)
    }

    pub fn completed_count(&self) -> usize {
        self.completed_entries.len()
    }

    pub fn passed_count(&self) -> usize {
        self.passed_entries.len()
    }

    pub fn is_passed(&self, agent: &str, dataset: &str, case_id: &str) -> bool {
        let key = (agent.to_string(), dataset.to_string(), case_id.to_string());
        self.passed_entries.contains(&key)
    }

    pub fn get_next_attempt(&self, agent: &str, dataset: &str, case_id: &str) -> u32 {
        self.get_attempt_count(agent, dataset, case_id) + 1
    }

    pub fn get_attempt_count(&self, agent: &str, dataset: &str, case_id: &str) -> u32 {
        let key = (agent.to_string(), dataset.to_string(), case_id.to_string());
        *self.case_attempts.get(&key).unwrap_or(&0)
    }

    fn update_metadata_status(&self, status: &str) -> Result<()> {
        let meta = RunMetadata {
            run_id: self.run_id.clone(),
            status: status.to_string(),
            config_fingerprint: self.config_fingerprint.clone(),
            total_cases: self.total_cases,
            completed_cases: self.completed_entries.len() as u64,
            start_time: None,
            completed_time: if status == "running" {
                None
            } else {
                Some(iso_timestamp_now())
            },
            system_error_count: self.system_error_count,
            extras: Default::default(),
        };
        self.store.write_metadata(&meta)
    }
}

fn resolve_run_dir(base: &Path, run_id: &str) -> PathBuf {
    if Path::new(run_id).is_absolute() {
        PathBuf::from(run_id)
    } else {
        base.join(run_id)
    }
}
