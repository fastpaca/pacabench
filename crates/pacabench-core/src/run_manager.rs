//! Run management: metadata, resume tracking, attempts.

use crate::config::BenchmarkConfig;
use crate::persistence::{compute_config_fingerprint, iso_timestamp_now, RunMetadata, RunStore};
use crate::types::{CaseResult, ErrorType};
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct RunManager {
    store: RunStore,
    pub run_id: String,
    pub resuming: bool,
    config_fingerprint: String,
    config_name: String,
    config_version: String,
    config_description: Option<String>,
    agent_names: Vec<String>,
    dataset_names: Vec<String>,
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

        let agent_names = config.agents.iter().map(|a| a.name.clone()).collect();
        let dataset_names = config.datasets.iter().map(|d| d.name.clone()).collect();

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
            config_name: config.name.clone(),
            config_version: config.version.clone(),
            config_description: config.description.clone(),
            agent_names,
            dataset_names,
            completed_entries,
            passed_entries,
            case_attempts,
            total_cases,
            system_error_count,
        })
    }

    pub fn initialize_metadata(&self) -> Result<()> {
        let mut meta = self
            .store
            .read_metadata()?
            .unwrap_or_else(|| self.default_metadata("running"));
        meta.status = "running".to_string();
        meta.total_cases = self.total_cases;
        meta.completed_cases = self.completed_entries.len() as u64;
        meta.system_error_count = self.system_error_count;
        if meta.start_time.is_none() {
            meta.start_time = Some(iso_timestamp_now());
        }
        meta.config_fingerprint = self.config_fingerprint.clone();
        self.merge_extras(&mut meta);
        self.store.write_metadata(&meta)
    }

    pub fn set_total_cases(&mut self, total: u64) -> Result<()> {
        self.total_cases = total;
        self.update_metadata_status("running")
    }

    pub fn append_result(&mut self, result: &CaseResult) -> Result<()> {
        let mut to_write = result.clone();
        if to_write.timestamp.is_none() {
            to_write.timestamp = Some(iso_timestamp_now());
        }

        self.store.append_result(&to_write)?;
        let key = (
            to_write.agent_name.clone(),
            to_write.dataset_name.clone(),
            to_write.case_id.clone(),
        );
        self.completed_entries.insert(key.clone());
        if to_write.passed {
            self.passed_entries.insert(key.clone());
        }
        self.case_attempts.insert(key, to_write.attempt.max(1));
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
        let mut meta = self
            .store
            .read_metadata()?
            .unwrap_or_else(|| self.default_metadata(status));
        meta.status = status.to_string();
        meta.config_fingerprint = self.config_fingerprint.clone();
        meta.total_cases = self.total_cases;
        meta.completed_cases = self.completed_entries.len() as u64;
        meta.system_error_count = self.system_error_count;
        if meta.start_time.is_none() {
            meta.start_time = Some(iso_timestamp_now());
        }
        if status != "running" {
            meta.completed_time = Some(iso_timestamp_now());
        }
        self.merge_extras(&mut meta);
        self.store.write_metadata(&meta)
    }

    fn merge_extras(&self, meta: &mut RunMetadata) {
        meta.extras.insert(
            "config_name".into(),
            Value::String(self.config_name.clone()),
        );
        meta.extras.insert(
            "config_version".into(),
            Value::String(self.config_version.clone()),
        );
        if let Some(desc) = &self.config_description {
            meta.extras
                .insert("config_description".into(), Value::String(desc.clone()));
        }
        meta.extras.insert(
            "agents".into(),
            Value::Array(
                self.agent_names
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        );
        meta.extras.insert(
            "datasets".into(),
            Value::Array(
                self.dataset_names
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        );
    }

    fn default_metadata(&self, status: &str) -> RunMetadata {
        let mut meta = RunMetadata {
            run_id: self.run_id.clone(),
            status: status.to_string(),
            config_fingerprint: self.config_fingerprint.clone(),
            total_cases: self.total_cases,
            completed_cases: self.completed_entries.len() as u64,
            start_time: Some(iso_timestamp_now()),
            completed_time: None,
            system_error_count: self.system_error_count,
            extras: Default::default(),
        };
        self.merge_extras(&mut meta);
        meta
    }
}

fn resolve_run_dir(base: &Path, run_id: &str) -> PathBuf {
    if Path::new(run_id).is_absolute() {
        PathBuf::from(run_id)
    } else {
        base.join(run_id)
    }
}
