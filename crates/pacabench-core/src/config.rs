//! Configuration loading and models for PacaBench.
//!
//! Configuration is loaded via figment from multiple layers:
//! 1. YAML file (base configuration)
//! 2. Environment variables (PACABENCH_ prefix, __ as nested separator)
//! 3. Runtime overrides (passed programmatically)

use figment::{
    providers::{Env, Format, Yaml},
    Figment,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use thiserror::Error;

/// Errors that can occur when loading or validating configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Failed to read the configuration file from disk.
    #[error("failed to read config: {0}")]
    Io(#[from] std::io::Error),
    /// Failed to parse the configuration (YAML syntax or type errors).
    #[error("failed to parse config: {0}")]
    Figment(#[source] Box<figment::Error>),
    /// Configuration is syntactically valid but semantically incorrect
    /// (e.g., missing required agents or datasets).
    #[error("invalid config: {0}")]
    Invalid(String),
}

impl From<figment::Error> for ConfigError {
    fn from(err: figment::Error) -> Self {
        ConfigError::Figment(Box::new(err))
    }
}

fn default_proxy_enabled() -> bool {
    true
}

fn default_proxy_provider() -> String {
    "openai".to_string()
}

fn default_concurrency() -> usize {
    4
}

fn default_timeout_seconds() -> f64 {
    60.0
}

fn default_max_retries() -> usize {
    2
}

fn default_version() -> String {
    "0.1.0".to_string()
}

fn default_output_directory() -> String {
    "./runs".to_string()
}

fn default_cache_directory() -> String {
    "~/.cache/pacabench/datasets".to_string()
}

fn default_f1_threshold() -> f64 {
    0.5
}

fn default_judge_model() -> String {
    "gpt-4o-mini".to_string()
}

fn default_judge_max_retries() -> usize {
    2
}

fn default_judge_backoff_ms() -> u64 {
    200
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    #[serde(default = "default_proxy_enabled")]
    pub enabled: bool,
    #[serde(default = "default_proxy_provider")]
    pub provider: String,
    #[serde(default)]
    pub base_url: Option<String>,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            enabled: default_proxy_enabled(),
            provider: default_proxy_provider(),
            base_url: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: f64,
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    #[serde(default)]
    pub proxy: ProxyConfig,
    #[serde(default = "default_cache_directory")]
    pub cache_directory: String,
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            concurrency: default_concurrency(),
            timeout_seconds: default_timeout_seconds(),
            max_retries: default_max_retries(),
            proxy: ProxyConfig::default(),
            cache_directory: default_cache_directory(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub setup: Option<String>,
    #[serde(default)]
    pub teardown: Option<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvaluatorConfig {
    ExactMatch,
    F1 {
        #[serde(default = "default_f1_threshold")]
        threshold: f64,
    },
    MultipleChoice {
        #[serde(default)]
        fallback: MultipleChoiceFallback,
        #[serde(default = "default_f1_threshold")]
        threshold: f64,
        #[serde(default = "default_judge_model")]
        model: String,
    },
    LlmJudge {
        #[serde(default = "default_judge_model")]
        model: String,
        #[serde(default)]
        base_url: Option<String>,
        #[serde(default)]
        api_key: Option<String>,
        #[serde(default = "default_judge_max_retries")]
        max_retries: usize,
        #[serde(default = "default_judge_backoff_ms")]
        backoff_ms: u64,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MultipleChoiceFallback {
    #[default]
    F1,
    Judge,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub name: String,
    pub source: String,
    #[serde(default)]
    pub split: Option<String>,
    #[serde(default)]
    pub prepare: Option<String>,
    #[serde(default)]
    pub input_map: HashMap<String, String>,
    #[serde(default)]
    pub evaluator: Option<EvaluatorConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    #[serde(default = "default_output_directory")]
    pub directory: String,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            directory: default_output_directory(),
        }
    }
}

/// The main configuration for a benchmark.
///
/// Load from a file with `Config::from_file()`, or construct programmatically
/// with `Config::new()` for tests.
#[derive(Debug, Clone, Serialize)]
pub struct Config {
    // YAML fields (included in fingerprint)
    pub name: String,
    pub description: Option<String>,
    pub version: String,
    pub author: Option<String>,
    pub global: GlobalConfig,
    pub agents: Vec<AgentConfig>,
    pub datasets: Vec<DatasetConfig>,
    pub output: OutputConfig,

    // Resolved paths (excluded from fingerprint)
    #[serde(skip)]
    pub root_dir: PathBuf,
    #[serde(skip)]
    pub cache_dir: PathBuf,
    #[serde(skip)]
    pub runs_dir: PathBuf,
    #[serde(skip)]
    pub config_path: Option<PathBuf>,
}

/// Runtime overrides for config values.
#[derive(Debug, Clone, Default)]
pub struct ConfigOverrides {
    pub concurrency: Option<usize>,
    pub timeout_seconds: Option<f64>,
    pub max_retries: Option<usize>,
    pub runs_dir: Option<String>,
    pub cache_dir: Option<String>,
}

impl Config {
    /// Load config from a file with optional overrides.
    pub fn from_file(
        path: impl AsRef<Path>,
        overrides: ConfigOverrides,
    ) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let raw = load_raw_config(path, &overrides)?;

        let root_dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));

        let runs_dir = overrides
            .runs_dir
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| resolve_path(&raw.output.directory, &root_dir));

        let cache_dir = overrides
            .cache_dir
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| expand_tilde(&raw.config.cache_directory));

        Ok(Self {
            name: raw.name,
            description: raw.description,
            version: raw.version,
            author: raw.author,
            global: raw.config,
            agents: raw.agents,
            datasets: raw.datasets,
            output: raw.output,
            root_dir,
            cache_dir,
            runs_dir,
            config_path: Some(path.to_path_buf()),
        })
    }

    /// Create a config programmatically (for tests).
    pub fn new(
        name: impl Into<String>,
        agents: Vec<AgentConfig>,
        datasets: Vec<DatasetConfig>,
        root_dir: PathBuf,
        runs_dir: PathBuf,
    ) -> Self {
        Self {
            name: name.into(),
            description: None,
            version: default_version(),
            author: None,
            global: GlobalConfig::default(),
            agents,
            datasets,
            output: OutputConfig::default(),
            root_dir: root_dir.clone(),
            cache_dir: root_dir.join("cache"),
            runs_dir,
            config_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawConfig {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default = "default_version")]
    version: String,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    config: GlobalConfig,
    #[serde(default)]
    agents: Vec<AgentConfig>,
    #[serde(default)]
    datasets: Vec<DatasetConfig>,
    #[serde(default)]
    output: OutputConfig,
}

fn load_raw_config(path: &Path, overrides: &ConfigOverrides) -> Result<RawConfig, ConfigError> {
    let mut cfg: RawConfig = Figment::new()
        .merge(Yaml::file(path))
        .merge(Env::prefixed("PACABENCH_").split("__"))
        .extract()?;

    if let Some(c) = overrides.concurrency {
        cfg.config.concurrency = c;
    }
    if let Some(t) = overrides.timeout_seconds {
        cfg.config.timeout_seconds = t;
    }
    if let Some(r) = overrides.max_retries {
        cfg.config.max_retries = r;
    }
    if let Some(dir) = &overrides.runs_dir {
        cfg.output.directory = dir.clone();
    }

    validate_raw_config(&cfg)?;
    Ok(cfg)
}

fn validate_raw_config(cfg: &RawConfig) -> Result<(), ConfigError> {
    if cfg.agents.is_empty() {
        return Err(ConfigError::Invalid(
            "at least one agent is required".into(),
        ));
    }
    if cfg.datasets.is_empty() {
        return Err(ConfigError::Invalid(
            "at least one dataset is required".into(),
        ));
    }
    if cfg
        .agents
        .iter()
        .any(|a| a.name.trim().is_empty() || a.command.trim().is_empty())
    {
        return Err(ConfigError::Invalid(
            "agents must have non-empty name and command".into(),
        ));
    }
    if cfg
        .datasets
        .iter()
        .any(|d| d.name.trim().is_empty() || d.source.trim().is_empty())
    {
        return Err(ConfigError::Invalid(
            "datasets must have non-empty name and source".into(),
        ));
    }
    Ok(())
}

use crate::utils::resolve_path;

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_overrides_default() {
        let overrides = ConfigOverrides::default();
        assert!(overrides.concurrency.is_none());
        assert!(overrides.timeout_seconds.is_none());
        assert!(overrides.max_retries.is_none());
        assert!(overrides.runs_dir.is_none());
    }

    #[test]
    fn test_config_overrides_with_values() {
        let overrides = ConfigOverrides {
            concurrency: Some(8),
            timeout_seconds: None,
            max_retries: Some(5),
            runs_dir: Some("./custom_runs".to_string()),
            cache_dir: None,
        };
        assert_eq!(overrides.concurrency, Some(8));
        assert!(overrides.timeout_seconds.is_none());
        assert_eq!(overrides.max_retries, Some(5));
    }

    #[test]
    fn test_expand_tilde() {
        std::env::set_var("HOME", "/home/test");
        assert_eq!(expand_tilde("~/foo"), PathBuf::from("/home/test/foo"));
        assert_eq!(expand_tilde("/absolute"), PathBuf::from("/absolute"));
        assert_eq!(expand_tilde("relative"), PathBuf::from("relative"));
    }
}
