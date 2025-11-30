//! Configuration loading and models for the Rust rewrite.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, fs, path::Path};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse config: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("invalid config: {0}")]
    Invalid(String),
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

fn default_worker_recycle_interval() -> usize {
    0
}

fn default_circuit_breaker_min_cases() -> usize {
    10
}

fn default_version() -> String {
    "0.1.0".to_string()
}

fn default_output_directory() -> String {
    "./runs".to_string()
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
    #[serde(default = "default_worker_recycle_interval")]
    pub worker_recycle_interval: usize,
    #[serde(default)]
    pub circuit_breaker_error_ratio: Option<f64>,
    #[serde(default = "default_circuit_breaker_min_cases")]
    pub circuit_breaker_min_cases: usize,
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            concurrency: default_concurrency(),
            timeout_seconds: default_timeout_seconds(),
            max_retries: default_max_retries(),
            proxy: ProxyConfig::default(),
            worker_recycle_interval: default_worker_recycle_interval(),
            circuit_breaker_error_ratio: None,
            circuit_breaker_min_cases: default_circuit_breaker_min_cases(),
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
pub struct EvaluatorConfig {
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub extra_config: HashMap<String, Value>,
    /// Any additional fields are retained to preserve forward compatibility.
    #[serde(flatten)]
    pub additional: HashMap<String, Value>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub config: GlobalConfig,
    #[serde(default)]
    pub agents: Vec<AgentConfig>,
    #[serde(default)]
    pub datasets: Vec<DatasetConfig>,
    #[serde(default)]
    pub output: OutputConfig,
}

pub fn load_config(path: impl AsRef<Path>) -> Result<BenchmarkConfig, ConfigError> {
    let contents = fs::read_to_string(path)?;
    let interpolated = interpolate_env_vars(&contents);
    let cfg: BenchmarkConfig = serde_yaml::from_str(&interpolated)?;
    validate_config(&cfg)?;
    Ok(cfg)
}

/// Interpolate environment variables in the config string.
/// Supports `${VAR}` and `${VAR:-default}` syntax.
pub fn interpolate_env_vars(input: &str) -> String {
    use once_cell::sync::Lazy;
    use regex::Regex;
    use std::env;

    // Match ${VAR} or ${VAR:-default}
    // Pattern is compile-time constant, panic on invalid regex is acceptable.
    static ENV_VAR_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
            .expect("valid regex pattern")
    });

    ENV_VAR_RE
        .replace_all(input, |caps: &regex::Captures| {
            let var_name = &caps[1];
            let default_val = caps.get(2).map(|m| m.as_str());

            match env::var(var_name) {
                Ok(val) => val,
                Err(_) => default_val.unwrap_or("").to_string(),
            }
        })
        .to_string()
}

/// Validate required fields and basic invariants.
pub fn validate_config(cfg: &BenchmarkConfig) -> Result<(), ConfigError> {
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
