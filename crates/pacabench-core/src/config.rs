//! Configuration loading and models for PacaBench.
//!
//! Configuration is loaded via figment from multiple layers:
//! 1. YAML file (base configuration)
//! 2. Environment variables (PACABENCH_ prefix, __ as nested separator)
//! 3. CLI overrides (passed programmatically)

use figment::{
    providers::{Env, Format, Serialized, Yaml},
    Figment,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse config: {0}")]
    Figment(#[from] figment::Error),
    #[error("invalid config: {0}")]
    Invalid(String),
}

// ============================================================================
// DEFAULTS (all in one place)
// ============================================================================

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

// ============================================================================
// PROXY CONFIG
// ============================================================================

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

// ============================================================================
// GLOBAL CONFIG
// ============================================================================

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

// ============================================================================
// AGENT CONFIG
// ============================================================================

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

// ============================================================================
// EVALUATOR CONFIG (typed, not HashMap)
// ============================================================================

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

// ============================================================================
// DATASET CONFIG
// ============================================================================

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

// ============================================================================
// OUTPUT CONFIG
// ============================================================================

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

// ============================================================================
// BENCHMARK CONFIG
// ============================================================================

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

impl BenchmarkConfig {
    /// Resolve the cache directory, expanding ~ to home.
    pub fn cache_dir(&self) -> std::path::PathBuf {
        let path = &self.config.cache_directory;
        if path.starts_with("~/") {
            if let Ok(home) = std::env::var("HOME") {
                return std::path::PathBuf::from(home).join(&path[2..]);
            }
        }
        std::path::PathBuf::from(path)
    }
}

// ============================================================================
// CLI OVERRIDES
// ============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CliOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runs_dir: Option<String>,
}

// ============================================================================
// LOADING
// ============================================================================

pub fn load_config(path: impl AsRef<Path>) -> Result<BenchmarkConfig, ConfigError> {
    load_config_with_overrides(path, CliOverrides::default())
}

pub fn load_config_with_overrides(
    path: impl AsRef<Path>,
    overrides: CliOverrides,
) -> Result<BenchmarkConfig, ConfigError> {
    let contents = std::fs::read_to_string(path.as_ref())?;
    let interpolated = interpolate_env_vars(&contents);

    let mut figment = Figment::new()
        .merge(Yaml::string(&interpolated))
        .merge(Env::prefixed("PACABENCH_").split("__"));

    if overrides.concurrency.is_some()
        || overrides.timeout_seconds.is_some()
        || overrides.max_retries.is_some()
    {
        let mut config_overrides = HashMap::new();
        if let Some(c) = overrides.concurrency {
            config_overrides.insert("concurrency".to_string(), serde_json::json!(c));
        }
        if let Some(t) = overrides.timeout_seconds {
            config_overrides.insert("timeout_seconds".to_string(), serde_json::json!(t));
        }
        if let Some(r) = overrides.max_retries {
            config_overrides.insert("max_retries".to_string(), serde_json::json!(r));
        }

        #[derive(Serialize)]
        struct ConfigOverride {
            config: HashMap<String, serde_json::Value>,
        }

        figment = figment.merge(Serialized::defaults(ConfigOverride {
            config: config_overrides,
        }));
    }

    if let Some(runs_dir) = overrides.runs_dir {
        #[derive(Serialize)]
        struct OutputOverride {
            output: OutputConfig,
        }

        figment = figment.merge(Serialized::defaults(OutputOverride {
            output: OutputConfig { directory: runs_dir },
        }));
    }

    let cfg: BenchmarkConfig = figment.extract()?;
    validate_config(&cfg)?;
    Ok(cfg)
}

fn interpolate_env_vars(input: &str) -> String {
    use once_cell::sync::Lazy;
    use regex::Regex;
    use std::env;

    static ENV_VAR_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}").expect("valid regex")
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

fn validate_config(cfg: &BenchmarkConfig) -> Result<(), ConfigError> {
    if cfg.agents.is_empty() {
        return Err(ConfigError::Invalid("at least one agent is required".into()));
    }
    if cfg.datasets.is_empty() {
        return Err(ConfigError::Invalid("at least one dataset is required".into()));
    }
    if cfg.agents.iter().any(|a| a.name.trim().is_empty() || a.command.trim().is_empty()) {
        return Err(ConfigError::Invalid("agents must have non-empty name and command".into()));
    }
    if cfg.datasets.iter().any(|d| d.name.trim().is_empty() || d.source.trim().is_empty()) {
        return Err(ConfigError::Invalid("datasets must have non-empty name and source".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_env_vars() {
        std::env::set_var("TEST_VAR", "hello");
        let input = "value: ${TEST_VAR}";
        let output = interpolate_env_vars(input);
        assert_eq!(output, "value: hello");
        std::env::remove_var("TEST_VAR");
    }

    #[test]
    fn test_interpolate_with_default() {
        std::env::remove_var("NONEXISTENT_VAR");
        let input = "value: ${NONEXISTENT_VAR:-default_value}";
        let output = interpolate_env_vars(input);
        assert_eq!(output, "value: default_value");
    }

    #[test]
    fn test_cli_overrides() {
        let overrides = CliOverrides {
            concurrency: Some(8),
            timeout_seconds: None,
            max_retries: Some(5),
            runs_dir: Some("./custom_runs".to_string()),
        };
        assert!(overrides.concurrency.is_some());
        assert!(overrides.timeout_seconds.is_none());
    }
}
