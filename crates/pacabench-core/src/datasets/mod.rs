//! Dataset loaders.

use crate::config::DatasetConfig;
use crate::error::{PacabenchError, Result};
use crate::types::Case;
use anyhow::anyhow;
use async_trait::async_trait;
use futures_util::stream::BoxStream;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncReadExt;

mod local;
pub use local::LocalDataset;

mod git;
pub use git::GitDataset;

mod huggingface;
pub use huggingface::HuggingFaceDataset;

/// Context passed to dataset loaders for path resolution and caching.
#[derive(Debug, Clone)]
pub struct DatasetContext {
    pub root_dir: PathBuf,
    pub cache_dir: PathBuf,
}

/// Trait for loading benchmark cases from various sources.
///
/// Implementations handle local files, git repositories, and HuggingFace datasets.
#[async_trait]
pub trait DatasetLoader: Send + Sync {
    /// Count how many cases are available, respecting an optional limit.
    async fn count_cases(&self, limit: Option<usize>) -> Result<usize>;

    /// Stream cases without materializing the entire dataset in memory.
    async fn stream_cases(&self, limit: Option<usize>) -> Result<BoxStream<'static, Result<Case>>>;
}

/// Create a dataset loader from configuration.
///
/// Dispatches to the appropriate loader based on the source prefix:
/// - `git:` - Clone from a git repository
/// - `huggingface:` - Download from HuggingFace Hub
/// - Otherwise - Load from local file system
pub fn get_dataset_loader(
    config: DatasetConfig,
    ctx: DatasetContext,
) -> Result<Box<dyn DatasetLoader>> {
    if config.source.starts_with("git:") {
        Ok(Box::new(GitDataset::new(config, ctx)))
    } else if config.source.starts_with("huggingface:") {
        Ok(Box::new(HuggingFaceDataset::new(config, ctx)))
    } else {
        Ok(Box::new(LocalDataset::new(config, ctx)))
    }
}

use crate::utils::resolve_path;

#[derive(Debug, Clone, Copy)]
enum DatasetFileFormat {
    Jsonl,
    JsonArray,
}

async fn detect_dataset_file_format(path: &Path) -> Result<DatasetFileFormat> {
    let mut file = fs::File::open(path)
        .await
        .map_err(PacabenchError::Persistence)?;
    let mut buffer = [0u8; 1024];
    loop {
        let bytes = file
            .read(&mut buffer)
            .await
            .map_err(PacabenchError::Persistence)?;
        if bytes == 0 {
            return Ok(DatasetFileFormat::Jsonl);
        }
        for byte in &buffer[..bytes] {
            if !byte.is_ascii_whitespace() {
                return Ok(if *byte == b'[' {
                    DatasetFileFormat::JsonArray
                } else {
                    DatasetFileFormat::Jsonl
                });
            }
        }
    }
}

async fn read_json_array(path: &Path) -> Result<Vec<serde_json::Value>> {
    let bytes = fs::read(path).await.map_err(PacabenchError::Persistence)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    match value {
        serde_json::Value::Array(items) => Ok(items),
        _ => Err(anyhow!("expected top-level JSON array in {}", path.display()).into()),
    }
}

/// Common helper to build a Case from a JSON-like record.
fn prepare_case(
    record: &serde_json::Map<String, serde_json::Value>,
    dataset_name: &str,
    fallback_id: &str,
    input_key: &str,
    expected_key: &str,
) -> Option<Case> {
    let input = record.get(input_key)?;
    let expected = record.get(expected_key);

    let case_id = record
        .get("case_id")
        .or_else(|| record.get("id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| fallback_id.to_string());

    let history = record
        .get("history")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let exclude_keys: HashSet<String> = COMMON_EXCLUDE_KEYS
        .iter()
        .map(|s| s.to_string())
        .chain([input_key.to_string(), expected_key.to_string()])
        .collect();

    let metadata: HashMap<String, serde_json::Value> = record
        .iter()
        .filter(|(k, _)| !exclude_keys.contains(*k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    Some(Case {
        case_id,
        dataset_name: dataset_name.to_string(),
        input: input.as_str().unwrap_or_default().to_string(),
        expected: expected.and_then(|v| v.as_str()).map(|s| s.to_string()),
        history,
        metadata,
    })
}

// Common fields to exclude from metadata to prevent leakage.
static COMMON_EXCLUDE_KEYS: &[&str] = &[
    "history",
    "case_id",
    "id",
    "ground_truth",
    "answer",
    "solution",
    "explanation",
    "reasoning",
    "correct_answer",
    "label",
    "target",
];
