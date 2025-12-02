//! Dataset loaders.

use crate::config::DatasetConfig;
use crate::error::Result;
use crate::types::Case;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

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

pub trait DatasetLoader {
    fn load(&self, limit: Option<usize>) -> Result<Vec<Case>>;
}

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

/// Resolve a path relative to the dataset context root if not absolute.
fn resolve_path(path_str: &str, root: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        root.join(p)
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

    let exclude_keys = {
        let mut set = COMMON_EXCLUDE_KEYS
            .iter()
            .map(|s| s.to_string())
            .collect::<HashSet<_>>();
        set.insert(input_key.to_string());
        set.insert(expected_key.to_string());
        set
    };

    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
    for (k, v) in record {
        if exclude_keys.contains(k) {
            continue;
        }
        metadata.insert(k.clone(), v.clone());
    }

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
