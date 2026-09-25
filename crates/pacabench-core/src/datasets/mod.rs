//! Dataset loaders.

use crate::config::DatasetConfig;
use crate::error::{PacabenchError, Result};
use crate::types::Case;
use crate::utils::resolve_path;
use anyhow::anyhow;
use async_trait::async_trait;
use futures_util::stream::{self, BoxStream, StreamExt, TryStreamExt};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio_stream::wrappers::LinesStream;

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

#[derive(Clone, Copy)]
enum CaseIdFallback {
    FullPath,
    FileStem,
}

#[derive(Clone, Copy)]
enum RecordSource {
    Jsonl,
    JsonlOrJsonArray,
}

#[derive(Clone)]
struct RecordKeys {
    input: String,
    expected: String,
}

fn record_keys(input_map: &HashMap<String, String>) -> RecordKeys {
    RecordKeys {
        input: input_map
            .get("input")
            .map(String::as_str)
            .unwrap_or("input")
            .to_string(),
        expected: input_map
            .get("expected")
            .map(String::as_str)
            .unwrap_or("expected")
            .to_string(),
    }
}

fn case_id_fallback(kind: CaseIdFallback, file: &Path, idx: usize) -> String {
    match kind {
        CaseIdFallback::FullPath => format!("{}-{idx}", file.display()),
        CaseIdFallback::FileStem => format!(
            "{}-{idx}",
            file.file_stem().unwrap_or_default().to_string_lossy()
        ),
    }
}

fn case_from_record(
    record: &serde_json::Map<String, Value>,
    dataset_name: &str,
    file: &Path,
    idx: usize,
    keys: &RecordKeys,
    fallback: CaseIdFallback,
) -> Option<Case> {
    prepare_case(
        record,
        dataset_name,
        &case_id_fallback(fallback, file, idx),
        &keys.input,
        &keys.expected,
    )
}

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

async fn count_prepared_cases(
    files: &[PathBuf],
    dataset_name: &str,
    keys: &RecordKeys,
    fallback: CaseIdFallback,
    source: RecordSource,
    limit: Option<usize>,
) -> Result<usize> {
    let mut count = 0usize;
    for file in files {
        count = match source {
            RecordSource::Jsonl => {
                count_jsonl_file(file, dataset_name, keys, fallback, limit, count).await?
            }
            RecordSource::JsonlOrJsonArray => match detect_dataset_file_format(file).await? {
                DatasetFileFormat::JsonArray => {
                    count_json_array_file(file, dataset_name, keys, fallback, limit, count).await?
                }
                DatasetFileFormat::Jsonl => {
                    count_jsonl_file(file, dataset_name, keys, fallback, limit, count).await?
                }
            },
        };
        if limit.is_some_and(|limit| count >= limit) {
            return Ok(count);
        }
    }
    Ok(count)
}

async fn count_json_array_file(
    file: &Path,
    dataset_name: &str,
    keys: &RecordKeys,
    fallback: CaseIdFallback,
    limit: Option<usize>,
    mut count: usize,
) -> Result<usize> {
    let items = read_json_array(file).await?;
    for (idx, item) in items.into_iter().enumerate() {
        if limit.is_some_and(|limit| count >= limit) {
            return Ok(count);
        }
        if let Value::Object(map) = item {
            if case_from_record(&map, dataset_name, file, idx, keys, fallback).is_some() {
                count += 1;
            }
        }
    }
    Ok(count)
}

async fn count_jsonl_file(
    file: &Path,
    dataset_name: &str,
    keys: &RecordKeys,
    fallback: CaseIdFallback,
    limit: Option<usize>,
    mut count: usize,
) -> Result<usize> {
    let opened = fs::File::open(file)
        .await
        .map_err(PacabenchError::Persistence)?;
    let reader = BufReader::new(opened);
    let mut lines = reader.lines();
    let mut idx = 0usize;
    while let Some(line) = lines
        .next_line()
        .await
        .map_err(PacabenchError::Persistence)?
    {
        if limit.is_some_and(|limit| count >= limit) {
            return Ok(count);
        }
        let current_idx = idx;
        idx += 1;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&line) {
            if case_from_record(&map, dataset_name, file, current_idx, keys, fallback).is_some() {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn stream_prepared_cases(
    files: Vec<PathBuf>,
    dataset_name: String,
    keys: RecordKeys,
    fallback: CaseIdFallback,
    source: RecordSource,
    limit: Option<usize>,
) -> BoxStream<'static, Result<Case>> {
    let stream = stream::iter(files)
        .then(move |file| {
            let dataset_name = dataset_name.clone();
            let keys = keys.clone();
            async move {
                match source {
                    RecordSource::Jsonl => {
                        jsonl_case_stream(file, dataset_name, keys, fallback).await
                    }
                    RecordSource::JsonlOrJsonArray => {
                        match detect_dataset_file_format(&file).await? {
                            DatasetFileFormat::JsonArray => {
                                json_array_case_stream(file, dataset_name, keys, fallback).await
                            }
                            DatasetFileFormat::Jsonl => {
                                jsonl_case_stream(file, dataset_name, keys, fallback).await
                            }
                        }
                    }
                }
            }
        })
        .try_flatten()
        .take(limit.unwrap_or(usize::MAX));

    Box::pin(stream)
}

async fn json_array_case_stream(
    file: PathBuf,
    dataset_name: String,
    keys: RecordKeys,
    fallback: CaseIdFallback,
) -> Result<BoxStream<'static, Result<Case>>> {
    let items = read_json_array(&file).await?;
    let cases = items
        .into_iter()
        .enumerate()
        .filter_map(|(idx, item)| {
            let Value::Object(map) = item else {
                return None;
            };
            case_from_record(&map, &dataset_name, &file, idx, &keys, fallback).map(Ok)
        })
        .collect::<Vec<_>>();
    Ok(stream::iter(cases).boxed())
}

async fn jsonl_case_stream(
    file: PathBuf,
    dataset_name: String,
    keys: RecordKeys,
    fallback: CaseIdFallback,
) -> Result<BoxStream<'static, Result<Case>>> {
    let file_for_id = file.clone();
    let opened = fs::File::open(&file)
        .await
        .map_err(PacabenchError::Persistence)?;
    let reader = BufReader::new(opened);
    let lines = LinesStream::new(reader.lines())
        .enumerate()
        .filter_map(move |(idx, line)| {
            let file = file_for_id.clone();
            let dataset_name = dataset_name.clone();
            let keys = keys.clone();
            async move {
                match line {
                    Ok(line) if !line.trim().is_empty() => {
                        match serde_json::from_str::<Value>(&line) {
                            Ok(Value::Object(map)) => {
                                case_from_record(&map, &dataset_name, &file, idx, &keys, fallback)
                                    .map(Ok)
                            }
                            _ => None,
                        }
                    }
                    Ok(_) => None,
                    Err(e) => Some(Err(PacabenchError::Internal(e.into()))),
                }
            }
        });
    Ok(lines.boxed())
}
