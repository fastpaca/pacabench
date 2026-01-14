use super::{
    detect_dataset_file_format, prepare_case, read_json_array, DatasetContext, DatasetFileFormat,
    DatasetLoader,
};
use crate::config::DatasetConfig;
use crate::error::{PacabenchError, Result};
use crate::types::Case;
use anyhow::anyhow;
use async_trait::async_trait;
use futures_util::stream::{self, BoxStream, StreamExt, TryStreamExt};
use parking_lot::Mutex;
use reqwest::Client;
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;
use tokio::fs::{self, File};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_stream::wrappers::LinesStream;

/// Minimal Hugging Face dataset loader.
///
/// This implementation expects the dataset repo to contain JSONL files or JSON
/// files with top-level arrays, downloading them to the cache directory.
/// It avoids full HF dataset semantics, keeping parity with the Python JSONL
/// ingestion path.
pub struct HuggingFaceDataset {
    config: DatasetConfig,
    ctx: DatasetContext,
    prepared_path: Mutex<Option<PathBuf>>,
}

impl HuggingFaceDataset {
    pub fn new(config: DatasetConfig, ctx: DatasetContext) -> Self {
        Self {
            config,
            ctx,
            prepared_path: Mutex::new(None),
        }
    }

    fn repo_id(&self) -> String {
        self.config
            .source
            .strip_prefix("huggingface:")
            .unwrap_or(&self.config.source)
            .to_string()
    }

    fn split_candidates(&self) -> Vec<String> {
        let split = self.config.split.as_deref().unwrap_or("train");
        if split.ends_with(".jsonl") || split.ends_with(".json") {
            vec![split.to_string()]
        } else {
            vec![format!("{split}.jsonl"), format!("{split}.json")]
        }
    }

    fn resolve_local_repo(&self) -> Option<PathBuf> {
        let id = self.repo_id();
        let path = PathBuf::from(&id);
        if path.exists() {
            return Some(path);
        }
        let relative = self.ctx.root_dir.join(&id);
        if relative.exists() {
            return Some(relative);
        }
        None
    }

    async fn download(&self) -> Result<PathBuf> {
        if let Some(existing) = self.prepared_path.lock().clone() {
            return Ok(existing);
        }

        if let Some(local) = self.resolve_local_repo() {
            *self.prepared_path.lock() = Some(local.clone());
            return Ok(local);
        }

        let repo_id = self.repo_id();
        let cache_dir = self
            .ctx
            .cache_dir
            .join("hf")
            .join(repo_id.replace('/', "_"));
        fs::create_dir_all(&cache_dir)
            .await
            .map_err(PacabenchError::Persistence)?;

        let split_candidates = self.split_candidates();
        for candidate in &split_candidates {
            let target_file = cache_dir.join(candidate);
            if fs::try_exists(&target_file)
                .await
                .map_err(PacabenchError::Persistence)?
            {
                *self.prepared_path.lock() = Some(cache_dir.clone());
                return Ok(cache_dir);
            }
        }

        let token = std::env::var("HF_TOKEN")
            .ok()
            .or_else(|| std::env::var("HUGGINGFACE_TOKEN").ok());
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| anyhow!("failed to build client: {e}"))?;

        for candidate in &split_candidates {
            let url = format!("https://huggingface.co/datasets/{repo_id}/resolve/main/{candidate}");
            let mut req = client.get(url);
            if let Some(t) = &token {
                req = req.bearer_auth(t);
            }
            let resp = req
                .send()
                .await
                .map_err(|e| anyhow!("download failed: {e}"))?;
            if resp.status().is_success() {
                let target_file = cache_dir.join(candidate);
                if let Some(parent) = target_file.parent() {
                    fs::create_dir_all(parent)
                        .await
                        .map_err(PacabenchError::Persistence)?;
                }
                let bytes = resp.bytes().await.map_err(|e| anyhow!("read body: {e}"))?;
                fs::write(&target_file, bytes)
                    .await
                    .map_err(PacabenchError::Persistence)?;
                *self.prepared_path.lock() = Some(cache_dir.clone());
                return Ok(cache_dir);
            }
            if resp.status() != reqwest::StatusCode::NOT_FOUND {
                return Err(anyhow!(
                    "failed to download dataset {repo_id} split {}: http {}",
                    self.config.split.as_deref().unwrap_or("train"),
                    resp.status()
                )
                .into());
            }
        }
        Err(anyhow!(
            "failed to download dataset {repo_id} split {}: no matching files",
            self.config.split.as_deref().unwrap_or("train")
        )
        .into())
    }
}

#[async_trait]
impl DatasetLoader for HuggingFaceDataset {
    async fn count_cases(&self, limit: Option<usize>) -> Result<usize> {
        let repo_dir = self.download().await?;
        let split_candidates = self.split_candidates();

        // Try split-specific file first, then all json/jsonl files.
        let mut files: Vec<PathBuf> = Vec::new();
        for candidate in &split_candidates {
            let split_file = repo_dir.join(candidate);
            if split_file.exists() {
                files.push(split_file);
                break;
            }
        }
        if files.is_empty() {
            for entry in
                globwalk::GlobWalkerBuilder::from_patterns(&repo_dir, &["**/*.jsonl", "**/*.json"])
                    .build()
                    .map_err(|e| PacabenchError::Internal(e.into()))?
                    .filter_map(|e| e.ok())
            {
                if entry.path().is_file() {
                    files.push(entry.path().to_path_buf());
                }
            }
        }

        if files.is_empty() {
            return Err(
                anyhow!("no JSON/JSONL files found in HF dataset {}", self.repo_id()).into(),
            );
        }

        let input_key = self
            .config
            .input_map
            .get("input")
            .map(String::as_str)
            .unwrap_or("input");
        let expected_key = self
            .config
            .input_map
            .get("expected")
            .map(String::as_str)
            .unwrap_or("expected");

        let mut count = 0usize;
        for file in files {
            match detect_dataset_file_format(&file).await? {
                DatasetFileFormat::JsonArray => {
                    let items = read_json_array(&file).await?;
                    for (idx, item) in items.into_iter().enumerate() {
                        if let Some(limit) = limit {
                            if count >= limit {
                                return Ok(count);
                            }
                        }
                        if let Value::Object(map) = item {
                            let fallback = format!(
                                "{}-{}",
                                file.file_stem().unwrap_or_default().to_string_lossy(),
                                idx
                            );
                            if prepare_case(
                                &map,
                                &self.config.name,
                                &fallback,
                                input_key,
                                expected_key,
                            )
                            .is_some()
                            {
                                count += 1;
                            }
                        }
                    }
                }
                DatasetFileFormat::Jsonl => {
                    let f = File::open(&file)
                        .await
                        .map_err(PacabenchError::Persistence)?;
                    let reader = BufReader::new(f);
                    let mut lines = reader.lines();
                    let mut idx = 0usize;
                    while let Some(line) = lines
                        .next_line()
                        .await
                        .map_err(PacabenchError::Persistence)?
                    {
                        if let Some(limit) = limit {
                            if count >= limit {
                                return Ok(count);
                            }
                        }
                        let current_idx = idx;
                        idx += 1;
                        if line.trim().is_empty() {
                            continue;
                        }
                        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&line) {
                            let fallback = format!(
                                "{}-{}",
                                file.file_stem().unwrap_or_default().to_string_lossy(),
                                current_idx
                            );
                            if prepare_case(
                                &map,
                                &self.config.name,
                                &fallback,
                                input_key,
                                expected_key,
                            )
                            .is_some()
                            {
                                count += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(count)
    }

    async fn stream_cases(&self, limit: Option<usize>) -> Result<BoxStream<'static, Result<Case>>> {
        let repo_dir = self.download().await?;
        let dataset_name = self.config.name.clone();
        let split_candidates = self.split_candidates();

        let mut files: Vec<PathBuf> = Vec::new();
        for candidate in &split_candidates {
            let split_file = repo_dir.join(candidate);
            if split_file.exists() {
                files.push(split_file);
                break;
            }
        }
        if files.is_empty() {
            for entry in
                globwalk::GlobWalkerBuilder::from_patterns(&repo_dir, &["**/*.jsonl", "**/*.json"])
                    .build()
                    .map_err(|e| PacabenchError::Internal(e.into()))?
                    .filter_map(|e| e.ok())
            {
                if entry.path().is_file() {
                    files.push(entry.path().to_path_buf());
                }
            }
        }

        if files.is_empty() {
            return Err(
                anyhow!("no JSON/JSONL files found in HF dataset {}", self.repo_id()).into(),
            );
        }

        let input_key = self
            .config
            .input_map
            .get("input")
            .map(String::as_str)
            .unwrap_or("input")
            .to_string();
        let expected_key = self
            .config
            .input_map
            .get("expected")
            .map(String::as_str)
            .unwrap_or("expected")
            .to_string();

        let stream = stream::iter(files)
            .then(move |file| {
                let dataset_name = dataset_name.clone();
                let input_key = input_key.clone();
                let expected_key = expected_key.clone();
                async move {
                    match detect_dataset_file_format(&file).await? {
                        DatasetFileFormat::JsonArray => {
                            let items = read_json_array(&file).await?;
                            let cases = items
                                .into_iter()
                                .enumerate()
                                .filter_map(|(idx, item)| {
                                    if let Value::Object(map) = item {
                                        let fallback = format!(
                                            "{}-{}",
                                            file.file_stem().unwrap_or_default().to_string_lossy(),
                                            idx
                                        );
                                        prepare_case(
                                            &map,
                                            &dataset_name,
                                            &fallback,
                                            &input_key,
                                            &expected_key,
                                        )
                                        .map(Ok)
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>();
                            Ok::<_, PacabenchError>(stream::iter(cases).boxed())
                        }
                        DatasetFileFormat::Jsonl => {
                            let file_clone = file.clone();
                            let f = File::open(&file)
                                .await
                                .map_err(PacabenchError::Persistence)?;
                            let reader = BufReader::new(f);
                            let lines = LinesStream::new(reader.lines()).enumerate().filter_map(
                                move |(idx, line)| {
                                    let file = file_clone.clone();
                                    let dataset_name = dataset_name.clone();
                                    let input_key = input_key.clone();
                                    let expected_key = expected_key.clone();
                                    async move {
                                        match line {
                                            Ok(line) if !line.trim().is_empty() => {
                                                match serde_json::from_str::<Value>(&line) {
                                                    Ok(Value::Object(map)) => {
                                                        let fallback = format!(
                                                            "{}-{}",
                                                            file.file_stem()
                                                                .unwrap_or_default()
                                                                .to_string_lossy(),
                                                            idx
                                                        );
                                                        prepare_case(
                                                            &map,
                                                            &dataset_name,
                                                            &fallback,
                                                            &input_key,
                                                            &expected_key,
                                                        )
                                                        .map(Ok)
                                                    }
                                                    _ => None,
                                                }
                                            }
                                            Ok(_) => None,
                                            Err(e) => Some(Err(PacabenchError::Internal(e.into()))),
                                        }
                                    }
                                },
                            );

                            Ok::<_, PacabenchError>(lines.boxed())
                        }
                    }
                }
            })
            .try_flatten()
            .take(limit.unwrap_or(usize::MAX));

        Ok(Box::pin(stream))
    }
}
