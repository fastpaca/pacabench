use super::{
    count_prepared_cases, record_keys, stream_prepared_cases, CaseIdFallback, DatasetContext,
    DatasetLoader, RecordSource,
};
use crate::config::DatasetConfig;
use crate::error::{PacabenchError, Result};
use crate::types::Case;
use anyhow::anyhow;
use async_trait::async_trait;
use futures_util::stream::BoxStream;
use parking_lot::Mutex;
use reqwest::Client;
use std::path::PathBuf;
use std::time::Duration;
use tokio::fs;

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

    async fn case_files(&self) -> Result<Vec<PathBuf>> {
        let repo_dir = self.download().await?;
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

        Ok(files)
    }
}

#[async_trait]
impl DatasetLoader for HuggingFaceDataset {
    async fn count_cases(&self, limit: Option<usize>) -> Result<usize> {
        let files = self.case_files().await?;
        let keys = record_keys(&self.config.input_map);
        count_prepared_cases(
            &files,
            &self.config.name,
            &keys,
            CaseIdFallback::FileStem,
            RecordSource::JsonlOrJsonArray,
            limit,
        )
        .await
    }

    async fn stream_cases(&self, limit: Option<usize>) -> Result<BoxStream<'static, Result<Case>>> {
        let files = self.case_files().await?;
        let keys = record_keys(&self.config.input_map);
        Ok(stream_prepared_cases(
            files,
            self.config.name.clone(),
            keys,
            CaseIdFallback::FileStem,
            RecordSource::JsonlOrJsonArray,
            limit,
        ))
    }
}
