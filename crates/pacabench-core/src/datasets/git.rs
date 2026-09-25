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
use git2::Repository;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;

pub struct GitDataset {
    config: DatasetConfig,
    ctx: DatasetContext,
    prepared_repo: Mutex<Option<PathBuf>>,
}

impl GitDataset {
    pub fn new(config: DatasetConfig, ctx: DatasetContext) -> Self {
        Self {
            config,
            ctx,
            prepared_repo: Mutex::new(None),
        }
    }

    fn normalize_repo_url(&self) -> String {
        self.config
            .source
            .strip_prefix("git:")
            .unwrap_or(&self.config.source)
            .to_string()
    }

    fn repo_cache_dir(&self, repo_url: &str) -> PathBuf {
        let name = repo_url
            .trim_end_matches(".git")
            .rsplit('/')
            .next()
            .unwrap_or("repo");
        self.ctx.cache_dir.join("repos").join(name)
    }

    fn ensure_repo(&self, repo_url: &str) -> Result<PathBuf> {
        let repo_dir = self.repo_cache_dir(repo_url);
        std::fs::create_dir_all(repo_dir.parent().unwrap_or_else(|| Path::new(".")))
            .map_err(PacabenchError::Persistence)?;
        if repo_dir.join(".git").exists() {
            if let Ok(repo) = Repository::open(&repo_dir) {
                let mut origin = repo
                    .find_remote("origin")
                    .map_err(|e| PacabenchError::Internal(e.into()))?;
                origin
                    .fetch(&["main"], None, None)
                    .map_err(|e| PacabenchError::Internal(e.into()))?;
                return Ok(repo_dir);
            }
        }
        Repository::clone(repo_url, &repo_dir).map_err(|e| PacabenchError::Internal(e.into()))?;
        Ok(repo_dir)
    }

    async fn run_prepare(&self, repo_dir: &Path) -> Result<()> {
        if let Some(cmd) = &self.config.prepare {
            let status = Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .current_dir(&self.ctx.root_dir)
                .env("PACABENCH_DATASET_PATH", repo_dir)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await
                .map_err(PacabenchError::Process)?;
            if !status.success() {
                return Err(anyhow!("prepare command failed with status {status}").into());
            }
        }
        Ok(())
    }

    async fn ensure_prepared_repo(&self) -> Result<PathBuf> {
        if let Some(path) = self.prepared_repo.lock().clone() {
            return Ok(path);
        }

        let repo_url = self.normalize_repo_url();
        let repo_dir = self.ensure_repo(&repo_url)?;
        self.run_prepare(&repo_dir).await?;

        *self.prepared_repo.lock() = Some(repo_dir.clone());
        Ok(repo_dir)
    }

    fn resolve_files(&self, repo_dir: &Path) -> Result<Vec<PathBuf>> {
        let files: Vec<PathBuf> =
            globwalk::GlobWalkerBuilder::from_patterns(repo_dir, &["**/*.jsonl"])
                .build()
                .map_err(|e| PacabenchError::Internal(e.into()))?
                .filter_map(|e| e.ok())
                .filter(|entry| entry.path().is_file())
                .map(|entry| entry.path().to_path_buf())
                .collect();

        // Filter by split if specified
        if let Some(split) = &self.config.split {
            let filtered: Vec<PathBuf> = files
                .iter()
                .filter(|p| {
                    let matches_stem = p
                        .file_stem()
                        .and_then(|f| f.to_str())
                        .is_some_and(|stem| stem == split);
                    let matches_name = p
                        .file_name()
                        .and_then(|f| f.to_str())
                        .is_some_and(|name| name.contains(split));
                    matches_stem || matches_name
                })
                .cloned()
                .collect();

            if !filtered.is_empty() {
                return Ok(filtered);
            }
        }

        Ok(files)
    }
}

#[async_trait]
impl DatasetLoader for GitDataset {
    async fn count_cases(&self, limit: Option<usize>) -> Result<usize> {
        let repo_dir = self.ensure_prepared_repo().await?;
        let files = self.resolve_files(&repo_dir)?;
        let keys = record_keys(&self.config.input_map);
        count_prepared_cases(
            &files,
            &self.config.name,
            &keys,
            CaseIdFallback::FileStem,
            RecordSource::Jsonl,
            limit,
        )
        .await
    }

    async fn stream_cases(&self, limit: Option<usize>) -> Result<BoxStream<'static, Result<Case>>> {
        let repo_dir = self.ensure_prepared_repo().await?;
        let files = self.resolve_files(&repo_dir)?;
        let keys = record_keys(&self.config.input_map);
        Ok(stream_prepared_cases(
            files,
            self.config.name.clone(),
            keys,
            CaseIdFallback::FileStem,
            RecordSource::Jsonl,
            limit,
        ))
    }
}
