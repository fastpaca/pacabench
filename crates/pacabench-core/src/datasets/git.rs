use super::{prepare_case, DatasetContext, DatasetLoader};
use crate::config::DatasetConfig;
use crate::error::{PacabenchError, Result};
use crate::types::Case;
use anyhow::anyhow;
use async_trait::async_trait;
use git2::Repository;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

pub struct GitDataset {
    config: DatasetConfig,
    ctx: DatasetContext,
}

impl GitDataset {
    pub fn new(config: DatasetConfig, ctx: DatasetContext) -> Self {
        Self { config, ctx }
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
        std::fs::create_dir_all(repo_dir.parent().unwrap_or_else(|| Path::new(".")))?;
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
                .status()
                .await?;
            if !status.success() {
                return Err(anyhow!("prepare command failed with status {status}").into());
            }
        }
        Ok(())
    }
}

#[async_trait]
impl DatasetLoader for GitDataset {
    async fn load(&self, limit: Option<usize>) -> Result<Vec<Case>> {
        let repo_url = self.normalize_repo_url();
        let repo_dir = self.ensure_repo(&repo_url)?;
        self.run_prepare(&repo_dir).await?;

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
        let split = self.config.split.clone();

        let mut files = Vec::new();
        for entry in globwalk::GlobWalkerBuilder::from_patterns(&repo_dir, &["**/*.jsonl"])
            .build()
            .map_err(|e| PacabenchError::Internal(e.into()))?
            .filter_map(|e| e.ok())
        {
            if entry.path().is_file() {
                files.push(entry.path().to_path_buf());
            }
        }

        if let Some(s) = split {
            let filtered: Vec<PathBuf> = files
                .iter()
                .filter(|p| {
                    p.file_stem()
                        .and_then(|f| f.to_str())
                        .map(|stem| stem == s)
                        .unwrap_or(false)
                        || p.file_name()
                            .and_then(|f| f.to_str())
                            .map(|name| name.contains(&s))
                            .unwrap_or(false)
                })
                .cloned()
                .collect();
            if !filtered.is_empty() {
                files = filtered;
            }
        }

        let mut cases = Vec::new();
        let mut count = 0usize;
        for file in files {
            let f = File::open(&file).await?;
            let reader = BufReader::new(f);
            let mut lines = reader.lines();
            let mut idx = 0usize;
            while let Some(line) = lines.next_line().await? {
                let current_idx = idx;
                idx += 1;
                if let Some(limit) = limit {
                    if count >= limit {
                        return Ok(cases);
                    }
                }
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&line) {
                    let fallback = format!(
                        "{}-{}",
                        file.file_stem().unwrap_or_default().to_string_lossy(),
                        current_idx
                    );
                    if let Some(case) =
                        prepare_case(&map, &self.config.name, &fallback, input_key, expected_key)
                    {
                        cases.push(case);
                        count += 1;
                    }
                }
            }
        }

        Ok(cases)
    }
}
