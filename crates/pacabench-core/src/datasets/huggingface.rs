use super::{prepare_case, DatasetContext, DatasetLoader};
use crate::config::DatasetConfig;
use crate::types::Case;
use anyhow::{anyhow, Result};
use reqwest::blocking::Client;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Duration;

/// Minimal Hugging Face dataset loader.
///
/// This implementation expects the dataset repo to contain JSONL files and will
/// download them to the cache directory. It avoids full HF dataset semantics,
/// keeping parity with the Python JSONL ingestion path.
pub struct HuggingFaceDataset {
    config: DatasetConfig,
    ctx: DatasetContext,
}

impl HuggingFaceDataset {
    pub fn new(config: DatasetConfig, ctx: DatasetContext) -> Self {
        Self { config, ctx }
    }

    fn repo_id(&self) -> String {
        self.config
            .source
            .strip_prefix("huggingface:")
            .unwrap_or(&self.config.source)
            .to_string()
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

    fn download(&self) -> Result<PathBuf> {
        if let Some(local) = self.resolve_local_repo() {
            return Ok(local);
        }

        let repo_id = self.repo_id();
        let cache_dir = self
            .ctx
            .cache_dir
            .join("hf")
            .join(&repo_id.replace('/', "_"));
        std::fs::create_dir_all(&cache_dir)?;

        let split = self.config.split.clone().unwrap_or_else(|| "train".into());
        let target_file = cache_dir.join(format!("{split}.jsonl"));
        if target_file.exists() {
            return Ok(cache_dir);
        }

        let token = std::env::var("HF_TOKEN")
            .ok()
            .or_else(|| std::env::var("HUGGINGFACE_TOKEN").ok());
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| anyhow!("failed to build client: {e}"))?;

        // Try to resolve huggingface dataset file via raw URL
        let url = format!("https://huggingface.co/datasets/{repo_id}/resolve/main/{split}.jsonl");
        let mut req = client.get(url);
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        let resp = req.send().map_err(|e| anyhow!("download failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(anyhow!(
                "failed to download dataset {repo_id} split {split}: http {}",
                resp.status()
            ));
        }
        let bytes = resp.bytes().map_err(|e| anyhow!("read body: {e}"))?;
        std::fs::write(&target_file, bytes)?;
        Ok(cache_dir)
    }
}

impl DatasetLoader for HuggingFaceDataset {
    fn load(&self, limit: Option<usize>) -> Result<Vec<Case>> {
        let repo_dir = self.download()?;
        let split = self.config.split.as_deref().unwrap_or("train");

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

        // Try split-specific file first, then all jsonl files.
        let mut files: Vec<PathBuf> = Vec::new();
        let split_file = repo_dir.join(format!("{split}.jsonl"));
        if split_file.exists() {
            files.push(split_file);
        } else {
            for entry in globwalk::GlobWalkerBuilder::from_patterns(&repo_dir, &["**/*.jsonl"])
                .build()?
                .filter_map(|e| e.ok())
            {
                if entry.path().is_file() {
                    files.push(entry.path().to_path_buf());
                }
            }
        }

        if files.is_empty() {
            return Err(anyhow!(
                "no JSONL files found in HF dataset {}",
                self.repo_id()
            ));
        }

        let mut cases = Vec::new();
        let mut count = 0usize;
        for file in files {
            let f = File::open(&file)?;
            let reader = BufReader::new(f);
            for (idx, line) in reader.lines().enumerate() {
                if let Some(limit) = limit {
                    if count >= limit {
                        return Ok(cases);
                    }
                }
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&line) {
                    let fallback = format!(
                        "{}-{}",
                        file.file_stem().unwrap_or_default().to_string_lossy(),
                        idx
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
