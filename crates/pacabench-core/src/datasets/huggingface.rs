use super::{prepare_case, DatasetContext, DatasetLoader};
use crate::config::DatasetConfig;
use crate::types::Case;
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

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

    fn download(&self) -> Result<PathBuf> {
        // Best-effort JSONL fetch using hf-hub style layout: assume repo has jsonl files.
        // To avoid additional dependencies, we rely on `git clone` under the hood.
        let repo_url = format!("https://huggingface.co/{}.git", self.repo_id());
        let cache_dir = self.ctx.cache_dir.join("hf");
        std::fs::create_dir_all(&cache_dir)?;
        let repo_dir = cache_dir.join(self.repo_id().rsplit('/').next().unwrap_or("dataset"));
        if repo_dir.join(".git").exists() {
            return Ok(repo_dir);
        }
        git2::Repository::clone(&repo_url, &repo_dir)
            .map_err(|e| anyhow!("failed to clone HF dataset: {e}"))?;
        Ok(repo_dir)
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
