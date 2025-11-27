use super::{prepare_case, DatasetContext, DatasetLoader};
use crate::config::DatasetConfig;
use crate::types::Case;
use anyhow::{anyhow, Result};
use git2::Repository;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;

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
                let mut origin = repo.find_remote("origin")?;
                origin.fetch(&["main"], None, None).ok(); // best-effort
                return Ok(repo_dir);
            }
        }
        Repository::clone(repo_url, &repo_dir)?;
        Ok(repo_dir)
    }

    fn run_prepare(&self, repo_dir: &Path) -> Result<()> {
        if let Some(cmd) = &self.config.prepare {
            let status = Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .current_dir(&self.ctx.root_dir)
                .env("PACABENCH_DATASET_PATH", repo_dir)
                .status()?;
            if !status.success() {
                return Err(anyhow!("prepare command failed with status {status}"));
            }
        }
        Ok(())
    }
}

impl DatasetLoader for GitDataset {
    fn load(&self, limit: Option<usize>) -> Result<Vec<Case>> {
        let repo_url = self.normalize_repo_url();
        let repo_dir = self.ensure_repo(&repo_url)?;
        self.run_prepare(&repo_dir)?;

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
            .build()?
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
