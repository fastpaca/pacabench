use super::{
    count_prepared_cases, record_keys, resolve_path, stream_prepared_cases, CaseIdFallback,
    DatasetContext, DatasetLoader, RecordSource,
};
use crate::config::DatasetConfig;
use crate::error::{PacabenchError, Result};
use crate::types::Case;
use async_trait::async_trait;
use futures_util::stream::BoxStream;
use globwalk::GlobWalkerBuilder;
use std::path::PathBuf;

pub struct LocalDataset {
    config: DatasetConfig,
    root: PathBuf,
}

impl LocalDataset {
    pub fn new(config: DatasetConfig, ctx: DatasetContext) -> Self {
        Self {
            config,
            root: ctx.root_dir,
        }
    }

    fn resolve_files(&self) -> Result<Vec<PathBuf>> {
        let source = &self.config.source;

        let mut files: Vec<PathBuf> = Vec::new();
        if source.contains('*') {
            let pattern = resolve_path(source, &self.root);
            let walker = GlobWalkerBuilder::from_patterns(
                pattern
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| self.root.clone()),
                &[pattern
                    .file_name()
                    .map(|os| os.to_string_lossy().to_string())
                    .unwrap_or_else(|| "*.jsonl".to_string())],
            )
            .build()
            .map_err(|e| PacabenchError::Internal(e.into()))?;
            for entry in walker.into_iter().filter_map(|e| e.ok()) {
                if entry.path().is_file() {
                    files.push(entry.path().to_path_buf());
                }
            }
        } else {
            let p = resolve_path(source, &self.root);
            if p.is_dir() {
                let walker = GlobWalkerBuilder::from_patterns(&p, &["*.jsonl", "*.json"])
                    .build()
                    .map_err(|e| PacabenchError::Internal(e.into()))?;
                for entry in walker.into_iter().filter_map(|e| e.ok()) {
                    if entry.path().is_file() {
                        files.push(entry.path().to_path_buf());
                    }
                }
            } else {
                files.push(p);
            }
        }

        if let Some(s) = self.config.split.clone() {
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
                return Ok(filtered);
            }
        }

        Ok(files)
    }
}

#[async_trait]
impl DatasetLoader for LocalDataset {
    async fn count_cases(&self, limit: Option<usize>) -> Result<usize> {
        let files = self.resolve_files()?;
        let keys = record_keys(&self.config.input_map);
        count_prepared_cases(
            &files,
            &self.config.name,
            &keys,
            CaseIdFallback::FullPath,
            RecordSource::JsonlOrJsonArray,
            limit,
        )
        .await
    }

    async fn stream_cases(&self, limit: Option<usize>) -> Result<BoxStream<'static, Result<Case>>> {
        let files = self.resolve_files()?;
        let keys = record_keys(&self.config.input_map);
        Ok(stream_prepared_cases(
            files,
            self.config.name.clone(),
            keys,
            CaseIdFallback::FullPath,
            RecordSource::JsonlOrJsonArray,
            limit,
        ))
    }
}
