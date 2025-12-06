//! Shared utility functions and constants.

use std::path::{Path, PathBuf};

/// Default OpenAI API base URL.
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com";

/// Resolve a path relative to a base directory if not absolute.
pub fn resolve_path(path: &str, base: &Path) -> PathBuf {
    let p = PathBuf::from(path);
    if p.is_absolute() {
        p
    } else {
        base.join(p)
    }
}
