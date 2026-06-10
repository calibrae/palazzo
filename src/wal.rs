use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Serialize;

use crate::util::now_rfc3339;

#[derive(Clone)]
pub struct Wal {
    path: Option<PathBuf>,
}

#[derive(Serialize)]
struct Entry<'a, T: Serialize> {
    timestamp: String,
    operation: &'a str,
    params: T,
}

impl Wal {
    pub fn from_env() -> Self {
        let path = std::env::var("PALAZZO_WAL")
            .ok()
            .map(PathBuf::from)
            .or_else(|| dirs_home().map(|h| h.join(".palazzo").join("wal.jsonl")));
        if let Some(p) = &path
            && let Some(parent) = p.parent()
        {
            let _ = std::fs::create_dir_all(parent);
        }
        Self { path }
    }

    /// Test-only constructor with an explicit path (or `None` for the
    /// unconfigured case) — avoids env-var pollution between parallel tests.
    #[cfg(test)]
    pub(crate) fn with_path(path: Option<PathBuf>) -> Self {
        Self { path }
    }

    /// Best-effort logging for non-destructive operations. No WAL path
    /// configured is a deliberate no-op; write failures only warn.
    pub fn log<T: Serialize>(&self, operation: &str, params: &T) {
        let Some(path) = &self.path else {
            return;
        };
        if let Err(e) = write_entry(path, operation, params) {
            tracing::warn!("wal: {e:#}");
        }
    }

    /// Like `log`, but errors when the entry cannot be durably appended —
    /// including when no WAL path is configured at all. Destructive operations
    /// (palace_delete, palace_delete_by_filter) call this and abort before
    /// touching Qdrant: the WAL is their only audit trail, so a delete that
    /// can't be logged must not happen.
    pub fn log_strict<T: Serialize>(&self, operation: &str, params: &T) -> anyhow::Result<()> {
        let path = self
            .path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("WAL path not configured (set PALAZZO_WAL or HOME)"))?;
        write_entry(path, operation, params)
    }
}

fn write_entry<T: Serialize>(path: &Path, operation: &str, params: &T) -> anyhow::Result<()> {
    let entry = Entry {
        timestamp: now_rfc3339(),
        operation,
        params,
    };
    let line = serde_json::to_string(&entry).context("wal serialize")?;
    append(path, &line).with_context(|| format!("wal append {}", path.display()))?;
    Ok(())
}

fn append(path: &Path, line: &str) -> std::io::Result<()> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{line}")
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::Wal;
    use serde_json::json;

    fn temp_wal_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "palazzo-wal-test-{tag}-{}.jsonl",
            std::process::id()
        ))
    }

    #[test]
    fn log_strict_errors_without_path() {
        let wal = Wal::with_path(None);
        let err = wal.log_strict("op", &json!({"k": "v"})).unwrap_err();
        assert!(err.to_string().contains("not configured"), "{err:#}");
    }

    #[test]
    fn log_strict_appends_jsonl() {
        let path = temp_wal_path("strict");
        let _ = std::fs::remove_file(&path);
        let wal = Wal::with_path(Some(path.clone()));
        wal.log_strict("palace_delete", &json!({"id": 42})).unwrap();
        wal.log_strict("palace_delete", &json!({"id": 43})).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["operation"], "palace_delete");
        assert_eq!(first["params"]["id"], 42);
        assert!(first["timestamp"].as_str().unwrap().ends_with('Z'));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn log_without_path_is_silent_noop() {
        let wal = Wal::with_path(None);
        wal.log("op", &json!({"k": "v"})); // must not panic
    }
}
