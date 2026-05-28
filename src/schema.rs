use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// category / wing / room / hall are deliberately free-text — the palace
// taxonomy is whatever the user makes of it. Conventional values are suggested
// in the `palace_store` tool-arg descriptions but never enforced. Light
// validation (non-empty, length-capped, trimmed) lives in `mcp.rs::validate_tag`.

/// The payload we write to Qdrant. Mirrors the palace schema established 2026-04-19,
/// extended 2026-04-24 with temporal-validity fields (valid_from / valid_until /
/// supersedes / superseded_by / superseded_reason) per the `palace_supersede` RFC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Payload {
    pub category: String,
    pub wing: String,
    pub room: String,
    pub hall: String,
    pub text: String,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    /// RFC3339 — when this memory started being authoritative. Defaults to
    /// `timestamp` at store time; only differs when callers want explicit
    /// temporal semantics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<String>,
    /// RFC3339 — when this memory stopped being authoritative. Absent means
    /// "still current." Set by `palace_supersede`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<String>,
    /// Point IDs that this memory replaces. Set on the new point during
    /// `palace_supersede`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<Vec<u64>>,
    /// Point ID of the successor. Set on the old point during `palace_supersede`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<u64>,
    /// Human explanation for the supersession. Set on the old point.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_reason: Option<String>,
}

/// A point as returned to MCP callers — the structured view of a palace memory.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Memory {
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    pub text: String,
    pub category: String,
    pub wing: String,
    pub room: String,
    pub hall: String,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_reason: Option<String>,
}

/// A point as returned by the `GET /export` HTTP endpoint — includes the vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportPoint {
    pub id: u64,
    pub text: String,
    pub category: String,
    pub wing: String,
    pub room: String,
    pub hall: String,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}
