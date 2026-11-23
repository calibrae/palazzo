//! MCP argument types for the image-memory tools. Kept out of `images.rs`
//! (fastembed-only) so the tool signatures compile in every build; without
//! PALAZZO_VISION the tools are removed from the router at startup.
// The ollama build has no image palace, so nothing reads these fields there.
#![cfg_attr(not(feature = "fastembed"), allow(dead_code))]

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, rmcp::schemars::JsonSchema)]
pub struct StoreImageArgs {
    /// The image: raw base64 or a `data:image/...;base64,` URL. PNG, JPEG, WebP
    /// or GIF; max 8 MiB and 40 MP. Large images cost a lot of context as tool
    /// arguments — for anything beyond a small screenshot, use the sibling
    /// `POST /images` HTTP endpoint instead.
    pub image: String,
    /// Where the image lives (URL, path, document reference). Stored as-is —
    /// palazzo keeps the vector and metadata, never the image bytes.
    pub source: String,
    /// Free-text, same conventions as palace_store.
    pub category: String,
    pub wing: String,
    pub room: String,
    pub hall: String,
    /// Optional human description. Stored and returned; not embedded.
    #[serde(default)]
    pub caption: Option<String>,
    /// Optional original time of the image (RFC3339 second-precision UTC).
    #[serde(default)]
    pub event_time: Option<String>,
    #[serde(default)]
    pub session: Option<String>,
}

#[derive(Debug, Deserialize, rmcp::schemars::JsonSchema)]
pub struct FindImagesArgs {
    /// Natural-language description of the image you are looking for. English
    /// queries match noticeably better than French.
    pub query: String,
    /// Max results. Default 5, max 20.
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub wing: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub room: Option<String>,
    #[serde(default)]
    pub hall: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    /// Inclusive lower bound on `time_field` (RFC3339 second-precision UTC).
    #[serde(default)]
    pub since: Option<String>,
    /// Inclusive upper bound on `time_field` (RFC3339 second-precision UTC).
    #[serde(default)]
    pub until: Option<String>,
    /// "timestamp" (default, when stored) or "event_time" (when taken).
    #[serde(default)]
    pub time_field: Option<String>,
}

#[derive(Debug, Deserialize, rmcp::schemars::JsonSchema)]
pub struct DeleteImagesArgs {
    /// Image point IDs to hard-delete. 1..=100.
    pub ids: Vec<u64>,
    /// Why — WAL-logged as the only audit trail. Required.
    pub reason: String,
    /// Must be `true`: asserts the human operator approved this delete.
    pub confirm: bool,
}
