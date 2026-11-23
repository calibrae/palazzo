//! Service `/docs` API: documents managed by the caller's own stable key.
//!
//! Meant for services, not MCP agents. Tokens are minted offline by whoever
//! holds the private key (`palazzo service-token mint`):
//!
//! - Ed25519-signed service tokens (see `svctoken`), each bound to ONE wing and a
//!   mode: `rw` (writes + reads) or `ro` (reads). A service can only write to and
//!   read from its own wing. No trusted public key → 404 `not_enabled`.
//! - Keyed identity: `external_id` is unique among a wing's *current* points
//!   (current = not superseded). No cosine/text dedup on this path.
//!   `PUT` with the same key: same text + same metadata → `unchanged`; same text,
//!   new metadata → `updated` (payload rewritten, no re-embed); new text →
//!   `replaced` (new point, old one superseded, kept for audit).
//! - `DELETE` hard-deletes every point of the key (current + superseded
//!   history), each WAL-logged with its full payload first.
//!
//! Single-replica assumption: concurrent writes to one key are serialised by an
//! in-process lock.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::embed::Embedder;
use crate::qdrant::Qdrant;
use crate::schema::Payload;
use crate::svctoken::{Verdict, Verifier};
use crate::util::{now_rfc3339, parse_rfc3339};
use crate::wal::Wal;

const MAX_TEXT_BYTES: usize = 32 * 1024;
const MAX_EXTERNAL_ID: usize = 128;
const MAX_META_BYTES: usize = 2048;
const MAX_META_KEY: usize = 64;
pub const MAX_BATCH: usize = 256;
/// Body cap for the /docs routes: a full batch of max-size texts plus metadata.
pub const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
const LOCK_SHARDS: usize = 64;
/// Path segments taken by static routes: can't be used as an external_id.
const RESERVED_IDS: [&str; 3] = ["batch", "delete", "search"];

// ---------------------------------------------------------------------------
// Errors: every /docs response that isn't 2xx is
// {"error": {"code": "<code>", "message": "<human text>"}}.

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }
    pub fn bad(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "bad_request", message)
    }
    fn internal(e: impl std::fmt::Display) -> Self {
        tracing::error!("docs: {e:#}");
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            format!("{e:#}"),
        )
    }
    fn body(&self) -> Value {
        json!({ "error": { "code": self.code, "message": self.message } })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}

// ---------------------------------------------------------------------------
// Caller identity (from a verified service token).

#[derive(Debug, Clone, PartialEq)]
pub struct Service {
    pub name: String,
    pub wing: String,
    pub rw: bool,
}

// ---------------------------------------------------------------------------
// Request shapes.

/// Body of `PUT /docs/{external_id}`. Unknown fields are rejected, and the wing
/// can't be set: it's the token's.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocIn {
    pub text: String,
    pub room: String,
    pub hall: String,
    pub category: String,
    #[serde(default)]
    pub meta: Option<BTreeMap<String, String>>,
    #[serde(default)]
    pub valid_from: Option<String>,
    #[serde(default)]
    pub valid_until: Option<String>,
    #[serde(default)]
    pub event_time: Option<String>,
}

/// One item of `POST /docs/batch` (DocIn + its key). Spelled out rather than
/// flattened: serde can't combine `flatten` with `deny_unknown_fields`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchItem {
    external_id: String,
    text: String,
    room: String,
    hall: String,
    category: String,
    #[serde(default)]
    meta: Option<BTreeMap<String, String>>,
    #[serde(default)]
    valid_from: Option<String>,
    #[serde(default)]
    valid_until: Option<String>,
    #[serde(default)]
    event_time: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchBody {
    items: Vec<BatchItem>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchDeleteBody {
    external_ids: Vec<String>,
    #[serde(default)]
    reason: Option<String>,
}

/// Body of `POST /docs/search`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchBody {
    query: String,
    /// Mandatory. `"<room>:<hall>"` or `"<room>:*"` (any hall of that room).
    scope: Vec<String>,
    #[serde(default)]
    limit: Option<u32>,
    /// Optional category filter (any-of).
    #[serde(default)]
    category: Option<Vec<String>>,
}

const MAX_SCOPE: usize = 1000;
const MAX_QUERY_BYTES: usize = 4096;

/// Parse scope pairs and group them by room: `{room: None}` = any hall,
/// `{room: Some(halls)}` = those halls only.
fn parse_scope(scope: &[String]) -> Result<BTreeMap<String, Option<Vec<String>>>, ApiError> {
    if scope.is_empty() {
        return Err(ApiError::bad("scope is required and must not be empty"));
    }
    if scope.len() > MAX_SCOPE {
        return Err(ApiError::bad(format!(
            "scope too large: {} pairs (max {MAX_SCOPE})",
            scope.len()
        )));
    }
    let mut out: BTreeMap<String, Option<Vec<String>>> = BTreeMap::new();
    for pair in scope {
        let (room, hall) = pair.split_once(':').ok_or_else(|| {
            ApiError::bad(format!(
                "scope entry {pair:?} must be \"<room>:<hall>\" or \"<room>:*\""
            ))
        })?;
        let room = crate::mcp::validate_tag("scope room", room)
            .map_err(|e| ApiError::bad(e.to_string()))?;
        let hall = hall.trim();
        if hall == "*" {
            out.insert(room, None);
            continue;
        }
        let hall = crate::mcp::validate_tag("scope hall", hall)
            .map_err(|e| ApiError::bad(e.to_string()))?;
        // A room already opened to any hall (None) stays open.
        if let Some(halls) = out.entry(room).or_insert_with(|| Some(Vec::new()))
            && !halls.contains(&hall)
        {
            halls.push(hall);
        }
    }
    Ok(out)
}

/// The full Qdrant filter for a scoped search: the token's wing, any of the
/// scope pairs, optional categories, current truth at `now`.
fn search_filter(
    wing: &str,
    scope: &BTreeMap<String, Option<Vec<String>>>,
    categories: &[String],
    now: &str,
) -> Value {
    let should: Vec<Value> = scope
        .iter()
        .map(|(room, halls)| {
            let mut must = vec![json!({"key": "room", "match": {"value": room}})];
            if let Some(h) = halls {
                must.push(json!({"key": "hall", "match": {"any": h}}));
            }
            json!({ "must": must })
        })
        .collect();
    let mut must = vec![
        json!({"key": "wing", "match": {"value": wing}}),
        json!({"should": should}),
    ];
    if !categories.is_empty() {
        must.push(json!({"key": "category", "match": {"any": categories}}));
    }
    json!({ "must": must, "must_not": crate::qdrant::current_truth_must_not(now) })
}

#[derive(Debug, Deserialize)]
pub struct ReasonQuery {
    #[serde(default)]
    reason: Option<String>,
}

/// A validated document, labels trimmed.
#[derive(Debug, Clone, PartialEq)]
struct Doc {
    text: String,
    room: String,
    hall: String,
    category: String,
    meta: Option<BTreeMap<String, String>>,
    valid_from: Option<String>,
    valid_until: Option<String>,
    event_time: Option<String>,
}

fn validate_external_id(raw: &str) -> Result<String, ApiError> {
    let s = raw.trim();
    if s.is_empty() || s.len() > MAX_EXTERNAL_ID {
        return Err(ApiError::bad(format!(
            "external_id must be 1-{MAX_EXTERNAL_ID} bytes"
        )));
    }
    if !s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "._:@-".contains(c))
    {
        return Err(ApiError::bad(
            "external_id may only contain [A-Za-z0-9._:@-]",
        ));
    }
    if RESERVED_IDS.contains(&s) {
        return Err(ApiError::bad(format!("external_id {s:?} is reserved")));
    }
    Ok(s.to_string())
}

fn validate_time(field: &str, v: &Option<String>) -> Result<Option<i64>, ApiError> {
    match v {
        None => Ok(None),
        Some(s) => parse_rfc3339(s).map(Some).ok_or_else(|| {
            ApiError::bad(format!(
                "{field} must be RFC3339 second-precision UTC (e.g. 2026-04-20T00:00:00Z), got {s:?}"
            ))
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_doc(
    text: String,
    room: &str,
    hall: &str,
    category: &str,
    meta: Option<BTreeMap<String, String>>,
    valid_from: Option<String>,
    valid_until: Option<String>,
    event_time: Option<String>,
) -> Result<Doc, ApiError> {
    if text.trim().is_empty() {
        return Err(ApiError::bad("text is empty"));
    }
    if text.len() > MAX_TEXT_BYTES {
        return Err(ApiError::bad(format!(
            "text too large: {} bytes (max {MAX_TEXT_BYTES})",
            text.len()
        )));
    }
    let tag =
        |f: &str, v: &str| crate::mcp::validate_tag(f, v).map_err(|e| ApiError::bad(e.to_string()));
    let room = tag("room", room)?;
    let hall = tag("hall", hall)?;
    let category = tag("category", category)?;
    let meta = match meta {
        Some(m) if !m.is_empty() => {
            for k in m.keys() {
                if k.is_empty() || k.len() > MAX_META_KEY {
                    return Err(ApiError::bad(format!(
                        "meta keys must be 1-{MAX_META_KEY} bytes, got {k:?}"
                    )));
                }
            }
            let size = serde_json::to_string(&m)
                .map(|s| s.len())
                .unwrap_or(usize::MAX);
            if size > MAX_META_BYTES {
                return Err(ApiError::bad(format!(
                    "meta too large: {size} bytes as JSON (max {MAX_META_BYTES})"
                )));
            }
            Some(m)
        }
        _ => None,
    };
    let from = validate_time("valid_from", &valid_from)?;
    let until = validate_time("valid_until", &valid_until)?;
    validate_time("event_time", &event_time)?;
    if let (Some(f), Some(u)) = (from, until)
        && u <= f
    {
        return Err(ApiError::bad("valid_until must be after valid_from"));
    }
    Ok(Doc {
        text,
        room,
        hall,
        category,
        meta,
        valid_from,
        valid_until,
        event_time,
    })
}

// ---------------------------------------------------------------------------
// Responses.

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DocStatus {
    Created,
    Unchanged,
    Updated,
    Replaced,
}

#[derive(Debug, Serialize)]
pub struct DocResult {
    pub external_id: String,
    pub id: u64,
    pub status: DocStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replaced_id: Option<u64>,
}

/// A stored point as returned by GET: `id` plus the payload keys (same keys as
/// a `/find` hit, minus `score`; optional keys absent when unset).
fn point_json(id: u64, payload: &Payload) -> Value {
    let mut v = serde_json::to_value(payload).unwrap_or_else(|_| json!({}));
    if let Some(o) = v.as_object_mut() {
        o.insert("id".into(), json!(id));
    }
    v
}

// ---------------------------------------------------------------------------
// Core.

#[derive(Clone)]
pub struct DocsState {
    embedder: Arc<Embedder>,
    qdrant: Arc<Qdrant>,
    wal: Arc<Wal>,
    verifier: Arc<Verifier>,
    locks: Arc<Vec<Mutex<()>>>,
}

fn key_filter(wing: &str, external_id: &str, current_only: bool) -> Value {
    let mut must = vec![
        json!({"key": "wing", "match": {"value": wing}}),
        json!({"key": "external_id", "match": {"value": external_id}}),
    ];
    if current_only {
        // Current = never superseded. `is_empty` matches a missing key.
        must.push(json!({"is_empty": {"key": "superseded_by"}}));
    }
    json!({ "must": must })
}

impl DocsState {
    pub fn new(
        embedder: Arc<Embedder>,
        qdrant: Arc<Qdrant>,
        wal: Arc<Wal>,
        verifier: Arc<Verifier>,
    ) -> Self {
        Self {
            embedder,
            qdrant,
            wal,
            verifier,
            locks: Arc::new((0..LOCK_SHARDS).map(|_| Mutex::new(())).collect()),
        }
    }

    fn lock_for(&self, wing: &str, external_id: &str) -> &Mutex<()> {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (wing, external_id).hash(&mut h);
        &self.locks[(h.finish() as usize) % LOCK_SHARDS]
    }

    fn auth(&self, headers: &HeaderMap, need_rw: bool) -> Result<Service, ApiError> {
        // Off on this instance: look like a missing route, whatever the caller sent.
        if !self.verifier.is_enabled() {
            return Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "not_enabled",
                "the service API is not enabled on this instance",
            ));
        }
        let raw = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let token = raw
            .split_once(' ')
            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
            .map(|(_, t)| t.trim())
            .filter(|t| !t.is_empty())
            .ok_or_else(|| {
                ApiError::new(
                    StatusCode::UNAUTHORIZED,
                    "unauthorized",
                    "missing bearer token",
                )
            })?;
        let claims = match self.verifier.check_service(token) {
            Verdict::Valid(c) => c,
            Verdict::NotEnabled => {
                return Err(ApiError::new(
                    StatusCode::NOT_FOUND,
                    "not_enabled",
                    "the service API is not enabled on this instance",
                ));
            }
            Verdict::Invalid => {
                return Err(ApiError::new(
                    StatusCode::UNAUTHORIZED,
                    "unauthorized",
                    "invalid, expired or revoked token",
                ));
            }
        };
        let svc = Service {
            name: claims.sub,
            wing: claims.wing.unwrap_or_default(),
            rw: claims.mode.as_deref() == Some("rw"),
        };
        if need_rw && !svc.rw {
            return Err(ApiError::new(
                StatusCode::FORBIDDEN,
                "forbidden",
                "this token is read-only",
            ));
        }
        Ok(svc)
    }

    async fn current(
        &self,
        wing: &str,
        external_id: &str,
    ) -> Result<Vec<(u64, Payload)>, ApiError> {
        let mut pts = self
            .qdrant
            .scroll_filter(key_filter(wing, external_id, true), 16)
            .await
            .map_err(ApiError::internal)?;
        pts.sort_by_key(|(id, _)| std::cmp::Reverse(*id)); // newest first (ids are monotonic)
        Ok(pts)
    }

    fn payload(
        svc: &Service,
        external_id: &str,
        doc: &Doc,
        supersedes: Option<Vec<u64>>,
    ) -> Payload {
        Payload {
            category: doc.category.clone(),
            wing: svc.wing.clone(),
            room: doc.room.clone(),
            hall: doc.hall.clone(),
            text: doc.text.clone(),
            timestamp: now_rfc3339(),
            event_time: doc.event_time.clone(),
            session: None,
            source_file: None,
            author: Some(format!("service:{}", svc.name).to_lowercase()),
            valid_from: doc.valid_from.clone(),
            valid_until: doc.valid_until.clone(),
            supersedes,
            superseded_by: None,
            superseded_reason: None,
            external_id: Some(external_id.to_string()),
            meta: doc.meta.clone(),
        }
    }

    fn same_metadata(p: &Payload, doc: &Doc) -> bool {
        p.room == doc.room
            && p.hall == doc.hall
            && p.category == doc.category
            && p.meta == doc.meta
            && p.valid_from == doc.valid_from
            && p.valid_until == doc.valid_until
            && p.event_time == doc.event_time
    }

    async fn supersede(&self, old: u64, new: u64, reason: String) -> Result<(), ApiError> {
        self.qdrant
            .set_payload(
                old,
                json!({ "valid_until": now_rfc3339(), "superseded_by": new, "superseded_reason": reason }),
            )
            .await
            .map_err(ApiError::internal)
    }

    async fn upsert(
        &self,
        svc: &Service,
        external_id: &str,
        doc: Doc,
    ) -> Result<DocResult, ApiError> {
        let _guard = self.lock_for(&svc.wing, external_id).lock().await;
        let current = self.current(&svc.wing, external_id).await?;

        // Repair: more than one current point for a key should never happen
        // (it would take a write outside this lock). Keep the newest.
        if let Some((keep, _)) = current.first() {
            for (extra, _) in current.iter().skip(1) {
                tracing::warn!(wing = %svc.wing, external_id, extra, keep, "docs: duplicate current point, superseding");
                self.supersede(*extra, *keep, "duplicate current point (repair)".into())
                    .await?;
            }
        }

        let result = match current.into_iter().next() {
            Some((cur_id, cur)) if cur.text == doc.text => {
                if Self::same_metadata(&cur, &doc) {
                    DocResult {
                        external_id: external_id.into(),
                        id: cur_id,
                        status: DocStatus::Unchanged,
                        replaced_id: None,
                    }
                } else {
                    let payload = Self::payload(svc, external_id, &doc, cur.supersedes.clone());
                    self.wal.log(
                        "docs_update",
                        &json!({"id": cur_id, "wing": svc.wing, "external_id": external_id, "service": svc.name}),
                    );
                    self.qdrant
                        .overwrite_payload(cur_id, &payload)
                        .await
                        .map_err(ApiError::internal)?;
                    DocResult {
                        external_id: external_id.into(),
                        id: cur_id,
                        status: DocStatus::Updated,
                        replaced_id: None,
                    }
                }
            }
            previous => {
                let vector = self
                    .embedder
                    .embed(&doc.text)
                    .await
                    .map_err(ApiError::internal)?;
                let id = crate::mcp::new_id();
                let replaced = previous.map(|(old, _)| old);
                let payload = Self::payload(svc, external_id, &doc, replaced.map(|o| vec![o]));
                self.wal.log(
                    if replaced.is_some() { "docs_replace" } else { "docs_create" },
                    &json!({"id": id, "replaces": replaced, "wing": svc.wing, "external_id": external_id,
                            "room": doc.room, "hall": doc.hall, "category": doc.category, "service": svc.name}),
                );
                // New point first, then retire the old one: never zero current versions.
                self.qdrant
                    .upsert(id, vector, payload)
                    .await
                    .map_err(ApiError::internal)?;
                if let Some(old) = replaced {
                    self.supersede(
                        old,
                        id,
                        format!("replaced via /docs by service {}", svc.name),
                    )
                    .await?;
                }
                DocResult {
                    external_id: external_id.into(),
                    id,
                    status: if replaced.is_some() {
                        DocStatus::Replaced
                    } else {
                        DocStatus::Created
                    },
                    replaced_id: replaced,
                }
            }
        };
        metrics::counter!("palazzo_docs_ops_total", "op" => "upsert", "status" => format!("{:?}", result.status).to_lowercase())
            .increment(1);
        Ok(result)
    }

    /// Hard-delete every point of the key (current + history). Returns the ids.
    async fn delete_key(
        &self,
        svc: &Service,
        external_id: &str,
        reason: &str,
    ) -> Result<Vec<u64>, ApiError> {
        let _guard = self.lock_for(&svc.wing, external_id).lock().await;
        let pts = self
            .qdrant
            .scroll_filter(key_filter(&svc.wing, external_id, false), 1000)
            .await
            .map_err(ApiError::internal)?;
        if pts.is_empty() {
            return Ok(vec![]);
        }
        for (id, payload) in &pts {
            self.wal
                .log_strict(
                    "docs_delete",
                    &json!({"id": id, "wing": svc.wing, "external_id": external_id, "service": svc.name,
                            "reason": reason, "payload": payload}),
                )
                .map_err(|e| {
                    ApiError::internal(format!("WAL append failed, refusing to delete without an audit trail: {e:#}"))
                })?;
        }
        let ids: Vec<u64> = pts.iter().map(|(id, _)| *id).collect();
        self.qdrant.delete(&ids).await.map_err(ApiError::internal)?;
        metrics::counter!("palazzo_docs_ops_total", "op" => "delete", "status" => "deleted")
            .increment(1);
        Ok(ids)
    }
}

fn parse_body<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, ApiError> {
    serde_json::from_slice(body).map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))
}

fn reason_or_default(r: Option<String>, svc: &Service) -> String {
    r.map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("deleted via /docs by service {}", svc.name))
}

// ---------------------------------------------------------------------------
// Handlers.

async fn put_doc(
    State(st): State<DocsState>,
    Path(external_id): Path<String>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<DocResult>, ApiError> {
    let svc = st.auth(&headers, true)?;
    let external_id = validate_external_id(&external_id)?;
    let d: DocIn = parse_body(&body)?;
    let doc = validate_doc(
        d.text,
        &d.room,
        &d.hall,
        &d.category,
        d.meta,
        d.valid_from,
        d.valid_until,
        d.event_time,
    )?;
    Ok(Json(st.upsert(&svc, &external_id, doc).await?))
}

async fn get_doc(
    State(st): State<DocsState>,
    Path(external_id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let svc = st.auth(&headers, false)?;
    let external_id = validate_external_id(&external_id)?;
    let current = st.current(&svc.wing, &external_id).await?;
    let (id, payload) = current.first().ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            format!("no current document for {external_id:?}"),
        )
    })?;
    Ok(Json(point_json(*id, payload)))
}

async fn delete_doc(
    State(st): State<DocsState>,
    Path(external_id): Path<String>,
    Query(q): Query<ReasonQuery>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let svc = st.auth(&headers, true)?;
    let external_id = validate_external_id(&external_id)?;
    let reason = reason_or_default(q.reason, &svc);
    let ids = st.delete_key(&svc, &external_id, &reason).await?;
    if ids.is_empty() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            format!("no document for {external_id:?}"),
        ));
    }
    Ok(Json(
        json!({ "external_id": external_id, "deleted_ids": ids }),
    ))
}

async fn batch(
    State(st): State<DocsState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<Value>, ApiError> {
    let svc = st.auth(&headers, true)?;
    let b: BatchBody = parse_body(&body)?;
    if b.items.is_empty() {
        return Err(ApiError::bad("items is empty"));
    }
    if b.items.len() > MAX_BATCH {
        return Err(ApiError::bad(format!(
            "too many items: {} (max {MAX_BATCH})",
            b.items.len()
        )));
    }
    let mut items = Vec::with_capacity(b.items.len());
    let mut counts: BTreeMap<&str, u32> = [
        ("created", 0),
        ("unchanged", 0),
        ("updated", 0),
        ("replaced", 0),
        ("failed", 0),
    ]
    .into();
    // Sequential, in order: a key repeated in one batch ends with its last item.
    for (index, it) in b.items.into_iter().enumerate() {
        let outcome = async {
            let ext = validate_external_id(&it.external_id)?;
            let doc = validate_doc(
                it.text,
                &it.room,
                &it.hall,
                &it.category,
                it.meta,
                it.valid_from,
                it.valid_until,
                it.event_time,
            )?;
            st.upsert(&svc, &ext, doc).await
        }
        .await;
        match outcome {
            Ok(r) => {
                *counts
                    .entry(match r.status {
                        DocStatus::Created => "created",
                        DocStatus::Unchanged => "unchanged",
                        DocStatus::Updated => "updated",
                        DocStatus::Replaced => "replaced",
                    })
                    .or_default() += 1;
                let mut v = serde_json::to_value(&r).unwrap_or_default();
                v["index"] = json!(index);
                v["ok"] = json!(true);
                items.push(v);
            }
            Err(e) => {
                *counts.entry("failed").or_default() += 1;
                items.push(json!({ "index": index, "ok": false, "external_id": it.external_id, "error": e.body()["error"] }));
            }
        }
    }
    Ok(Json(json!({ "items": items, "counts": counts })))
}

async fn batch_delete(
    State(st): State<DocsState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<Value>, ApiError> {
    let svc = st.auth(&headers, true)?;
    let b: BatchDeleteBody = parse_body(&body)?;
    if b.external_ids.is_empty() {
        return Err(ApiError::bad("external_ids is empty"));
    }
    if b.external_ids.len() > MAX_BATCH {
        return Err(ApiError::bad(format!(
            "too many external_ids: {} (max {MAX_BATCH})",
            b.external_ids.len()
        )));
    }
    let keys: Vec<String> = b
        .external_ids
        .iter()
        .map(|k| validate_external_id(k))
        .collect::<Result<_, _>>()?;
    let reason = reason_or_default(b.reason, &svc);
    let (mut deleted, mut missing) = (Vec::new(), Vec::new());
    for k in keys {
        let ids = st.delete_key(&svc, &k, &reason).await?;
        if ids.is_empty() {
            missing.push(json!(k));
        } else {
            deleted.push(json!({ "external_id": k, "ids": ids }));
        }
    }
    Ok(Json(json!({ "deleted": deleted, "missing": missing })))
}

async fn search(
    State(st): State<DocsState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<Vec<crate::schema::Memory>>, ApiError> {
    let svc = st.auth(&headers, false)?;
    let b: SearchBody = parse_body(&body)?;
    if b.query.trim().is_empty() {
        return Err(ApiError::bad("query is empty"));
    }
    if b.query.len() > MAX_QUERY_BYTES {
        return Err(ApiError::bad(format!(
            "query too large: {} bytes (max {MAX_QUERY_BYTES})",
            b.query.len()
        )));
    }
    let scope = parse_scope(&b.scope)?;
    let categories: Vec<String> = b
        .category
        .unwrap_or_default()
        .iter()
        .map(|c| crate::mcp::validate_tag("category", c).map_err(|e| ApiError::bad(e.to_string())))
        .collect::<Result<_, _>>()?;
    let limit = b.limit.unwrap_or(10).clamp(1, 20);
    // Same embedding as palace_find (prefix-free nomic).
    let vector = st
        .embedder
        .embed(b.query.trim())
        .await
        .map_err(ApiError::internal)?;
    let filter = search_filter(&svc.wing, &scope, &categories, &now_rfc3339());
    let hits = st
        .qdrant
        .search_filter(vector, limit, filter)
        .await
        .map_err(ApiError::internal)?;
    metrics::counter!("palazzo_docs_ops_total", "op" => "search", "status" => "ok").increment(1);
    Ok(Json(hits))
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|raw| raw.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, t)| t.trim())
        .filter(|t| !t.is_empty())
}

fn admin_auth(st: &DocsState, headers: &HeaderMap) -> Result<(), ApiError> {
    let token = bearer(headers).ok_or_else(|| {
        ApiError::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "missing bearer token",
        )
    })?;
    match st.verifier.check_admin(token) {
        Verdict::Valid(_) => Ok(()),
        Verdict::NotEnabled => Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_enabled",
            "the service API is not enabled on this instance",
        )),
        Verdict::Invalid => Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "invalid, expired or revoked admin token",
        )),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RevokeBody {
    jti: String,
}

/// `POST /admin/service-tokens/revoke`: admin token; jti → revocation list.
async fn admin_revoke(
    State(st): State<DocsState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<Value>, ApiError> {
    admin_auth(&st, &headers)?;
    let b: RevokeBody = parse_body(&body)?;
    st.verifier
        .revoke(&b.jti)
        .map_err(|e| ApiError::bad(format!("{e:#}")))?;
    st.wal
        .log("service_token_revoke", &json!({ "jti": b.jti.trim() }));
    tracing::info!(jti = %b.jti.trim(), "service token revoked");
    Ok(Json(json!({ "revoked": b.jti.trim() })))
}

/// `GET /admin/service-tokens/revoked`: admin token; the revocation list.
async fn admin_revoked(
    State(st): State<DocsState>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    admin_auth(&st, &headers)?;
    Ok(Json(json!({ "revoked": st.verifier.revoked_list() })))
}

/// The `/docs` + `/admin/service-tokens` routes. Always mounted; they answer
/// 404 `not_enabled` when the build/env trusts no public key.
pub fn router(state: DocsState) -> axum::Router {
    use axum::routing::{post, put};
    axum::Router::new()
        .route("/docs/batch", post(batch))
        .route("/docs/delete", post(batch_delete))
        .route("/docs/search", post(search))
        .route("/admin/service-tokens/revoke", post(admin_revoke))
        .route(
            "/admin/service-tokens/revoked",
            axum::routing::get(admin_revoked),
        )
        .route(
            "/docs/{external_id}",
            put(put_doc).get(get_doc).delete(delete_doc),
        )
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testmock::MockQdrant;
    use std::path::PathBuf;

    /// A fresh keypair + rw/ro tokens signed with it.
    struct Fx {
        key: crate::svctoken::SigningKey,
        rw: String,
        ro: String,
    }

    fn fx() -> Fx {
        let key = crate::svctoken::SigningKey::generate();
        let (rw, _) = key
            .mint_service("demo-svc", "demo-svc", "rw", None, Some(1))
            .unwrap();
        let (ro, _) = key
            .mint_service("demo-svc-ro", "demo-svc", "ro", None, Some(1))
            .unwrap();
        Fx { key, rw, ro }
    }

    fn verifier_for(key: Option<&crate::svctoken::SigningKey>) -> Verifier {
        let keys = key
            .map(|k| crate::svctoken::parse_pubkeys(&k.public_b64()).unwrap())
            .unwrap_or_default();
        let revoked = std::env::temp_dir().join(format!(
            "palazzo-docs-revoked-{}-{}",
            std::process::id(),
            crate::mcp::new_id()
        ));
        Verifier::new(keys, "test", revoked)
    }

    fn temp_wal(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "palazzo-docs-wal-{tag}-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn state_with(url: &str, wal: Option<PathBuf>, verifier: Verifier) -> DocsState {
        DocsState::new(
            Arc::new(Embedder::fake()),
            Arc::new(Qdrant::with_api_key(url, "test", None)),
            Arc::new(Wal::with_path(wal)),
            Arc::new(verifier),
        )
    }

    /// Core-logic tests call upsert/delete_key directly, so no key is needed.
    fn state(url: &str, wal: Option<PathBuf>) -> DocsState {
        state_with(url, wal, verifier_for(None))
    }

    fn svc() -> Service {
        Service {
            name: "demo-svc".into(),
            wing: "demo-svc".into(),
            rw: true,
        }
    }

    fn doc(text: &str) -> Doc {
        validate_doc(
            text.into(),
            "11111111-2222-3333-4444-555555555555",
            "billing",
            "sheet",
            Some(BTreeMap::from([("revision".to_string(), "3".to_string())])),
            Some("2026-01-01T00:00:00Z".into()),
            None,
            None,
        )
        .unwrap()
    }

    fn stored(text: &str, meta_rev: &str) -> Value {
        json!({"id": 42, "payload": {
            "text": text, "wing": "demo-svc", "room": "11111111-2222-3333-4444-555555555555",
            "hall": "billing", "category": "sheet", "timestamp": "2026-11-01T00:00:00Z",
            "valid_from": "2026-01-01T00:00:00Z", "external_id": "k1",
            "meta": {"revision": meta_rev}}})
    }

    // ---------- tokens ----------

    #[test]
    fn auth_rules() {
        let f = fx();
        let st = state_with("http://127.0.0.1:9", None, verifier_for(Some(&f.key)));
        let h = |v: &str| {
            let mut m = HeaderMap::new();
            m.insert(axum::http::header::AUTHORIZATION, v.parse().unwrap());
            m
        };
        let code = |r: Result<Service, ApiError>| r.unwrap_err().status;
        assert_eq!(
            code(st.auth(&HeaderMap::new(), false)),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            code(st.auth(&h("Bearer wrong"), false)),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            code(st.auth(&h(&format!("Basic {}", f.rw)), false)),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            code(st.auth(&h(&format!("Bearer {}", f.ro)), true)),
            StatusCode::FORBIDDEN
        );
        let ro = st.auth(&h(&format!("Bearer {}", f.ro)), false).unwrap();
        assert_eq!((ro.wing.as_str(), ro.rw), ("demo-svc", false));
        assert!(st.auth(&h(&format!("bearer {}", f.rw)), true).unwrap().rw);
        // a token signed by another key is invalid here
        let other = fx();
        assert_eq!(
            code(st.auth(&h(&format!("Bearer {}", other.rw)), true)),
            StatusCode::UNAUTHORIZED
        );
    }

    #[test]
    fn no_key_file_means_not_enabled() {
        let st = state_with("http://127.0.0.1:9", None, verifier_for(None));
        let e = st.auth(&HeaderMap::new(), true).unwrap_err();
        assert_eq!((e.status, e.code), (StatusCode::NOT_FOUND, "not_enabled"));
    }

    // ---------- validation ----------

    #[test]
    fn external_id_and_doc_validation() {
        assert!(validate_external_id("6f1c2a9e-0b1d-4c55-9a77-2f3e4d5c6b7a").is_ok());
        assert!(validate_external_id("").is_err());
        assert!(validate_external_id("has space").is_err());
        assert!(validate_external_id("a/b").is_err());
        assert!(validate_external_id("batch").is_err());
        assert!(validate_external_id(&"x".repeat(129)).is_err());
        let v = |from: Option<&str>, until: Option<&str>| {
            validate_doc(
                "t".into(),
                "r",
                "h",
                "c",
                None,
                from.map(Into::into),
                until.map(Into::into),
                None,
            )
        };
        assert!(v(Some("2026-01-01T00:00:00Z"), Some("2025-01-01T00:00:00Z")).is_err());
        assert!(v(Some("2026-01-01T00:00:00Z"), Some("2027-01-01T00:00:00Z")).is_ok());
        assert!(v(Some("yesterday"), None).is_err());
        let big = BTreeMap::from([("k".to_string(), "x".repeat(MAX_META_BYTES))]);
        assert!(validate_doc("t".into(), "r", "h", "c", Some(big), None, None, None).is_err());
        assert!(validate_doc(" ".into(), "r", "h", "c", None, None, None, None).is_err());
        // wing is not a field: unknown fields are rejected
        assert!(
            serde_json::from_str::<DocIn>(
                r#"{"text":"t","room":"r","hall":"h","category":"c","wing":"other"}"#
            )
            .is_err()
        );
    }

    // ---------- upsert ----------

    #[tokio::test]
    async fn create_forces_wing_and_stamps_key() {
        let q = MockQdrant::start().await;
        let wal = temp_wal("create");
        let r = state(&q.url, Some(wal.clone()))
            .upsert(&svc(), "k1", doc("hello"))
            .await
            .unwrap();
        assert_eq!(r.status, DocStatus::Created);
        let scroll = &q.requests_for("POST /points/scroll")[0];
        let must = scroll["filter"]["must"].as_array().unwrap();
        assert!(
            must.iter()
                .any(|c| c["key"] == "external_id" && c["match"]["value"] == "k1")
        );
        assert!(must.iter().any(|c| c["is_empty"]["key"] == "superseded_by"));
        let pl = &q.requests_for("PUT /points")[0]["points"][0]["payload"];
        assert_eq!(pl["wing"], "demo-svc");
        assert_eq!(pl["external_id"], "k1");
        assert_eq!(pl["meta"]["revision"], "3");
        assert_eq!(pl["author"], "service:demo-svc");
        assert!(
            std::fs::read_to_string(&wal)
                .unwrap()
                .contains("docs_create")
        );
    }

    #[tokio::test]
    async fn same_text_same_meta_is_unchanged() {
        let q = MockQdrant::start().await;
        q.push(
            "POST /points/scroll",
            json!({"result": {"points": [stored("hello", "3")], "next_page_offset": null}}),
        );
        let r = state(&q.url, None)
            .upsert(&svc(), "k1", doc("hello"))
            .await
            .unwrap();
        assert_eq!((r.status, r.id), (DocStatus::Unchanged, 42));
        assert!(q.requests_for("PUT /points").is_empty());
        assert!(q.requests_for("PUT /points/payload").is_empty());
    }

    #[tokio::test]
    async fn same_text_new_meta_is_updated_without_reembed() {
        let q = MockQdrant::start().await;
        q.push(
            "POST /points/scroll",
            json!({"result": {"points": [stored("hello", "2")], "next_page_offset": null}}),
        );
        let r = state(&q.url, None)
            .upsert(&svc(), "k1", doc("hello"))
            .await
            .unwrap();
        assert_eq!((r.status, r.id), (DocStatus::Updated, 42));
        assert!(
            q.requests_for("PUT /points").is_empty(),
            "no new point / no re-embed"
        );
        let ow = &q.requests_for("PUT /points/payload")[0];
        assert_eq!(ow["points"], json!([42]));
        assert_eq!(ow["payload"]["meta"]["revision"], "3");
    }

    #[tokio::test]
    async fn new_text_replaces_and_supersedes_old() {
        let q = MockQdrant::start().await;
        q.push(
            "POST /points/scroll",
            json!({"result": {"points": [stored("old text", "2")], "next_page_offset": null}}),
        );
        let r = state(&q.url, None)
            .upsert(&svc(), "k1", doc("new text"))
            .await
            .unwrap();
        assert_eq!(r.status, DocStatus::Replaced);
        assert_eq!(r.replaced_id, Some(42));
        let new_pt = &q.requests_for("PUT /points")[0]["points"][0];
        assert_eq!(new_pt["payload"]["supersedes"], json!([42]));
        let mark = &q.requests_for("POST /points/payload")[0];
        assert_eq!(mark["points"], json!([42]));
        assert_eq!(mark["payload"]["superseded_by"], json!(r.id));
    }

    // ---------- delete ----------

    #[tokio::test]
    async fn delete_removes_history_after_wal() {
        let q = MockQdrant::start().await;
        q.push("POST /points/scroll", json!({"result": {"points": [stored("v2", "2"), {"id": 41, "payload": {
            "text": "v1", "wing": "demo-svc", "room": "r", "hall": "h", "category": "c",
            "timestamp": "2026-10-01T00:00:00Z", "external_id": "k1", "superseded_by": 42}}], "next_page_offset": null}}));
        let wal = temp_wal("delete");
        let ids = state(&q.url, Some(wal.clone()))
            .delete_key(&svc(), "k1", "withdrawn")
            .await
            .unwrap();
        assert_eq!(ids, vec![42, 41]);
        let scroll = &q.requests_for("POST /points/scroll")[0];
        assert!(
            !scroll.to_string().contains("is_empty"),
            "delete covers history too"
        );
        assert_eq!(
            q.requests_for("POST /points/delete")[0]["points"],
            json!([42, 41])
        );
        let log = std::fs::read_to_string(&wal).unwrap();
        assert_eq!(log.matches("docs_delete").count(), 2);
        assert!(log.contains("withdrawn"));
    }

    #[tokio::test]
    async fn delete_refuses_without_wal_and_missing_is_empty() {
        let q = MockQdrant::start().await;
        assert!(
            state(&q.url, None)
                .delete_key(&svc(), "k1", "x")
                .await
                .unwrap()
                .is_empty()
        );
        q.push(
            "POST /points/scroll",
            json!({"result": {"points": [stored("v", "1")], "next_page_offset": null}}),
        );
        assert!(
            state(&q.url, None)
                .delete_key(&svc(), "k1", "x")
                .await
                .is_err()
        );
        assert!(q.requests_for("POST /points/delete").is_empty());
    }

    // ---------- search ----------

    #[test]
    fn scope_parsing_groups_by_room() {
        let s = |v: &[&str]| parse_scope(&v.iter().map(|x| x.to_string()).collect::<Vec<_>>());
        let g = s(&[
            "P1:billing",
            "P1:grants",
            "P2:*",
            "P2:billing",
            "P1:billing",
        ])
        .unwrap();
        assert_eq!(
            g["P1"],
            Some(vec!["billing".to_string(), "grants".to_string()])
        );
        assert_eq!(g["P2"], None, "a wildcard wins over specific halls");
        assert!(s(&[]).unwrap_err().message.contains("required"));
        assert!(s(&["no-colon"]).is_err());
        assert!(s(&["P1:"]).is_err());
        assert!(s(&[":billing"]).is_err());
    }

    #[test]
    fn search_filter_shape() {
        let scope = parse_scope(&["P1:billing".into(), "P2:*".into()]).unwrap();
        let f = search_filter(
            "demo-svc",
            &scope,
            &["sheet".into()],
            "2026-11-23T10:00:00Z",
        );
        let must = f["must"].as_array().unwrap();
        assert_eq!(
            must[0],
            json!({"key": "wing", "match": {"value": "demo-svc"}})
        );
        let should = must[1]["should"].as_array().unwrap();
        assert_eq!(should.len(), 2);
        assert_eq!(
            should[0]["must"][1],
            json!({"key": "hall", "match": {"any": ["billing"]}})
        );
        assert_eq!(
            should[1]["must"].as_array().unwrap().len(),
            1,
            "P2:* = room only"
        );
        assert_eq!(
            must[2],
            json!({"key": "category", "match": {"any": ["sheet"]}})
        );
        let must_not = f["must_not"].as_array().unwrap();
        assert!(must_not.iter().any(|c| c["key"] == "valid_until"));
        assert!(
            must_not
                .iter()
                .any(|c| c["key"] == "valid_from" && c["range"]["gt"] == "2026-11-23T10:00:00Z")
        );
    }

    // ---------- HTTP: exact error shapes ----------

    #[tokio::test]
    async fn http_routes_and_error_shapes() {
        let q = MockQdrant::start().await;
        let f = fx();
        let (tok_rw, tok_ro) = (f.rw.clone(), f.ro.clone());
        let (admin, _) = f.key.mint_admin(600).unwrap();
        let app = router(state_with(
            &q.url,
            Some(temp_wal("http")),
            verifier_for(Some(&f.key)),
        ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let c = reqwest::Client::new();
        let body = json!({"text": "t", "room": "r", "hall": "h", "category": "c"});

        let r = c
            .put(format!("{base}/docs/k1"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 401);
        let e: Value = r.json().await.unwrap();
        assert_eq!(
            e,
            json!({"error": {"code": "unauthorized", "message": "missing bearer token"}})
        );

        let r = c
            .put(format!("{base}/docs/k1"))
            .bearer_auth(&tok_ro)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 403);
        assert_eq!(
            r.json::<Value>().await.unwrap()["error"]["code"],
            "forbidden"
        );

        let r = c
            .put(format!("{base}/docs/k1"))
            .bearer_auth(&tok_rw)
            .body("{bad")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400);
        assert_eq!(
            r.json::<Value>().await.unwrap()["error"]["code"],
            "bad_request"
        );

        let r = c
            .put(format!("{base}/docs/k1"))
            .bearer_auth(&tok_rw)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        let ok: Value = r.json().await.unwrap();
        assert_eq!(ok["status"], "created");
        assert_eq!(ok["external_id"], "k1");

        let r = c
            .get(format!("{base}/docs/nothing"))
            .bearer_auth(&tok_ro)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 404);
        assert_eq!(
            r.json::<Value>().await.unwrap()["error"]["code"],
            "not_found"
        );

        let r = c
            .delete(format!("{base}/docs/nothing"))
            .bearer_auth(&tok_rw)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 404);

        let r = c
            .post(format!("{base}/docs/batch"))
            .bearer_auth(&tok_rw)
            .json(&json!({"items": [
                {"external_id": "a", "text": "x", "room": "r", "hall": "h", "category": "c"},
                {"external_id": "bad id", "text": "x", "room": "r", "hall": "h", "category": "c"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        let b: Value = r.json().await.unwrap();
        assert_eq!(b["counts"]["created"], 1);
        assert_eq!(b["counts"]["failed"], 1);
        assert_eq!(b["items"][1]["ok"], false);
        assert_eq!(b["items"][1]["error"]["code"], "bad_request");

        // admin: revoke the ro token; service tokens can't call admin routes
        let jti = crate::svctoken::decode_unverified(&tok_ro).unwrap().jti;
        let r = c
            .post(format!("{base}/admin/service-tokens/revoke"))
            .bearer_auth(&tok_rw)
            .json(&json!({"jti": jti}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 401, "a service token is not an admin token");
        let r = c
            .post(format!("{base}/admin/service-tokens/revoke"))
            .bearer_auth(&admin)
            .json(&json!({"jti": jti}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        let r = c
            .get(format!("{base}/admin/service-tokens/revoked"))
            .bearer_auth(&admin)
            .send()
            .await
            .unwrap();
        assert_eq!(r.json::<Value>().await.unwrap()["revoked"], json!([jti]));
        let r = c
            .get(format!("{base}/docs/k1"))
            .bearer_auth(&tok_ro)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 401, "revoked immediately");
        let (tok_ro, _) = f
            .key
            .mint_service("demo-svc-ro", "demo-svc", "ro", None, Some(1))
            .unwrap();

        let r = c
            .post(format!("{base}/docs/search"))
            .bearer_auth(&tok_ro)
            .json(&json!({"query": "billing", "scope": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400, "scope is mandatory");
        q.push(
            "POST /points/query",
            json!({"result": [{"id": 7, "score": 0.8, "payload": {"text": "t", "wing": "demo-svc", "room": "P1",
                "hall": "billing", "category": "sheet", "timestamp": "2026-11-01T00:00:00Z", "external_id": "k7",
                "meta": {"revision": "2"}}}]}),
        );
        let r = c
            .post(format!("{base}/docs/search"))
            .bearer_auth(&tok_ro)
            .json(&json!({"query": "billing", "scope": ["P1:billing", "P2:*"], "limit": 99}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        let hits: Value = r.json().await.unwrap();
        assert_eq!(hits[0]["external_id"], "k7");
        assert_eq!(hits[0]["meta"]["revision"], "2");
        assert_eq!(hits[0]["score"], 0.8);
        let sent = q.requests_for("POST /points/query").pop().unwrap();
        assert_eq!(sent["limit"], 20, "limit clamps to 20");
        assert_eq!(
            sent["filter"]["must"][0]["match"]["value"], "demo-svc",
            "wing forced"
        );

        let r = c
            .post(format!("{base}/docs/delete"))
            .bearer_auth(&tok_rw)
            .json(&json!({"external_ids": ["gone"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            r.json::<Value>().await.unwrap(),
            json!({"deleted": [], "missing": ["gone"]})
        );
    }
}
