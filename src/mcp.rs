use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use mcp_gain::Tracker;
use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::baselines;
use crate::embed::Embedder;
use crate::qdrant::{FindFilter, PointUpsert, Qdrant};
use crate::schema::{Category, Hall, Memory, Payload, Wing};
use crate::util::now_rfc3339;
use crate::wal::Wal;

const DUPLICATE_THRESHOLD: f32 = 0.95;
/// Upper bound on stored/searched text. nomic-embed-text has ~8k token ctx; 32KB is well above
/// what a sane memory should be, and keeps pathological inputs from flooding Ollama.
const MAX_TEXT_BYTES: usize = 32 * 1024;
/// Cap on a single palace_recall batch. Keeps one tool call from fetching the whole palace.
const MAX_RECALL_IDS: usize = 100;
/// Cap on how many points can be superseded in one tool call. Large batches are usually a
/// design smell — revisit the model or run multiple calls.
const MAX_SUPERSEDES: usize = 50;
/// Cap on items per `palace_store_batch` call. At 32 KB/text * 256 items the upper bound
/// is ~8 MB request payload — well under any sensible HTTP limit and tractable for the
/// embedder in one batch. Bigger bulk loads should issue multiple calls.
pub const MAX_STORE_BATCH: usize = 256;

#[derive(Clone)]
pub struct Palace {
    embedder: Arc<Embedder>,
    qdrant: Arc<Qdrant>,
    wal: Arc<Wal>,
    tracker: Arc<Tracker>,
    // `tool_router` is read via the derived `Clone` impl and by the `#[tool_router]` macro,
    // but clippy can't see that — silence the warning.
    #[allow(dead_code)]
    tool_router: ToolRouter<Palace>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StoreArgs {
    /// The memory to file. Store verbatim — do not summarise.
    pub text: String,
    /// Palace category.
    pub category: Category,
    /// Palace wing.
    pub wing: Wing,
    /// Room — free-text topic or project (e.g. "palazzo", "hermytt", "family").
    pub room: String,
    /// Hall — facts / events / decisions / discoveries / preferences.
    pub hall: Hall,
    /// Optional session identifier — the conversation that produced this memory.
    #[serde(default)]
    pub session: Option<String>,
    /// Optional source path if the memory was imported from a markdown file.
    #[serde(default)]
    pub source_file: Option<String>,
}

/// One element of a `palace_store_batch` request. Same fields as `StoreArgs`
/// minus the wrapper — kept as its own type so the batch tool's schema is
/// inspectable independently of single-store.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StoreBatchItem {
    pub text: String,
    pub category: Category,
    pub wing: Wing,
    pub room: String,
    pub hall: Hall,
    #[serde(default)]
    pub session: Option<String>,
    #[serde(default)]
    pub source_file: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StoreBatchArgs {
    /// Up to `MAX_STORE_BATCH` memories to ingest in one call. Each item is
    /// embedded, dedup-checked, and upserted; results are returned in input
    /// order with per-item status.
    pub items: Vec<StoreBatchItem>,
    /// When true, items that match an existing memory above the 0.95 cosine
    /// threshold (and are an exact text match) are reported as duplicates
    /// without writing anything new — `duplicate_of` carries the existing ID.
    /// When false (default), the same dedup logic runs but the response just
    /// surfaces it as informational. Either way, no new point is created for
    /// a true exact duplicate.
    #[serde(default)]
    pub skip_duplicates: Option<bool>,
}

#[derive(Debug, Serialize)]
struct StoreResult {
    id: u64,
    duplicate_of: Option<u64>,
    score: Option<f32>,
    text: String,
    wing: String,
    room: String,
    hall: String,
    timestamp: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FindArgs {
    /// Natural-language query. Embedded with nomic-embed-text before search.
    pub query: String,
    /// Max results. Default 5, max 20.
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub wing: Option<Wing>,
    #[serde(default)]
    pub category: Option<Category>,
    /// Exact-match room filter.
    #[serde(default)]
    pub room: Option<String>,
    #[serde(default)]
    pub hall: Option<Hall>,
    /// Inclusive lower bound on memory timestamp (RFC3339 second-precision, e.g.
    /// "2026-04-01T00:00:00Z"). Memories older than this are excluded.
    #[serde(default)]
    pub since: Option<String>,
    /// Inclusive upper bound on memory timestamp (RFC3339 second-precision).
    #[serde(default)]
    pub until: Option<String>,
    /// Optional recency boost: after top-N cosine retrieval, re-rank by
    /// `score * exp(-age_days / half_life)`. Set to a positive number of days to
    /// enable (e.g. 365 = year-long half-life). Omit or 0 for pure cosine.
    #[serde(default)]
    pub recency_half_life_days: Option<f64>,
    /// Include memories that have been superseded by a newer entry. Default
    /// false — only current-truth memories are returned. Set to true for
    /// archaeology / auditing.
    #[serde(default)]
    pub include_superseded: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SupersedeArgs {
    /// Point ID(s) that this new memory replaces. Each is marked with
    /// `valid_until = now`, `superseded_by = <new_id>`, `superseded_reason`.
    pub supersedes: Vec<u64>,
    /// The corrected / updated memory text (stored verbatim, embedded).
    pub text: String,
    pub category: Category,
    pub wing: Wing,
    pub room: String,
    pub hall: Hall,
    #[serde(default)]
    pub session: Option<String>,
    #[serde(default)]
    pub source_file: Option<String>,
    /// Short human explanation recorded on each superseded point.
    pub reason: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecallArgs {
    /// Point IDs to fetch verbatim. No embedding needed.
    pub ids: Vec<u64>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CheckDuplicateArgs {
    /// Candidate text. Returns the closest existing memory and whether it's above the duplicate threshold (0.95).
    pub text: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GainArgs {
    /// Optional inclusive lower bound (RFC3339 second-precision UTC). Aggregates only
    /// events at or after this time. Useful for "gain since deploy" / "gain today".
    #[serde(default)]
    pub since: Option<String>,
    /// If true, return the human-friendly text rendering as well as the structured JSON.
    /// Default false — agents almost always want structured.
    #[serde(default)]
    pub include_text: Option<bool>,
}

#[tool_router]
impl Palace {
    pub fn new(embedder: Embedder, qdrant: Qdrant, wal: Wal, tracker: Tracker) -> Self {
        Self {
            embedder: Arc::new(embedder),
            qdrant: Arc::new(qdrant),
            wal: Arc::new(wal),
            tracker: Arc::new(tracker),
            tool_router: Self::tool_router(),
        }
    }

    /// Bridge between an `anyhow::Result<T>` body and an MCP `CallToolResult`,
    /// recording one row in the gain log along the way. Pattern lifted from
    /// prompto's `finish_tool` so siblings stay consistent.
    fn finish_tool<T: Serialize>(
        &self,
        tool: &'static str,
        started: Instant,
        res: anyhow::Result<T>,
    ) -> Result<CallToolResult, McpError> {
        let elapsed = started.elapsed();
        let exec_ms = elapsed.as_millis() as u64;
        let secs = elapsed.as_secs_f64();
        match res {
            Ok(v) => {
                let body = serde_json::to_value(&v).unwrap_or_default().to_string();
                self.tracker
                    .record(tool, None, true, exec_ms, body.len() as u64);
                metrics::counter!("palazzo_tool_calls_total", "tool" => tool, "status" => "ok")
                    .increment(1);
                metrics::histogram!("palazzo_tool_duration_seconds", "tool" => tool).record(secs);
                Ok(CallToolResult::success(vec![Content::text(body)]))
            }
            Err(e) => {
                let msg = format!("{e:#}");
                self.tracker
                    .record(tool, None, false, exec_ms, msg.len() as u64);
                metrics::counter!("palazzo_tool_calls_total", "tool" => tool, "status" => "error")
                    .increment(1);
                Err(McpError::internal_error(msg, None))
            }
        }
    }

    #[tool(
        description = "File a verbatim memory into the palace. Categorise by wing, room, hall. Returns the new point ID (or the existing one if a near-duplicate is found above 0.95 cosine)."
    )]
    async fn palace_store(
        &self,
        Parameters(args): Parameters<StoreArgs>,
    ) -> Result<CallToolResult, McpError> {
        let started = Instant::now();
        let res = self.do_store(args).await;
        self.finish_tool("palace_store", started, res)
    }

    #[tool(
        description = "Bulk-store up to 256 memories in one call. Embeds all items in one ONNX/Ollama batch inference for ~3-5× speedup over single-item calls, then bulk-upserts to Qdrant in one HTTP roundtrip. Each item is dedup-checked against the existing palace using the same 0.95-cosine + exact-text-match rule as palace_store; duplicates short-circuit and return the existing ID. Best-effort per-item error reporting — if item N fails embedding, items 1..N-1 stay stored and item N+1.. continue. Designed for migrations and bulk imports where the per-call overhead of palace_store would dominate."
    )]
    async fn palace_store_batch(
        &self,
        Parameters(args): Parameters<StoreBatchArgs>,
    ) -> Result<CallToolResult, McpError> {
        let started = Instant::now();
        let res = self.do_store_batch(args).await;
        self.finish_tool("palace_store_batch", started, res)
    }

    #[tool(
        description = "Semantic search over the palace. Optional typed filters narrow the search before vector comparison: wing/category/room/hall for faceted filtering, since/until (RFC3339) for time-range filtering, recency_half_life_days to bias scores toward recent memories. By default, points that have been superseded by a newer memory (via palace_supersede) are hidden; pass include_superseded=true to surface them for archaeology."
    )]
    async fn palace_find(
        &self,
        Parameters(args): Parameters<FindArgs>,
    ) -> Result<CallToolResult, McpError> {
        let started = Instant::now();
        let res = self.do_find(args).await;
        self.finish_tool("palace_find", started, res)
    }

    #[tool(
        description = "Fetch palace points by explicit IDs. No embedding — cheap lookup when you already know what you want."
    )]
    async fn palace_recall(
        &self,
        Parameters(args): Parameters<RecallArgs>,
    ) -> Result<CallToolResult, McpError> {
        let started = Instant::now();
        let res = self.do_recall(args).await;
        self.finish_tool("palace_recall", started, res)
    }

    #[tool(
        description = "Palace status: total point count plus breakdown by wing and by hall. Useful for agents orienting themselves before searching."
    )]
    async fn palace_status(&self) -> Result<CallToolResult, McpError> {
        let started = Instant::now();
        let res = self.do_status().await;
        self.finish_tool("palace_status", started, res)
    }

    #[tool(
        description = "Faceted taxonomy: value → count for wing, room, hall, category. Same data as palace_status but flatter — good for dump-the-layout queries."
    )]
    async fn palace_taxonomy(&self) -> Result<CallToolResult, McpError> {
        let started = Instant::now();
        let res = self.do_taxonomy().await;
        self.finish_tool("palace_taxonomy", started, res)
    }

    #[tool(
        description = "Replace one or more existing memories with a corrected/updated version. Embeds and stores the new text, marks each old point with valid_until=now, superseded_by=<new_id>, and the given reason. Use this when a fact changed over time (e.g. infrastructure reshuffle, decision reversal) instead of deleting the old entry — the old point stays in the palace for archaeology but is hidden from default palace_find."
    )]
    async fn palace_supersede(
        &self,
        Parameters(args): Parameters<SupersedeArgs>,
    ) -> Result<CallToolResult, McpError> {
        let started = Instant::now();
        let res = self.do_supersede(args).await;
        self.finish_tool("palace_supersede", started, res)
    }

    #[tool(
        description = "Check whether candidate text is already in the palace. Returns the closest match and a flag if cosine ≥ 0.95. Call this before palace_store to avoid duplicates."
    )]
    async fn palace_check_duplicate(
        &self,
        Parameters(args): Parameters<CheckDuplicateArgs>,
    ) -> Result<CallToolResult, McpError> {
        let started = Instant::now();
        let res = self.do_check_duplicate(args).await;
        self.finish_tool("palace_check_duplicate", started, res)
    }

    #[tool(
        description = "Token-savings report: how many tokens of agent context this palazzo has saved versus a hand-coded SSH+curl+jq equivalent. Aggregates the per-tool gain log; pass `since` (RFC3339) to scope to a recent window. Set include_text=true for a human-friendly text block alongside the structured numbers."
    )]
    async fn palace_gain(
        &self,
        Parameters(args): Parameters<GainArgs>,
    ) -> Result<CallToolResult, McpError> {
        let started = Instant::now();
        let res = self.do_gain(args).await;
        self.finish_tool("palace_gain", started, res)
    }
}

impl Palace {
    async fn do_store(&self, args: StoreArgs) -> anyhow::Result<StoreResult> {
        if args.text.len() > MAX_TEXT_BYTES {
            anyhow::bail!(
                "text too large: {} bytes (max {})",
                args.text.len(),
                MAX_TEXT_BYTES
            );
        }
        if args.text.trim().is_empty() {
            anyhow::bail!("text is empty");
        }
        let vec = self.embedder.embed(&args.text).await?;

        // Duplicate check — if the top hit is above threshold, skip the write and return the existing ID.
        let existing = self
            .qdrant
            .search(vec.clone(), 1, &FindFilter::default())
            .await?;
        if let Some(top) = existing.first()
            && top.score.unwrap_or(0.0) >= DUPLICATE_THRESHOLD
            && top.text == args.text
        {
            tracing::info!(id = top.id, "skipping store — exact duplicate");
            return Ok(StoreResult {
                id: top.id,
                duplicate_of: Some(top.id),
                score: top.score,
                text: top.text.clone(),
                wing: top.wing.clone(),
                room: top.room.clone(),
                hall: top.hall.clone(),
                timestamp: top.timestamp.clone(),
            });
        }

        let id = new_id();
        let timestamp = now_rfc3339();
        let payload = Payload {
            category: args.category.as_str().to_string(),
            wing: args.wing.as_str().to_string(),
            room: args.room.clone(),
            hall: args.hall.as_str().to_string(),
            text: args.text.clone(),
            timestamp: timestamp.clone(),
            session: args.session.clone(),
            source_file: args.source_file.clone(),
            valid_from: None,
            valid_until: None,
            supersedes: None,
            superseded_by: None,
            superseded_reason: None,
        };

        self.wal.log(
            "palace_store",
            &json!({
                "id": id,
                "wing": payload.wing,
                "room": payload.room,
                "hall": payload.hall,
                "category": payload.category,
                "text_preview": preview(&payload.text),
                "session": payload.session,
            }),
        );

        self.qdrant.upsert(id, vec, payload.clone()).await?;

        Ok(StoreResult {
            id,
            duplicate_of: None,
            score: None,
            text: payload.text,
            wing: payload.wing,
            room: payload.room,
            hall: payload.hall,
            timestamp: payload.timestamp,
        })
    }

    pub(crate) async fn do_store_batch(
        &self,
        args: StoreBatchArgs,
    ) -> anyhow::Result<BatchStoreResult> {
        if args.items.is_empty() {
            anyhow::bail!("items is empty — nothing to store");
        }
        if args.items.len() > MAX_STORE_BATCH {
            anyhow::bail!(
                "too many items: {} (max {MAX_STORE_BATCH}). Split into multiple calls.",
                args.items.len()
            );
        }

        // Validate inputs up-front so we don't spend embedder time on a doomed batch.
        let n = args.items.len();
        let mut item_errors: Vec<Option<String>> = vec![None; n];
        for (i, item) in args.items.iter().enumerate() {
            if item.text.len() > MAX_TEXT_BYTES {
                item_errors[i] = Some(format!(
                    "text too large: {} bytes (max {MAX_TEXT_BYTES})",
                    item.text.len()
                ));
            } else if item.text.trim().is_empty() {
                item_errors[i] = Some("text is empty".into());
            }
        }

        // Gather the texts that survived validation, embed them in one batch.
        let valid_indexes: Vec<usize> = (0..n).filter(|i| item_errors[*i].is_none()).collect();
        let valid_texts: Vec<String> = valid_indexes
            .iter()
            .map(|&i| args.items[i].text.clone())
            .collect();

        let vectors = match self.embedder.embed_batch(&valid_texts).await {
            Ok(v) => v,
            Err(e) => {
                // Embedder failure poisons the whole batch — every item gets the error.
                let msg = format!("{e:#}");
                for slot in &mut item_errors {
                    if slot.is_none() {
                        *slot = Some(format!("embedder failed: {msg}"));
                    }
                }
                return Ok(self.assemble_batch_result(args, item_errors, vec![], vec![]));
            }
        };
        if vectors.len() != valid_texts.len() {
            anyhow::bail!(
                "embedder returned {} vectors for {} inputs",
                vectors.len(),
                valid_texts.len()
            );
        }

        // Per-item dedup check against the live collection. Cheap (single top-1
        // search per item) and correct, but does serialize. For a batch of 256
        // that's ~256 ms total round-trip on localhost Qdrant — acceptable.
        let mut dedup_status: Vec<DedupStatus> = vec![DedupStatus::Fresh; n];
        for (slot, &idx) in valid_indexes.iter().enumerate() {
            let vec_for_search = vectors[slot].clone();
            let hits = match self
                .qdrant
                .search(vec_for_search, 1, &FindFilter::default())
                .await
            {
                Ok(h) => h,
                Err(e) => {
                    item_errors[idx] = Some(format!("dedup search: {e:#}"));
                    continue;
                }
            };
            if let Some(top) = hits.first()
                && top.score.unwrap_or(0.0) >= DUPLICATE_THRESHOLD
                && top.text == args.items[idx].text
            {
                dedup_status[idx] = DedupStatus::Duplicate {
                    of: top.id,
                    score: top.score,
                    text: top.text.clone(),
                    wing: top.wing.clone(),
                    room: top.room.clone(),
                    hall: top.hall.clone(),
                    timestamp: top.timestamp.clone(),
                };
            }
        }

        // Build the upsert batch from items that are still fresh and error-free.
        let now = now_rfc3339();
        let mut to_upsert: Vec<PointUpsert> = Vec::new();
        let mut new_ids: Vec<Option<u64>> = vec![None; n];

        for (slot, &idx) in valid_indexes.iter().enumerate() {
            if item_errors[idx].is_some() {
                continue;
            }
            if matches!(dedup_status[idx], DedupStatus::Duplicate { .. }) {
                continue;
            }
            let item = &args.items[idx];
            let id = new_id_for_index(slot);
            let payload = Payload {
                category: item.category.as_str().to_string(),
                wing: item.wing.as_str().to_string(),
                room: item.room.clone(),
                hall: item.hall.as_str().to_string(),
                text: item.text.clone(),
                timestamp: now.clone(),
                session: item.session.clone(),
                source_file: item.source_file.clone(),
                valid_from: None,
                valid_until: None,
                supersedes: None,
                superseded_by: None,
                superseded_reason: None,
            };
            self.wal.log(
                "palace_store_batch:item",
                &json!({
                    "id": id,
                    "wing": payload.wing,
                    "room": payload.room,
                    "hall": payload.hall,
                    "category": payload.category,
                    "text_preview": preview(&payload.text),
                    "session": payload.session,
                }),
            );
            new_ids[idx] = Some(id);
            to_upsert.push(PointUpsert {
                id,
                vector: vectors[slot].clone(),
                payload,
            });
        }

        if let Err(e) = self.qdrant.upsert_batch(to_upsert).await {
            let msg = format!("qdrant upsert_batch: {e:#}");
            for (idx, id_slot) in new_ids.iter_mut().enumerate() {
                if id_slot.is_some() {
                    *id_slot = None;
                    item_errors[idx] = Some(msg.clone());
                }
            }
        }

        Ok(self.assemble_batch_result(args, item_errors, dedup_status, new_ids))
    }

    fn assemble_batch_result(
        &self,
        args: StoreBatchArgs,
        item_errors: Vec<Option<String>>,
        dedup_status: Vec<DedupStatus>,
        new_ids: Vec<Option<u64>>,
    ) -> BatchStoreResult {
        let skip_duplicates = args.skip_duplicates.unwrap_or(false);
        let n = args.items.len();
        let mut items = Vec::with_capacity(n);
        let mut counts = BatchCounts::default();
        let now = now_rfc3339();
        for (idx, item) in args.items.into_iter().enumerate() {
            let entry_idx = idx as u32;
            if let Some(err) = item_errors.get(idx).and_then(|e| e.clone()) {
                counts.failed += 1;
                items.push(BatchStoreEntry {
                    index: entry_idx,
                    ok: false,
                    error: Some(err),
                    id: None,
                    duplicate_of: None,
                    matched_score: None,
                    text: None,
                    wing: None,
                    room: None,
                    hall: None,
                    timestamp: None,
                });
                continue;
            }
            match dedup_status.get(idx).cloned().unwrap_or(DedupStatus::Fresh) {
                DedupStatus::Duplicate {
                    of,
                    score,
                    text,
                    wing,
                    room,
                    hall,
                    timestamp,
                } => {
                    if skip_duplicates {
                        counts.skipped_duplicates += 1;
                    } else {
                        counts.duplicates_returned += 1;
                    }
                    items.push(BatchStoreEntry {
                        index: entry_idx,
                        ok: true,
                        error: None,
                        id: Some(of),
                        duplicate_of: Some(of),
                        matched_score: score,
                        text: Some(text),
                        wing: Some(wing),
                        room: Some(room),
                        hall: Some(hall),
                        timestamp: Some(timestamp),
                    });
                }
                DedupStatus::Fresh => {
                    let new_id = new_ids.get(idx).copied().flatten();
                    if new_id.is_some() {
                        counts.stored += 1;
                    }
                    items.push(BatchStoreEntry {
                        index: entry_idx,
                        ok: new_id.is_some(),
                        error: None,
                        id: new_id,
                        duplicate_of: None,
                        matched_score: None,
                        text: Some(item.text),
                        wing: Some(item.wing.as_str().to_string()),
                        room: Some(item.room),
                        hall: Some(item.hall.as_str().to_string()),
                        timestamp: Some(now.clone()),
                    });
                }
            }
        }
        BatchStoreResult { items, counts }
    }

    async fn do_find(&self, args: FindArgs) -> anyhow::Result<Vec<Memory>> {
        if args.query.len() > MAX_TEXT_BYTES {
            anyhow::bail!(
                "query too large: {} bytes (max {})",
                args.query.len(),
                MAX_TEXT_BYTES
            );
        }
        for (name, val) in [("since", &args.since), ("until", &args.until)] {
            if let Some(s) = val
                && crate::util::parse_rfc3339(s).is_none()
            {
                anyhow::bail!(
                    "{name} must be RFC3339 second-precision UTC (e.g. 2026-04-20T00:00:00Z), got {s:?}"
                );
            }
        }
        let limit = args.limit.unwrap_or(5).clamp(1, 20);
        let exclude_superseded_before = if args.include_superseded.unwrap_or(false) {
            None
        } else {
            Some(now_rfc3339())
        };
        let filter = FindFilter {
            wing: args.wing.map(|w| w.as_str().to_string()),
            category: args.category.map(|c| c.as_str().to_string()),
            room: args.room,
            hall: args.hall.map(|h| h.as_str().to_string()),
            since: args.since,
            until: args.until,
            exclude_superseded_before,
        };
        let vec = self.embedder.embed(&args.query).await?;

        let half_life = args.recency_half_life_days.filter(|h| *h > 0.0);
        let fetch_limit = match half_life {
            Some(_) => (limit.saturating_mul(4)).min(80),
            None => limit,
        };
        let mut hits = self.qdrant.search(vec, fetch_limit, &filter).await?;

        if let Some(hl) = half_life {
            let now_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            let half_life_secs = hl * 86_400.0;
            for m in &mut hits {
                let ts = crate::util::parse_rfc3339(&m.timestamp).unwrap_or(now_secs);
                let age = (now_secs - ts).max(0) as f64;
                let decay = (-age / half_life_secs).exp() as f32;
                if let Some(s) = m.score.as_mut() {
                    *s *= decay;
                }
            }
            hits.sort_by(|a, b| {
                b.score
                    .unwrap_or(0.0)
                    .partial_cmp(&a.score.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        hits.truncate(limit as usize);
        Ok(hits)
    }

    async fn do_supersede(&self, args: SupersedeArgs) -> anyhow::Result<SupersedeResult> {
        if args.text.len() > MAX_TEXT_BYTES {
            anyhow::bail!(
                "text too large: {} bytes (max {})",
                args.text.len(),
                MAX_TEXT_BYTES
            );
        }
        if args.text.trim().is_empty() {
            anyhow::bail!("text is empty");
        }
        if args.reason.trim().is_empty() {
            anyhow::bail!("reason is empty — say why the supersession happened");
        }
        if args.supersedes.is_empty() {
            anyhow::bail!("supersedes is empty — nothing to supersede");
        }
        if args.supersedes.len() > MAX_SUPERSEDES {
            anyhow::bail!(
                "too many supersedes: {} (max {})",
                args.supersedes.len(),
                MAX_SUPERSEDES
            );
        }

        let vec = self.embedder.embed(&args.text).await?;
        let id = new_id();
        let now = now_rfc3339();
        let payload = Payload {
            category: args.category.as_str().to_string(),
            wing: args.wing.as_str().to_string(),
            room: args.room.clone(),
            hall: args.hall.as_str().to_string(),
            text: args.text.clone(),
            timestamp: now.clone(),
            session: args.session.clone(),
            source_file: args.source_file.clone(),
            valid_from: Some(now.clone()),
            valid_until: None,
            supersedes: Some(args.supersedes.clone()),
            superseded_by: None,
            superseded_reason: None,
        };

        self.wal.log(
            "palace_supersede",
            &json!({
                "id": id,
                "supersedes": args.supersedes,
                "reason": args.reason,
                "wing": payload.wing,
                "room": payload.room,
                "hall": payload.hall,
                "category": payload.category,
                "text_preview": preview(&payload.text),
                "session": payload.session,
            }),
        );

        self.qdrant.upsert(id, vec, payload).await?;

        // Mark each old point. Non-atomic across points — if any patch fails,
        // we report it and leave the caller to retry with the same supersedes list.
        let mut marked = Vec::with_capacity(args.supersedes.len());
        for old_id in &args.supersedes {
            let fields = json!({
                "valid_until": now.clone(),
                "superseded_by": id,
                "superseded_reason": args.reason.clone(),
            });
            let result = self.qdrant.set_payload(*old_id, fields).await;
            match result {
                Ok(()) => marked.push(SupersededEntry {
                    id: *old_id,
                    ok: true,
                    error: None,
                }),
                Err(e) => {
                    tracing::warn!(id = *old_id, "supersede mark failed: {e:#}");
                    marked.push(SupersededEntry {
                        id: *old_id,
                        ok: false,
                        error: Some(format!("{e:#}")),
                    });
                }
            }
        }

        Ok(SupersedeResult {
            id,
            text: args.text,
            wing: args.wing.as_str().to_string(),
            room: args.room,
            hall: args.hall.as_str().to_string(),
            timestamp: now,
            supersedes: args.supersedes,
            reason: args.reason,
            marked,
        })
    }

    async fn do_recall(&self, args: RecallArgs) -> anyhow::Result<Vec<Memory>> {
        if args.ids.len() > MAX_RECALL_IDS {
            anyhow::bail!("too many ids: {} (max {MAX_RECALL_IDS})", args.ids.len());
        }
        self.qdrant.retrieve(args.ids).await
    }

    async fn do_status(&self) -> anyhow::Result<serde_json::Value> {
        let total = self.qdrant.count(&FindFilter::default()).await?;
        let wings = self.qdrant.facet("wing").await?;
        let halls = self.qdrant.facet("hall").await?;
        let categories = self.qdrant.facet("category").await?;
        Ok(json!({
            "collection": self.qdrant.collection(),
            "total": total,
            "wings": facet_map(&wings),
            "halls": facet_map(&halls),
            "categories": facet_map(&categories),
        }))
    }

    async fn do_taxonomy(&self) -> anyhow::Result<serde_json::Value> {
        let wings = self.qdrant.facet("wing").await?;
        let rooms = self.qdrant.facet("room").await?;
        let halls = self.qdrant.facet("hall").await?;
        let categories = self.qdrant.facet("category").await?;
        Ok(json!({
            "wings": facet_map(&wings),
            "rooms": facet_map(&rooms),
            "halls": facet_map(&halls),
            "categories": facet_map(&categories),
        }))
    }

    async fn do_check_duplicate(
        &self,
        args: CheckDuplicateArgs,
    ) -> anyhow::Result<serde_json::Value> {
        let vec = self.embedder.embed(&args.text).await?;
        let hits = self.qdrant.search(vec, 1, &FindFilter::default()).await?;
        let top = hits.into_iter().next();
        let is_duplicate = top
            .as_ref()
            .and_then(|m| m.score)
            .map(|s| s >= DUPLICATE_THRESHOLD)
            .unwrap_or(false);
        Ok(json!({
            "is_duplicate": is_duplicate,
            "threshold": DUPLICATE_THRESHOLD,
            "closest": top,
        }))
    }

    async fn do_gain(&self, args: GainArgs) -> anyhow::Result<serde_json::Value> {
        let since = match args.since.as_deref() {
            None => None,
            Some(s) => {
                let secs = crate::util::parse_rfc3339(s).ok_or_else(|| {
                    anyhow::anyhow!("since must be RFC3339 second-precision UTC, got {s:?}")
                })?;
                Some(
                    chrono::DateTime::<chrono::Utc>::from_timestamp(secs, 0)
                        .ok_or_else(|| anyhow::anyhow!("since out of range"))?,
                )
            }
        };
        let summary = self.tracker.summary(since)?;
        let mut value = serde_json::to_value(&summary)?;
        if args.include_text.unwrap_or(false)
            && let Some(obj) = value.as_object_mut()
        {
            obj.insert(
                "text".into(),
                serde_json::Value::String(mcp_gain::render_text(&summary, &baselines::header())),
            );
        }
        Ok(value)
    }
}

#[derive(Debug, Serialize)]
struct SupersedeResult {
    id: u64,
    text: String,
    wing: String,
    room: String,
    hall: String,
    timestamp: String,
    supersedes: Vec<u64>,
    reason: String,
    /// Per-old-point result of the payload patch. If any `ok: false` entries,
    /// the caller can retry `palace_supersede` with that smaller supersedes
    /// list — the new point is already created, so the retry is cheap.
    marked: Vec<SupersededEntry>,
}

#[derive(Debug, Serialize)]
struct SupersededEntry {
    id: u64,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[tool_handler]
impl ServerHandler for Palace {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder().enable_tools().build(),
        )
        .with_server_info(Implementation::from_build_env())
        .with_protocol_version(ProtocolVersion::LATEST)
        .with_instructions(
            "palazzo — Cali's memory palace over MCP. \
             Every memory has a wing (projects/infrastructure/nexpublica/personal/career/vibe), \
             a room (free-text project or topic), and a hall (facts/events/decisions/discoveries/preferences). \
             Tools: palace_store, palace_store_batch, palace_find, palace_recall, palace_status, palace_taxonomy, palace_check_duplicate, palace_supersede, palace_gain. \
             For bulk migrations of pre-existing data (>~10K tokens of payload), prefer the sibling REST endpoint POST /ingest on the same host:port \
             (Content-Type: application/x-ndjson, body = JSONL of palace_store items). Invoke via Bash(curl --data-binary @file) — the bytes flow through curl's body and never enter the MCP transcript, \
             unlike palace_store_batch tool args which do. Same backend (embed, dedup, WAL, upsert), zero context cost for the payload. \
             Error handling: if any tool call returns 'Session not found', the MCP session was reset (server restarted). \
             Retry the exact same call once — the client will re-establish the session automatically and the retry will succeed.".to_string(),
        )
    }
}

fn new_id() -> u64 {
    // Unix millis, guaranteed above the 1_000_000_000 floor.
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0) as u64;
    millis.max(1_000_000_000)
}

/// Stable, collision-free IDs for a single `palace_store_batch` call. The
/// per-item index is added on top of the millis floor so a 256-item batch
/// can complete inside a single millisecond without two items sharing an ID.
fn new_id_for_index(slot: usize) -> u64 {
    new_id().saturating_add(slot as u64)
}

#[derive(Debug, Clone)]
enum DedupStatus {
    Fresh,
    Duplicate {
        of: u64,
        score: Option<f32>,
        text: String,
        wing: String,
        room: String,
        hall: String,
        timestamp: String,
    },
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct BatchCounts {
    /// Items newly written to Qdrant.
    pub stored: u32,
    /// Duplicate items returned with the existing point's ID.
    pub duplicates_returned: u32,
    /// Duplicate items that the caller asked to skip silently.
    pub skipped_duplicates: u32,
    /// Items that failed validation, embedding, or upsert.
    pub failed: u32,
}

#[derive(Debug, Serialize)]
pub(crate) struct BatchStoreEntry {
    pub index: u32,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duplicate_of: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wing: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub room: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hall: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct BatchStoreResult {
    pub items: Vec<BatchStoreEntry>,
    pub counts: BatchCounts,
}

fn facet_map(items: &[(String, u64)]) -> serde_json::Value {
    let mut m = serde_json::Map::with_capacity(items.len());
    for (k, v) in items {
        m.insert(k.clone(), json!(v));
    }
    serde_json::Value::Object(m)
}

fn preview(s: &str) -> String {
    const MAX: usize = 120;
    if s.chars().count() <= MAX {
        return s.to_string();
    }
    let truncated: String = s.chars().take(MAX).collect();
    format!("{truncated}…")
}
