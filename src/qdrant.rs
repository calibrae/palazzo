use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::schema::{ExportPoint, Memory, Payload};

#[derive(Clone)]
pub struct Qdrant {
    client: reqwest::Client,
    base_url: String,
    collection: String,
}

#[derive(Debug, Clone)]
pub struct FindFilter {
    pub wing: Option<String>,
    pub category: Option<String>,
    pub room: Option<String>,
    pub hall: Option<String>,
    /// Exact-match author (the email stamped at write time). Stored lowercased.
    pub author: Option<String>,
    /// Inclusive lower bound on `time_field` (RFC3339).
    pub since: Option<String>,
    /// Inclusive upper bound on `time_field` (RFC3339).
    pub until: Option<String>,
    /// Hide points whose `valid_until` is at-or-before this instant (RFC3339).
    /// Typically set to "now" by callers that want only current-truth memories.
    /// Points without `valid_until` are always kept.
    pub exclude_superseded_before: Option<String>,
    /// Which payload key `since`/`until` range against — `"timestamp"` (write
    /// time, the default) or `"event_time"` (original event time). Qdrant range
    /// conditions only match points that have the key, so `"event_time"` +
    /// since/until naturally excludes points with no `event_time` set.
    pub time_field: String,
}

impl Default for FindFilter {
    fn default() -> Self {
        Self {
            wing: None,
            category: None,
            room: None,
            hall: None,
            author: None,
            since: None,
            until: None,
            exclude_superseded_before: None,
            time_field: "timestamp".to_string(),
        }
    }
}

impl FindFilter {
    fn is_empty(&self) -> bool {
        self.wing.is_none()
            && self.category.is_none()
            && self.room.is_none()
            && self.hall.is_none()
            && self.author.is_none()
            && self.since.is_none()
            && self.until.is_none()
            && self.exclude_superseded_before.is_none()
    }

    fn to_qdrant_filter(&self) -> Option<Value> {
        if self.is_empty() {
            return None;
        }
        let mut must = Vec::new();
        let mut must_not = Vec::new();
        let pairs = [
            ("wing", &self.wing),
            ("category", &self.category),
            ("room", &self.room),
            ("hall", &self.hall),
            ("author", &self.author),
        ];
        for (key, val) in pairs {
            if let Some(v) = val {
                must.push(json!({"key": key, "match": {"value": v}}));
            }
        }
        if self.since.is_some() || self.until.is_some() {
            let mut range = serde_json::Map::new();
            if let Some(s) = &self.since {
                range.insert("gte".into(), json!(s));
            }
            if let Some(u) = &self.until {
                range.insert("lte".into(), json!(u));
            }
            must.push(json!({ "key": self.time_field, "range": range }));
        }
        if let Some(now) = &self.exclude_superseded_before {
            must_not.extend(current_truth_must_not(now));
        }
        let mut body = serde_json::Map::new();
        if !must.is_empty() {
            body.insert("must".into(), Value::Array(must));
        }
        if !must_not.is_empty() {
            body.insert("must_not".into(), Value::Array(must_not));
        }
        Some(Value::Object(body))
    }
}

/// "Current truth" at `now`: hide points whose `valid_until` is at-or-before
/// `now` (superseded or expired) and points whose `valid_from` is still in the
/// future. Range conditions only match when the field exists, so points
/// without these fields are unaffected.
pub fn current_truth_must_not(now: &str) -> [Value; 2] {
    [
        json!({ "key": "valid_until", "range": { "lte": now } }),
        json!({ "key": "valid_from", "range": { "gt": now } }),
    ]
}

/// Payload indexes of the main (text) collection.
pub const TEXT_INDEXES: [(&str, &str); 10] = [
    ("wing", "keyword"),
    ("category", "keyword"),
    ("room", "keyword"),
    ("hall", "keyword"),
    ("author", "keyword"),
    ("timestamp", "datetime"),
    ("valid_until", "datetime"),
    ("valid_from", "datetime"),
    ("event_time", "datetime"),
    // Service /docs API key; absent on ordinary memories (index stays tiny).
    ("external_id", "keyword"),
];

#[derive(Debug, Serialize)]
struct UpsertBody {
    points: Vec<PointUpsert>,
}

#[derive(Debug, Serialize)]
pub struct PointUpsert {
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: Payload,
}

#[derive(Debug, Deserialize)]
struct ScoredPoint {
    id: Value,
    score: f32,
    payload: Option<Payload>,
}

#[derive(Debug, Deserialize)]
struct RetrievedPoint {
    id: Value,
    payload: Option<Payload>,
}

#[derive(Debug, Deserialize)]
struct ScrolledPoint {
    id: Value,
    payload: Option<Payload>,
    #[serde(default)]
    vector: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct ScrollResult {
    points: Vec<ScrolledPoint>,
    #[serde(default)]
    next_page_offset: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ResultWrapper<T> {
    result: T,
}

/// Query API (`POST /points/query`) result envelope: `{"points": [...]}`.
/// Replaces the legacy `/points/search`, which Qdrant 1.19 drops from its API.
#[derive(Debug, Deserialize)]
struct QueryResult<T> {
    points: Vec<T>,
}

/// Nearest-neighbour request for the Query API (available since Qdrant 1.10).
fn query_body(vector: Vec<f32>, limit: u32, filter: &FindFilter) -> Value {
    let mut body = json!({ "query": vector, "limit": limit, "with_payload": true });
    if let Some(f) = filter.to_qdrant_filter() {
        body["filter"] = f;
    }
    body
}

/// `QDRANT_API_KEY` (trimmed; unset or empty = no key, today's behaviour).
pub fn api_key_from_env() -> Option<String> {
    std::env::var("QDRANT_API_KEY")
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
}

impl Qdrant {
    /// Client for `collection`, authenticating with `QDRANT_API_KEY` when set.
    pub fn new(base_url: impl Into<String>, collection: impl Into<String>) -> Self {
        Self::with_api_key(base_url, collection, api_key_from_env())
    }

    /// Every request through this client carries `api-key: <key>` — one default
    /// header, so search/upsert/delete/scroll/count/facet and collection/index
    /// provisioning are all covered. Marked sensitive so it never shows up in
    /// reqwest debug output.
    pub fn with_api_key(
        base_url: impl Into<String>,
        collection: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        let mut builder = reqwest::Client::builder().timeout(std::time::Duration::from_secs(30));
        if let Some(key) = api_key {
            let mut value = reqwest::header::HeaderValue::from_str(&key).expect(
                "QDRANT_API_KEY contains characters that are not allowed in an HTTP header",
            );
            value.set_sensitive(true);
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert("api-key", value);
            builder = builder.default_headers(headers);
        }
        Self {
            client: builder.build().expect("reqwest client"),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            collection: collection.into(),
        }
    }

    pub fn collection(&self) -> &str {
        &self.collection
    }

    fn url(&self, path: &str) -> String {
        format!("{}/collections/{}{}", self.base_url, self.collection, path)
    }

    pub async fn upsert(&self, id: u64, vector: Vec<f32>, payload: Payload) -> Result<()> {
        self.upsert_batch(vec![PointUpsert {
            id,
            vector,
            payload,
        }])
        .await
    }

    /// Upsert many points in one HTTP roundtrip. Caller is responsible for
    /// ensuring the batch fits the Qdrant request-size limit (we cap at 256
    /// items per `palace_store_batch` call upstream, which is well under the
    /// default 30 MB limit for any sane payload size).
    pub async fn upsert_batch(&self, points: Vec<PointUpsert>) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let url = self.url("/points?wait=true");
        let body = UpsertBody { points };
        let resp = self
            .client
            .put(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("PUT {url}"))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("qdrant upsert {}: {}", status, text));
        }
        metrics::histogram!("palazzo_qdrant_duration_seconds", "op" => "upsert")
            .record(t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Merge `fields` into the payload of point `id`. Qdrant's set-payload
    /// endpoint leaves untouched keys alone — useful for `palace_supersede`
    /// marking an old point without rewriting the rest of its payload.
    pub async fn set_payload(&self, id: u64, fields: Value) -> Result<()> {
        let url = self.url("/points/payload?wait=true");
        let body = json!({ "payload": fields, "points": [id] });
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("qdrant set_payload {id}: {status} {text}"));
        }
        Ok(())
    }

    /// Replace the whole payload of point `id` (Qdrant overwrite-payload). The
    /// vector is untouched, so metadata-only changes need no re-embedding.
    pub async fn overwrite_payload(&self, id: u64, payload: &Payload) -> Result<()> {
        let url = self.url("/points/payload?wait=true");
        let body = json!({ "payload": payload, "points": [id] });
        let resp = self
            .client
            .put(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("PUT {url}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("qdrant overwrite_payload {id}: {status} {text}"));
        }
        Ok(())
    }

    /// Points matching a raw Qdrant `filter`, with typed payloads (max `limit`).
    /// For exact-key lookups (service `/docs` API), not for search.
    pub async fn scroll_filter(&self, filter: Value, limit: usize) -> Result<Vec<(u64, Payload)>> {
        let url = self.url("/points/scroll");
        let body = json!({
            "limit": limit,
            "with_payload": true,
            "with_vector": false,
            "filter": filter,
        });
        let resp: ResultWrapper<ScrollResult> = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?
            .error_for_status()
            .context("qdrant scroll status")?
            .json()
            .await
            .context("qdrant scroll decode")?;
        Ok(resp
            .result
            .points
            .into_iter()
            .filter_map(|p| Some((id_as_u64(&p.id)?, p.payload?)))
            .collect())
    }

    /// Hard-delete one or more points by ID. Missing IDs are silently ok.
    pub async fn delete(&self, ids: &[u64]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let url = self.url("/points/delete?wait=true");
        let body = json!({ "points": ids });
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("qdrant delete: {status} {text}"));
        }
        metrics::histogram!("palazzo_qdrant_duration_seconds", "op" => "delete")
            .record(t0.elapsed().as_secs_f64());
        Ok(())
    }

    pub async fn search(
        &self,
        vector: Vec<f32>,
        limit: u32,
        filter: &FindFilter,
    ) -> Result<Vec<Memory>> {
        let t0 = std::time::Instant::now();
        let url = self.url("/points/query");
        let body = query_body(vector, limit, filter);
        let resp: ResultWrapper<QueryResult<ScoredPoint>> = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?
            .error_for_status()
            .context("qdrant search status")?
            .json()
            .await
            .context("qdrant search decode")?;
        metrics::histogram!("palazzo_qdrant_duration_seconds", "op" => "search")
            .record(t0.elapsed().as_secs_f64());
        Ok(resp
            .result
            .points
            .into_iter()
            .filter_map(to_memory_scored)
            .collect())
    }

    /// Vector search with a raw Qdrant `filter`, typed results. For filters
    /// `FindFilter` can't express (e.g. the `/docs/search` scope pairs).
    pub async fn search_filter(
        &self,
        vector: Vec<f32>,
        limit: u32,
        filter: Value,
    ) -> Result<Vec<Memory>> {
        let url = self.url("/points/query");
        let body =
            json!({ "query": vector, "limit": limit, "with_payload": true, "filter": filter });
        let resp: ResultWrapper<QueryResult<ScoredPoint>> = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?
            .error_for_status()
            .context("qdrant search status")?
            .json()
            .await
            .context("qdrant search decode")?;
        Ok(resp
            .result
            .points
            .into_iter()
            .filter_map(to_memory_scored)
            .collect())
    }

    pub async fn retrieve(&self, ids: Vec<u64>) -> Result<Vec<Memory>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let url = self.url("/points");
        let body = json!({
            "ids": ids,
            "with_payload": true,
        });
        let resp: ResultWrapper<Vec<RetrievedPoint>> = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?
            .error_for_status()
            .context("qdrant retrieve status")?
            .json()
            .await
            .context("qdrant retrieve decode")?;
        Ok(resp
            .result
            .into_iter()
            .filter_map(to_memory_plain)
            .collect())
    }

    pub async fn count(&self, filter: &FindFilter) -> Result<u64> {
        let url = self.url("/points/count");
        let mut body = json!({ "exact": true });
        if let Some(f) = filter.to_qdrant_filter() {
            body["filter"] = f;
        }
        let resp: ResultWrapper<Value> = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?
            .error_for_status()
            .context("qdrant count status")?
            .json()
            .await
            .context("qdrant count decode")?;
        Ok(resp
            .result
            .get("count")
            .and_then(Value::as_u64)
            .unwrap_or(0))
    }

    /// Facet a single key. Returns (value, count) pairs.
    pub async fn facet(&self, key: &str) -> Result<Vec<(String, u64)>> {
        let url = self.url("/facet");
        let body = json!({ "key": key, "limit": 100, "exact": true });
        let resp: ResultWrapper<Value> = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?
            .error_for_status()
            .context("qdrant facet status")?
            .json()
            .await
            .context("qdrant facet decode")?;
        let hits = resp
            .result
            .get("hits")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::with_capacity(hits.len());
        for h in hits {
            let val = h
                .get("value")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            let count = h.get("count").and_then(Value::as_u64).unwrap_or(0);
            if let Some(v) = val {
                out.push((v, count));
            }
        }
        Ok(out)
    }

    /// Scroll through points in the collection with optional filtering and pagination.
    /// Returns a page of points and the next page offset (if more pages exist).
    /// Useful for streaming large result sets without loading everything into memory.
    pub async fn scroll(
        &self,
        limit: usize,
        offset: Option<Value>,
        filter: &FindFilter,
        with_vector: bool,
    ) -> Result<(Vec<ExportPoint>, Option<Value>)> {
        let url = self.url("/points/scroll");
        let mut body = json!({
            "limit": limit,
            "with_payload": true,
            "with_vector": with_vector,
        });
        if let Some(o) = offset {
            body["offset"] = o;
        }
        if let Some(f) = filter.to_qdrant_filter() {
            body["filter"] = f;
        }
        let resp: ResultWrapper<ScrollResult> = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?
            .error_for_status()
            .context("qdrant scroll status")?
            .json()
            .await
            .context("qdrant scroll decode")?;
        let points = resp
            .result
            .points
            .into_iter()
            .filter_map(to_export_point)
            .collect();
        Ok((points, resp.result.next_page_offset))
    }

    /// Ensure keyword indexes exist on wing, category, room, hall, plus a
    /// datetime index on `timestamp` for since/until range queries. Idempotent —
    /// Qdrant accepts re-creation as no-op.
    /// Create the collection (768-dim, Cosine — the nomic contract) if it
    /// doesn't exist yet. Idempotent: an existing collection is left untouched,
    /// so this never affects data — it only self-provisions a fresh deployment
    /// (any COLLECTION name) so there's no manual Qdrant step before first boot.
    pub async fn ensure_collection(&self) -> Result<()> {
        let url = format!("{}/collections/{}", self.base_url, self.collection);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("GET {url}"))?;
        if resp.status().is_success() {
            return Ok(()); // already exists — do not touch
        }
        if resp.status() != reqwest::StatusCode::NOT_FOUND {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!(
                "check collection {}: {status} {text}",
                self.collection
            ));
        }
        tracing::warn!(
            collection = %self.collection,
            "collection not found — creating it (768-dim, Cosine)"
        );
        let body = json!({ "vectors": { "size": 768, "distance": "Cosine" } });
        let resp = self
            .client
            .put(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("PUT {url}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!(
                "create collection {}: {status} {text}",
                self.collection
            ));
        }
        Ok(())
    }

    pub async fn ensure_indexes(&self) -> Result<()> {
        self.ensure_indexes_for(&TEXT_INDEXES).await
    }

    /// Self-provision the collection, then create the given payload indexes
    /// (idempotent — Qdrant accepts re-creation as a no-op).
    pub async fn ensure_indexes_for(&self, fields: &[(&str, &str)]) -> Result<()> {
        // Self-provision the collection first, so a fresh deploy with any
        // COLLECTION name works with no manual Qdrant step.
        self.ensure_collection().await?;
        let url = self.url("/index?wait=true");
        for &(field, schema) in fields {
            let body = json!({ "field_name": field, "field_schema": schema });
            let resp = self.client.put(&url).json(&body).send().await?;
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(anyhow!("index {field}: {status} {text}"));
            }
        }
        Ok(())
    }
    // ---- Raw-JSON payload access ------------------------------------------
    // The text collection's payload is the typed `Payload`; the image
    // collection carries a different shape, so it goes through these.
    // Only the image palace (fastembed builds) uses them.

    /// Upsert points whose payload is arbitrary JSON.
    #[cfg(feature = "fastembed")]
    pub async fn upsert_raw(&self, points: Vec<(u64, Vec<f32>, Value)>) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let url = self.url("/points?wait=true");
        let pts: Vec<Value> = points
            .into_iter()
            .map(|(id, vector, payload)| json!({"id": id, "vector": vector, "payload": payload}))
            .collect();
        let resp = self
            .client
            .put(&url)
            .json(&json!({ "points": pts }))
            .send()
            .await
            .with_context(|| format!("PUT {url}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("qdrant upsert {status}: {text}"));
        }
        Ok(())
    }

    /// Vector search returning `(id, score, payload)`.
    #[cfg(feature = "fastembed")]
    pub async fn search_raw(
        &self,
        vector: Vec<f32>,
        limit: u32,
        filter: &FindFilter,
    ) -> Result<Vec<(u64, f32, Value)>> {
        let url = self.url("/points/query");
        let body = query_body(vector, limit, filter);
        let resp: ResultWrapper<QueryResult<Value>> = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?
            .error_for_status()
            .context("qdrant search status")?
            .json()
            .await
            .context("qdrant search decode")?;
        Ok(resp
            .result
            .points
            .into_iter()
            .filter_map(|p| {
                let id = id_as_u64(p.get("id")?)?;
                let score = p.get("score")?.as_f64()? as f32;
                Some((id, score, p.get("payload")?.clone()))
            })
            .collect())
    }

    /// Fetch `(id, payload)` for existing IDs; missing IDs are skipped.
    #[cfg(feature = "fastembed")]
    pub async fn retrieve_raw(&self, ids: Vec<u64>) -> Result<Vec<(u64, Value)>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let url = self.url("/points");
        let resp: ResultWrapper<Vec<Value>> = self
            .client
            .post(&url)
            .json(&json!({ "ids": ids, "with_payload": true }))
            .send()
            .await
            .with_context(|| format!("POST {url}"))?
            .error_for_status()
            .context("qdrant retrieve status")?
            .json()
            .await
            .context("qdrant retrieve decode")?;
        Ok(resp
            .result
            .into_iter()
            .filter_map(|p| Some((id_as_u64(p.get("id")?)?, p.get("payload")?.clone())))
            .collect())
    }

    /// ID of the first point whose keyword payload `key` equals `value`.
    /// Exact match on an indexed field — used for content-hash dedup.
    #[cfg(feature = "fastembed")]
    pub async fn first_id_where(&self, key: &str, value: &str) -> Result<Option<u64>> {
        let url = self.url("/points/scroll");
        let body = json!({
            "limit": 1,
            "with_payload": false,
            "with_vector": false,
            "filter": { "must": [ { "key": key, "match": { "value": value } } ] },
        });
        let resp: ResultWrapper<Value> = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?
            .error_for_status()
            .context("qdrant scroll status")?
            .json()
            .await
            .context("qdrant scroll decode")?;
        Ok(resp
            .result
            .get("points")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(|p| p.get("id"))
            .and_then(id_as_u64))
    }
}

fn id_as_u64(v: &Value) -> Option<u64> {
    match v {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn to_memory_scored(p: ScoredPoint) -> Option<Memory> {
    let id = id_as_u64(&p.id)?;
    Some(hydrate(id, Some(p.score), p.payload?))
}

fn to_memory_plain(p: RetrievedPoint) -> Option<Memory> {
    let id = id_as_u64(&p.id)?;
    Some(hydrate(id, None, p.payload?))
}

fn hydrate(id: u64, score: Option<f32>, pl: Payload) -> Memory {
    Memory {
        id,
        score,
        text: pl.text,
        category: pl.category,
        wing: pl.wing,
        room: pl.room,
        hall: pl.hall,
        timestamp: pl.timestamp,
        event_time: pl.event_time,
        session: pl.session,
        source_file: pl.source_file,
        author: pl.author,
        valid_from: pl.valid_from,
        valid_until: pl.valid_until,
        supersedes: pl.supersedes,
        superseded_by: pl.superseded_by,
        superseded_reason: pl.superseded_reason,
        external_id: pl.external_id,
        meta: pl.meta,
    }
}

fn to_export_point(p: ScrolledPoint) -> Option<ExportPoint> {
    let id = id_as_u64(&p.id)?;
    let pl = p.payload?;
    Some(ExportPoint {
        id,
        text: pl.text,
        category: pl.category,
        wing: pl.wing,
        room: pl.room,
        hall: pl.hall,
        timestamp: pl.timestamp,
        event_time: pl.event_time,
        session: pl.session,
        source_file: pl.source_file,
        author: pl.author,
        valid_from: pl.valid_from,
        valid_until: pl.valid_until,
        supersedes: pl.supersedes,
        superseded_by: pl.superseded_by,
        superseded_reason: pl.superseded_reason,
        external_id: pl.external_id,
        meta: pl.meta,
        vector: p.vector,
    })
}

#[cfg(test)]
mod tests {
    use super::{FindFilter, Qdrant};
    use crate::testmock::MockQdrant;
    use serde_json::json;

    #[test]
    fn empty_filter_is_none() {
        assert!(FindFilter::default().to_qdrant_filter().is_none());
    }

    #[test]
    fn tag_filters_become_must_matches() {
        let f = FindFilter {
            wing: Some("projects".into()),
            hall: Some("facts".into()),
            ..FindFilter::default()
        };
        let q = f.to_qdrant_filter().unwrap();
        let must = q["must"].as_array().unwrap();
        assert_eq!(must.len(), 2);
        assert!(must.contains(&json!({"key": "wing", "match": {"value": "projects"}})));
        assert!(must.contains(&json!({"key": "hall", "match": {"value": "facts"}})));
        assert!(q.get("must_not").is_none());
    }

    #[test]
    fn author_filter_becomes_must_match() {
        let f = FindFilter {
            author: Some("nico@calii.net".into()),
            ..FindFilter::default()
        };
        let q = f.to_qdrant_filter().unwrap();
        let must = q["must"].as_array().unwrap();
        assert_eq!(must.len(), 1);
        assert_eq!(
            must[0],
            json!({"key": "author", "match": {"value": "nico@calii.net"}})
        );
    }

    #[test]
    fn since_until_become_timestamp_range() {
        let f = FindFilter {
            since: Some("2026-01-01T00:00:00Z".into()),
            until: Some("2026-02-01T00:00:00Z".into()),
            ..FindFilter::default()
        };
        let q = f.to_qdrant_filter().unwrap();
        let must = q["must"].as_array().unwrap();
        assert_eq!(must.len(), 1);
        assert_eq!(must[0]["key"], "timestamp");
        assert_eq!(must[0]["range"]["gte"], "2026-01-01T00:00:00Z");
        assert_eq!(must[0]["range"]["lte"], "2026-02-01T00:00:00Z");
    }

    #[test]
    fn since_until_use_selected_time_field() {
        let f = FindFilter {
            since: Some("2026-01-01T00:00:00Z".into()),
            until: Some("2026-02-01T00:00:00Z".into()),
            time_field: "event_time".into(),
            ..FindFilter::default()
        };
        let q = f.to_qdrant_filter().unwrap();
        let must = q["must"].as_array().unwrap();
        assert_eq!(must.len(), 1);
        assert_eq!(must[0]["key"], "event_time");
        assert_eq!(must[0]["range"]["gte"], "2026-01-01T00:00:00Z");
        assert_eq!(must[0]["range"]["lte"], "2026-02-01T00:00:00Z");
    }

    #[test]
    fn exclude_superseded_becomes_must_not_valid_until() {
        let f = FindFilter {
            exclude_superseded_before: Some("2026-06-10T00:00:00Z".into()),
            ..FindFilter::default()
        };
        let q = f.to_qdrant_filter().unwrap();
        assert!(q.get("must").is_none());
        let must_not = q["must_not"].as_array().unwrap();
        assert_eq!(must_not.len(), 2);
        assert_eq!(must_not[0]["key"], "valid_until");
        assert_eq!(must_not[0]["range"]["lte"], "2026-06-10T00:00:00Z");
        // not-yet-valid points are hidden too
        assert_eq!(must_not[1]["key"], "valid_from");
        assert_eq!(must_not[1]["range"]["gt"], "2026-06-10T00:00:00Z");
    }

    fn payload_json(text: &str) -> serde_json::Value {
        json!({
            "category": "technical", "wing": "projects", "room": "palazzo",
            "hall": "facts", "text": text, "timestamp": "2026-04-20T00:00:00Z",
        })
    }

    #[tokio::test]
    async fn ensure_collection_creates_when_missing() {
        let mock = MockQdrant::start().await;
        mock.push(
            "GET /collections/test",
            json!({"__status": 404, "__body": {"status": {"error": "Not found"}}}),
        );
        let q = Qdrant::new(&mock.url, "test");
        q.ensure_collection().await.unwrap();
        let puts = mock.requests_for("PUT /collections/test");
        assert_eq!(puts.len(), 1, "should create the missing collection");
        assert_eq!(puts[0]["vectors"]["size"], 768);
        assert_eq!(puts[0]["vectors"]["distance"], "Cosine");
    }

    #[tokio::test]
    async fn ensure_collection_skips_when_exists() {
        let mock = MockQdrant::start().await;
        // default GET response is 200 → collection "exists"
        let q = Qdrant::new(&mock.url, "test");
        q.ensure_collection().await.unwrap();
        assert!(
            mock.requests_for("PUT /collections/test").is_empty(),
            "must not touch an existing collection"
        );
    }

    #[tokio::test]
    async fn search_decodes_numeric_and_string_ids() {
        let mock = MockQdrant::start().await;
        mock.push(
            "POST /points/query",
            json!({"result": [
                {"id": 7, "score": 0.9, "payload": payload_json("a")},
                {"id": "12", "score": 0.8, "payload": payload_json("b")},
                {"id": [1], "score": 0.7, "payload": payload_json("dropped — bad id")},
            ]}),
        );
        let q = Qdrant::new(&mock.url, "test");
        let hits = q
            .search(vec![0.0; 768], 5, &FindFilter::default())
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 7);
        assert_eq!(hits[0].score, Some(0.9));
        assert_eq!(hits[1].id, 12);
    }

    #[tokio::test]
    async fn retrieve_skips_payloadless_points() {
        let mock = MockQdrant::start().await;
        mock.push(
            "POST /points",
            json!({"result": [
                {"id": 1, "payload": payload_json("kept")},
                {"id": 2},
            ]}),
        );
        let q = Qdrant::new(&mock.url, "test");
        let got = q.retrieve(vec![1, 2]).await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, 1);
        assert_eq!(got[0].text, "kept");
    }

    #[tokio::test]
    async fn api_key_sent_on_every_request_and_absent_without_one() {
        let mock = MockQdrant::start().await;
        let q = Qdrant::with_api_key(&mock.url, "c", Some("s3cret".into()));
        q.ensure_indexes().await.unwrap();
        q.search(vec![0.0; 768], 1, &FindFilter::default())
            .await
            .unwrap();
        q.count(&FindFilter::default()).await.unwrap();
        q.delete(&[1]).await.unwrap();
        let keys = mock.api_keys();
        assert!(keys.len() > 4);
        assert!(
            keys.iter().all(|k| k.as_deref() == Some("s3cret")),
            "{keys:?}"
        );

        let mock = MockQdrant::start().await;
        Qdrant::with_api_key(&mock.url, "c", None)
            .count(&FindFilter::default())
            .await
            .unwrap();
        assert_eq!(mock.api_keys(), vec![None]);
    }

    #[test]
    fn api_key_header_is_sensitive_in_debug_output() {
        let q = Qdrant::with_api_key("http://127.0.0.1:9", "c", Some("s3cret".into()));
        assert!(!format!("{:?}", q.client).contains("s3cret"));
    }

    #[tokio::test]
    async fn upsert_error_status_surfaces() {
        let mock = MockQdrant::start().await;
        mock.push(
            "PUT /points",
            json!({"__status": 500, "__body": {"status": {"error": "boom"}}}),
        );
        let q = Qdrant::new(&mock.url, "test");
        let err = q
            .upsert(
                1,
                vec![0.0; 768],
                serde_json::from_value(payload_json("x")).unwrap(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("500"), "{err:#}");
    }
}
