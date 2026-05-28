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

#[derive(Debug, Default, Clone)]
pub struct FindFilter {
    pub wing: Option<String>,
    pub category: Option<String>,
    pub room: Option<String>,
    pub hall: Option<String>,
    /// Inclusive lower bound on `timestamp` (RFC3339).
    pub since: Option<String>,
    /// Inclusive upper bound on `timestamp` (RFC3339).
    pub until: Option<String>,
    /// Hide points whose `valid_until` is at-or-before this instant (RFC3339).
    /// Typically set to "now" by callers that want only current-truth memories.
    /// Points without `valid_until` are always kept.
    pub exclude_superseded_before: Option<String>,
}

impl FindFilter {
    fn is_empty(&self) -> bool {
        self.wing.is_none()
            && self.category.is_none()
            && self.room.is_none()
            && self.hall.is_none()
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
            must.push(json!({ "key": "timestamp", "range": range }));
        }
        if let Some(now) = &self.exclude_superseded_before {
            // Exclude points that have a `valid_until` at-or-before `now`.
            // Qdrant range conditions only match when the field exists, so
            // points without `valid_until` slip through unaffected.
            must_not.push(json!({
                "key": "valid_until",
                "range": { "lte": now }
            }));
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

impl Qdrant {
    pub fn new(base_url: impl Into<String>, collection: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("reqwest client"),
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

    pub async fn search(
        &self,
        vector: Vec<f32>,
        limit: u32,
        filter: &FindFilter,
    ) -> Result<Vec<Memory>> {
        let t0 = std::time::Instant::now();
        let url = self.url("/points/search");
        let mut body = json!({
            "vector": vector,
            "limit": limit,
            "with_payload": true,
        });
        if let Some(f) = filter.to_qdrant_filter() {
            body["filter"] = f;
        }
        let resp: ResultWrapper<Vec<ScoredPoint>> = self
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
    pub async fn ensure_indexes(&self) -> Result<()> {
        let url = self.url("/index?wait=true");
        let fields: [(&str, &str); 6] = [
            ("wing", "keyword"),
            ("category", "keyword"),
            ("room", "keyword"),
            ("hall", "keyword"),
            ("timestamp", "datetime"),
            ("valid_until", "datetime"),
        ];
        for (field, schema) in fields {
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
        session: pl.session,
        source_file: pl.source_file,
        valid_from: pl.valid_from,
        valid_until: pl.valid_until,
        supersedes: pl.supersedes,
        superseded_by: pl.superseded_by,
        superseded_reason: pl.superseded_reason,
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
        session: pl.session,
        source_file: pl.source_file,
        valid_from: pl.valid_from,
        valid_until: pl.valid_until,
        supersedes: pl.supersedes,
        superseded_by: pl.superseded_by,
        superseded_reason: pl.superseded_reason,
        vector: p.vector,
    })
}
