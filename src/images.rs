//! Image memories: a separate Qdrant collection (`<COLLECTION>-images`, or
//! `PALAZZO_IMAGE_COLLECTION`) of nomic-embed-vision-v1.5 vectors. Enabled
//! with `PALAZZO_VISION=1`.
//!
//! Why a separate collection and not more points in the palace:
//!   * the vision space is aligned with *prefixed* text queries
//!     (`search_query: …`), while the palace embeds prefix-free;
//!   * text↔image cosines (~0.03-0.10) are an order of magnitude below
//!     text↔text, so images would sink to the bottom of every mixed search.
//!
//! An image memory stores the vector plus metadata and a `source` reference
//! (URL or path) — never the image bytes. Dedup is by SHA-256 of the bytes.

use std::sync::Arc;

use anyhow::Context as _;
use serde::Serialize;

pub use crate::image_args::{DeleteImagesArgs, FindImagesArgs, StoreImageArgs};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::embed::Embedder;
use crate::qdrant::{FindFilter, Qdrant};
use crate::util::now_rfc3339;
use crate::vision::{self, VisionEmbedder};
use crate::wal::Wal;

/// nomic's query instruction — the vision model is aligned with prefixed queries.
const QUERY_PREFIX: &str = "search_query: ";
const MAX_SOURCE_BYTES: usize = 1024;
const MAX_CAPTION_BYTES: usize = 4096;
const MAX_QUERY_BYTES: usize = 4096;
const MAX_DELETE_IDS: usize = 100;
/// Images per `POST /images` request — same cap as `/v1/image-embeddings`.
pub const MAX_IMAGES_PER_REQUEST: usize = vision::MAX_IMAGES;

pub const IMAGE_INDEXES: [(&str, &str); 8] = [
    ("wing", "keyword"),
    ("category", "keyword"),
    ("room", "keyword"),
    ("hall", "keyword"),
    ("author", "keyword"),
    ("sha256", "keyword"),
    ("timestamp", "datetime"),
    ("event_time", "datetime"),
];

/// `PALAZZO_IMAGE_COLLECTION`, else `<COLLECTION>-images`.
pub fn collection_name(text_collection: &str) -> String {
    std::env::var("PALAZZO_IMAGE_COLLECTION")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{text_collection}-images"))
}

#[derive(Debug, Serialize, PartialEq)]
pub struct StoreImageResult {
    pub id: u64,
    /// Set when the exact same image bytes were already stored; `id` is then
    /// the existing point and nothing was written.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duplicate_of: Option<u64>,
    pub sha256: String,
}

#[derive(Debug, Serialize)]
pub struct ImageHit {
    pub id: u64,
    pub score: f32,
    #[serde(flatten)]
    pub payload: Value,
}

#[derive(Debug, Serialize)]
pub struct DeleteImagesResult {
    pub deleted: Vec<u64>,
    pub missing: Vec<u64>,
    pub reason: String,
}

#[derive(Clone)]
pub struct ImagePalace {
    vision: VisionEmbedder,
    text: Arc<Embedder>,
    qdrant: Arc<Qdrant>,
    wal: Arc<Wal>,
}

impl ImagePalace {
    pub fn new(vision: VisionEmbedder, text: Arc<Embedder>, qdrant: Qdrant, wal: Arc<Wal>) -> Self {
        Self {
            vision,
            text,
            qdrant: Arc::new(qdrant),
            wal,
        }
    }

    pub fn collection(&self) -> &str {
        self.qdrant.collection()
    }

    /// The shared vision embedder — `/v1/image-embeddings` reuses it so the
    /// model is loaded once per process.
    pub fn vision(&self) -> &VisionEmbedder {
        &self.vision
    }

    pub async fn ensure_indexes(&self) -> anyhow::Result<()> {
        self.qdrant.ensure_indexes_for(&IMAGE_INDEXES).await
    }

    pub async fn store(
        &self,
        args: StoreImageArgs,
        author: Option<String>,
    ) -> anyhow::Result<StoreImageResult> {
        let category = tag("category", &args.category)?;
        let wing = tag("wing", &args.wing)?;
        let room = tag("room", &args.room)?;
        let hall = tag("hall", &args.hall)?;
        let source = args.source.trim();
        if source.is_empty() {
            anyhow::bail!("source is empty — say where the image lives (URL, path, reference)");
        }
        if source.len() > MAX_SOURCE_BYTES {
            anyhow::bail!(
                "source too long: {} bytes (max {MAX_SOURCE_BYTES})",
                source.len()
            );
        }
        if let Some(c) = &args.caption
            && c.len() > MAX_CAPTION_BYTES
        {
            anyhow::bail!(
                "caption too long: {} bytes (max {MAX_CAPTION_BYTES})",
                c.len()
            );
        }
        if let Some(et) = &args.event_time
            && crate::util::parse_rfc3339(et).is_none()
        {
            anyhow::bail!(
                "event_time must be RFC3339 second-precision UTC (e.g. 2026-04-20T00:00:00Z), got {et:?}"
            );
        }
        let bytes = vision::decode_one(&args.image).map_err(|e| anyhow::anyhow!("image: {e}"))?;
        let info = vision::check_image(&bytes).map_err(|e| anyhow::anyhow!("image: {e}"))?;
        let sha256 = hex(&Sha256::digest(&bytes));

        if let Some(existing) = self.qdrant.first_id_where("sha256", &sha256).await? {
            return Ok(StoreImageResult {
                id: existing,
                duplicate_of: Some(existing),
                sha256,
            });
        }

        let vector = self
            .vision
            .embed(vec![bytes.clone()])
            .await?
            .pop()
            .context("vision embedder returned no vector")?;
        let id = crate::mcp::new_id();
        let mut payload = json!({
            "source": source,
            "category": category,
            "wing": wing,
            "room": room,
            "hall": hall,
            "sha256": sha256,
            "format": info.format,
            "width": info.width,
            "height": info.height,
            "bytes": bytes.len(),
            "timestamp": now_rfc3339(),
        });
        for (k, v) in [
            (
                "caption",
                args.caption
                    .map(|c| c.trim().to_string())
                    .filter(|c| !c.is_empty()),
            ),
            ("event_time", args.event_time),
            ("session", args.session),
            ("author", author.map(|a| a.to_lowercase())),
        ] {
            if let Some(v) = v {
                payload[k] = json!(v);
            }
        }
        self.wal.log(
            "palace_store_image",
            &json!({
                "id": id,
                "collection": self.qdrant.collection(),
                "source": payload["source"],
                "sha256": payload["sha256"],
                "wing": payload["wing"],
                "room": payload["room"],
                "hall": payload["hall"],
                "category": payload["category"],
                "author": payload.get("author"),
            }),
        );
        self.qdrant.upsert_raw(vec![(id, vector, payload)]).await?;
        Ok(StoreImageResult {
            id,
            duplicate_of: None,
            sha256,
        })
    }

    pub async fn find(&self, args: FindImagesArgs) -> anyhow::Result<Vec<ImageHit>> {
        if args.query.trim().is_empty() {
            anyhow::bail!("query is empty");
        }
        if args.query.len() > MAX_QUERY_BYTES {
            anyhow::bail!(
                "query too large: {} bytes (max {MAX_QUERY_BYTES})",
                args.query.len()
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
        let time_field = match args.time_field.as_deref() {
            None | Some("timestamp") => "timestamp",
            Some("event_time") => "event_time",
            Some(other) => {
                anyhow::bail!("time_field must be \"timestamp\" or \"event_time\", got {other:?}")
            }
        };
        let filter = FindFilter {
            wing: args.wing.map(|w| w.trim().to_string()),
            category: args.category.map(|c| c.trim().to_string()),
            room: args.room.map(|r| r.trim().to_string()),
            hall: args.hall.map(|h| h.trim().to_string()),
            author: args.author.map(|a| a.trim().to_lowercase()),
            since: args.since,
            until: args.until,
            exclude_superseded_before: None,
            time_field: time_field.to_string(),
        };
        let limit = args.limit.unwrap_or(5).clamp(1, 20);
        let vector = self
            .text
            .embed(&format!("{QUERY_PREFIX}{}", args.query.trim()))
            .await?;
        Ok(self
            .qdrant
            .search_raw(vector, limit, &filter)
            .await?
            .into_iter()
            .map(|(id, score, payload)| ImageHit { id, score, payload })
            .collect())
    }

    pub async fn delete(&self, args: DeleteImagesArgs) -> anyhow::Result<DeleteImagesResult> {
        if !args.confirm {
            anyhow::bail!(
                "confirm must be true — palace_delete_images requires explicit operator approval (irreversible; only the WAL keeps the metadata)"
            );
        }
        if args.reason.trim().is_empty() {
            anyhow::bail!("reason is empty — say why this delete is happening");
        }
        if args.ids.is_empty() {
            anyhow::bail!("ids is empty — nothing to delete");
        }
        if args.ids.len() > MAX_DELETE_IDS {
            anyhow::bail!("too many ids: {} (max {MAX_DELETE_IDS})", args.ids.len());
        }
        let found: std::collections::HashMap<u64, Value> = self
            .qdrant
            .retrieve_raw(args.ids.clone())
            .await?
            .into_iter()
            .collect();
        let (mut deleted, mut missing) = (Vec::new(), Vec::new());
        for id in &args.ids {
            match found.get(id) {
                Some(payload) => {
                    // Strict: the WAL is the only audit trail.
                    self.wal
                        .log_strict(
                            "palace_delete_image",
                            &json!({
                                "id": id,
                                "collection": self.qdrant.collection(),
                                "payload": payload,
                                "reason": args.reason,
                            }),
                        )
                        .context("WAL append failed — refusing to delete without an audit trail")?;
                    deleted.push(*id);
                }
                None => missing.push(*id),
            }
        }
        self.qdrant.delete(&deleted).await?;
        Ok(DeleteImagesResult {
            deleted,
            missing,
            reason: args.reason,
        })
    }
}

fn tag(field: &str, value: &str) -> anyhow::Result<String> {
    let t = value.trim();
    if t.is_empty() {
        anyhow::bail!("{field} is empty");
    }
    if t.len() > 64 {
        anyhow::bail!("{field} too long: {} bytes (max 64)", t.len());
    }
    Ok(t.to_string())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// One NDJSON line of `POST /images`.
pub fn parse_ndjson(body: &str) -> Result<Vec<StoreImageArgs>, String> {
    let mut items = Vec::new();
    for (n, line) in body.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let item: StoreImageArgs = serde_json::from_str(line)
            .map_err(|e| format!("parse error on line {}: {e}", n + 1))?;
        items.push(item);
        if items.len() > MAX_IMAGES_PER_REQUEST {
            return Err(format!(
                "too many images: max {MAX_IMAGES_PER_REQUEST} per request — split the file"
            ));
        }
    }
    if items.is_empty() {
        return Err("no items in body".into());
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testmock::MockQdrant;
    use base64::Engine;
    use image::{DynamicImage, ImageFormat, RgbImage};
    use std::io::Cursor;
    use std::path::PathBuf;

    fn png_b64(w: u32, h: u32) -> String {
        let mut out = Vec::new();
        DynamicImage::ImageRgb8(RgbImage::new(w, h))
            .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
            .unwrap();
        base64::engine::general_purpose::STANDARD.encode(out)
    }

    fn temp_wal(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "palazzo-images-test-wal-{tag}-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn palace(url: &str, wal: Option<PathBuf>) -> ImagePalace {
        ImagePalace::new(
            VisionEmbedder::fake(),
            Arc::new(Embedder::fake()),
            Qdrant::new(url, "test-images"),
            Arc::new(Wal::with_path(wal)),
        )
    }

    fn store_args() -> StoreImageArgs {
        StoreImageArgs {
            image: png_b64(3, 2),
            source: "https://example.org/a.png".into(),
            category: "diagram".into(),
            wing: "projects".into(),
            room: "palazzo".into(),
            hall: "facts".into(),
            caption: Some(" architecture sketch ".into()),
            event_time: Some("2026-10-01T10:00:00Z".into()),
            session: None,
        }
    }

    #[tokio::test]
    async fn store_writes_vector_and_metadata_not_bytes() {
        let q = MockQdrant::start().await;
        let wal = temp_wal("store");
        let ip = palace(&q.url, Some(wal.clone()));
        let r = ip
            .store(store_args(), Some("Dev@Example.org".into()))
            .await
            .unwrap();
        assert!(r.id >= 1_000_000_000);
        assert_eq!(r.duplicate_of, None);
        assert_eq!(r.sha256.len(), 64);

        let ups = q.requests_for("PUT /points");
        assert_eq!(ups.len(), 1);
        let pt = &ups[0]["points"][0];
        assert_eq!(pt["vector"].as_array().unwrap().len(), 768);
        let pl = &pt["payload"];
        assert_eq!(pl["source"], "https://example.org/a.png");
        assert_eq!(pl["sha256"], r.sha256);
        assert_eq!(pl["format"], "png");
        assert_eq!(
            (pl["width"].as_u64(), pl["height"].as_u64()),
            (Some(3), Some(2))
        );
        assert_eq!(pl["caption"], "architecture sketch");
        assert_eq!(pl["author"], "dev@example.org");
        assert_eq!(pl["event_time"], "2026-10-01T10:00:00Z");
        assert!(
            pl.get("image").is_none(),
            "image bytes must never be stored"
        );
        assert!(
            std::fs::read_to_string(&wal)
                .unwrap()
                .contains("palace_store_image")
        );
    }

    #[tokio::test]
    async fn store_dedups_on_sha256_without_embedding_or_upsert() {
        let q = MockQdrant::start().await;
        q.push(
            "POST /points/scroll",
            json!({"result": {"points": [{"id": 42}], "next_page_offset": null}}),
        );
        let r = palace(&q.url, None)
            .store(store_args(), None)
            .await
            .unwrap();
        assert_eq!((r.id, r.duplicate_of), (42, Some(42)));
        let scroll = &q.requests_for("POST /points/scroll")[0];
        assert_eq!(scroll["filter"]["must"][0]["key"], "sha256");
        assert!(q.requests_for("PUT /points").is_empty());
    }

    #[tokio::test]
    async fn store_validates_before_touching_qdrant() {
        let q = MockQdrant::start().await;
        let ip = palace(&q.url, None);
        type Mutate = Box<dyn Fn(&mut StoreImageArgs)>;
        let cases: Vec<(&str, Mutate)> = vec![
            ("source", Box::new(|a| a.source = "  ".into())),
            ("wing", Box::new(|a| a.wing = "".into())),
            (
                "event_time",
                Box::new(|a| a.event_time = Some("yesterday".into())),
            ),
            ("image", Box::new(|a| a.image = "bm90IGFuIGltYWdl".into())), // "not an image"
            (
                "caption",
                Box::new(|a| a.caption = Some("x".repeat(MAX_CAPTION_BYTES + 1))),
            ),
        ];
        for (field, mutate) in cases {
            let mut a = store_args();
            mutate(&mut a);
            let err = format!("{:#}", ip.store(a, None).await.unwrap_err());
            assert!(err.contains(field), "{field}: {err}");
        }
        assert!(q.requests_for("POST /points/scroll").is_empty());
        assert!(q.requests_for("PUT /points").is_empty());
    }

    #[tokio::test]
    async fn find_embeds_prefixed_query_and_filters() {
        let q = MockQdrant::start().await;
        q.push(
            "POST /points/query",
            json!({"result": [{"id": 7, "score": 0.08, "payload": {"source": "s3://x.png", "wing": "projects"}}]}),
        );
        let hits = palace(&q.url, None)
            .find(FindImagesArgs {
                query: " a red cat ".into(),
                limit: Some(99),
                wing: Some(" projects ".into()),
                category: None,
                room: None,
                hall: None,
                author: None,
                since: Some("2026-01-01T00:00:00Z".into()),
                until: None,
                time_field: Some("event_time".into()),
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        let out = serde_json::to_value(&hits[0]).unwrap();
        assert_eq!(out["id"], 7);
        assert_eq!(out["source"], "s3://x.png"); // payload flattened

        let body = &q.requests_for("POST /points/query")[0];
        assert_eq!(body["limit"], 20); // clamped
        let expected = Embedder::fake()
            .embed("search_query: a red cat")
            .await
            .unwrap();
        let sent: Vec<f32> = serde_json::from_value(body["query"].clone()).unwrap();
        assert_eq!(
            sent, expected,
            "query must be embedded with the search_query: prefix"
        );
        let must = body["filter"]["must"].as_array().unwrap();
        assert!(
            must.iter()
                .any(|c| c["key"] == "wing" && c["match"]["value"] == "projects")
        );
        assert!(
            must.iter()
                .any(|c| c["key"] == "event_time" && c["range"]["gte"] == "2026-01-01T00:00:00Z")
        );
    }

    #[tokio::test]
    async fn find_rejects_bad_inputs() {
        let q = MockQdrant::start().await;
        let ip = palace(&q.url, None);
        let args = |query: &str, tf: Option<&str>, since: Option<&str>| FindImagesArgs {
            query: query.into(),
            limit: None,
            wing: None,
            category: None,
            room: None,
            hall: None,
            author: None,
            since: since.map(Into::into),
            until: None,
            time_field: tf.map(Into::into),
        };
        assert!(ip.find(args("", None, None)).await.is_err());
        assert!(ip.find(args("cat", Some("mtime"), None)).await.is_err());
        assert!(ip.find(args("cat", None, Some("last week"))).await.is_err());
        assert!(q.requests_for("POST /points/query").is_empty());
    }

    #[tokio::test]
    async fn delete_wal_logs_then_deletes_only_existing() {
        let q = MockQdrant::start().await;
        q.push(
            "POST /points",
            json!({"result": [{"id": 1, "payload": {"source": "a.png", "sha256": "ab"}}]}),
        );
        let wal = temp_wal("delete");
        let r = palace(&q.url, Some(wal.clone()))
            .delete(DeleteImagesArgs {
                ids: vec![1, 2],
                reason: "operator approved, test".into(),
                confirm: true,
            })
            .await
            .unwrap();
        assert_eq!((r.deleted, r.missing), (vec![1], vec![2]));
        assert_eq!(
            q.requests_for("POST /points/delete")[0]["points"],
            json!([1])
        );
        let log = std::fs::read_to_string(&wal).unwrap();
        assert!(log.contains("palace_delete_image") && log.contains("a.png"));
    }

    #[tokio::test]
    async fn delete_refuses_without_confirm_reason_or_wal() {
        let q = MockQdrant::start().await;
        let ip = palace(&q.url, None);
        let del = |confirm: bool, reason: &str| DeleteImagesArgs {
            ids: vec![1],
            reason: reason.into(),
            confirm,
        };
        assert!(ip.delete(del(false, "x")).await.is_err());
        assert!(ip.delete(del(true, " ")).await.is_err());
        // existing point but no WAL configured → strict log refuses, nothing deleted
        q.push(
            "POST /points",
            json!({"result": [{"id": 1, "payload": {}}]}),
        );
        assert!(ip.delete(del(true, "approved")).await.is_err());
        assert!(q.requests_for("POST /points/delete").is_empty());
    }

    #[test]
    fn ndjson_parses_skips_comments_and_caps() {
        let line = serde_json::to_string(&json!({
            "image": "x", "source": "s", "category": "c", "wing": "w", "room": "r", "hall": "h"
        }))
        .unwrap();
        let body = format!("# header\n\n{line}\n{line}\n");
        assert_eq!(parse_ndjson(&body).unwrap().len(), 2);
        let too_many = vec![line.as_str(); MAX_IMAGES_PER_REQUEST + 1].join("\n");
        assert!(parse_ndjson(&too_many).unwrap_err().contains("too many"));
        assert!(parse_ndjson("{oops").unwrap_err().contains("line 1"));
        assert!(parse_ndjson("\n# only comments\n").is_err());
    }

    #[test]
    fn collection_name_defaults_to_suffix() {
        // Env-free path only: PALAZZO_IMAGE_COLLECTION is not set in tests.
        if std::env::var("PALAZZO_IMAGE_COLLECTION").is_err() {
            assert_eq!(collection_name("claude-memory"), "claude-memory-images");
        }
    }
}
