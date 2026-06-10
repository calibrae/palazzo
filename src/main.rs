mod baselines;
mod embed;
mod mcp;
mod qdrant;
mod schema;
#[cfg(test)]
mod testmock;
mod util;
mod wal;

#[cfg(all(feature = "ollama", feature = "fastembed"))]
compile_error!(
    "features `ollama` and `fastembed` are mutually exclusive — pick one with --features, and --no-default-features if you want fastembed."
);

#[cfg(not(any(feature = "ollama", feature = "fastembed")))]
compile_error!("enable one of the embedding features: `ollama` (default) or `fastembed`.");

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use mcp_gain::Tracker;
use rmcp::{
    ServiceExt,
    transport::{
        stdio,
        streamable_http_server::{
            StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
        },
    },
};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

use crate::embed::Embedder;
use crate::mcp::{MAX_STORE_BATCH, Palace, StoreBatchArgs, StoreBatchItem};
use crate::qdrant::{FindFilter, Qdrant};
use crate::wal::Wal;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

struct Config {
    #[cfg(all(feature = "ollama", not(feature = "fastembed")))]
    ollama_url: String,
    #[cfg(all(feature = "ollama", not(feature = "fastembed")))]
    ollama_model: String,
    qdrant_url: String,
    collection: String,
}

impl Config {
    fn from_env() -> Self {
        Self {
            #[cfg(all(feature = "ollama", not(feature = "fastembed")))]
            ollama_url: env_or("OLLAMA_URL", "http://localhost:11434"),
            #[cfg(all(feature = "ollama", not(feature = "fastembed")))]
            ollama_model: env_or("OLLAMA_MODEL", "nomic-embed-text"),
            qdrant_url: env_or("QDRANT_URL", "http://localhost:6333"),
            collection: env_or("COLLECTION", "claude-memory"),
        }
    }

    fn make_palace(&self) -> Result<Palace> {
        self.make_palace_with_embedder(make_embedder(self)?)
    }

    fn make_palace_with_embedder(&self, embedder: Embedder) -> Result<Palace> {
        let qdrant = Qdrant::new(&self.qdrant_url, &self.collection);
        let wal = Wal::from_env();
        let tracker = make_tracker();
        Ok(Palace::new(embedder, qdrant, wal, tracker))
    }

    fn make_qdrant(&self) -> Qdrant {
        Qdrant::new(&self.qdrant_url, &self.collection)
    }
}

/// Where the gain analytics log lives. Defaults to `/var/lib/palazzo/usage.jsonl`
/// (matches the systemd unit's ReadWritePaths) but is overridable for local dev
/// or relocated deployments.
fn usage_log_path() -> PathBuf {
    std::env::var("PALAZZO_USAGE_LOG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/palazzo/usage.jsonl"))
}

fn gain_enabled() -> bool {
    !matches!(
        std::env::var("PALAZZO_GAIN_ENABLED").as_deref(),
        Ok("0" | "false" | "no" | "off")
    )
}

fn make_tracker() -> Tracker {
    Tracker::new(usage_log_path(), gain_enabled(), baselines::BASELINES)
}

const BACKEND: &str = if cfg!(feature = "fastembed") {
    "fastembed:NomicEmbedTextV15"
} else {
    "ollama"
};

#[cfg(all(feature = "ollama", not(feature = "fastembed")))]
fn make_embedder(cfg: &Config) -> Result<Embedder> {
    Ok(Embedder::new(&cfg.ollama_url, &cfg.ollama_model))
}

#[cfg(feature = "fastembed")]
fn make_embedder(_cfg: &Config) -> Result<Embedder> {
    Embedder::new()
}

fn print_help() {
    eprintln!(
        "palazzo {} — MCP server over Qdrant memory palace
Usage:
  palazzo                      Serve MCP over stdio (default)
  palazzo serve [--bind ADDR]  Serve MCP over Streamable HTTP at POST /mcp,
                               and a sibling NDJSON bulk-ingest endpoint at POST /ingest.
                               (default ADDR: 127.0.0.1:6334, override with PALAZZO_BIND)
  palazzo gain [--since-secs N] [--json]
                               Render the token-savings report from PALAZZO_USAGE_LOG.
                               Defaults to all-time text rendering; --json emits the structured Summary.
  palazzo ingest [--file PATH] [--json]
                               Bulk-ingest JSONL of palace_store_batch items. One item per line:
                               {{\"text\":...,\"category\":...,\"wing\":...,\"room\":...,\"hall\":...}}.
                               Reads stdin when --file is omitted. Chunks into MAX_STORE_BATCH groups.
                               Bypasses the MCP transcript — use this instead of palace_store_batch
                               when the agent context can't afford the round-trip cost of the texts.
  palazzo --help               Show this message

Environment:
  OLLAMA_URL    (default http://localhost:11434)
  OLLAMA_MODEL  (default nomic-embed-text)
  QDRANT_URL    (default http://localhost:6333)
  COLLECTION    (default claude-memory)
  PALAZZO_WAL   (default ~/.palazzo/wal.jsonl)
  PALAZZO_BIND  (default 127.0.0.1:6334 in serve mode)
  PALAZZO_ALLOWED_HOSTS (default localhost,127.0.0.1,::1 — set to \"*\" to disable DNS rebinding check)
  PALAZZO_MAX_INGEST_BYTES (default 67108864 = 64 MiB — body cap for POST /ingest; 32KB/item still enforced)
  FASTEMBED_BATCH_CHUNK (default 16 — fastembed sub-batch size; smaller = lower RSS, larger = faster)
  FASTEMBED_RECYCLE_RSS_MB (default 1500 — recycle the embedder when process RSS exceeds this; 0 disables)
  FASTEMBED_RECYCLE_IDLE_SECS (default 30 — only recycle after this many seconds with no embed activity)
  PALAZZO_USAGE_LOG (default /var/lib/palazzo/usage.jsonl — gain analytics JSONL)
  PALAZZO_GAIN_ENABLED (default 1; set 0/false/no/off to disable per-call recording)
  RUST_LOG      (default palazzo=info)
",
        env!("CARGO_PKG_VERSION")
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    // Tracing goes to stderr only — stdout is the stdio MCP transport channel.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("palazzo=info")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => run_stdio().await,
        Some("serve") => run_http(&args[1..]).await,
        Some("gain") => run_gain(&args[1..]),
        Some("ingest") => run_ingest(&args[1..]).await,
        Some("--help" | "-h") => {
            print_help();
            Ok(())
        }
        Some("--version" | "-V") => {
            println!("palazzo {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(other) => {
            eprintln!("unknown argument: {other}");
            print_help();
            std::process::exit(2);
        }
    }
}

fn run_gain(rest: &[String]) -> Result<()> {
    let mut since_secs: Option<u64> = None;
    let mut as_json = false;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--since-secs" => {
                since_secs = Some(
                    rest.get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--since-secs requires a value"))?
                        .parse()
                        .context("--since-secs must be a non-negative integer")?,
                );
                i += 2;
            }
            "--json" => {
                as_json = true;
                i += 1;
            }
            other => anyhow::bail!("unknown gain argument: {other}"),
        }
    }
    let since = since_secs.map(|s| chrono::Utc::now() - chrono::Duration::seconds(s as i64));
    let tracker = make_tracker();
    let summary = tracker.summary(since)?;
    if as_json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print!("{}", mcp_gain::render_text(&summary, &baselines::header()));
    }
    Ok(())
}

async fn run_ingest(rest: &[String]) -> Result<()> {
    use std::io::Read;

    let mut path: Option<PathBuf> = None;
    let mut as_json = false;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--file" => {
                path = Some(PathBuf::from(
                    rest.get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--file requires a path"))?,
                ));
                i += 2;
            }
            "--json" => {
                as_json = true;
                i += 1;
            }
            other => anyhow::bail!("unknown ingest argument: {other}"),
        }
    }

    let mut raw = String::new();
    match path.as_ref() {
        Some(p) => {
            std::fs::File::open(p)
                .with_context(|| format!("open {p:?}"))?
                .read_to_string(&mut raw)?;
        }
        None => {
            std::io::stdin().read_to_string(&mut raw)?;
        }
    }
    let mut items: Vec<StoreBatchItem> = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let item: StoreBatchItem = serde_json::from_str(trimmed)
            .with_context(|| format!("parse item on line {}", lineno + 1))?;
        items.push(item);
    }
    if items.is_empty() {
        anyhow::bail!("no items to ingest (input had no JSONL records)");
    }

    let cfg = Config::from_env();
    tracing::info!(
        backend = BACKEND,
        qdrant = %cfg.qdrant_url,
        collection = %cfg.collection,
        items = items.len(),
        mode = "ingest",
        "palazzo ingest"
    );
    if let Err(e) = cfg.make_qdrant().ensure_indexes().await {
        tracing::warn!("ensure_indexes: {e:#}");
    }
    let palace = cfg.make_palace()?;

    let total = items.len();
    let mut all_entries: Vec<crate::mcp::BatchStoreEntry> = Vec::with_capacity(total);
    let mut totals = crate::mcp::BatchCounts::default();
    let mut base_index: u32 = 0;
    while !items.is_empty() {
        let take = items.len().min(MAX_STORE_BATCH);
        let chunk: Vec<StoreBatchItem> = items.drain(..take).collect();
        let args = StoreBatchArgs {
            items: chunk,
            skip_duplicates: None,
        };
        let result = palace.do_store_batch(args).await?;
        totals.stored += result.counts.stored;
        totals.duplicates_returned += result.counts.duplicates_returned;
        totals.skipped_duplicates += result.counts.skipped_duplicates;
        totals.failed += result.counts.failed;
        for mut entry in result.items {
            entry.index += base_index;
            all_entries.push(entry);
        }
        base_index += MAX_STORE_BATCH as u32;
    }

    if as_json {
        let result = crate::mcp::BatchStoreResult {
            items: all_entries,
            counts: totals,
        };
        println!("{}", serde_json::to_string(&result)?);
    } else {
        eprintln!(
            "ingest: total={total} stored={} duplicates_returned={} skipped_duplicates={} failed={}",
            totals.stored, totals.duplicates_returned, totals.skipped_duplicates, totals.failed,
        );
        if totals.failed > 0 {
            for entry in all_entries.iter().filter(|e| !e.ok) {
                eprintln!(
                    "  [{}] FAILED: {}",
                    entry.index,
                    entry.error.as_deref().unwrap_or("(no error message)"),
                );
            }
            anyhow::bail!("{} item(s) failed", totals.failed);
        }
    }
    Ok(())
}

/// POST /ingest handler. Accepts NDJSON in the body — one StoreBatchItem per line.
///
/// Streams progress as NDJSON: one `{"chunk":N,"counts":{...},"running":{...}}` line
/// per processed batch (default 256 items each), then a final
/// `{"done":true,"total":N,"counts":{...}}` line. Errors during processing emit a
/// `{"error":"..."}` line and close the response. Body parse failures still come
/// back as 400 with a plain-text body before any streaming starts.
///
/// The payload bytes flow through the HTTP body and never need to enter an MCP
/// transcript. Use this from agents for bulk migrations: `Bash(curl --data-binary)`
/// only puts the curl command in the conversation, not the file content.
async fn ingest_handler(
    axum::extract::State(palace): axum::extract::State<std::sync::Arc<Palace>>,
    body: String,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    let mut items: Vec<StoreBatchItem> = Vec::new();
    for (lineno, line) in body.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        match serde_json::from_str::<StoreBatchItem>(trimmed) {
            Ok(it) => items.push(it),
            Err(e) => {
                return (
                    axum::http::StatusCode::BAD_REQUEST,
                    format!("parse error on line {}: {e}\n", lineno + 1),
                )
                    .into_response();
            }
        }
    }
    if items.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            "empty body — expected NDJSON of StoreBatchItem records\n",
        )
            .into_response();
    }

    let total = items.len();
    metrics::counter!("palazzo_ingest_items_total").increment(total as u64);
    tracing::info!(items = total, "POST /ingest streaming start");

    let stream = async_stream::stream! {
        let mut totals = crate::mcp::BatchCounts::default();
        let mut chunk_idx: u32 = 0;
        while !items.is_empty() {
            let take = items.len().min(MAX_STORE_BATCH);
            let chunk: Vec<StoreBatchItem> = items.drain(..take).collect();
            let chunk_len = chunk.len();
            let args = StoreBatchArgs {
                items: chunk,
                skip_duplicates: None,
            };
            match palace.do_store_batch(args).await {
                Ok(result) => {
                    totals.stored += result.counts.stored;
                    totals.duplicates_returned += result.counts.duplicates_returned;
                    totals.skipped_duplicates += result.counts.skipped_duplicates;
                    totals.failed += result.counts.failed;
                    let dedup_against: Vec<u64> = result
                        .items
                        .iter()
                        .filter_map(|e| e.duplicate_of)
                        .collect();
                    let line = serde_json::json!({
                        "chunk": chunk_idx,
                        "items_in_chunk": chunk_len,
                        "counts": result.counts,
                        "dedup_against": dedup_against,
                        "running": totals,
                    });
                    yield Ok::<_, std::io::Error>(axum::body::Bytes::from(format!("{line}\n")));
                }
                Err(e) => {
                    let line = serde_json::json!({
                        "chunk": chunk_idx,
                        "error": format!("{e:#}"),
                        "running": totals,
                    });
                    yield Ok(axum::body::Bytes::from(format!("{line}\n")));
                    return;
                }
            }
            chunk_idx += 1;
        }
        let line = serde_json::json!({
            "done": true,
            "total": total,
            "counts": totals,
        });
        yield Ok(axum::body::Bytes::from(format!("{line}\n")));
    };

    axum::response::Response::builder()
        .status(axum::http::StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "application/x-ndjson")
        .header("X-Accel-Buffering", "no")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

struct HealthState {
    qdrant_up: Arc<AtomicBool>,
}

async fn health_handler(
    axum::extract::State(state): axum::extract::State<Arc<HealthState>>,
) -> axum::Json<serde_json::Value> {
    let qdrant = if state.qdrant_up.load(Ordering::Relaxed) {
        "up"
    } else {
        "down"
    };
    axum::Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "qdrant": qdrant,
        "embedder": "ready",
    }))
}

/// Query params for `GET /export`.
#[derive(serde::Deserialize)]
struct ExportParams {
    #[serde(default = "default_true")]
    vectors: bool,
    #[serde(default)]
    wing: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    room: Option<String>,
    #[serde(default)]
    hall: Option<String>,
    #[serde(default)]
    since: Option<String>,
    #[serde(default)]
    until: Option<String>,
    #[serde(default)]
    include_superseded: bool,
}

fn default_true() -> bool {
    true
}

/// GET /export handler. Streams the collection as NDJSON via the scroll API.
async fn export_handler(
    axum::extract::State(qdrant): axum::extract::State<Arc<Qdrant>>,
    axum::extract::Query(params): axum::extract::Query<ExportParams>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    // Validate RFC3339 timestamps up front.
    for (name, val) in [("since", &params.since), ("until", &params.until)] {
        if let Some(s) = val
            && crate::util::parse_rfc3339(s).is_none()
        {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                format!(
                    "{} must be RFC3339 second-precision UTC (e.g. 2026-04-20T00:00:00Z), got {:?}\n",
                    name, s
                ),
            )
                .into_response();
        }
    }

    // Build the filter.
    let exclude_superseded_before = if params.include_superseded {
        None
    } else {
        Some(crate::util::now_rfc3339())
    };
    let filter = FindFilter {
        wing: params.wing.map(|w| w.trim().to_string()),
        category: params.category.map(|c| c.trim().to_string()),
        room: params.room.map(|r| r.trim().to_string()),
        hall: params.hall.map(|h| h.trim().to_string()),
        since: params.since,
        until: params.until,
        exclude_superseded_before,
    };

    metrics::counter!("palazzo_export_requests_total").increment(1);
    tracing::info!(vectors = params.vectors, "GET /export streaming start");

    const PAGE_SIZE: usize = 256;
    let qdrant_for_stream = qdrant.clone();
    let stream = async_stream::stream! {
        let mut offset: Option<serde_json::Value> = None;
        let mut total_points = 0u64;

        loop {
            match qdrant_for_stream.scroll(PAGE_SIZE, offset.clone(), &filter, params.vectors).await {
                Ok((points, next_offset)) => {
                    let page_count = points.len();
                    for pt in points {
                        total_points += 1;
                        match serde_json::to_string(&pt) {
                            Ok(line) => {
                                yield Ok::<_, std::io::Error>(
                                    axum::body::Bytes::from(format!("{line}\n"))
                                );
                                metrics::counter!("palazzo_export_points_total").increment(1);
                            }
                            Err(e) => {
                                let err_line = serde_json::json!({
                                    "error": format!("serialize point: {e}"),
                                });
                                yield Ok(axum::body::Bytes::from(format!("{err_line}\n")));
                                return;
                            }
                        }
                    }
                    if next_offset.is_none() || page_count < PAGE_SIZE {
                        // No more pages.
                        break;
                    }
                    offset = next_offset;
                }
                Err(e) => {
                    let err_line = serde_json::json!({
                        "error": format!("{e:#}"),
                    });
                    yield Ok(axum::body::Bytes::from(format!("{err_line}\n")));
                    return;
                }
            }
        }

        tracing::info!(total = total_points, "GET /export streaming complete");
    };

    axum::response::Response::builder()
        .status(axum::http::StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "application/x-ndjson")
        .header("X-Accel-Buffering", "no")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

async fn run_stdio() -> Result<()> {
    let cfg = Config::from_env();
    tracing::info!(
        backend = BACKEND,
        qdrant = %cfg.qdrant_url,
        collection = %cfg.collection,
        mode = "stdio",
        "palazzo starting"
    );

    if let Err(e) = cfg.make_qdrant().ensure_indexes().await {
        tracing::warn!("ensure_indexes: {e:#}");
    }

    let palace = cfg.make_palace()?;
    let service = palace.serve(stdio()).await.inspect_err(|e| {
        tracing::error!("mcp serve: {e:?}");
    })?;
    service.waiting().await?;
    Ok(())
}

async fn run_http(rest: &[String]) -> Result<()> {
    let mut bind = std::env::var("PALAZZO_BIND").unwrap_or_else(|_| "127.0.0.1:6334".into());
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--bind" => {
                bind = rest
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--bind requires an address"))?
                    .clone();
                i += 2;
            }
            other => anyhow::bail!("unknown serve argument: {other}"),
        }
    }

    let cfg = Config::from_env();
    tracing::info!(
        backend = BACKEND,
        qdrant = %cfg.qdrant_url,
        collection = %cfg.collection,
        bind = %bind,
        mode = "streamable-http",
        "palazzo starting"
    );

    if let Err(e) = cfg.make_qdrant().ensure_indexes().await {
        tracing::warn!("ensure_indexes: {e:#}");
    }

    let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .expect("prometheus recorder");

    let ct = CancellationToken::new();
    let ct_child = ct.child_token();
    let mut http_config = StreamableHttpServerConfig::default()
        .with_cancellation_token(ct_child)
        .with_accept_unknown_sessions(true);
    match std::env::var("PALAZZO_ALLOWED_HOSTS") {
        Ok(raw) if raw.trim() == "*" => {
            tracing::warn!(
                "PALAZZO_ALLOWED_HOSTS=* — DNS rebinding protection DISABLED. Ensure the listener is behind a trusted reverse proxy or firewall."
            );
            http_config = http_config.disable_allowed_hosts();
        }
        Ok(raw) => {
            let hosts: Vec<String> = raw
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            tracing::info!(?hosts, "Host header allowlist");
            http_config = http_config.with_allowed_hosts(hosts);
        }
        Err(_) => {
            tracing::info!(
                "Host header allowlist defaults to localhost/127.0.0.1/::1 — set PALAZZO_ALLOWED_HOSTS to accept remote clients."
            );
        }
    }

    // Single Embedder shared across all MCP sessions and /ingest.
    // Embedder::Clone shares the inner Arc<Mutex<TextEmbedding>> and Arc<AtomicU64>
    // so there is exactly one TextEmbedding in memory and one background watcher,
    // regardless of how many concurrent sessions are open. Per-session Palace
    // instances each hold a clone of this same Arc — no additional model loads.
    let shared_embedder = make_embedder(&cfg)?;
    let ingest_palace =
        std::sync::Arc::new(cfg.make_palace_with_embedder(shared_embedder.clone())?);
    let cfg = std::sync::Arc::new(cfg);
    let mcp_cfg = cfg.clone();
    let mcp_embedder = shared_embedder;
    // Disable the session activity timeout. rmcp's default is 5 min — agents
    // idle longer than that between tool calls get 404 Session not found.
    // Zombie sessions from crashed clients are cheap (one idle tokio task + a
    // HashMap entry) and cleared on the next server restart / deployment.
    let mut session_mgr = LocalSessionManager::default();
    session_mgr.session_config.keep_alive = None;
    let service = StreamableHttpService::new(
        move || {
            mcp_cfg
                .make_palace_with_embedder(mcp_embedder.clone())
                .map_err(std::io::Error::other)
        },
        std::sync::Arc::new(session_mgr),
        http_config,
    );

    let max_ingest = std::env::var("PALAZZO_MAX_INGEST_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(64 * 1024 * 1024);
    tracing::info!(max_ingest_bytes = max_ingest, "POST /ingest body cap");

    // Qdrant liveness poller — cached, refreshed every 30 s. /health reads the flag
    // without touching Qdrant, so the probe is cheap regardless of poll cadence.
    let qdrant_for_health = cfg.make_qdrant();
    let qdrant_up = Arc::new(AtomicBool::new(false));
    {
        let ok = qdrant_for_health
            .count(&FindFilter::default())
            .await
            .is_ok();
        qdrant_up.store(ok, Ordering::Relaxed);
        tracing::info!(qdrant_up = ok, "initial qdrant health check");
    }
    {
        let poller_qdrant = qdrant_for_health;
        let flag = qdrant_up.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(30));
            ticker.tick().await; // skip first immediate tick — already checked above
            loop {
                ticker.tick().await;
                let ok = poller_qdrant.count(&FindFilter::default()).await.is_ok();
                flag.store(ok, Ordering::Relaxed);
                tracing::debug!(qdrant_up = ok, "qdrant health poll");
            }
        });
    }
    let health_state = Arc::new(HealthState { qdrant_up });

    let ingest_route = axum::Router::new()
        .route("/ingest", axum::routing::post(ingest_handler))
        .layer(axum::extract::DefaultBodyLimit::max(max_ingest))
        .with_state(ingest_palace);
    let health_route = axum::Router::new()
        .route("/health", axum::routing::get(health_handler))
        .with_state(health_state);
    let qdrant_for_export = Arc::new(cfg.make_qdrant());
    let export_route = axum::Router::new()
        .route("/export", axum::routing::get(export_handler))
        .with_state(qdrant_for_export);
    let metrics_route = axum::Router::new().route(
        "/metrics",
        axum::routing::get(move || {
            let body = metrics_handle.render();
            async move { body }
        }),
    );
    let router = axum::Router::new()
        .nest_service("/mcp", service)
        .merge(ingest_route)
        .merge(health_route)
        .merge(export_route)
        .merge(metrics_route);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    tracing::info!(
        "listening on {bind}: POST /mcp (MCP), POST /ingest (NDJSON bulk), GET /export (NDJSON stream)"
    );

    let shutdown = ct.clone();
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutdown signal received");
            shutdown.cancel();
        })
        .await?;
    Ok(())
}
