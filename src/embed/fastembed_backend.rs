use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use tokio::sync::Mutex;

/// Local ONNX embedding backend backed by fastembed-rs and
/// `nomic-embed-text-v1.5` (768-dim). Model is fetched on first run into the
/// fastembed cache (default `~/.cache/fastembed`) — set `FASTEMBED_CACHE_DIR`
/// to override (for packaged / read-only deployments).
///
/// The ONNX runtime's CPU arena grows monotonically with cumulative inference
/// work and never shrinks — a long-lived process creeps from ~440 MB cold to
/// multiple GB. To bound that, a background watcher recycles the underlying
/// `TextEmbedding` when **both** conditions hold:
///
///   * process RSS exceeds `FASTEMBED_RECYCLE_RSS_MB` (default 1500, 0 disables)
///   * no embed has finished in the last `FASTEMBED_RECYCLE_IDLE_SECS` (default 30)
///
/// The idle gate is the important part — Cali's hard requirement: never
/// recycle while MCP traffic is in flight. The watcher polls every 10 s and
/// only fires when both predicates are true, so a burst of requests delays
/// the recycle until the next lull.
///
/// The actual recycle builds a fresh `TextEmbedding` off-lock (model loads
/// from the on-disk cache, ~0.5-1 s) and only takes the `Mutex` for the
/// pointer swap, which is sub-millisecond. Even if a request DOES arrive
/// during the rebuild window, it just blocks on the lock for that single ms.
#[derive(Clone)]
pub struct Embedder {
    inner: Arc<Mutex<Inner>>,
    /// Unix-seconds timestamp of the last embed completion. Read by the
    /// background watcher to decide whether traffic is quiet.
    last_embed_at: Arc<AtomicU64>,
}

/// The actual embedding implementation behind the mutex. `Fake` exists only
/// for unit tests — deterministic 768-dim vectors with no model download.
/// (Variant size difference is irrelevant: exactly one `Inner` exists per
/// process, behind an `Arc<Mutex<_>>`.)
#[allow(clippy::large_enum_variant)]
enum Inner {
    Real(TextEmbedding),
    #[cfg(test)]
    Fake,
}

impl Inner {
    fn embed(&mut self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        match self {
            Inner::Real(m) => m.embed(texts, None).map_err(|e| anyhow!("fastembed: {e}")),
            #[cfg(test)]
            Inner::Fake => Ok(texts.iter().map(|t| fake_vec(t)).collect()),
        }
    }
}

/// Deterministic pseudo-embedding: same text → same vector, different text →
/// different vector. Not semantically meaningful — tests control similarity
/// through canned Qdrant responses, not vector geometry.
#[cfg(test)]
fn fake_vec(text: &str) -> Vec<f32> {
    let mut h: u32 = 2_166_136_261;
    for b in text.bytes() {
        h = (h ^ u32::from(b)).wrapping_mul(16_777_619);
    }
    (0..768)
        .map(|i| {
            h = h
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223 + i as u32);
            (h as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

/// Build a fresh `TextEmbedding`. Blocking — loads the model from the on-disk
/// cache (or downloads it on the very first run). Call from `spawn_blocking`.
fn build_model() -> Result<TextEmbedding> {
    // INT8 dynamic-quantised variant of nomic-embed-text-v1.5. Output stays 768-dim
    // and same vector space as V15, so collections embedded with V15 stay searchable;
    // resident weights drop ~330 MB and ONNX Runtime arenas shrink with the smaller
    // intermediate tensors. Empirical same-text cosine vs V15: ~0.98-0.99.
    let mut init = InitOptions::new(EmbeddingModel::NomicEmbedTextV15Q);
    if let Ok(dir) = std::env::var("FASTEMBED_CACHE_DIR") {
        init = init.with_cache_dir(dir.into());
    }
    TextEmbedding::try_new(init).context("initialise fastembed NomicEmbedTextV15")
}

/// Current process resident set size in MiB. Linux-only (reads the `VmRSS`
/// line of `/proc/self/status`, which is already in kB — no page-size
/// assumption, correct on 16K-page ARM kernels too). Returns `None` elsewhere,
/// which disables recycling on non-Linux dev boxes.
fn current_rss_mb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Embedder {
    /// Construct a new embedder. Blocks to load / download the model — do this
    /// once at startup, not per request. Spawns a background watcher that
    /// recycles the embedder during idle windows when RSS has grown too much.
    pub fn new() -> Result<Self> {
        let model = build_model()?;
        let me = Self {
            inner: Arc::new(Mutex::new(Inner::Real(model))),
            last_embed_at: Arc::new(AtomicU64::new(now_secs())),
        };

        let recycle_rss_mb: u64 = std::env::var("FASTEMBED_RECYCLE_RSS_MB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1500);
        let recycle_idle_secs: u64 = std::env::var("FASTEMBED_RECYCLE_IDLE_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);

        if recycle_rss_mb == 0 {
            tracing::info!("fastembed embedder recycling disabled (FASTEMBED_RECYCLE_RSS_MB=0)");
        } else {
            tracing::info!(
                recycle_rss_mb,
                recycle_idle_secs,
                "fastembed embedder will recycle when RSS exceeds threshold AND no embed in last N seconds"
            );
            me.spawn_watcher(recycle_rss_mb, recycle_idle_secs);
        }

        Ok(me)
    }

    /// Test-only embedder: deterministic vectors, no model load, no watcher.
    #[cfg(test)]
    pub fn fake() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::Fake)),
            last_embed_at: Arc::new(AtomicU64::new(now_secs())),
        }
    }

    /// Embed a single string.
    ///
    /// ONNX inference is synchronous and CPU-bound — 50-200 ms+ per call. It
    /// MUST run on `spawn_blocking`, not the async worker: blocking a Tokio
    /// worker thread for that long stalls every other future it is responsible
    /// for, including the Streamable-HTTP MCP sessions' keep-alive and read
    /// tasks. That stall is exactly what made agents "lose MCP mid-insertion".
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // nomic-embed-text-v1.5 expects an instruction prefix for search-corpus
        // documents; without it embeddings are still usable but slightly off
        // from the model card's reference. The Ollama-side pipeline does NOT
        // add the prefix, so we omit it here too — the point is to stay as
        // close as possible to the existing vectors in `claude-memory`.
        let inner = self.inner.clone();
        let text = text.to_string();
        let t0 = std::time::Instant::now();
        let out = tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            let mut out = guard.embed(vec![text.as_str()])?;
            out.pop()
                .ok_or_else(|| anyhow!("fastembed returned zero embeddings"))
        })
        .await
        .context("fastembed embed task join")?;
        metrics::histogram!("palazzo_embed_duration_seconds", "mode" => "single")
            .record(t0.elapsed().as_secs_f64());
        self.last_embed_at.store(now_secs(), Ordering::Release);
        out
    }

    /// Embed a batch of strings, sub-chunked so the ONNX runtime arenas stay
    /// bounded. Empirically a single batch of 256 long sequences peaks at ~6 GB
    /// RSS on an AVX2 worker — well past what a small VM can absorb. We chunk
    /// into groups of `FASTEMBED_BATCH_CHUNK` (default 16, override with the
    /// env var of the same name) and concatenate; per-call ONNX overhead is
    /// dwarfed by the matmul itself, so throughput barely moves.
    ///
    /// Runs on `spawn_blocking` for the same reason as [`Self::embed`] — a
    /// multi-chunk batch can occupy a CPU for seconds, which must never happen
    /// on an async worker thread.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let chunk_size: usize = std::env::var("FASTEMBED_BATCH_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|n: &usize| *n > 0)
            .unwrap_or(16);
        let inner = self.inner.clone();
        let texts = texts.to_vec();
        let t0 = std::time::Instant::now();
        let out = tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
            for chunk in texts.chunks(chunk_size) {
                let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
                let mut vecs = guard.embed(refs).context("fastembed batch")?;
                out.append(&mut vecs);
            }
            Ok(out)
        })
        .await
        .context("fastembed embed_batch task join")?;
        metrics::histogram!("palazzo_embed_duration_seconds", "mode" => "batch")
            .record(t0.elapsed().as_secs_f64());
        self.last_embed_at.store(now_secs(), Ordering::Release);
        out
    }

    /// Background task: every 10 s, check RSS and idle time; recycle if both
    /// trigger. Single-flight via `AtomicBool` (only one recycle in progress
    /// at a time, even though under normal conditions the watcher's own tick
    /// rate already serializes them).
    fn spawn_watcher(&self, recycle_rss_mb: u64, recycle_idle_secs: u64) {
        let inner = self.inner.clone();
        let last_embed_at = self.last_embed_at.clone();
        let recycling = Arc::new(AtomicBool::new(false));
        tokio::spawn(async move {
            // Light cadence — we're checking RSS, not driving the recycle.
            // 10 s gives a recycle a few-tick window to fire once RSS crosses
            // the line and traffic stays quiet, without spinning.
            let mut tick = tokio::time::interval(Duration::from_secs(10));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;

                if recycling.load(Ordering::Acquire) {
                    continue;
                }
                let Some(rss) = current_rss_mb() else {
                    continue; // non-Linux: recycling silently disabled
                };
                if rss < recycle_rss_mb {
                    continue;
                }
                let idle = now_secs().saturating_sub(last_embed_at.load(Ordering::Acquire));
                if idle < recycle_idle_secs {
                    tracing::debug!(
                        rss_mb = rss,
                        idle_secs = idle,
                        recycle_idle_secs,
                        "RSS over threshold but embedder is busy — deferring recycle"
                    );
                    continue;
                }
                if recycling
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    continue;
                }

                tracing::info!(
                    rss_mb = rss,
                    threshold_mb = recycle_rss_mb,
                    idle_secs = idle,
                    "fastembed RSS over threshold and quiet — recycling embedder"
                );
                let build_res = tokio::task::spawn_blocking(build_model).await;
                match build_res {
                    Ok(Ok(fresh)) => {
                        {
                            let mut guard = inner.lock().await;
                            *guard = Inner::Real(fresh);
                        }
                        let after = current_rss_mb().unwrap_or(0);
                        metrics::counter!("palazzo_embedder_recycles_total").increment(1);
                        tracing::info!(rss_mb = after, "fastembed embedder recycled");
                    }
                    Ok(Err(e)) => {
                        tracing::error!("fastembed recycle: rebuild failed: {e:#}");
                    }
                    Err(e) => {
                        tracing::error!("fastembed recycle: rebuild task join failed: {e}");
                    }
                }
                recycling.store(false, Ordering::Release);
            }
        });
    }
}
