use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use tokio::sync::Mutex;

/// Local ONNX embedding backend backed by fastembed-rs and
/// `nomic-embed-text-v1.5` (768-dim). Model is fetched on first run into the
/// fastembed cache (default `~/.cache/fastembed`) — set `FASTEMBED_CACHE_DIR`
/// to override (for packaged / read-only deployments).
#[derive(Clone)]
pub struct Embedder {
    inner: Arc<Mutex<TextEmbedding>>,
}

impl Embedder {
    /// Construct a new embedder. Blocks to load / download the model — do this
    /// once at startup, not per request.
    pub fn new() -> Result<Self> {
        // INT8 dynamic-quantised variant of nomic-embed-text-v1.5. Output stays 768-dim
        // and same vector space as V15, so collections embedded with V15 stay searchable;
        // resident weights drop ~330 MB and ONNX Runtime arenas shrink with the smaller
        // intermediate tensors. Empirical same-text cosine vs V15: ~0.98-0.99.
        let mut init = InitOptions::new(EmbeddingModel::NomicEmbedTextV15Q);
        if let Ok(dir) = std::env::var("FASTEMBED_CACHE_DIR") {
            init = init.with_cache_dir(dir.into());
        }
        let model =
            TextEmbedding::try_new(init).context("initialise fastembed NomicEmbedTextV15")?;
        Ok(Self {
            inner: Arc::new(Mutex::new(model)),
        })
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
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            let mut out = guard
                .embed(vec![text.as_str()], None)
                .map_err(|e| anyhow!("fastembed: {e}"))?;
            out.pop()
                .ok_or_else(|| anyhow!("fastembed returned zero embeddings"))
        })
        .await
        .context("fastembed embed task join")?
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
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
            for chunk in texts.chunks(chunk_size) {
                let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
                let mut vecs = guard
                    .embed(refs, None)
                    .map_err(|e| anyhow!("fastembed batch: {e}"))?;
                out.append(&mut vecs);
            }
            Ok(out)
        })
        .await
        .context("fastembed embed_batch task join")?
    }
}
