//! `POST /v1/image-embeddings` — nomic-embed-vision-v1.5 over fastembed.
//!
//! The vision model was trained into the SAME 768-dim latent space as
//! nomic-embed-text-v1.5, so an image vector is directly comparable to a text
//! vector from `POST /v1/embeddings` with `input_type: "query"` (text→image
//! search, image→document search). Like the text endpoint, nothing is stored:
//! bytes in, vectors out.
//!
//! Opt-in (`PALAZZO_VISION=1`): the model is ~100M parameters and a few
//! hundred MB of RSS that most deployments don't need. It loads lazily on the
//! first request (from the fastembed cache — `palazzo warm` pre-fetches it when
//! the flag is set) and has its own lock, so image traffic never queues behind
//! text embeds.
//!
//! Untrusted input: every image is size-capped, format-allowlisted (PNG, JPEG,
//! WebP, GIF) and pixel-capped from its header BEFORE fastembed decodes it, so a
//! small compressed file can't expand into a multi-GB bitmap.

use std::io::Cursor;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use fastembed::{ImageEmbedding, ImageEmbeddingModel, ImageInitOptions};
use image::{ImageFormat, ImageReader};
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::embeddings::{EmbeddingData, EmbeddingsResponse, Input};

pub const MODEL_ID: &str = "nomic-embed-vision-v1.5";
/// Images per request. Vision inference is far heavier than text.
pub const MAX_IMAGES: usize = 16;
/// Decoded (post-base64) bytes per image.
pub const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
/// Width × height per image, read from the header before any decode.
pub const MAX_PIXELS: u64 = 40_000_000;
/// Request body cap for the route: worst case all images at the byte cap,
/// base64-inflated (4/3), plus JSON slack.
pub const MAX_BODY_BYTES: usize = MAX_IMAGES * MAX_IMAGE_BYTES * 4 / 3 + 64 * 1024;

const ALLOWED: [ImageFormat; 4] = [
    ImageFormat::Png,
    ImageFormat::Jpeg,
    ImageFormat::WebP,
    ImageFormat::Gif,
];

/// `PALAZZO_VISION=1|true|on|yes` enables the endpoint (and `warm` fetches the model).
pub fn enabled() -> bool {
    matches!(
        std::env::var("PALAZZO_VISION").as_deref().map(str::trim),
        Ok("1" | "true" | "on" | "yes")
    )
}

#[derive(Clone)]
pub struct VisionEmbedder {
    inner: Arc<Mutex<Option<Inner>>>,
}

#[allow(clippy::large_enum_variant)]
enum Inner {
    Real(ImageEmbedding),
    #[cfg(test)]
    Fake,
}

impl Inner {
    fn embed(&mut self, images: &[&[u8]]) -> Result<Vec<Vec<f32>>> {
        match self {
            Inner::Real(m) => m
                .embed_bytes(images, None)
                .map_err(|e| anyhow!("fastembed vision: {e}")),
            #[cfg(test)]
            Inner::Fake => Ok(images.iter().map(|b| fake_vec(b)).collect()),
        }
    }
}

#[cfg(test)]
fn fake_vec(bytes: &[u8]) -> Vec<f32> {
    let mut h: u32 = 2_166_136_261;
    for b in bytes {
        h = (h ^ u32::from(*b)).wrapping_mul(16_777_619);
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

/// Blocking: loads (or on the very first run downloads) the vision model.
pub fn build_model() -> Result<ImageEmbedding> {
    let mut init = ImageInitOptions::new(ImageEmbeddingModel::NomicEmbedVisionV15);
    if let Ok(dir) = std::env::var("FASTEMBED_CACHE_DIR") {
        init = init.with_cache_dir(dir.into());
    }
    ImageEmbedding::try_new(init).context("initialise fastembed NomicEmbedVisionV15")
}

impl VisionEmbedder {
    /// Lazy: no model is loaded until the first embed.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
        }
    }

    #[cfg(test)]
    pub fn fake() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(Inner::Fake))),
        }
    }

    /// Embed already-validated image bytes. CPU-bound ONNX work runs on
    /// `spawn_blocking` (same reason as the text embedder: never stall a Tokio
    /// worker that is also driving MCP sessions).
    pub async fn embed(&self, images: Vec<Vec<u8>>) -> Result<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        let t0 = std::time::Instant::now();
        let out = tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            if guard.is_none() {
                tracing::info!("loading nomic-embed-vision-v1.5 (first image request)");
                *guard = Some(Inner::Real(build_model()?));
            }
            let refs: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
            guard
                .as_mut()
                .expect("vision model initialised above")
                .embed(&refs)
        })
        .await
        .context("vision embed task join")?;
        metrics::histogram!("palazzo_embed_duration_seconds", "mode" => "image")
            .record(t0.elapsed().as_secs_f64());
        out
    }
}

#[derive(Debug, Deserialize)]
pub struct ImageEmbeddingsRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// One or more images, each raw base64 or a `data:image/...;base64,` URL.
    pub input: Input,
}

/// Strip an optional data-URL header and decode base64 (padded or not).
pub(crate) fn decode_one(s: &str) -> Result<Vec<u8>, String> {
    let b64 = match s.strip_prefix("data:") {
        Some(rest) => {
            let (meta, data) = rest
                .split_once(',')
                .ok_or("malformed data URL (no comma)")?;
            if !meta.ends_with(";base64") {
                return Err("data URL must be base64-encoded".into());
            }
            data
        }
        None => s,
    };
    let b64: String = b64.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    // Cheap pre-check so a huge string is rejected before allocating its decode.
    if b64.len() / 4 * 3 > MAX_IMAGE_BYTES + 3 {
        return Err(format!("image too large (max {MAX_IMAGE_BYTES} bytes)"));
    }
    base64::engine::general_purpose::STANDARD
        .decode(&b64)
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(&b64))
        .map_err(|e| format!("invalid base64: {e}"))
}

/// What `check_image` learned from the header.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ImageInfo {
    pub format: &'static str,
    pub width: u32,
    pub height: u32,
}

/// Format allowlist + pixel cap, from the header only (no full decode).
pub(crate) fn check_image(bytes: &[u8]) -> Result<ImageInfo, String> {
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(format!(
            "image too large: {} bytes (max {MAX_IMAGE_BYTES})",
            bytes.len()
        ));
    }
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("unreadable image: {e}"))?;
    let fmt = reader.format().ok_or("unrecognised image format")?;
    if !ALLOWED.contains(&fmt) {
        return Err(format!(
            "unsupported image format {fmt:?} (allowed: PNG, JPEG, WebP, GIF)"
        ));
    }
    let (w, h) = reader
        .into_dimensions()
        .map_err(|e| format!("unreadable image header: {e}"))?;
    if w == 0 || h == 0 {
        return Err("empty image".into());
    }
    if u64::from(w) * u64::from(h) > MAX_PIXELS {
        return Err(format!(
            "image too large: {w}x{h} (max {MAX_PIXELS} pixels)"
        ));
    }
    let format = match fmt {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::WebP => "webp",
        _ => "gif",
    };
    Ok(ImageInfo {
        format,
        width: w,
        height: h,
    })
}

#[derive(Debug)]
pub enum VisionError {
    BadRequest(String),
    Embedder(String),
}

/// Validate, decode, embed, shape — separated from the handler for tests.
pub async fn do_image_embeddings(
    embedder: &VisionEmbedder,
    req: ImageEmbeddingsRequest,
) -> Result<EmbeddingsResponse, VisionError> {
    let model = req.model.unwrap_or_else(|| MODEL_ID.to_string());
    let inputs = req.input.into_vec();
    if inputs.is_empty() {
        return Err(VisionError::BadRequest("input is empty".into()));
    }
    if inputs.len() > MAX_IMAGES {
        return Err(VisionError::BadRequest(format!(
            "too many images: {} (max {MAX_IMAGES})",
            inputs.len()
        )));
    }
    let mut images = Vec::with_capacity(inputs.len());
    for (i, s) in inputs.iter().enumerate() {
        let bytes =
            decode_one(s).map_err(|e| VisionError::BadRequest(format!("input[{i}]: {e}")))?;
        check_image(&bytes).map_err(|e| VisionError::BadRequest(format!("input[{i}]: {e}")))?;
        images.push(bytes);
    }
    let n = images.len();
    let vectors = embedder
        .embed(images)
        .await
        .map_err(|e| VisionError::Embedder(format!("{e:#}")))?;
    if vectors.len() != n {
        return Err(VisionError::Embedder(format!(
            "embedder returned {} vectors for {n} images",
            vectors.len()
        )));
    }
    Ok(EmbeddingsResponse {
        object: "list",
        data: vectors
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| EmbeddingData {
                object: "embedding",
                index,
                embedding,
            })
            .collect(),
        model,
        usage: crate::embeddings::Usage::default(),
    })
}

/// `POST /v1/image-embeddings`.
pub async fn image_embeddings_handler(
    State(embedder): State<VisionEmbedder>,
    body: axum::body::Bytes,
) -> Response {
    let req: ImageEmbeddingsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return crate::embeddings::error_response(
                StatusCode::BAD_REQUEST,
                &format!("invalid request body: {e}"),
            );
        }
    };
    match do_image_embeddings(&embedder, req).await {
        Ok(resp) => (StatusCode::OK, axum::Json(resp)).into_response(),
        Err(VisionError::BadRequest(m)) => {
            crate::embeddings::error_response(StatusCode::BAD_REQUEST, &m)
        }
        Err(VisionError::Embedder(m)) => {
            tracing::error!("image embeddings: {m}");
            crate::embeddings::error_response(StatusCode::INTERNAL_SERVER_ERROR, &m)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    fn png(w: u32, h: u32) -> Vec<u8> {
        let mut out = Vec::new();
        DynamicImage::ImageRgb8(RgbImage::new(w, h))
            .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
            .unwrap();
        out
    }

    fn b64(bytes: &[u8]) -> String {
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn req(inputs: Vec<String>) -> ImageEmbeddingsRequest {
        ImageEmbeddingsRequest {
            model: None,
            input: Input::Many(inputs),
        }
    }

    #[tokio::test]
    async fn embeds_raw_base64_and_data_url() {
        let e = VisionEmbedder::fake();
        let p = png(4, 4);
        let resp = do_image_embeddings(
            &e,
            req(vec![b64(&p), format!("data:image/png;base64,{}", b64(&p))]),
        )
        .await
        .unwrap();
        assert_eq!(resp.model, MODEL_ID);
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[1].index, 1);
        assert_eq!(resp.data[0].embedding.len(), 768);
        // same bytes either way → same vector
        assert_eq!(resp.data[0].embedding, resp.data[1].embedding);
    }

    #[tokio::test]
    async fn rejects_bad_inputs() {
        let e = VisionEmbedder::fake();
        let bad = |inputs: Vec<String>| {
            let e = e.clone();
            async move { do_image_embeddings(&e, req(inputs)).await }
        };
        assert!(matches!(bad(vec![]).await, Err(VisionError::BadRequest(_))));
        let too_many = vec![b64(&png(1, 1)); MAX_IMAGES + 1];
        assert!(matches!(
            bad(too_many).await,
            Err(VisionError::BadRequest(_))
        ));
        assert!(matches!(
            bad(vec!["%%%not base64".into()]).await,
            Err(VisionError::BadRequest(_))
        ));
        // valid base64, not an image
        assert!(matches!(
            bad(vec![b64(b"hello world, not a picture")]).await,
            Err(VisionError::BadRequest(_))
        ));
        // data URL without ;base64
        assert!(matches!(
            bad(vec!["data:image/png,abc".into()]).await,
            Err(VisionError::BadRequest(_))
        ));
    }

    #[test]
    fn rejects_disallowed_format() {
        let mut bmp = Vec::new();
        DynamicImage::ImageRgb8(RgbImage::new(2, 2))
            .write_to(&mut Cursor::new(&mut bmp), ImageFormat::Bmp)
            .unwrap();
        let err = check_image(&bmp).unwrap_err();
        assert!(err.contains("unsupported image format"), "{err}");
    }

    #[test]
    fn pixel_cap_reads_header_not_body() {
        // A PNG header claiming 10000x10000 (100 MP) with no real pixel data:
        // must be rejected from the header alone, before any decode.
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&10_000u32.to_be_bytes());
        ihdr.extend_from_slice(&10_000u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB
        let mut hdr = b"\x89PNG\r\n\x1a\n".to_vec();
        for (kind, data) in [(&b"IHDR"[..], &ihdr[..]), (b"IDAT", &[]), (b"IEND", &[])] {
            hdr.extend_from_slice(&(data.len() as u32).to_be_bytes());
            let mut body = kind.to_vec();
            body.extend_from_slice(data);
            hdr.extend_from_slice(&body);
            hdr.extend_from_slice(&crc32(&body).to_be_bytes());
        }
        assert!(hdr.len() < 100, "fixture is header-only");
        let err = check_image(&hdr).unwrap_err();
        assert!(err.contains("max 40000000 pixels"), "{err}");
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut c = 0xFFFF_FFFFu32;
        for &b in data {
            c ^= u32::from(b);
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
        }
        !c
    }

    #[test]
    fn size_precheck_before_decode() {
        let huge = "A".repeat(MAX_IMAGE_BYTES * 2);
        assert!(decode_one(&huge).unwrap_err().contains("too large"));
    }

    #[test]
    fn body_cap_covers_worst_case_request() {
        let one = MAX_IMAGE_BYTES.div_ceil(3) * 4; // base64 length of a max image
        assert!(MAX_BODY_BYTES >= MAX_IMAGES * one);
    }
}
