//! OpenAI-compatible `POST /v1/embeddings` shim over the shared fastembed
//! embedder. Lets any client speaking the OpenAI embeddings wire shape (e.g.
//! `rmcp-guide`) borrow palazzo's embedder as "just a URL" — no palazzo-specific
//! code on the caller side.
//!
//! **Prefix asymmetry.** nomic-embed-text-v1.5 is an asymmetric model: it wants
//! `search_query:` / `search_document:` instruction prefixes. The palace's own
//! pipeline (`palace_*`) deliberately runs PREFIX-FREE to stay vector-compatible
//! with the Ollama-embedded points already in `claude-memory`. This endpoint is
//! the opposite: it ADDS the prefix (keyed off `input_type`, default document)
//! so external corpora get the model's intended retrieval quality. The two live
//! in different vector spaces on purpose and must never share a collection.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::embed::Embedder;

const QUERY_PREFIX: &str = "search_query: ";
const DOC_PREFIX: &str = "search_document: ";
/// Per-item input cap. Generous — nomic ctx is ~8k tokens; 32 KB is well past
/// any sane chunk and stops a pathological body from flooding the embedder.
const MAX_ITEM_BYTES: usize = 32 * 1024;
/// Cap on items per request. Keeps one call from monopolising the embedder.
const MAX_INPUTS: usize = 512;

/// OpenAI `input` accepts a bare string or an array of strings. (Token-array
/// inputs are not supported — palazzo embeds text, not pre-tokenised ids.)
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Input {
    Single(String),
    Many(Vec<String>),
}

impl Input {
    fn into_vec(self) -> Vec<String> {
        match self {
            Input::Single(s) => vec![s],
            Input::Many(v) => v,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    /// Echoed back in the response. palazzo runs a single model, so the value is
    /// informational — any string is accepted.
    #[serde(default)]
    pub model: Option<String>,
    pub input: Input,
    /// `"query"` or `"document"` (default). Selects the nomic instruction prefix.
    /// Not part of the OpenAI spec, but a no-op for callers that omit it.
    #[serde(default)]
    pub input_type: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
}

#[derive(Debug)]
pub enum EmbedError {
    BadRequest(String),
    Embedder(String),
}

impl EmbedError {
    fn status(&self) -> StatusCode {
        match self {
            EmbedError::BadRequest(_) => StatusCode::BAD_REQUEST,
            EmbedError::Embedder(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
    fn message(&self) -> &str {
        match self {
            EmbedError::BadRequest(m) | EmbedError::Embedder(m) => m,
        }
    }
}

fn prefix_for(input_type: Option<&str>) -> &'static str {
    match input_type {
        Some("query") => QUERY_PREFIX,
        _ => DOC_PREFIX,
    }
}

/// Core logic: validate, prefix, embed, shape the response. Separated from the
/// axum handler so it's unit-testable with a fake embedder.
pub async fn do_embeddings(
    embedder: &Embedder,
    req: EmbeddingsRequest,
) -> Result<EmbeddingsResponse, EmbedError> {
    let model = req.model.unwrap_or_else(|| "nomic-embed-text".to_string());
    let prefix = prefix_for(req.input_type.as_deref());
    let inputs = req.input.into_vec();

    if inputs.is_empty() {
        return Err(EmbedError::BadRequest("input is empty".into()));
    }
    if inputs.len() > MAX_INPUTS {
        return Err(EmbedError::BadRequest(format!(
            "too many inputs: {} (max {MAX_INPUTS})",
            inputs.len()
        )));
    }
    for (i, s) in inputs.iter().enumerate() {
        if s.len() > MAX_ITEM_BYTES {
            return Err(EmbedError::BadRequest(format!(
                "input[{i}] too large: {} bytes (max {MAX_ITEM_BYTES})",
                s.len()
            )));
        }
    }

    let prefixed: Vec<String> = inputs.iter().map(|s| format!("{prefix}{s}")).collect();
    let vectors = embedder
        .embed_batch(&prefixed)
        .await
        .map_err(|e| EmbedError::Embedder(format!("{e:#}")))?;
    if vectors.len() != inputs.len() {
        return Err(EmbedError::Embedder(format!(
            "embedder returned {} vectors for {} inputs",
            vectors.len(),
            inputs.len()
        )));
    }

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingData {
            object: "embedding",
            index,
            embedding,
        })
        .collect();
    Ok(EmbeddingsResponse {
        object: "list",
        data,
        model,
    })
}

/// `POST /v1/embeddings`. State is a clone of the process-wide shared embedder.
pub async fn embeddings_handler(
    State(embedder): State<Embedder>,
    body: axum::body::Bytes,
) -> Response {
    let req: EmbeddingsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("invalid request body: {e}"),
            );
        }
    };
    match do_embeddings(&embedder, req).await {
        Ok(resp) => (StatusCode::OK, axum::Json(resp)).into_response(),
        Err(e) => error_response(e.status(), e.message()),
    }
}

/// `GET /v1/models` — OpenAI-compatible model list. palazzo serves one model;
/// discovery/validation tooling expects a non-empty `data` array here.
pub async fn models_handler() -> axum::Json<serde_json::Value> {
    axum::Json(json!({
        "object": "list",
        "data": [{
            "id": "nomic-embed-text",
            "object": "model",
            "owned_by": "palazzo",
        }],
    }))
}

fn error_response(status: StatusCode, msg: &str) -> Response {
    (
        status,
        axum::Json(json!({
            "error": { "message": msg, "type": "invalid_request_error" }
        })),
    )
        .into_response()
}

#[cfg(all(test, feature = "fastembed"))]
mod tests {
    use super::*;

    fn req(input: Input, input_type: Option<&str>) -> EmbeddingsRequest {
        EmbeddingsRequest {
            model: None,
            input,
            input_type: input_type.map(str::to_string),
        }
    }

    #[tokio::test]
    async fn shapes_response_openai_style() {
        let e = Embedder::fake();
        let resp = do_embeddings(&e, req(Input::Many(vec!["a".into(), "b".into()]), None))
            .await
            .unwrap();
        assert_eq!(resp.object, "list");
        assert_eq!(resp.model, "nomic-embed-text"); // default echo
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].object, "embedding");
        assert_eq!(resp.data[0].index, 0);
        assert_eq!(resp.data[1].index, 1);
        assert_eq!(resp.data[0].embedding.len(), 768);
    }

    #[tokio::test]
    async fn single_string_input_accepted() {
        let e = Embedder::fake();
        let resp = do_embeddings(&e, req(Input::Single("solo".into()), None))
            .await
            .unwrap();
        assert_eq!(resp.data.len(), 1);
    }

    #[tokio::test]
    async fn query_and_document_prefixes_differ() {
        // Same text, different input_type → different prefix → different vector.
        // Proves the prefix is applied and input_type is routed.
        let e = Embedder::fake();
        let q = do_embeddings(&e, req(Input::Single("hello".into()), Some("query")))
            .await
            .unwrap();
        let d = do_embeddings(&e, req(Input::Single("hello".into()), Some("document")))
            .await
            .unwrap();
        assert_ne!(q.data[0].embedding, d.data[0].embedding);
    }

    #[tokio::test]
    async fn default_input_type_is_document() {
        let e = Embedder::fake();
        let none = do_embeddings(&e, req(Input::Single("x".into()), None))
            .await
            .unwrap();
        let doc = do_embeddings(&e, req(Input::Single("x".into()), Some("document")))
            .await
            .unwrap();
        assert_eq!(none.data[0].embedding, doc.data[0].embedding);
    }

    #[tokio::test]
    async fn echoes_requested_model() {
        let e = Embedder::fake();
        let mut r = req(Input::Single("x".into()), None);
        r.model = Some("nomic-embed-text-v1.5".into());
        let resp = do_embeddings(&e, r).await.unwrap();
        assert_eq!(resp.model, "nomic-embed-text-v1.5");
    }

    #[tokio::test]
    async fn rejects_empty_and_oversized() {
        let e = Embedder::fake();
        assert!(matches!(
            do_embeddings(&e, req(Input::Many(vec![]), None)).await,
            Err(EmbedError::BadRequest(_))
        ));
        let too_many: Vec<String> = (0..=MAX_INPUTS).map(|i| i.to_string()).collect();
        assert!(matches!(
            do_embeddings(&e, req(Input::Many(too_many), None)).await,
            Err(EmbedError::BadRequest(_))
        ));
        let huge = "x".repeat(MAX_ITEM_BYTES + 1);
        assert!(matches!(
            do_embeddings(&e, req(Input::Single(huge), None)).await,
            Err(EmbedError::BadRequest(_))
        ));
    }
}
