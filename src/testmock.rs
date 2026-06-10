//! Test-only mock Qdrant: a real HTTP server with canned responses and full
//! request capture, so unit tests exercise the actual reqwest/serde paths in
//! `qdrant.rs` instead of a hand-rolled trait double.
//!
//! Responses are keyed by `"METHOD /path-suffix"` where the suffix is the part
//! after `/collections/{name}` (e.g. `"POST /points/search"`). Each key holds
//! a FIFO queue — push one canned value per expected call. Unmatched calls get
//! a sensible empty default. A canned object may carry `"__status"` (HTTP
//! status override) and `"__body"` (response body when using `__status`).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{Method, StatusCode, Uri};
use serde_json::{Value, json};

#[derive(Default)]
pub struct MockState {
    /// Every request received: (method, full path, parsed JSON body).
    requests: Mutex<Vec<(String, String, Value)>>,
    /// Canned responses, keyed by `"METHOD /suffix"`.
    responses: Mutex<HashMap<String, VecDeque<Value>>>,
}

pub struct MockQdrant {
    pub url: String,
    state: Arc<MockState>,
}

/// `/collections/{name}/points/search` → `/points/search`. Paths that don't
/// have the prefix come back unchanged.
fn suffix_of(path: &str) -> String {
    let mut parts = path.splitn(4, '/');
    // ["", "collections", "{name}", "rest..."]
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(""), Some("collections"), Some(_), Some(rest)) => format!("/{rest}"),
        _ => path.to_string(),
    }
}

impl MockQdrant {
    pub async fn start() -> Self {
        let state = Arc::new(MockState::default());
        let router = axum::Router::new()
            .fallback(handler)
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        Self {
            url: format!("http://{addr}"),
            state,
        }
    }

    /// Queue a canned response for `key` (e.g. `"POST /points/search"`).
    pub fn push(&self, key: &str, resp: Value) {
        self.state
            .responses
            .lock()
            .unwrap()
            .entry(key.to_string())
            .or_default()
            .push_back(resp);
    }

    /// All request bodies received for `key`, in order.
    pub fn requests_for(&self, key: &str) -> Vec<Value> {
        self.state
            .requests
            .lock()
            .unwrap()
            .iter()
            .filter(|(m, p, _)| format!("{m} {}", suffix_of(p)) == key)
            .map(|(_, _, b)| b.clone())
            .collect()
    }
}

async fn handler(
    State(state): State<Arc<MockState>>,
    method: Method,
    uri: Uri,
    body: Bytes,
) -> (StatusCode, axum::Json<Value>) {
    let path = uri.path().to_string();
    let body_val: Value = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body).unwrap_or(Value::Null)
    };
    state
        .requests
        .lock()
        .unwrap()
        .push((method.to_string(), path.clone(), body_val));

    let suffix = suffix_of(&path);
    let key = format!("{method} {suffix}");
    let canned = state
        .responses
        .lock()
        .unwrap()
        .get_mut(&key)
        .and_then(|q| q.pop_front());

    match canned {
        Some(v) => {
            let status = v
                .get("__status")
                .and_then(Value::as_u64)
                .and_then(|s| StatusCode::from_u16(s as u16).ok())
                .unwrap_or(StatusCode::OK);
            let resp_body = v.get("__body").cloned().unwrap_or(v);
            (status, axum::Json(resp_body))
        }
        None => {
            let default = match suffix.as_str() {
                "/points/search" => json!({"result": []}),
                "/points/scroll" => {
                    json!({"result": {"points": [], "next_page_offset": null}})
                }
                "/points" if method == Method::POST => json!({"result": []}),
                "/points/count" => json!({"result": {"count": 0}}),
                "/facet" => json!({"result": {"hits": []}}),
                _ => json!({"result": {"status": "ok"}}),
            };
            (StatusCode::OK, axum::Json(default))
        }
    }
}
