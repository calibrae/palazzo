//! Fixed-window rate limiter for palazzo's auth endpoints.
//!
//! A brake against floods and brute-force attacks on `/whoami`, `/authorize`,
//! `/token`, and `/register`. Keyed on the first hop of `X-Forwarded-For`
//! (spoofable — this is a brake, not a per-client shield). Not a DDoS defense.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

struct Window {
    start: Instant,
    count: u32,
}

#[derive(Clone)]
pub struct RateLimiter {
    inner: Arc<Mutex<HashMap<String, Window>>>,
    max: u32,
    window: Duration,
}

impl RateLimiter {
    pub fn new(max: u32, window_secs: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            max,
            window: Duration::from_secs(window_secs),
        }
    }

    fn check(&self, key: &str) -> bool {
        let now = Instant::now();
        let mut map = self.inner.lock().unwrap();
        // Prune expired windows to bound memory when the key space is large.
        if map.len() > 10_000 {
            map.retain(|_, w| now.duration_since(w.start) < self.window);
        }
        let w = map.entry(key.to_string()).or_insert(Window {
            start: now,
            count: 0,
        });
        if now.duration_since(w.start) >= self.window {
            w.start = now;
            w.count = 0;
        }
        w.count += 1;
        w.count <= self.max
    }
}

fn client_key(req: &Request) -> String {
    req.headers()
        .get("x-forwarded-for")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "global".to_string())
}

pub async fn rate_limit(State(rl): State<RateLimiter>, req: Request, next: Next) -> Response {
    let key = client_key(&req);
    if rl.check(&key) {
        next.run(req).await
    } else {
        (
            StatusCode::TOO_MANY_REQUESTS,
            "rate limit exceeded — slow down\n",
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_window_limits_and_isolates_keys() {
        let rl = RateLimiter::new(2, 60);
        assert!(rl.check("k"));
        assert!(rl.check("k"));
        assert!(!rl.check("k"));
        assert!(rl.check("other"));
    }
}
