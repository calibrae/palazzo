//! Self-hosted OAuth 2.1 Authorization Server for the "Authenticate" button.
//!
//! Turns palazzo into its own AS so MCP clients (Claude Code / Copilot / OpenCode)
//! drive the native flow: `/mcp` 401 → discovery → browser `/authorize` (email
//! login) → loopback redirect with a code → `/token` (PKCE) → access + refresh
//! tokens. Refresh is the low-maintenance bit: the client renews silently, the
//! dev never re-touches it.
//!
//! Login is email-only for now (unverified — attribution). The `authenticate()`
//! seam is where Entra/OIDC federation drops in later: swap the email form for a
//! redirect to the upstream IdP and resolve the email from its callback.
//!
//! Identity is still minted as palazzo's own HS256 token (see `auth.rs`), so the
//! bearer middleware and `author` stamping are unchanged regardless of how the
//! email was obtained.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Form, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Redirect, Response};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::auth::AuthConfig;

/// Authorization codes live here briefly between `/authorize` and `/token`.
/// Single process (one replica) — fine for staging/prod-single. Single-use,
/// short TTL. Stateless tokens mean this is the only server-side auth state.
#[derive(Clone)]
pub struct OAuthState {
    pub cfg: AuthConfig,
    codes: Arc<Mutex<HashMap<String, AuthCode>>>,
}

struct AuthCode {
    email: String,
    code_challenge: String,
    redirect_uri: String,
    expires_at: u64,
}

impl OAuthState {
    pub fn new(cfg: AuthConfig) -> Self {
        Self {
            cfg,
            codes: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

const CODE_TTL_SECS: u64 = 600;
const MAX_PENDING_CODES: usize = 10_000;

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn random_b64(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    getrandom::getrandom(&mut buf).expect("getrandom");
    B64.encode(buf)
}

/// PKCE S256: base64url(sha256(verifier)) must equal the stored challenge.
fn pkce_ok(verifier: &str, challenge: &str) -> bool {
    let mut h = Sha256::new();
    h.update(verifier.as_bytes());
    B64.encode(h.finalize()) == challenge
}

/// Only loopback (http) or https redirect targets — never an arbitrary http
/// origin, so palazzo can't be turned into an open redirector.
fn redirect_uri_ok(uri: &str) -> bool {
    // Split into scheme and the rest of the URI.
    let Some((scheme, rest)) = uri.split_once("://") else {
        return false;
    };
    // Authority is everything before the first path/query/fragment delimiter.
    let authority = match rest.find(['/', '?', '#']) {
        Some(pos) => &rest[..pos],
        None => rest,
    };
    // Strip optional "userinfo@" prefix (take the part after the last '@').
    let authority = if let Some(pos) = authority.rfind('@') {
        &authority[pos + 1..]
    } else {
        authority
    };
    // Extract the host, distinguishing IPv6 literals from host[:port].
    let host = if authority.starts_with('[') {
        // IPv6 literal: "[::1]" or "[::1]:port"
        let Some(end) = authority.find(']') else {
            return false;
        };
        &authority[1..end]
    } else {
        // Hostname or IPv4: take the part before the optional ":port".
        match authority.find(':') {
            Some(pos) => &authority[..pos],
            None => authority,
        }
    };
    match scheme {
        "https" => !host.is_empty(),
        "http" => matches!(host, "localhost" | "127.0.0.1" | "::1"),
        _ => false,
    }
}

// ---------- discovery metadata ----------

/// `GET /.well-known/oauth-protected-resource` (RFC 9728).
pub async fn protected_resource(State(st): State<OAuthState>, headers: HeaderMap) -> Response {
    let base = st.cfg.issuer_base(&headers);
    axum::Json(json!({
        "resource": base,
        "authorization_servers": [base],
    }))
    .into_response()
}

/// `GET /.well-known/oauth-authorization-server` (RFC 8414).
pub async fn authorization_server(State(st): State<OAuthState>, headers: HeaderMap) -> Response {
    let base = st.cfg.issuer_base(&headers);
    axum::Json(json!({
        "issuer": base,
        "authorization_endpoint": format!("{base}/authorize"),
        "token_endpoint": format!("{base}/token"),
        "registration_endpoint": format!("{base}/register"),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": ["palace"],
    }))
    .into_response()
}

/// `POST /register` (RFC 7591 Dynamic Client Registration). We're our own AS and
/// every client is public + PKCE-protected, so we accept any registration and
/// hand back a client_id. Nothing is stored — PKCE is what secures the exchange.
pub async fn register(body: Option<axum::Json<serde_json::Value>>) -> Response {
    let redirect_uris = body
        .as_ref()
        .and_then(|b| b.0.get("redirect_uris").cloned())
        .unwrap_or_else(|| json!([]));
    (
        StatusCode::CREATED,
        axum::Json(json!({
            "client_id": format!("palazzo-{}", random_b64(12)),
            "redirect_uris": redirect_uris,
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        })),
    )
        .into_response()
}

// ---------- authorize ----------

#[derive(Deserialize)]
pub struct AuthorizeQuery {
    #[serde(default)]
    redirect_uri: String,
    #[serde(default)]
    code_challenge: String,
    #[serde(default)]
    code_challenge_method: String,
    #[serde(default)]
    state: String,
    #[serde(default)]
    client_id: String,
}

/// `GET /authorize` — serve the email login form, carrying the OAuth params as
/// hidden fields so the POST can complete the flow. (The Entra seam: this is
/// where you'd 302 to the upstream IdP instead of rendering the form.)
pub async fn authorize_get(Query(q): Query<AuthorizeQuery>) -> Response {
    if q.code_challenge.is_empty() || q.code_challenge_method != "S256" {
        return (
            StatusCode::BAD_REQUEST,
            Html("<p>this server requires PKCE with code_challenge_method=S256</p>".to_string()),
        )
            .into_response();
    }
    if !redirect_uri_ok(&q.redirect_uri) {
        return (
            StatusCode::BAD_REQUEST,
            Html("<p>invalid redirect_uri (loopback or https only)</p>".to_string()),
        )
            .into_response();
    }
    Html(authorize_form(&q)).into_response()
}

#[derive(Deserialize)]
pub struct AuthorizeForm {
    email: String,
    redirect_uri: String,
    code_challenge: String,
    state: String,
    #[serde(default)]
    client_id: String,
}

/// `POST /authorize` — the login submit. Validate the email/domain, mint a
/// one-time code bound to the PKCE challenge + redirect_uri, redirect back.
pub async fn authorize_post(
    State(st): State<OAuthState>,
    Form(f): Form<AuthorizeForm>,
) -> Response {
    let _ = &f.client_id; // public client; recorded for symmetry, not trusted
    let email = f.email.trim().to_lowercase();
    if !st.cfg.email_allowed(&email) {
        return (
            StatusCode::BAD_REQUEST,
            Html(format!(
                "<!doctype html><meta charset=utf-8><body style=\"font-family:system-ui;max-width:40rem;margin:3rem auto\">\
                 <h2>Rejected</h2><p><code>{}</code> is not an accepted address.</p></body>",
                html_escape(&email)
            )),
        )
            .into_response();
    }
    if !redirect_uri_ok(&f.redirect_uri) {
        return (StatusCode::BAD_REQUEST, "invalid redirect_uri").into_response();
    }

    let code = random_b64(32);
    let at_capacity = {
        let mut codes = st.codes.lock().unwrap();
        let cutoff = now();
        codes.retain(|_, c| c.expires_at > cutoff); // prune expired
        if codes.len() >= MAX_PENDING_CODES {
            true
        } else {
            codes.insert(
                code.clone(),
                AuthCode {
                    email,
                    code_challenge: f.code_challenge,
                    redirect_uri: f.redirect_uri.clone(),
                    expires_at: now() + CODE_TTL_SECS,
                },
            );
            false
        }
    }; // lock released here
    if at_capacity {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "too many pending authorizations, retry shortly\n",
        )
            .into_response();
    }
    let sep = if f.redirect_uri.contains('?') {
        '&'
    } else {
        '?'
    };
    let url = format!(
        "{}{}code={}&state={}",
        f.redirect_uri,
        sep,
        urlencode(&code),
        urlencode(&f.state)
    );
    Redirect::to(&url).into_response()
}

// ---------- token ----------

#[derive(Deserialize)]
pub struct TokenForm {
    #[serde(default)]
    grant_type: String,
    #[serde(default)]
    code: String,
    #[serde(default)]
    code_verifier: String,
    #[serde(default)]
    redirect_uri: String,
    #[serde(default)]
    refresh_token: String,
}

/// `POST /token` — authorization_code (with PKCE) and refresh_token grants.
/// Issues a short access token + a long refresh token (both palazzo HS256).
pub async fn token(State(st): State<OAuthState>, Form(f): Form<TokenForm>) -> Response {
    let email = match f.grant_type.as_str() {
        "authorization_code" => {
            let entry = { st.codes.lock().unwrap().remove(&f.code) }; // single-use
            let Some(c) = entry else {
                return token_error("invalid_grant", "unknown or used code");
            };
            if now() >= c.expires_at {
                return token_error("invalid_grant", "code expired");
            }
            if c.redirect_uri != f.redirect_uri {
                return token_error("invalid_grant", "redirect_uri mismatch");
            }
            if !pkce_ok(&f.code_verifier, &c.code_challenge) {
                return token_error("invalid_grant", "PKCE verification failed");
            }
            c.email
        }
        "refresh_token" => match st.cfg.verify_refresh(&f.refresh_token) {
            Some(email) => email,
            None => return token_error("invalid_grant", "invalid or expired refresh token"),
        },
        other => {
            return token_error(
                "unsupported_grant_type",
                &format!("unsupported grant_type: {other}"),
            );
        }
    };

    axum::Json(json!({
        "access_token": st.cfg.mint_access(&email),
        "token_type": "Bearer",
        "expires_in": st.cfg.access_ttl_secs(),
        "refresh_token": st.cfg.mint_refresh(&email),
        "scope": "palace",
    }))
    .into_response()
}

fn token_error(error: &str, desc: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        axum::Json(json!({ "error": error, "error_description": desc })),
    )
        .into_response()
}

// ---------- helpers ----------

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Minimal percent-encoding for the redirect query (code is base64url so only
/// `+`/`/`-free already, but `state` is opaque client data).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn authorize_form(q: &AuthorizeQuery) -> String {
    format!(
        "<!doctype html><meta charset=utf-8><title>palazzo — sign in</title>\
<body style=\"font-family:system-ui;max-width:40rem;margin:3rem auto;line-height:1.5\">\
<h2>palazzo — sign in</h2>\
<p>Enter your work email to connect this MCP client. This attributes what you store.</p>\
<form method=post action=\"/authorize\">\
<input name=email type=email placeholder=you@example.com required \
style=\"padding:.5rem;width:20rem;font-size:1rem\">\
<button type=submit style=\"padding:.5rem 1rem;font-size:1rem\">Authorize</button>\
<input type=hidden name=redirect_uri value=\"{redirect_uri}\">\
<input type=hidden name=code_challenge value=\"{code_challenge}\">\
<input type=hidden name=state value=\"{state}\">\
<input type=hidden name=client_id value=\"{client_id}\">\
</form></body>",
        redirect_uri = html_escape(&q.redirect_uri),
        code_challenge = html_escape(&q.code_challenge),
        state = html_escape(&q.state),
        client_id = html_escape(&q.client_id),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_roundtrip() {
        // verifier → challenge per RFC 7636, then verify.
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let mut h = Sha256::new();
        h.update(verifier.as_bytes());
        let challenge = B64.encode(h.finalize());
        assert!(pkce_ok(verifier, &challenge));
        assert!(!pkce_ok("wrong-verifier", &challenge));
    }

    #[test]
    fn redirect_uri_policy() {
        assert!(redirect_uri_ok("http://localhost:8765/cb"));
        assert!(redirect_uri_ok("http://127.0.0.1:1/x"));
        assert!(redirect_uri_ok("https://app.example.com/cb"));
        assert!(!redirect_uri_ok("http://evil.example.com/cb"));
        assert!(!redirect_uri_ok("ftp://x"));
    }

    #[test]
    fn redirect_uri_rejects_lookalike_hosts() {
        assert!(!redirect_uri_ok("http://localhost.evil.com/cb"));
        assert!(!redirect_uri_ok("http://127.0.0.1.evil.com/cb"));
        assert!(redirect_uri_ok("http://[::1]:8080/cb"));
    }

    #[test]
    fn urlencode_keeps_unreserved_escapes_rest() {
        assert_eq!(urlencode("aZ09-_.~"), "aZ09-_.~");
        assert_eq!(urlencode("a b&c"), "a%20b%26c");
    }
}
