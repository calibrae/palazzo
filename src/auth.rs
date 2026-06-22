//! Optional self-hosted attribution (v1): a token-mint page (`/whoami`) plus a
//! bearer middleware that stamps a claimed identity onto writes. See
//! `docs/auth-v1.md`.
//!
//! **This is attribution, not authentication.** The email is unverified (no OTP
//! in v1) — it records who wrote what and soft-gates by domain. It is NOT a
//! defence against a malicious insider. Disabled by default (`PALAZZO_AUTH`
//! unset/`off`): no middleware, no identity, today's behaviour.
//!
//! Tokens are HS256 JWTs we assemble by hand over vetted RustCrypto primitives
//! (`hmac`/`sha2`) — no `ring`/`openssl`, no version tangle with reqwest.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Form, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{Html, IntoResponse, Response};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
use hmac::{Hmac, Mac};
use serde::Deserialize;
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Identity resolved from a valid bearer token, injected into the request so
/// tool handlers (`palace_store` etc.) can stamp `author`. Cloned into the
/// request's extensions; rmcp surfaces it to handlers via the HTTP `Parts`.
#[derive(Debug, Clone)]
pub struct AuthIdentity(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    Off,
    Email,
}

#[derive(Clone)]
pub struct AuthConfig {
    pub mode: AuthMode,
    signing_key: Arc<Vec<u8>>,
    allowed_domains: Arc<Vec<String>>,
    ttl_secs: u64,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl AuthConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let mode = match std::env::var("PALAZZO_AUTH").ok().as_deref() {
            Some("email") => AuthMode::Email,
            None | Some("off") | Some("") => AuthMode::Off,
            Some(other) => {
                anyhow::bail!("PALAZZO_AUTH must be 'off' or 'email', got {other:?}")
            }
        };
        let signing_key = std::env::var("PALAZZO_AUTH_SIGNING_KEY").unwrap_or_default();
        if mode == AuthMode::Email && signing_key.len() < 16 {
            anyhow::bail!(
                "PALAZZO_AUTH=email requires PALAZZO_AUTH_SIGNING_KEY (>= 16 bytes); generate one with `openssl rand -base64 48`"
            );
        }
        let allowed_domains: Vec<String> = std::env::var("PALAZZO_ALLOWED_EMAIL_DOMAINS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        let ttl_secs = std::env::var("PALAZZO_TOKEN_TTL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30 * 24 * 3600);
        Ok(Self {
            mode,
            signing_key: Arc::new(signing_key.into_bytes()),
            allowed_domains: Arc::new(allowed_domains),
            ttl_secs,
        })
    }

    #[cfg(test)]
    fn test(key: &str, domains: &[&str]) -> Self {
        Self {
            mode: AuthMode::Email,
            signing_key: Arc::new(key.as_bytes().to_vec()),
            allowed_domains: Arc::new(domains.iter().map(|d| d.to_lowercase()).collect()),
            ttl_secs: 3600,
        }
    }

    pub fn enabled(&self) -> bool {
        self.mode == AuthMode::Email
    }

    /// Basic shape check + domain allowlist. Empty allowlist accepts any domain.
    fn email_allowed(&self, email: &str) -> bool {
        let parts: Vec<&str> = email.split('@').collect();
        if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
            return false;
        }
        if self.allowed_domains.is_empty() {
            return true;
        }
        let domain = parts[1].to_lowercase();
        self.allowed_domains.contains(&domain)
    }

    fn sign(&self, signing_input: &str) -> String {
        let mut mac =
            HmacSha256::new_from_slice(&self.signing_key).expect("HMAC accepts any key length");
        mac.update(signing_input.as_bytes());
        B64.encode(mac.finalize().into_bytes())
    }

    /// Mint an HS256 token carrying the email as `sub`, with `iat`/`exp`.
    pub fn mint(&self, email: &str) -> String {
        let header = B64.encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let iat = now();
        let exp = iat + self.ttl_secs;
        let claims = serde_json::json!({ "sub": email, "iat": iat, "exp": exp }).to_string();
        let payload = B64.encode(claims.as_bytes());
        let signing_input = format!("{header}.{payload}");
        let sig = self.sign(&signing_input);
        format!("{signing_input}.{sig}")
    }

    /// Validate a token: constant-time signature check, then `exp`. Returns the
    /// `sub` (email) on success, `None` on any failure.
    pub fn verify(&self, token: &str) -> Option<String> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        let signing_input = format!("{}.{}", parts[0], parts[1]);
        let mut mac = HmacSha256::new_from_slice(&self.signing_key).ok()?;
        mac.update(signing_input.as_bytes());
        let sig = B64.decode(parts[2]).ok()?;
        mac.verify_slice(&sig).ok()?; // constant-time
        let claims: serde_json::Value = serde_json::from_slice(&B64.decode(parts[1]).ok()?).ok()?;
        let exp = claims.get("exp").and_then(serde_json::Value::as_u64)?;
        if now() >= exp {
            return None;
        }
        claims
            .get("sub")
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned)
    }
}

/// Bearer middleware applied to write routes when auth is enabled. Validates the
/// token, injects [`AuthIdentity`] into the request extensions (which rmcp
/// surfaces to tool handlers), and 401s otherwise.
pub async fn require_auth(State(cfg): State<AuthConfig>, mut req: Request, next: Next) -> Response {
    let token = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "));
    match token.and_then(|t| cfg.verify(t)) {
        Some(email) => {
            req.extensions_mut().insert(AuthIdentity(email));
            next.run(req).await
        }
        None => (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "unauthorized: present a valid bearer token — get one from /whoami\n",
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
pub struct WhoamiForm {
    email: String,
}

/// `GET /whoami` — the token-mint form.
pub async fn whoami_get() -> Html<&'static str> {
    Html(WHOAMI_FORM)
}

/// `POST /whoami` — validate the email against the domain allowlist, mint a
/// token, show it for the dev to paste into their MCP client config.
pub async fn whoami_post(
    State(cfg): State<AuthConfig>,
    headers: axum::http::HeaderMap,
    Form(form): Form<WhoamiForm>,
) -> Response {
    let email = form.email.trim().to_lowercase();
    if !cfg.email_allowed(&email) {
        return (
            StatusCode::BAD_REQUEST,
            Html(format!(
                "<!doctype html><meta charset=utf-8><body style=\"font-family:system-ui;max-width:40rem;margin:3rem auto\">\
                 <h2>Rejected</h2><p><code>{}</code> is not an accepted address (wrong domain or malformed).</p>\
                 <p><a href=\"/whoami\">back</a></p>",
                html_escape(&email)
            )),
        )
            .into_response();
    }
    let token = cfg.mint(&email);
    Html(token_page(&email, &token, &base_url(&headers))).into_response()
}

/// Best-effort public base URL for the example commands, from the request: the
/// `Host` header for authority, `X-Forwarded-Proto` for scheme (nginx sets it
/// in front of palazzo; absent → plain http, as on the LAN staging box).
fn base_url(headers: &axum::http::HeaderMap) -> String {
    let host = headers
        .get(axum::http::header::HOST)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("PALAZZO");
    let scheme = headers
        .get("x-forwarded-proto")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("http");
    format!("{scheme}://{host}")
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

const WHOAMI_FORM: &str = "<!doctype html><meta charset=utf-8>\
<title>palazzo — identify</title>\
<body style=\"font-family:system-ui;max-width:40rem;margin:3rem auto;line-height:1.5\">\
<h2>palazzo — get your access token</h2>\
<p>Enter your work email. You'll get a bearer token to paste into your MCP client config. \
This attributes what you store; it is not a password.</p>\
<form method=post action=/whoami>\
<input name=email type=email placeholder=you@example.com required \
style=\"padding:.5rem;width:20rem;font-size:1rem\">\
<button type=submit style=\"padding:.5rem 1rem;font-size:1rem\">Get token</button>\
</form></body>";

fn token_page(email: &str, token: &str, base: &str) -> String {
    // Full token embedded in every copy target — the buttons copy exactly what
    // you'd paste, token and all. Built here, escaped for display; the copy JS
    // reads textContent/value (entities decoded) so the clipboard gets raw text.
    let claude = format!(
        "claude mcp add --transport http palazzo {base}/mcp --header \"Authorization: Bearer {token}\""
    );
    let opencode = format!(
        r#"{{ "mcp": {{ "palazzo": {{ "type":"remote", "url":"{base}/mcp", "headers": {{ "Authorization":"Bearer {token}" }} }} }} }}"#
    );
    TOKEN_PAGE_TMPL
        .replace("__EMAIL__", &html_escape(email))
        .replace("__TOKEN__", &html_escape(token))
        .replace("__CLAUDE__", &html_escape(&claude))
        .replace("__OPENCODE__", &html_escape(&opencode))
}

const TOKEN_PAGE_TMPL: &str = r##"<!doctype html><meta charset=utf-8><title>palazzo — token</title>
<style>
body{font-family:system-ui;max-width:52rem;margin:3rem auto;line-height:1.5;padding:0 1rem}
.row{position:relative;margin:.6rem 0}
pre,textarea{background:#f4f4f8;border:1px solid #ddd;border-radius:6px;padding:1rem;
  width:100%;font-family:ui-monospace,monospace;font-size:.85rem;box-sizing:border-box;overflow:auto;white-space:pre-wrap;word-break:break-all}
button.copy{position:absolute;top:.5rem;right:.5rem;padding:.3rem .7rem;font-size:.8rem;
  border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
button.copy:hover{background:#eee}
h3{margin-top:1.6rem}
</style>
<body>
<h2>Token for __EMAIL__</h2>
<p>Paste as <code>Authorization: Bearer &lt;token&gt;</code> in your MCP client. Valid ~30 days; revisit when it expires.</p>
<div class=row><button class=copy data-target=tok>Copy token</button>
<textarea id=tok readonly rows=3>__TOKEN__</textarea></div>
<h3>Claude Code</h3>
<div class=row><button class=copy data-target=cc>Copy</button><pre id=cc>__CLAUDE__</pre></div>
<h3>OpenCode <span style="font-weight:normal;color:#666">(opencode.json)</span></h3>
<div class=row><button class=copy data-target=oc>Copy</button><pre id=oc>__OPENCODE__</pre></div>
<script>
document.querySelectorAll('button.copy').forEach(function(b){
  b.addEventListener('click',function(){
    var el=document.getElementById(b.dataset.target);
    var text=el.tagName==='TEXTAREA'?el.value:el.textContent;
    function done(){var o=b.textContent;b.textContent='Copied!';setTimeout(function(){b.textContent=o;},1200);}
    function fallback(){
      var ta=document.createElement('textarea');ta.value=text;ta.style.position='fixed';ta.style.opacity='0';
      document.body.appendChild(ta);ta.focus();ta.select();
      try{document.execCommand('copy');}catch(e){}
      document.body.removeChild(ta);done();
    }
    if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(text).then(done,fallback);}
    else{fallback();}
  });
});
</script>
</body>"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mint_then_verify_roundtrips() {
        let c = AuthConfig::test("super-secret-key-of-some-length", &[]);
        let t = c.mint("ada@example.com");
        assert_eq!(c.verify(&t).as_deref(), Some("ada@example.com"));
    }

    #[test]
    fn tampered_token_rejected() {
        let c = AuthConfig::test("super-secret-key-of-some-length", &[]);
        let t = c.mint("ada@example.com");
        // Flip a char inside the payload segment — changes the HMAC input, so
        // the signature no longer matches. (Tampering the signature's trailing
        // base64 char can hit unused bits and decode to the same bytes.)
        let dot = t.find('.').unwrap();
        let mut chars: Vec<char> = t.chars().collect();
        let idx = dot + 3;
        chars[idx] = if chars[idx] == 'A' { 'B' } else { 'A' };
        let tampered: String = chars.into_iter().collect();
        assert_eq!(c.verify(&tampered), None);
    }

    #[test]
    fn wrong_key_rejected() {
        let a = AuthConfig::test("key-aaaaaaaaaaaaaaaaaaaa", &[]);
        let b = AuthConfig::test("key-bbbbbbbbbbbbbbbbbbbb", &[]);
        let t = a.mint("ada@example.com");
        assert_eq!(b.verify(&t), None);
    }

    #[test]
    fn expired_token_rejected() {
        let mut c = AuthConfig::test("super-secret-key-of-some-length", &[]);
        c.ttl_secs = 0; // exp == iat == now → now() >= exp
        let t = c.mint("ada@example.com");
        assert_eq!(c.verify(&t), None);
    }

    #[test]
    fn domain_allowlist() {
        let c = AuthConfig::test("super-secret-key-of-some-length", &["nexpublica.fr"]);
        assert!(c.email_allowed("dev@nexpublica.fr"));
        assert!(c.email_allowed("dev@NexPublica.FR")); // case-insensitive
        assert!(!c.email_allowed("dev@gmail.com"));
        assert!(!c.email_allowed("not-an-email"));
        assert!(!c.email_allowed("@nexpublica.fr"));
    }

    #[test]
    fn empty_allowlist_accepts_any_valid_shape() {
        let c = AuthConfig::test("super-secret-key-of-some-length", &[]);
        assert!(c.email_allowed("anyone@anywhere.io"));
        assert!(!c.email_allowed("garbage"));
    }
}
