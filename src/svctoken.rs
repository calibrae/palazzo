//! Service tokens for the `/docs` API: Ed25519-signed (JWT, `alg: EdDSA`).
//!
//! Root of trust = a keypair whose PRIVATE half stays with the operator (minted
//! offline, e.g. `palazzo service-token mint` on a laptop) and whose PUBLIC half
//! ships with the build: `service-trust.pub` in the repo (read by build.rs) or
//! `PALAZZO_SERVICE_PUBKEYS` at runtime. Nothing to configure per service, no
//! cluster access needed to mint.
//!
//! - Service token (`typ: palazzo-svc`): `sub` (service), `wing`, `mode` rw|ro,
//!   optional `aud` (= the instance's collection; absent = any instance),
//!   optional `exp`, `jti`.
//! - Admin token (`typ: palazzo-admin`): short-lived (≤ 1 h), only for
//!   `POST /admin/service-tokens/revoke`. Revoked jtis go to a file on the
//!   instance's state volume, re-read every few seconds.
//!
//! No trusted public key → the service API is off (404 `not_enabled`).

use std::collections::HashSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use aws_lc_rs::signature::{ED25519, Ed25519KeyPair, KeyPair, UnparsedPublicKey};
use base64::Engine;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD as B64};
use serde::{Deserialize, Serialize};

pub const TYP_SERVICE: &str = "palazzo-svc";
pub const TYP_ADMIN: &str = "palazzo-admin";
/// Admin tokens can't outlive this.
const MAX_ADMIN_TTL: u64 = 3600;
const RELOAD_EVERY: Duration = Duration::from_secs(5);
/// Public keys baked in at build time (build.rs ← service-trust.pub), comma-separated.
const BUILTIN_TRUST: &str = env!("PALAZZO_SERVICE_TRUST");

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn random_hex(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    getrandom::getrandom(&mut buf).expect("getrandom");
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Claims {
    pub typ: String,
    pub sub: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wing: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Instance (collection) the token is limited to; absent = any instance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,
    pub iat: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exp: Option<u64>,
    pub jti: String,
}

// ---------------------------------------------------------------------------
// Keys.

/// Raw 32-byte Ed25519 public keys, from comma/whitespace-separated base64.
pub fn parse_pubkeys(raw: &str) -> Result<Vec<Vec<u8>>> {
    raw.lines()
        .map(|line| line.split('#').next().unwrap_or("")) // drop comments
        .flat_map(|line| line.split(|c: char| c == ',' || c.is_whitespace()))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            let k = STANDARD
                .decode(s)
                .with_context(|| format!("public key {s:?} is not base64"))?;
            if k.len() != 32 {
                bail!(
                    "public key {s:?} is {} bytes, an Ed25519 key is 32",
                    k.len()
                );
            }
            Ok(k)
        })
        .collect()
}

/// Trusted public keys: `PALAZZO_SERVICE_PUBKEYS` if set (even empty = none),
/// else the ones baked into the build.
pub fn trusted_pubkeys() -> Result<Vec<Vec<u8>>> {
    match std::env::var("PALAZZO_SERVICE_PUBKEYS") {
        Ok(raw) => parse_pubkeys(&raw),
        Err(_) => parse_pubkeys(BUILTIN_TRUST),
    }
}

pub struct SigningKey(Ed25519KeyPair);

impl SigningKey {
    /// PKCS#8 (v1 or v2) PEM, as written by `openssl genpkey -algorithm ed25519`.
    pub fn from_pem(pem: &str) -> Result<Self> {
        let b64: String = pem
            .lines()
            .filter(|l| !l.starts_with("-----"))
            .map(str::trim)
            .collect();
        let der = STANDARD
            .decode(b64)
            .context("private key PEM is not base64")?;
        Ed25519KeyPair::from_pkcs8_maybe_unchecked(&der)
            .map(Self)
            .map_err(|e| anyhow!("not an Ed25519 PKCS#8 private key: {e}"))
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_pem(
            &std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
        )
    }

    #[cfg(test)]
    pub fn generate() -> Self {
        let rng = aws_lc_rs::rand::SystemRandom::new();
        let doc = Ed25519KeyPair::generate_pkcs8(&rng).expect("keygen");
        Self(Ed25519KeyPair::from_pkcs8(doc.as_ref()).expect("pkcs8"))
    }

    pub fn public_b64(&self) -> String {
        STANDARD.encode(self.0.public_key().as_ref())
    }

    pub fn encode(&self, claims: &Claims) -> String {
        let header = B64.encode(br#"{"alg":"EdDSA","typ":"JWT"}"#);
        let payload = B64.encode(serde_json::to_vec(claims).expect("claims serialise"));
        let input = format!("{header}.{payload}");
        let sig = B64.encode(self.0.sign(input.as_bytes()).as_ref());
        format!("{input}.{sig}")
    }

    pub fn mint_service(
        &self,
        name: &str,
        wing: &str,
        mode: &str,
        aud: Option<&str>,
        days: Option<u64>,
    ) -> Result<(String, Claims)> {
        let name = name.trim();
        if name.is_empty()
            || name.len() > 64
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
        {
            bail!("--name must be 1-64 chars of [A-Za-z0-9._-]");
        }
        let wing = crate::mcp::validate_tag("wing", wing)?;
        if mode != "rw" && mode != "ro" {
            bail!("--mode must be rw or ro");
        }
        let aud = aud
            .map(|a| crate::mcp::validate_tag("aud", a))
            .transpose()?;
        let iat = now();
        let c = Claims {
            typ: TYP_SERVICE.into(),
            sub: name.into(),
            wing: Some(wing),
            mode: Some(mode.into()),
            aud,
            iat,
            exp: days.map(|d| iat + d * 86_400),
            jti: random_hex(8),
        };
        Ok((self.encode(&c), c))
    }

    pub fn mint_admin(&self, ttl_secs: u64) -> Result<(String, Claims)> {
        if ttl_secs == 0 || ttl_secs > MAX_ADMIN_TTL {
            bail!("admin tokens live 1..={MAX_ADMIN_TTL} seconds");
        }
        let iat = now();
        let c = Claims {
            typ: TYP_ADMIN.into(),
            sub: "admin".into(),
            wing: None,
            mode: None,
            aud: None,
            iat,
            exp: Some(iat + ttl_secs),
            jti: random_hex(8),
        };
        Ok((self.encode(&c), c))
    }
}

/// Claims without checking the signature. For `inspect` only.
pub fn decode_unverified(token: &str) -> Result<Claims> {
    let mid = token
        .split('.')
        .nth(1)
        .ok_or_else(|| anyhow!("not a JWT"))?;
    serde_json::from_slice(&B64.decode(mid).context("payload is not base64url")?)
        .context("payload is not palazzo service claims")
}

/// Signature against any trusted key, header alg, expiry. Type-specific checks
/// are the caller's.
fn verify_sig(keys: &[Vec<u8>], token: &str, at: u64) -> Option<Claims> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let header: serde_json::Value = serde_json::from_slice(&B64.decode(parts[0]).ok()?).ok()?;
    if header.get("alg").and_then(|a| a.as_str()) != Some("EdDSA") {
        return None;
    }
    let input = format!("{}.{}", parts[0], parts[1]);
    let sig = B64.decode(parts[2]).ok()?;
    if !keys.iter().any(|k| {
        UnparsedPublicKey::new(&ED25519, k)
            .verify(input.as_bytes(), &sig)
            .is_ok()
    }) {
        return None;
    }
    let c: Claims = serde_json::from_slice(&B64.decode(parts[1]).ok()?).ok()?;
    if c.exp.is_some_and(|exp| at >= exp) {
        return None;
    }
    Some(c)
}

// ---------------------------------------------------------------------------
// Server-side verifier.

pub fn default_revoked_path() -> PathBuf {
    std::env::var("PALAZZO_SERVICE_REVOKED_FILE")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/var/lib/palazzo/service-revoked.txt"))
}

struct Revoked {
    checked: Option<Instant>,
    mtime: Option<SystemTime>,
    jtis: HashSet<String>,
}

pub struct Verifier {
    keys: Vec<Vec<u8>>,
    audience: String,
    revoked_path: PathBuf,
    revoked: Mutex<Revoked>,
}

pub enum Verdict {
    NotEnabled,
    Invalid,
    Valid(Claims),
}

impl Verifier {
    /// `audience` = this instance's collection, matched against a token's `aud`.
    pub fn new(keys: Vec<Vec<u8>>, audience: impl Into<String>, revoked_path: PathBuf) -> Self {
        Self {
            keys,
            audience: audience.into(),
            revoked_path,
            revoked: Mutex::new(Revoked {
                checked: None,
                mtime: None,
                jtis: HashSet::new(),
            }),
        }
    }

    pub fn is_enabled(&self) -> bool {
        !self.keys.is_empty()
    }

    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    fn is_revoked(&self, jti: &str) -> bool {
        let mut r = self.revoked.lock().expect("revoked lock");
        if r.checked.is_none_or(|t| t.elapsed() >= RELOAD_EVERY) {
            r.checked = Some(Instant::now());
            let m = std::fs::metadata(&self.revoked_path)
                .and_then(|m| m.modified())
                .ok();
            if m != r.mtime {
                r.jtis = std::fs::read_to_string(&self.revoked_path)
                    .map(|s| {
                        s.lines()
                            .map(|l| l.trim().to_string())
                            .filter(|l| !l.is_empty())
                            .collect()
                    })
                    .unwrap_or_default();
                r.mtime = m;
            }
        }
        r.jtis.contains(jti)
    }

    pub fn check_service(&self, token: &str) -> Verdict {
        if !self.is_enabled() {
            return Verdict::NotEnabled;
        }
        match verify_sig(&self.keys, token, now()) {
            Some(c)
                if c.typ == TYP_SERVICE
                    && c.wing.as_deref().is_some_and(|w| !w.trim().is_empty())
                    && matches!(c.mode.as_deref(), Some("rw" | "ro"))
                    && c.aud.as_deref().is_none_or(|a| a == self.audience)
                    && !self.is_revoked(&c.jti) =>
            {
                Verdict::Valid(c)
            }
            _ => Verdict::Invalid,
        }
    }

    pub fn check_admin(&self, token: &str) -> Verdict {
        if !self.is_enabled() {
            return Verdict::NotEnabled;
        }
        match verify_sig(&self.keys, token, now()) {
            Some(c)
                if c.typ == TYP_ADMIN
                    && c.exp
                        .is_some_and(|e| e.saturating_sub(c.iat) <= MAX_ADMIN_TTL)
                    && !self.is_revoked(&c.jti) =>
            {
                Verdict::Valid(c)
            }
            _ => Verdict::Invalid,
        }
    }

    /// Append a jti to the revocation list (state volume).
    pub fn revoke(&self, jti: &str) -> Result<()> {
        let jti = jti.trim();
        if jti.is_empty() || jti.len() > 64 || !jti.chars().all(|c| c.is_ascii_hexdigit()) {
            bail!("jti must be hex (see `palazzo service-token inspect <token>`)");
        }
        if let Some(dir) = self.revoked_path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.revoked_path)
            .with_context(|| format!("open {}", self.revoked_path.display()))?;
        writeln!(f, "{jti}")?;
        self.revoked.lock().expect("revoked lock").checked = None; // effective now
        Ok(())
    }

    pub fn revoked_list(&self) -> Vec<String> {
        self.is_revoked(""); // refresh
        let mut v: Vec<String> = self
            .revoked
            .lock()
            .expect("revoked lock")
            .jtis
            .iter()
            .cloned()
            .collect();
        v.sort();
        v
    }
}

// ---------------------------------------------------------------------------
// CLI: runs where the PRIVATE key is (the operator's machine), not on the server.

fn default_signing_key_path() -> PathBuf {
    std::env::var("PALAZZO_SERVICE_SIGNING_KEY_FILE")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".config/palazzo/service-signing.key")
        })
}

pub async fn run_cli(args: &[String]) -> Result<()> {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let key = || -> Result<SigningKey> {
        SigningKey::from_file(
            &flag("--key")
                .map(PathBuf::from)
                .unwrap_or_else(default_signing_key_path),
        )
    };
    match args.first().map(String::as_str) {
        Some("mint") => {
            let name = flag("--name").ok_or_else(|| anyhow!("--name is required"))?;
            let wing = flag("--wing").ok_or_else(|| anyhow!("--wing is required"))?;
            let mode = flag("--mode").unwrap_or_else(|| "ro".into());
            let days = if args.iter().any(|a| a == "--no-expiry") {
                None
            } else {
                Some(
                    flag("--days")
                        .map(|d| d.parse::<u64>().context("--days must be a number"))
                        .transpose()?
                        .unwrap_or(365),
                )
            };
            let (token, c) =
                key()?.mint_service(&name, &wing, &mode, flag("--aud").as_deref(), days)?;
            eprintln!(
                "minted {} token for {} on wing {:?}{} (jti {}, {})",
                c.mode.as_deref().unwrap_or(""),
                c.sub,
                c.wing.as_deref().unwrap_or(""),
                c.aud
                    .as_deref()
                    .map_or(" on any instance".into(), |a| format!(
                        " for instance {a:?}"
                    )),
                c.jti,
                c.exp
                    .map_or("never expires".into(), |e| format!("expires at unix {e}"))
            );
            println!("{token}");
            Ok(())
        }
        Some("admin") => {
            let mins = flag("--minutes")
                .map(|m| m.parse::<u64>().context("--minutes must be a number"))
                .transpose()?
                .unwrap_or(10);
            let (token, c) = key()?.mint_admin(mins * 60)?;
            eprintln!("minted admin token (jti {}, valid {mins} min)", c.jti);
            println!("{token}");
            Ok(())
        }
        Some("revoke") => {
            let url =
                flag("--url").ok_or_else(|| anyhow!("--url https://<instance> is required"))?;
            let jti = args
                .iter()
                .skip(1)
                .filter(|a| !a.starts_with("--"))
                .find(|a| {
                    Some(a.as_str()) != flag("--url").as_deref()
                        && Some(a.as_str()) != flag("--key").as_deref()
                })
                .ok_or_else(|| anyhow!("usage: service-token revoke --url <instance> <jti>"))?;
            let (admin, _) = key()?.mint_admin(600)?;
            let resp = reqwest::Client::new()
                .post(format!(
                    "{}/admin/service-tokens/revoke",
                    url.trim_end_matches('/')
                ))
                .bearer_auth(admin)
                .json(&serde_json::json!({ "jti": jti }))
                .send()
                .await
                .context("POST /admin/service-tokens/revoke")?;
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                bail!("revoke failed: HTTP {status}: {body}");
            }
            eprintln!("revoked {jti} on {url}");
            Ok(())
        }
        Some("inspect") => {
            let token = args
                .get(1)
                .ok_or_else(|| anyhow!("usage: service-token inspect <token>"))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&decode_unverified(token)?)?
            );
            Ok(())
        }
        Some("pubkey") => {
            println!("{}", key()?.public_b64());
            Ok(())
        }
        _ => bail!(
            "usage (run where the PRIVATE key is; --key defaults to ~/.config/palazzo/service-signing.key):\n  \
             palazzo service-token mint --name N --wing W [--mode ro|rw] [--aud <collection>] [--days 365 | --no-expiry]\n  \
             palazzo service-token revoke --url https://<instance> <jti>\n  \
             palazzo service-token admin [--minutes 10]\n  \
             palazzo service-token inspect <token>\n  \
             palazzo service-token pubkey"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("palazzo-revoked-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn verifier(k: &SigningKey, tag: &str) -> Verifier {
        Verifier::new(
            parse_pubkeys(&k.public_b64()).unwrap(),
            "collection-a",
            tmp(tag),
        )
    }

    #[test]
    fn service_token_lifecycle() {
        let k = SigningKey::generate();
        let v = verifier(&k, "svc");
        let (t, c) = k
            .mint_service("demo-svc", "demo-svc", "rw", None, Some(1))
            .unwrap();
        assert!(matches!(v.check_service(&t), Verdict::Valid(ref got) if *got == c));
        // tampered claims
        let mut parts: Vec<String> = t.split('.').map(String::from).collect();
        let mut forged = c.clone();
        forged.wing = Some("other".into());
        parts[1] = B64.encode(serde_json::to_vec(&forged).unwrap());
        assert!(matches!(
            v.check_service(&parts.join(".")),
            Verdict::Invalid
        ));
        // signed by another key
        let (other, _) = SigningKey::generate()
            .mint_service("x", "w", "rw", None, None)
            .unwrap();
        assert!(matches!(v.check_service(&other), Verdict::Invalid));
        // expired
        let mut old = c.clone();
        old.exp = Some(1);
        assert!(matches!(v.check_service(&k.encode(&old)), Verdict::Invalid));
        // revoked
        v.revoke(&c.jti).unwrap();
        assert!(matches!(v.check_service(&t), Verdict::Invalid));
        assert_eq!(v.revoked_list(), vec![c.jti.clone()]);
    }

    #[test]
    fn audience_and_types() {
        let k = SigningKey::generate();
        let v = verifier(&k, "aud");
        let (here, _) = k
            .mint_service("s", "w", "ro", Some("collection-a"), None)
            .unwrap();
        let (there, _) = k
            .mint_service("s", "w", "ro", Some("collection-b"), None)
            .unwrap();
        assert!(matches!(v.check_service(&here), Verdict::Valid(_)));
        assert!(
            matches!(v.check_service(&there), Verdict::Invalid),
            "other instance's token"
        );
        let (admin, _) = k.mint_admin(600).unwrap();
        assert!(
            matches!(v.check_service(&admin), Verdict::Invalid),
            "admin token isn't a service token"
        );
        assert!(matches!(v.check_admin(&admin), Verdict::Valid(_)));
        assert!(
            matches!(v.check_admin(&here), Verdict::Invalid),
            "service token isn't an admin token"
        );
        assert!(k.mint_admin(MAX_ADMIN_TTL + 1).is_err());
        // a forged long-lived admin token (signed, but exp - iat > 1h) is refused
        let mut long = decode_unverified(&admin).unwrap();
        long.exp = Some(long.iat + 7 * 86_400);
        assert!(matches!(v.check_admin(&k.encode(&long)), Verdict::Invalid));
    }

    #[test]
    fn no_trusted_keys_means_not_enabled() {
        let v = Verifier::new(vec![], "c", tmp("none"));
        assert!(!v.is_enabled());
        assert!(matches!(v.check_service("x.y.z"), Verdict::NotEnabled));
        assert!(matches!(v.check_admin("x.y.z"), Verdict::NotEnabled));
    }

    #[test]
    fn pubkey_parsing() {
        let k = SigningKey::generate();
        let two = format!(
            "{}, {}\n# comment\n",
            k.public_b64(),
            SigningKey::generate().public_b64()
        );
        assert_eq!(parse_pubkeys(&two).unwrap().len(), 2);
        assert_eq!(parse_pubkeys("").unwrap().len(), 0);
        assert!(parse_pubkeys("bm90IGEga2V5").is_err(), "wrong length");
        assert!(parse_pubkeys("%%%").is_err());
    }

    #[test]
    fn pem_roundtrip_and_inspect() {
        let rng = aws_lc_rs::rand::SystemRandom::new();
        let doc = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let pem = format!(
            "-----BEGIN PRIVATE KEY-----\n{}\n-----END PRIVATE KEY-----\n",
            STANDARD.encode(doc.as_ref())
        );
        let k = SigningKey::from_pem(&pem).unwrap();
        let (t, c) = k.mint_service("s", "w", "rw", None, None).unwrap();
        assert_eq!(decode_unverified(&t).unwrap(), c);
        assert!(
            SigningKey::from_pem(
                "-----BEGIN PRIVATE KEY-----\nbm9wZQ==\n-----END PRIVATE KEY-----"
            )
            .is_err()
        );
    }
}
