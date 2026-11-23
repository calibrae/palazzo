//! Embeds the commit the binary was built from as `PALAZZO_GIT_SHA` (8 chars,
//! like the `palazzo-<sha>` image tags), shown as `<version>+<sha>` in
//! `/health`, `/stats` and `--version`.
//!
//! Sources, first hit wins:
//!   1. `PALAZZO_GIT_SHA` env: explicit override.
//!   2. `CI_COMMIT_SHA` env: GitLab CI.
//!   3. `git rev-parse HEAD` in a checkout, plus `-dirty` with uncommitted changes.
//!   4. the `GIT_COMMIT` file, which `git archive` stamps via export-subst
//!      (tarball builds have no `.git`).
//!   5. `unknown`.

use std::process::Command;

fn main() {
    // Service-token trust anchors (public keys only), baked in when present.
    // A build without the file trusts nothing unless PALAZZO_SERVICE_PUBKEYS is set.
    println!("cargo:rerun-if-changed=service-trust.pub");
    let trust = std::fs::read_to_string("service-trust.pub")
        .map(|s| {
            s.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    println!("cargo:rustc-env=PALAZZO_SERVICE_TRUST={trust}");

    println!("cargo:rerun-if-env-changed=PALAZZO_GIT_SHA");
    println!("cargo:rerun-if-env-changed=CI_COMMIT_SHA");
    println!("cargo:rerun-if-changed=GIT_COMMIT");
    if let Ok(head) = std::fs::read_to_string(".git/HEAD") {
        println!("cargo:rerun-if-changed=.git/HEAD");
        println!("cargo:rerun-if-changed=.git/index");
        if let Some(r) = head.strip_prefix("ref: ") {
            println!("cargo:rerun-if-changed=.git/{}", r.trim());
        }
    }

    let sha = env_sha("PALAZZO_GIT_SHA")
        .or_else(|| env_sha("CI_COMMIT_SHA"))
        .or_else(git_sha)
        .or_else(file_sha)
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=PALAZZO_GIT_SHA={sha}");
}

fn short(full: &str) -> Option<String> {
    let s = full.trim();
    (s.len() >= 8 && s.chars().all(|c| c.is_ascii_hexdigit())).then(|| s[..8].to_lowercase())
}

fn env_sha(var: &str) -> Option<String> {
    std::env::var(var).ok().and_then(|v| short(&v))
}

fn git_sha() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = short(&String::from_utf8_lossy(&out.stdout))?;
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .map(|o| o.status.success() && !o.stdout.is_empty())
        .unwrap_or(false);
    Some(if dirty { format!("{sha}-dirty") } else { sha })
}

fn file_sha() -> Option<String> {
    // Unsubstituted in a normal checkout ("$Format:%H$"), so short() rejects it.
    short(&std::fs::read_to_string("GIT_COMMIT").ok()?)
}
