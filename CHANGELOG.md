# Release notes

One entry per iteration, newest first. Every commit that changes behaviour adds
its note here in the same commit (see `CLAUDE.md` → "Release notes"). Entries
under **Unreleased** roll up into the next tagged version. Pre-v0.14.2 history
lives in the git log.

## Unreleased

- **feat: `GET /find` REST endpoint** — HTTP sibling of the `palace_find` MCP
  tool. Semantic search over the palace for non-MCP clients (built for the fugo
  workflow engine's palazzo RAG node). Query params map 1:1 to `palace_find`
  (`query` + `wing`/`category`/`room`/`hall`/`author`/`since`/`until`/`limit`/
  `recency_half_life_days`/`include_superseded`); returns a JSON array of
  `Memory` hits. Server-side fastembed; shares the `/ingest` Palace. Gated by
  `require_auth` when `PALAZZO_AUTH=email`, open when auth is off.

## v0.14.2 — 2026-07-02

- Security + hardening pass: OAuth `redirect_uri` now host-exact validated
  (open-redirect / auth-code-capture fix), dep-free fixed-window rate limiting
  on `/authorize` `/token` `/whoami` `/register`, dedup widened top-1 → top-K
  plus intra-batch dedup, `palace_supersede` reports `ok:false` on nonexistent
  IDs, `Dockerfile.cloud` bakes the fastembed model at build time (no runtime
  HuggingFace egress). Bumped anyhow for RUSTSEC-2026-0190.
