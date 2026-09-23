# Release notes

One entry per iteration, newest first. Every commit that changes behaviour adds
its note here in the same commit (see `CLAUDE.md` → "Release notes"). Entries
under **Unreleased** roll up into the next tagged version. Pre-v0.14.2 history
lives in the git log.

## Unreleased

- **feat: nullable `event_time` on stored memories + `time_field` selector on `palace_find`/`GET /find`** —
  `palace_store`/`palace_store_batch`/`POST /ingest` accept an optional `event_time`
  (RFC3339), the original time the underlying event happened (e.g. an email's Date
  header), independent of `timestamp` (still always the write-time stamp, unchanged).
  Omitted `event_time` writes no key to Qdrant — old points are untouched, no backfill.
  `palace_find`/`GET /find` gain `time_field` ("timestamp" default, or "event_time") to
  redirect `since`/`until` range filtering and the recency re-rank onto whichever field
  you mean; `since`/`until` with `time_field=event_time` naturally excludes points with
  no `event_time` set, and such points get no recency boost. Default behavior is
  byte-for-byte unchanged.
- **feat: `GET /stats` endpoint** — JSON palace stats (the `palace_status` view over
  HTTP): `collection`, `total`, facet counts by `wings`/`halls`/`categories`, plus
  `version` and `embedder`. Reuses `do_status`; shares the `/ingest` Palace. Auth-gated
  when `PALAZZO_AUTH=email`, open otherwise. Returns 503 if Qdrant is down. (`GET
  /metrics` remains the Prometheus scrape endpoint for time-series.)
- **feat!: migrate MCP transport to vanilla rmcp 3.1.2** — drop the `calibrae/rmcp`
  fork + the `[patch.crates-io]` pin. `legacy_session_mode` stays ON by default
  (stateful): access logs show ~997 real `GET /mcp` SSE-stream opens from live Claude
  Code clients, which sessionless would 405 — so we keep serving them; modern
  2026-07-28 clients get the stateless path regardless (the flag only governs legacy
  clients). Override with `PALAZZO_LEGACY_SESSION_MODE=false` for sessionless without
  a rebuild. The retired fork's `accept_unknown_sessions` is gone, so the redeploy-404
  blip returns (a stale session 404s once, then the client re-initializes) — accepted.
  Pins `ProtocolVersion::V_2025_11_25`. One shared `Palace` built once and cloned per
  request (shared embedder/qdrant/wal/tracker), required under the per-request factory.
  `Content`→`ContentBlock` (3.x rename).
- **fix(deps): bump `crossbeam-epoch` 0.9.18 → 0.9.20** — RUSTSEC-2026-0204
  (invalid pointer deref in the `fmt::Pointer` impl for `Atomic`/`Shared`).
  Semver-compatible patch bump; clears the `cargo_audit` gate that had been
  failing the CI `security` stage since early July and blocking the
  downstream package/deploy stages.
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
