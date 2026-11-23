# Release notes

One entry per iteration, newest first. Every commit that changes behaviour adds
its note here in the same commit (see `CLAUDE.md` → "Release notes"). Entries
under **Unreleased** roll up into the next tagged version. Pre-v0.14.2 history
lives in the git log.

## Unreleased

## v0.15.0 — 2026-11-23

Highlights:

- palazzo can now be a knowledge store for services. They write documents under their own keys and search them by scope through the `/docs` API, using Ed25519 tokens minted offline, so there's no per-service config.
- Optional image support (`PALAZZO_VISION=1`): image embeddings and searchable image memories in a separate collection.
- Original-time support (`event_time`) and REST siblings of the MCP tools (`GET /find`, `GET /stats`).
- Qdrant hardening: API key support (`QDRANT_API_KEY`), and compatibility with Qdrant 1.19+ through the Query API.
- MCP transport on vanilla rmcp 3.x (3.11), plus a steady run of security bumps. `/health` now shows the exact build (`<version>+<sha>`).

All new features are opt-in: an instance that doesn't enable them behaves as before, apart from the two items under **Changes** and **Breaking**.

### Breaking

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

### Features

- **feat: `POST /docs/search`**: scoped search for services in their token's wing. The `scope` list of `room:hall` / `room:*` pairs is mandatory (up to 1000, grouped into one Qdrant filter), with an optional category any-of and `limit` 1–20. Hits come back in the `/find` shape, including `external_id` and `meta`.
- **feat: service document API `/docs` with Ed25519 service tokens**: services can create, replace and retract documents by their own `external_id` in one wing bound to a token (`rw` or `ro`, optionally limited to one instance with `aud`). `PUT`/`GET`/`DELETE /docs/{external_id}`, `POST /docs/batch` and `POST /docs/delete`. Keyed identity reports created, unchanged, updated (metadata only, no re-embed) or replaced (old version superseded). Deletes remove all versions after logging each to the WAL. Tokens are minted offline by the private-key holder (`palazzo service-token mint|revoke|inspect|pubkey`). The server trusts the public keys in `service-trust.pub` (baked in at build) or `PALAZZO_SERVICE_PUBKEYS`, so there's no per-service config and no cluster access. Revocation goes through `POST /admin/service-tokens/revoke`, using short-lived admin tokens, with the list stored on the state volume. Points gain two optional fields, `external_id` and a `meta` map, returned in `/find` hits when set. With no trusted key, `/docs` answers 404 `not_enabled`.
- **feat: commit hash in the version**: `/health`, `/stats` and `--version` report `<version>+<8-char commit>` (e.g. `0.14.2+4aafed5a`, matching the `palazzo-<sha>` image tag), so you can tell which build a pod or host is serving. `build.rs` takes it from `PALAZZO_GIT_SHA`, then GitLab's `CI_COMMIT_SHA`, then `git rev-parse` (with `-dirty` for uncommitted changes), then the `GIT_COMMIT` file that `git archive` stamps. The MCP server version stays plain semver.
- **feat: `QDRANT_API_KEY`**: when set, every Qdrant request carries an `api-key` header, so palazzo works with a Qdrant that has `service.api_key` enabled. That covers MCP tools, `/ingest`, `/find`, `/export`, `/stats`, the health poll, collection provisioning and image memories. The header is marked sensitive and never logged. Unset keeps today's behaviour. `scripts/smoke.py` honours it too.
- **feat: `/v1/embeddings` works with stock OpenAI clients**: responses now include the `usage` block the OpenAI SDK expects (tokens aren't counted, so both values are 0). Clients that can't send `input_type` can pick the nomic prefix by model name instead: `nomic-embed-text-query` gets the query prefix, `nomic-embed-text` / `nomic-embed-text-document` the document prefix (the default). `input_type` still wins when given.
- **feat: image memories (opt-in with `PALAZZO_VISION=1`)**: three new MCP tools, `palace_store_image`, `palace_find_images` (search by text description) and `palace_delete_images`, plus `POST /images` (NDJSON, 16 per request) for bulk uploads. Images go into a separate `<COLLECTION>-images` collection and are never mixed into text search. Each point stores the vision vector, metadata and a `source` reference, never the bytes. Dedup is by SHA-256, and deletes are WAL-logged first. Without the flag, the tools aren't listed and the route isn't mounted.
- **feat: image embeddings (`POST /v1/image-embeddings`, opt-in `PALAZZO_VISION=1`)**: nomic-embed-vision-v1.5, in the same 768-dim space as the text model, so images can be searched with text queries. Nothing is stored. Input is base64 or a data URL; PNG/JPEG/WebP/GIF only; 16 images, 8 MiB and 40 MP per image, with the pixel cap checked from the header before decoding. The model loads on first use; `Dockerfile.cloud` bakes it only with `--build-arg PALAZZO_VISION=1`.
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
- **feat: `GET /find` REST endpoint** — HTTP sibling of the `palace_find` MCP
  tool. Semantic search over the palace for non-MCP clients (built for the fugo
  workflow engine's palazzo RAG node). Query params map 1:1 to `palace_find`
  (`query` + `wing`/`category`/`room`/`hall`/`author`/`since`/`until`/`limit`/
  `recency_half_life_days`/`include_superseded`); returns a JSON array of
  `Memory` hits. Server-side fastembed; shares the `/ingest` Palace. Gated by
  `require_auth` when `PALAZZO_AUTH=email`, open when auth is off.

### Changes

- **change (all instances): searches hide not-yet-valid points**: default `palace_find`, `GET /find` and `/docs/search` now also exclude points whose `valid_from` is in the future, on top of those whose `valid_until` has passed. palazzo itself only ever writes `valid_from` as "now", so existing data is unaffected. New `valid_from` datetime index.

### Fixes

- **fix: Qdrant 1.19+ compatibility**: vector search moves from the legacy `POST /points/search` to the Query API (`POST /points/query`, available since Qdrant 1.10), because Qdrant 1.19 drops the search endpoints from its API. That covers `palace_find`, `GET /find`, dedup checks and `palace_find_images`. The smoke test passes against Qdrant 1.13.5, 1.17.1, 1.18.3, 1.19.2 and 1.20.2.
- **fix: image tools no longer advertised when vision is off**: `#[tool_handler]` defaults to a fresh full router on every call, so `palace_store_image`, `palace_find_images` and `palace_delete_images` were listed even without `PALAZZO_VISION` (calls failed with "disabled"). It now serves the trimmed router.
- **fix: embedder recycle loop**: after a recycle, the RSS watcher waits for 256 MB of new growth before recycling again, so a baseline above the threshold no longer rebuilds the text model on every idle tick. The default threshold rises to 2000 MB when vision is enabled.

### Security

- **security: bump `cookie` 0.18.1 → 0.18.3 (RUSTSEC-2026-0594, private-cookie nonces can repeat after fork) and `url` 2.5.8 → 2.5.9 (RUSTSEC-2026-0592, stack overflow in `Url::origin()` on nested `blob:` URLs)**.
- **security: bump `rmcp` 3.1.2 → 3.11.0** (RUSTSEC-2026-0461: unbounded tool-schema cache in the Streamable HTTP server transport; also -0457 and -0462 in client code we don't use) **and `quinn-udp` 0.5.14 → 0.5.16** (RUSTSEC-2026-0538). Behaviour changes from rmcp: methods the server doesn't advertise (e.g. `resources/list`) return -32601. Checked: stdio smoke test, plus HTTP on protocols 2025-06-18 and 2025-11-25 and stale-session-after-restart (404, then re-initialize).
- **security: bump `tokio` 1.52.1 → 1.54.1** (RUSTSEC-2026-0386: `AsyncFd::new` and related constructors violate I/O safety). CI runs `cargo audit --deny warnings`, so this was blocking the pipeline.
- **security: bump `quinn-proto` 0.11.15 → 0.11.19 and `rustls` 0.23.45 → 0.23.46** (RUSTSEC-2026-0338..0342, -0346..0349: QUIC frame-handling panics/memory exhaustion; rustls secret-leaking `Debug`, ECH data leak, pre-auth Brotli allocation, key-exchange downgrade).
- **security: eMudhra CA distrust** — bump `rustls-webpki` 0.103.15 → 0.103.16 and `webpki-roots` / `webpki-root-certs` 1.0.7 → 1.0.10 (RUSTSEC-2026-0382 / CVE-2026-111943: compromised eMudhra intermediates). Outbound TLS (Qdrant, Ollama, HF model download) now refuses certs chaining to them.
- **security: hardened container image** — `Dockerfile.cloud` runtime moves from `debian:trixie-slim`
  to Chainguard `glibc-dynamic` (glibc, libstdc++, libgcc_s, CA certs; no shell, no package
  manager, no curl). Trivy: 188 OS vulns → 0. The Docker `HEALTHCHECK` is dropped (no curl in the
  image; Kubernetes uses its own HTTP probes on `/health`). Also `fastembed` now uses its rustls
  features instead of native-tls and drops the unused `image-models` default, so the binary no
  longer links OpenSSL (`libssl`/`libcrypto`) and the dependency tree shrinks 456 → 377 crates.
- **fix(deps): clear `cargo audit` gate** — bump `h2` 0.4.13 → 0.4.19 (RUSTSEC-2026-0258,
  unbounded empty DATA frames) and `rustls` 0.23.38 → 0.23.45 (RUSTSEC-2026-0285, TLS 1.3
  handshake messages accepted across encryption levels; pulls `aws-lc-rs`/`aws-lc-sys` and
  `rustls-webpki` forward). Also replaces yanked `chacha20` 0.10.0 → 0.10.2 and `der` 0.8.0 →
  0.8.2. Lockfile-only; unblocks the CI security stage.
- **fix(deps): bump `crossbeam-epoch` 0.9.18 → 0.9.20** — RUSTSEC-2026-0204
  (invalid pointer deref in the `fmt::Pointer` impl for `Atomic`/`Shared`).
  Semver-compatible patch bump; clears the `cargo_audit` gate that had been
  failing the CI `security` stage since early July and blocking the
  downstream package/deploy stages.

### Maintenance

- **chore: `.cargo/audit.toml`**: ignores RUSTSEC-2026-0543 (chrono unmaintained; informational, pulled in by rmcp, schemars and mcp-gain). It would otherwise fail CI's `cargo audit --deny warnings`.
- **chore: Rust 1.99 compatibility** — `AtomicU64::fetch_update` → `try_update` (deprecated in 1.99; CI builds with `-D warnings`).

## v0.14.2 — 2026-07-02

- Security + hardening pass: OAuth `redirect_uri` now host-exact validated
  (open-redirect / auth-code-capture fix), dep-free fixed-window rate limiting
  on `/authorize` `/token` `/whoami` `/register`, dedup widened top-1 → top-K
  plus intra-batch dedup, `palace_supersede` reports `ok:false` on nonexistent
  IDs, `Dockerfile.cloud` bakes the fastembed model at build time (no runtime
  HuggingFace egress). Bumped anyhow for RUSTSEC-2026-0190.
