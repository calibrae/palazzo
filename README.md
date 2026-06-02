<p align="center">
  <img src=".assets/hero.png" alt="palazzo" width="100%">
</p>

<h1 align="center">
  <img src=".assets/logo.png" alt="" width="64" align="middle">
  &nbsp;palazzo
</h1>

MCP server exposing a Qdrant-backed memory palace — typed wings, rooms, and halls instead of a generic blob store.

Cali's Rust daemon. Stdio or Streamable HTTP. No web UI, no auth, no drama.

## What it is

A single-binary Rust MCP server that:

- Speaks MCP over **stdio** (run locally) or **Streamable HTTP** (run as a service for a team or a homelab)
- Embeds text via one of two backends — local ONNX (`fastembed-rs`, fully self-contained) or a remote Ollama (`nomic-embed-text`). Either way: 768-dim, same vector space.
- Stores and retrieves points in [Qdrant](https://qdrant.tech/) with a **structured palace schema** (wing → room → hall) and **temporal validity** (`valid_until` / `superseded_by`) so the palace is a journal, not a snapshot
- Detects near-duplicates before writing (cosine ≥ 0.95 + exact text match)
- Keeps an append-only JSONL write-ahead log for every mutation

It is intentionally opinionated. If you want a generic `(text, metadata)` store, use [`qdrant/mcp-server-qdrant`](https://github.com/qdrant/mcp-server-qdrant) — this project starts from that interface and replaces the untyped metadata with an enum-validated palace schema.

## Inspiration and prior art

- [`qdrant/mcp-server-qdrant`](https://github.com/qdrant/mcp-server-qdrant) — the official Qdrant MCP server (Python, FastMCP). palazzo borrows its `store` / `find` tool shape, collection configuration, and filter-wrapping pattern.
- [`MemPalace/mempalace`](https://github.com/MemPalace/mempalace) — the wing / room / drawer terminology and the read-tool set (`status`, `taxonomy`, `check_duplicate`) are lifted from MemPalace's 29-tool MCP server. If you want a full palace with an agentic knowledge graph, cross-wing tunnels, and 96.6% R@5 retrieval on LongMemEval, go use MemPalace directly. palazzo is the minimum-viable single-user flavour of the same idea, Rust-native, Qdrant-backed.

Neither upstream is vendored. Both are linked above; please follow and star their work.

## Tools

| Tool | What it does |
|---|---|
| `palace_store` | File a verbatim memory into a wing/room/hall. Returns a new point ID or the existing one on near-duplicate. |
| `palace_find` | Semantic search. Optional typed filters: `wing`, `category`, `room`, `hall`, `since`, `until`, `recency_half_life_days`. |
| `palace_recall` | Fetch by explicit IDs. Cheap — no embedding. |
| `palace_status` | Total point count plus facet breakdown by wing, hall, category. |
| `palace_taxonomy` | Flat facet dump of wing / room / hall / category counts. |
| `palace_check_duplicate` | Probe whether candidate text already exists above the 0.95 cosine threshold. |
| `palace_supersede` | Replace one or more existing memories with a corrected version. Marks the old points with `valid_until`, `superseded_by`, `superseded_reason`; default `palace_find` hides them. |
| `palace_delete` | **DESTRUCTIVE — hard-delete by ID.** Requires explicit operator approval; both `confirm: true` and a `reason` are required and WAL-logged before the Qdrant call. Use for PII scrubs / garbage / mistakes only — prefer `palace_supersede` for fact corrections. Vectors are NOT recoverable. Cap: 100 IDs per call. |
| `palace_store_batch` | Bulk-ingest up to 256 memories in one call. Embeds the whole batch in one ONNX/Ollama inference pass and bulk-upserts to Qdrant in one HTTP call (~3-5× faster than N single-item calls). Per-item dedup against the live palace; result returns per-item status, IDs, and dedup hits. Designed for migrations and bulk imports. |
| `palace_gain` | Token-savings report. Aggregates the per-tool gain log and returns a `Summary` of how many tokens of agent context this server saved versus a hand-coded SSH+curl+jq equivalent. Optional `since` (RFC3339) and `include_text` flags. |

Input caps: 32 KB per text body, 100 IDs per recall batch, 1–20 results per find.

### Temporal filtering on `palace_find`

- `since` / `until` — inclusive RFC3339 second-precision UTC timestamps (e.g. `2026-04-01T00:00:00Z`). Filter memories by when they were stored. Bad format is rejected with an explicit error.
- `recency_half_life_days` (f64) — opt-in recency bias. When set, palazzo fetches up to 4× the requested limit from Qdrant (capped at 80), re-ranks each hit by `score × exp(-age_days / half_life)`, then returns the top `limit`. Omit or pass `0` for pure cosine. Typical values: `30` (aggressive), `90` (moderate), `365` (gentle — a year-old memory gets half its raw score).

Both knobs work alongside the wing/category/room/hall filters — they compose.

### Destructive operations

`palace_delete` is the only tool that physically removes data. It requires both `confirm: true` and a `reason`, each call is WAL-logged before the Qdrant call, but vectors are not recoverable from the WAL. Configure your MCP client to require human approval for this tool — in Claude Code, set `palace_delete` to ask each time via `~/.claude/settings.json` or project settings. For any reversible change, use `palace_supersede` instead.

### Temporal validity (`palace_supersede`)

Memories become wrong over time — infra gets renamed, services get rebuilt, decisions get reversed. `palace_supersede` lets you replace an old entry with a corrected one without losing the history:

- The new text is embedded and stored as a fresh point with `supersedes: [<old_id>, ...]`.
- Each old point gets marked with `valid_until = now`, `superseded_by = <new_id>`, and your free-text `reason`.
- Default `palace_find` excludes any point with a past `valid_until` — agents only see current truth. Pass `include_superseded: true` to surface the full timeline for archaeology.
- `palace_recall` always exposes `valid_until` / `superseded_by` / `superseded_reason` on the returned point, so you can tell current-from-stale at a glance.

The palace becomes a journal, not a snapshot — every correction is an append, never a delete.

## Palace schema

Every point carries:

```
category:    free-text — conventionally person | career | technical | infrastructure | project-memory | vibe | project
wing:        free-text — conventionally projects | infrastructure | personal | career | vibe
room:        free-text (project or topic)
hall:        free-text — conventionally facts | events | decisions | discoveries | preferences
text:        the memory itself, verbatim
timestamp:   RFC3339 UTC
session:     optional conversation identifier
source_file: optional MD path when imported
```

`category` / `wing` / `room` / `hall` are all free-text — the conventional values above are suggestions, not constraints; organise the palace however suits you. They are validated only for non-emptiness and a 64-byte length cap. IDs ≥ `1_000_000_000` are reserved for auto-generation (unix-millis). The palace `Payload` / `Memory` structs are defined in [`src/schema.rs`](src/schema.rs).

## Config

All via environment variables:

| Variable | Default | Notes |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | |
| `COLLECTION` | `claude-memory` | |
| `PALAZZO_WAL` | `~/.palazzo/wal.jsonl` | |
| `PALAZZO_BIND` | `127.0.0.1:6334` | only used by `serve` |
| `PALAZZO_ALLOWED_HOSTS` | `localhost,127.0.0.1,::1` | DNS-rebinding guard for `serve`; set to `*` to disable |
| `OLLAMA_URL` | `http://localhost:11434` | only read by the `ollama` backend |
| `OLLAMA_MODEL` | `nomic-embed-text` | only read by the `ollama` backend |
| `FASTEMBED_CACHE_DIR` | `~/.cache/fastembed` | only used by the `fastembed` backend |
| `PALAZZO_USAGE_LOG` | `/var/lib/palazzo/usage.jsonl` | append-only JSONL backing `palace_gain` |
| `PALAZZO_GAIN_ENABLED` | `1` | set to `0`/`false`/`no`/`off` to disable per-call recording |
| `RUST_LOG` | `palazzo=info` | |

Logging goes to **stderr only**. Stdout is the MCP transport — anything written there corrupts the JSON-RPC stream.

On startup, palazzo creates keyword payload indexes on `wing`, `category`, `room`, `hall` if they're missing. Idempotent; required for the facet-based tools. Adding indexes to an existing collection is non-destructive — Qdrant builds them in place and existing points stay.

## Embedding backends

palazzo ships two backends behind mutually-exclusive cargo features. Pick one at build time.

| Feature | How it embeds | When to use |
|---|---|---|
| `fastembed` (default) | Local ONNX inference of `nomic-embed-text-v1.5-Q` (INT8 dynamic-quantised) via [`fastembed-rs`](https://github.com/Anush008/fastembed-rs) | You want palazzo fully self-contained — zero external services. Static binary, ~110 MB one-time model download into `FASTEMBED_CACHE_DIR`, ~1 GB resident. **This is what every deployed palazzo runs.** |
| `ollama` | HTTP calls to an Ollama server running `nomic-embed-text` | You already run Ollama on your LAN and prefer a tiny no-native-deps binary. Useful for dev rigs that don't want to pay the model-download / RSS cost. |

Select the variant via cargo features (release archives publish both per-platform):

```
cargo build --release                                          # fastembed (default)
cargo build --release --no-default-features --features ollama
```

Both backends produce 768-dim vectors in the **same vector space** (nomic-embed-text-v1.5 architecture). Existing points embedded with one backend stay searchable with the other — including across the f32 → INT8-quantised swap that landed in v0.5.1. Expect ~0.98–0.99 cosine on the same text between any two precision combos, which is below the noise floor of typical palace queries.

## Build

```
cargo build --release
```

Release profile is LTO-thin, single codegen unit, stripped. Binary ~28 MB with `fastembed` (default — static ONNX runtime included), ~8 MB with `ollama`. Resident memory at idle: ~1 GB (fastembed, model loaded) or ~30 MB (ollama).

## Running

palazzo speaks two transports; pick one.

### stdio (local)

```
palazzo
```

Stdout is the MCP channel — logging always goes to stderr. This is the default mode when the binary is invoked with no arguments. Best for single-user laptop use: no port to bind, no service to manage.

Register with Claude Code:

```
claude mcp add palazzo -- /path/to/target/release/palazzo
claude mcp list
```

Override env vars with `-e KEY=VALUE` before the `--`:

```
claude mcp add palazzo \
  -e COLLECTION=my-palace \
  -e OLLAMA_URL=http://localhost:11434 \
  -- /path/to/target/release/palazzo
```

### Streamable HTTP (service)

```
palazzo serve --bind 0.0.0.0:6334
```

Serves MCP over Streamable HTTP at `POST /mcp`. Useful when the binary lives on a server co-located with Qdrant + Ollama, and your laptop (or multiple clients) connect over the network.

Register with Claude Code as a remote server:

```
claude mcp add --transport http palazzo http://your-server:6334/mcp
```

Bind address can also be set via `PALAZZO_BIND`. Default is `127.0.0.1:6334`.

### Bulk ingest over HTTP (`POST /ingest`)

The `serve` mode exposes a sibling REST endpoint alongside `/mcp`:

```
curl -X POST http://palazzo-host:6334/ingest \
  -H 'Content-Type: application/x-ndjson' \
  --data-binary @batch.jsonl
```

Same backend as `palace_store_batch` — embed, dedup, WAL, upsert — but delivered as a plain HTTP request. When invoked from an MCP client via `Bash(curl)`, the agent transcript only carries the curl command and the JSON response summary; the file's bytes flow through curl's body stream and never touch the LLM tokenizer. Use this for any bulk migration where the source data already exists on disk or a reachable URL. Same `PALAZZO_ALLOWED_HOSTS` allowlist as `/mcp`.

The response is **streamed NDJSON** — one progress line per processed batch (default 256 items each), then a final `{"done": true, ...}` line:

```
{"chunk":0,"items_in_chunk":256,"counts":{"stored":256,...},"dedup_against":[],"running":{"stored":256,...}}
{"chunk":1,"items_in_chunk":256,"counts":{...},"dedup_against":[1000001,1000002],"running":{"stored":512,...}}
{"chunk":2,"items_in_chunk":88,"counts":{...},"dedup_against":[],"running":{"stored":600,...}}
{"done":true,"total":600,"counts":{"stored":600,"duplicates_returned":2,"skipped_duplicates":0,"failed":0}}
```

Each per-chunk progress line carries a `dedup_against` array — the existing point IDs that the chunk's items matched (cosine ≥ 0.95 + exact text). Empty when no dedup hits; non-empty when `counts.duplicates_returned > 0`. Use this to distinguish "all items deduplicated against existing memories" from "nothing happened". Use `curl -N` to disable client-side buffering and watch progress live. Errors during processing emit a `{"chunk":N,"error":"..."}` line and close the stream. Body parse failures still return `400` with a plain-text body before any streaming starts.

### Export collection as NDJSON (`GET /export`)

Stream the entire collection (or filtered subset) as **NDJSON**, one point per line. Pagination via Qdrant's scroll API keeps memory bounded:

```
curl 'http://palazzo-host:6334/export?vectors=false&wing=projects' | head -5
```

Query params (all optional):
- `vectors=true|false` (default **true** — include the 768-dim embedding vector)
- `wing`, `category`, `room`, `hall`, `since`, `until` — same semantics as `palace_find`. Values are trimmed.
- `include_superseded=true|false` (default **false** — by default, only current-truth memories)

Each line is a JSON object with all point fields plus optional `vector`. Errors emit `{"error":"..."}` and close the stream. Bad RFC3339 timestamps return `400` before streaming starts. Same `PALAZZO_ALLOWED_HOSTS` allowlist as `/mcp`.

### Bulk ingest from a file (`palazzo ingest`)

```
palazzo ingest --file batch.jsonl
palazzo ingest --json < batch.jsonl
```

Same backend as `palace_store_batch` — embedding, dedup, WAL, upsert — but the texts never round-trip through the MCP transcript. Use this from migration scripts when the agent context can't afford the per-call cost of carrying the payloads. Input is JSON-Lines (`{"text":..., "category":..., "wing":..., "room":..., "hall":...}` per line, blank/`#`-prefixed lines ignored). Items are chunked into `MAX_STORE_BATCH` (256) groups and processed sequentially. Default output is a one-line summary on stderr; `--json` emits the full per-item result on stdout.

## Deploy

palazzo is a single binary; two supported paths to put it on a host.

### Deploy as a systemd service

The `deploy/` directory contains a hardened systemd unit, an env-file template, and an installer. On Debian / Ubuntu / any systemd host:

```
# On the target host, after placing the binary at e.g. ~/palazzo
sudo ./deploy/install.sh ~/palazzo
# Review /etc/palazzo/env, then:
sudo systemctl enable --now palazzo
```

The unit runs as a dedicated `palazzo` user, drops all needless privileges (`ProtectSystem=strict`, `MemoryDenyWriteExecute=true`, `RestrictNamespaces=true`, etc.), and persists the WAL at `/var/lib/palazzo/wal.jsonl`.

If you expose the service beyond a trusted LAN, put a reverse proxy with TLS + auth (e.g. nginx + basic auth, or an identity-aware proxy) in front of `:6334`. There is no built-in authentication — palazzo assumes a trusted network.

Pre-built release binaries are published on each tag at <https://github.com/calibrae/palazzo/releases>. The `palazzo-fastembed-<version>-x86_64-unknown-linux-gnu.tar.gz` artifact is the production target (glibc ≥ 2.38).

### Deploy with Docker Compose

`Dockerfile` + `docker-compose.yml` at the repo root deploy the whole stack — palazzo plus, optionally, a bundled Qdrant and/or Ollama. The runtime image is Debian trixie (glibc 2.41); fastembed's ONNX prebuilts need glibc ≥ 2.38.

**Quick start** — fully self-contained (fastembed + bundled Qdrant, zero external dependencies):

```
git clone https://github.com/calibrae/palazzo && cd palazzo
cp .env.example .env             # default: COMPOSE_PROFILES=qdrant
docker compose up -d --build
curl http://localhost:6334/health
```

Then register the MCP endpoint with your client (Claude Code shown):

```
claude mcp add --transport http palazzo http://<host>:6334/mcp
```

**Three common shapes** — pick one in `.env`:

| Want | `.env` settings |
|---|---|
| Self-contained (default) | `PALAZZO_BACKEND=fastembed` · `COMPOSE_PROFILES=qdrant` |
| Against an existing Qdrant | `PALAZZO_BACKEND=fastembed` · `COMPOSE_PROFILES=` · `QDRANT_URL=http://your-qdrant:6333` |
| Ollama backend, bundled | `PALAZZO_BACKEND=ollama` · `COMPOSE_PROFILES=qdrant,ollama` · then `docker compose exec ollama ollama pull nomic-embed-text` |

**Two axes**, both driven by `.env`:

- **Embedding backend** — `PALAZZO_BACKEND=fastembed` (local ONNX, default) or `ollama` (HTTP to an Ollama server). **Baked into the image at build time** via the `BACKEND` build arg — switching backends requires `docker compose build`.
- **Bundled services** — `COMPOSE_PROFILES` activates the `qdrant` and/or `ollama` containers. Omit a profile to point at an external service via its `*_URL` instead.

**Persistence** — three named volumes survive `docker compose down`; only `docker compose down -v` wipes them:

- `palazzo-data` — WAL (`wal.jsonl`), fastembed model cache, usage log.
- `qdrant-data` — Qdrant storage (only when the `qdrant` profile is active).
- `ollama-data` — Ollama model cache (only when the `ollama` profile is active).

**Updating to a new palazzo release**:

```
git pull
docker compose up -d --build       # rebuilds the image, restarts the service
```

palazzo tolerates Qdrant being unreachable at boot, so container start order is not constrained. The container binds `0.0.0.0:6334` internally and publishes to the host port `PALAZZO_PORT` (default `6334`) — change it in `.env` if that port is taken on the host.

## Testing

End-to-end smoke test against a throwaway Qdrant collection:

```
cargo build --release
python3 scripts/smoke.py
```

It creates `palazzo-test`, boots the binary, round-trips store / find / recall / status / check_duplicate / duplicate-skip / filtered find / since-until / recency-boost / supersede / superseded-hidden / superseded-surfaced / recall-temporal-metadata, and drops the collection. Fails loudly on any mismatch.

Requires live Qdrant and (for the `ollama` backend) Ollama reachable at the configured URLs. The `fastembed` backend has no external service requirement once the model cache is warm.

## Security notes

- Stdio transport, single-user threat model. No network listener, no auth.
- Dependencies audited with `cargo audit` on every build bump.
- Every write goes through a WAL (`~/.palazzo/wal.jsonl` by default) with content previews truncated to 120 chars.
- `OLLAMA_URL` and `QDRANT_URL` are environment-controlled — anyone who can set env vars on this binary can already execute code as you, so the SSRF surface is accepted.
- MCP tool outputs (including stored text) are echoed back through the protocol; treat them as untrusted input to whatever LLM consumes them. This is a generic MCP concern, not specific to palazzo.

## Non-goals

- **No multi-tenant auth.** Anyone reachable on the listener can read and write the palace. Put it behind a tailnet, a reverse proxy with auth, or a localhost-only bind. palazzo assumes a trusted network.
- **No web UI.** Use the Qdrant dashboard for raw inspection; the MCP tools are the supported interface.
- **No knowledge graph, no agent diaries, no LLM rerankers, no embedding-model swaps to a different architecture.** The palace stays a single 768-dim collection on nomic-embed-text. If you want any of those layers, [MemPalace](https://github.com/MemPalace/mempalace) is purpose-built for it.
- **No automatic collection migrations across architecture changes.** Compatible variants of the same model (V15 ↔ V15Q) work in the same collection because the vector space is identical. A different architecture would invalidate the existing points and isn't supported.

## License

MIT — see [LICENSE](LICENSE).

## Credits

- [`qdrant/mcp-server-qdrant`](https://github.com/qdrant/mcp-server-qdrant) (Apache-2.0) for the MCP-over-Qdrant baseline.
- [`MemPalace/mempalace`](https://github.com/MemPalace/mempalace) (MIT) for the palace terminology, read-tool set, and the idea that verbatim beats summarised.
- The [MCP Rust SDK](https://github.com/modelcontextprotocol/rust-sdk) for the server harness.
