# Be the agnostic embedder backend for `rmcp-guide` (expose `/v1/embeddings`)

**From:** rmcp-guide build session (Claude Code, cali's machine)
**Date:** 2026-06-22
**Effort:** ~half a day for the `/v1/embeddings` shim; the rest is optional
**Sibling repo:** `../rmcp-guide` (new crate, built around upstream rmcp — *not* the rmcp fork)

## TL;DR

I built a new crate, **`rmcp-guide`**, that fronts any rmcp MCP server with a single
`guide` tool: the agent asks in natural language, `guide` semantically searches the
server's own docs/tools and returns which tool to use (and, in *dynamic mode*, reveals
that tool natively via `notifications/tools/list_changed` — zero pre-use tool exposure,
no proxying, no restart). It's green: 9 tests, including an end-to-end client↔server
proof of the dynamic reveal.

`rmcp-guide` carries **no embedder of its own** — on purpose. It defines an agnostic
`Embedder` contract and an `EmbedderConfig` setting. The maximally-agnostic remote
option is **anything speaking the OpenAI `/v1/embeddings` wire shape**. So the single
highest-leverage thing palazzo can do is **expose a `/v1/embeddings` endpoint**. Then
palazzo becomes "just a URL" — guide (and atrium later) point at it with zero
palazzo-specific code, and you (palazzo) reuse the fastembed stack you already run.

## Why this shape (and the firewall principle)

We deliberately did **not** hard-wire palazzo into guide. palazzo is your off-work fun
project that's quietly becoming load-bearing across Nexpublica — so the architecture
keeps it behind a contract:

- guide/atrium/NXP depend on the **`Embedder` trait + `/v1/embeddings` wire shape**, never
  on palazzo-the-binary.
- palazzo is *an implementation* of that contract, not a mandatory chokepoint. NXP can
  swap a governed/supported backend in prod; the homelab points at palazzo. Same config.
- **Dependency arrow points away from palazzo**: the palazzo-side adapter (if any) depends
  on guide's trait crate, never the reverse. Nothing downstream pins your hobby repo.

This is the firewall that lets palazzo stay personal and unsupported while still powering
the fleet. Don't break it by making guide `use palazzo::...`.

## The concrete ask

### 1. Expose `POST /v1/embeddings` (OpenAI-compatible) — the main thing

Request body guide will send:

```json
{ "model": "nomic-embed-text", "input": ["text one", "text two", "..."] }
```

Response guide expects:

```json
{ "data": [ { "index": 0, "embedding": [0.01, ...] }, { "index": 1, "embedding": [...] } ],
  "model": "nomic-embed-text" }
```

You already embed with fastembed (`NomicEmbedTextV15Q`, 768-dim per the
quantization note in this inbox) — this endpoint is a thin HTTP shim over the embedder you
already load. Batch `input` is required (corpus embedding is one round-trip).

### 2. Mind the query/document asymmetry

nomic-embed-text is **asymmetric** — it wants `search_query:` / `search_document:`
prefixes. Two options, pick one and document it:

- **(preferred) Handle prefixes server-side** keyed off an optional `input_type`
  field (`"query"` | `"document"`), defaulting to document. Then callers stay dumb.
- Or do nothing and let the caller prepend; guide's config has `query_prefix` /
  `document_prefix` knobs for exactly this. But then every caller must know palazzo's
  model quirk — worse. Prefer server-side.

Whatever palazzo does for `palace_find` today (it must already prefix queries) is the
behavior to mirror here.

### 3. (Optional, later) A guide-scoped semantic find

If we later want palazzo to hold the guide *corpus* too (not just embed), we'd want a
find scoped to a dedicated namespace so it never pollutes the memory palace — e.g.
`wing="guide"`, `room=<server-name>`, tool name in the payload — queryable without the
results bleeding into normal `palace_find`. This is the "palazzo as atrium's index store"
idea; **not needed for v1**. For v1, guide keeps its own tiny corpus and only borrows
palazzo's *embedder* via `/v1/embeddings`. File this under "if/when atrium."

## What the contract looks like on guide's side (FYI, so the endpoint matches)

```rust
// rmcp-guide/src/embed.rs
pub enum EmbedderConfig {
    None,                                  // lexical, no embedder (default)
    Fastembed { model },                   // sovereign single-binary (eGov)
    OpenAiHttp {                           // ← palazzo lands here
        base_url, model, api_key_env,
        query_prefix, document_prefix,
    },
}

pub trait Embedder {                       // object-safe, no async-trait
    fn model_id(&self) -> &str;            // store/cache partition key
    fn dims(&self) -> usize;               // 768 for nomic
    async fn embed_documents(&self, &[String]) -> Result<Vec<Vec<f32>>, EmbedError>;
    async fn embed_query(&self, &str)       -> Result<Vec<f32>, EmbedError>;
}
```

A homelab guide server would then be configured with literally:

```toml
[embedder]
kind = "openai-http"
base_url = "http://palazzo.<lan>:<port>/v1"
model = "nomic-embed-text"
# query_prefix/document_prefix only if you DON'T handle input_type server-side
```

## Status / where to read the code

- `../rmcp-guide` — crate, 9 tests green. `src/embed.rs` is the contract; `src/server.rs`
  is the `GuideServer` wrapper; `tests/dynamic_reveal.rs` is the end-to-end proof.
- The `openai-http` `Embedder` impl on guide's side is stubbed (feature-gated, errors
  loudly until built). I'll wire it the moment palazzo's endpoint exists — easiest to
  validate first against a plain `ollama` `/v1` to prove "just a URL", then point at
  palazzo.

Background (full arc — the RC/stateless analysis, the dynamic-reveal/spec findings, the
"contract not chokepoint" decision) is in the palace under
`technical/discoveries/rmcp` and `technical/decisions/mcp-design-patterns`
(`guide` id 1781988926889, `atrium` id 1781989325687).
