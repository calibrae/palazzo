# Palazzo Auth v1 — self-hosted login + email attribution

Build doc for the **nxp palazzo** production deploy (hundreds of devs, shared store).
Target: ship this week with the SRE. Scope is deliberately small.

---

## 1. Goal & non-goals

**Goal.** When enabled, every write to palazzo is attributed to a claimed identity
(an email), and access is soft-gated to one or more allowed email domains. Off by
default — the homelab instance keeps working with zero auth.

**This is attribution, not authentication.** There is no email verification in v1
(OTP is explicitly dropped). Anyone can type any address in the allowed domain. The
value is provenance ("who wrote this memory") + keeping casual outsiders / wrong-domain
users out — **not** a defense against a malicious insider. Decide this with the SRE
on purpose; don't back into it.

**No read isolation.** Single collection, multi-tenant dropped → `palace_find` returns
the whole pool to every authenticated user. Fine for a shared org brain; make it a
conscious decision before hundreds of people write into it.

**Non-goals (v1):** OTP / email verification, per-tenant data isolation, per-user
revocation UI, HA/multi-replica token state.

---

## 2. Architecture: palazzo as its own OAuth 2.1 Authorization Server

Palazzo already is the OAuth **resource server**. v1 makes it the **authorization
server** too, serving its own login page. Why self-AS instead of Entra:

- **Dissolves the DCR blocker.** MCP clients bootstrap a `client_id` via Dynamic
  Client Registration (RFC 7591); Entra refuses anonymous DCR. A self-hosted AS just
  says "yes" to every registration. The thing that made real OAuth flaky goes away.
- No external IdP dependency, no app registration, no redirect-URI exact-match dance
  with a third party. The only reachable URL needed is the one clients already use for
  `/mcp`.

**Cost:** palazzo now mints and signs its own tokens, so it holds a **signing key** —
a new server-side secret (treat as an app secret: keep out of logs/args; not a vault
token, but don't leak it).

---

## 2b. Recommended v1 path — **static bearer token + self-service page**, NOT the OAuth dance

Client research (Claude Code, Copilot CLI, OpenCode — June 2026, sourced in §7) changed
the recommendation. Two facts dominate:

1. **All three clients support a static `Authorization: Bearer …` header** in their MCP
   config (confirmed for Copilot CLI and OpenCode; very likely for Claude Code via
   `--header`). That bypasses the entire OAuth flow.
2. **The OAuth flow is currently buggy across all three** against a self-hosted AS —
   Claude Code has open token-persistence (#52565) and scope-reuse (#67714) bugs;
   Copilot CLI ignores static client config and forces DCR (#2717); OpenCode uses a
   random callback port (#18955) and drops custom headers mid-handshake (#20286).

So for a **hundreds-of-devs launch this week**, don't bet it on the OAuth machinery.
Get the identical outcome (a palazzo-signed JWT carrying the email, domain-gated) with
a fraction of the code and none of the per-client OAuth quirks:

- **Self-service token page** (plain web, *not* OAuth): dev visits `https://palazzo-nxp/whoami`,
  types their email → palazzo validates the domain, mints a signed JWT, shows it → dev
  pastes it into their MCP client's `headers` once. This *is* your "fake login page"
  idea — just served as a token page instead of routed through OAuth.
- **Bearer middleware** on `/mcp` `/ingest` `/export` validates the JWT and stamps `author`.

That's the **whole** server surface: `GET /whoami` (form), `POST /whoami` (mint), and the
middleware. No DCR, no `/authorize`, no `/token`, no PKCE, no metadata documents, no
browser dance. Works identically on every client because static headers are universal.

**Tradeoff vs full OAuth:** token sits in a config file rather than the OS keychain, and
the dev pastes it once instead of an auto-browser flow. For unverified-attribution at a
trusted-internal shop — where the token only attributes writes and there's no read
isolation anyway — that's an easy trade. The full OAuth AS (§4) stays on the roadmap as
the "nicer UX once the clients' OAuth bugs settle" option.

> Net: build §2b + §5 + §6 now. Treat §4 (full OAuth AS) as deferred.

---

## 3. Config (env vars, SRE-owned via `/etc/palazzo/env`)

Mirrors the existing `PALAZZO_ALLOWED_HOSTS` / `PALAZZO_BIND` pattern in `main.rs`.

| Var | Default | Meaning |
|---|---|---|
| `PALAZZO_AUTH` | `off` | `off` = no auth (homelab). `email` = self-AS + email attribution. |
| `PALAZZO_AUTH_SIGNING_KEY` | — | Required when `auth=email`. ≥32 random bytes (HS256). Rotating it invalidates all live tokens. |
| `PALAZZO_ALLOWED_EMAIL_DOMAINS` | _(empty = any)_ | Comma-separated, e.g. `nexpublica.fr`. Login rejects addresses outside the list. |
| `PALAZZO_TOKEN_TTL_SECS` | `2592000` (30d) | Access-token lifetime. |
| `PALAZZO_AUTH_ISSUER` | derived from bind/host | Public base URL palazzo advertises as the AS issuer (must be the HTTPS URL clients reach, e.g. `https://palazzo-nxp.<domain>`). |

> The **allowed-domain list is config, not an admin API.** It changes ~never; an
> admin endpoint to mutate it would need its own auth bootstrap (turtles). The SRE
> editing this env var *is* the admin action, gated by deploy access. Defer any
> admin surface (`PALAZZO_ADMIN_EMAILS=…`) until there's a concrete per-user need
> (e.g. revoke one bad actor).

---

## 4. Endpoints — Option B: full OAuth AS (DEFERRED — see §2b)

> Keep for the "auto-login UX later" milestone. The recommended v1 (§2b) needs only
> `GET/POST /whoami` + the bearer middleware, not the list below.

All gated behind `PALAZZO_AUTH=email`; when `off`, none are mounted and `/mcp` is bare
(one `if` branch in `run_http`).

| Method / path | Purpose |
|---|---|
| `GET /.well-known/oauth-protected-resource` | RFC 9728. Points clients at palazzo-as-AS. |
| `GET /.well-known/oauth-authorization-server` | RFC 8414. Advertises authorize/token/register endpoints + PKCE support. |
| `POST /register` | RFC 7591 DCR. Accept any client, store its `redirect_uri`(s), return a `client_id`. |
| `GET /authorize` | Serve the **login HTML form** (single email field). Carries `state`, PKCE `code_challenge`, `redirect_uri`. |
| `POST /authorize` | Validate email **shape + allowed domain**. Mint a single-use auth code bound to `(email, code_challenge, redirect_uri, expiry)`. 302 → `redirect_uri?code=…&state=…`. |
| `POST /token` | PKCE verify (`code_verifier` vs stored `code_challenge`). Exchange code → signed JWT (`sub`/`email`, `exp`). |
| middleware on `/mcp`, `/ingest`, `/export` | Require `Authorization: Bearer`; validate JWT (sig, exp, issuer). On missing/invalid → `401 + WWW-Authenticate` pointing at the protected-resource metadata. Inject identity downstream. |

**Auth-code store:** in-memory map, 60s TTL, single-use. Fine for **one process**.
⚠️ If the SRE runs >1 replica behind a load balancer, this breaks (code minted on box A,
redeemed on box B) — either pin sticky sessions, share the store (Redis/sled), or run
single-process for v1. Tokens themselves are stateless JWTs, so only the brief
authorize→token window is process-affine.

---

## 5. Identity plumbing (the valuable core — do this even before the login is pretty)

- Add `author: Option<String>` to `Payload` (`schema.rs`), and surface it on `Memory`
  and `ExportPoint`.
- Bind identity from the validated token at the **per-session factory seam**
  (`make_palace_with_embedder` in `run_http`) — the `Palace` for that session carries
  the author, so every tool call is stamped without threading args through each handler.
  (Verify the pinned rmcp rev surfaces request/auth context to the factory; if not,
  fall back to an axum request-extension read in the middleware.)
- Stamp `author` on `palace_store`, `palace_store_batch`, `palace_supersede`. The WAL
  already records `session`; add `author` alongside.
- Optional follow-ups: facet/filter `palace_find` by author; per-author gain stats.

This is also exactly the hook the future multi-tenant work reuses (identity → tenant),
and it's the "per-agent insert tracking" idea from the homelab — same field.

---

## 6. Security notes for the SRE

- **Signing key** is a real secret. Generate with `openssl rand -base64 48`, inject via
  the env file (mode 0640, root:palazzo like the existing env). Keep out of logs.
- **TLS.** The AS issuer + all endpoints must be HTTPS (you have nginx terminating
  TLS). The client-side redirect is `http://localhost:PORT` (loopback), which is allowed
  unauthenticated by spec — nothing inbound to palazzo for that.
- **Rate-limit** `POST /authorize` and `POST /token` (cheap tower layer) — stops a bored
  dev brute-forcing the form or replaying codes.
- **Verified identity later** = the broker path (Authentik/Keycloak federated to Entra).
  Same `author` plumbing, but the email is proven. Roadmap, not critical path.

---

## Compromise response / token revocation

Palazzo's JWT tokens are stateless HS256. There is no per-token revocation list — the server does not track issued tokens.

**Access tokens** default to ~1 hour TTL; **refresh tokens** default to ~90 days. If either leaks, the only remedy is key rotation: change `PALAZZO_AUTH_SIGNING_KEY` to a fresh value and restart (or redeploy). This invalidates every outstanding token simultaneously — every user must re-authenticate on their next request. Via the OAuth flow that is one browser click on the "Authenticate" button; via the `/whoami` paste path it is a revisit and re-paste. A blunt instrument (logs everyone out) but immediate and total.

**Future option — per-user revoke without a global logout:** add a `jti` (JWT ID) claim to every issued token and maintain a small server-side denylist. Each verify call checks the list; revoking a user means adding their `jti`; they get a new token on next auth. Requires a fast persistent store (sled, Redis, or a `tokio::sync::RwLock<HashSet>` for single-process). Not implemented in v1 — punt until there is a concrete bad-actor case.

---

## 7. Client setup (verified June 2026)

### Recommended: static bearer header (works on all three, no OAuth)

**Claude Code** — *very likely* (`--header` flag on http transport; confirm in a 5-min test):
```bash
claude mcp add --transport http palazzo https://palazzo-nxp/mcp \
  --header "Authorization: Bearer <token-from-/whoami>"
```

**GitHub Copilot CLI** — CONFIRMED. `~/.copilot/mcp-config.json`:
```json
{ "palazzo": { "type": "http", "url": "https://palazzo-nxp/mcp",
    "headers": { "Authorization": "Bearer <token>" }, "tools": ["*"] } }
```

**OpenCode** — CONFIRMED. `opencode.json` (note: repo moved `sst/opencode` → `anomalyco/opencode`):
```json
{ "$schema": "https://opencode.ai/config.json",
  "mcp": { "palazzo": { "type": "remote", "url": "https://palazzo-nxp/mcp",
      "enabled": true, "oauth": false,
      "headers": { "Authorization": "Bearer {env:PALAZZO_TOKEN}" } } } }
```

### If you ever do the full OAuth AS (§4), the client reality you must build to

| Constraint | Why | Source |
|---|---|---|
| **Must implement RFC 7591 DCR** | Copilot CLI ignores static `clientId`, always DCRs; all three prefer it. Self-AS makes "say yes to everyone" trivial. | Copilot #2717 |
| **Accept loopback redirect on ANY port** | OpenCode uses a random callback port, not configurable. Honor RFC 8252 §7.3 wildcard loopback. | OpenCode #18955, #7377 |
| **No header-based routing during the handshake** | OpenCode drops custom `headers` on the auth/metadata requests → your AS sees no realm/tenant header. | OpenCode #20286 |
| **Expect Claude Code token bugs** | Tokens may not persist across restart (#52565); scope reuse conflicts on re-auth (#67714). | Claude Code #52565, #67714 |
| AS endpoints must be HTTPS; loopback redirect may be http | Standard. You have nginx TLS. | Claude Code docs (RFC 9728/8414) |

This matrix is exactly why §2b skips OAuth for the launch — every client has at least one
open auth bug against a self-hosted AS, while the static-header path has none.

---

## 8. Effort & sequencing

**Recommended v1 (§2b — static bearer + token page): ~1 day.**
- `author` field + stamping (§5) — useful immediately, independent of any login.
- `GET/POST /whoami` token page (domain check + JWT mint).
- Bearer middleware on `/mcp` `/ingest` `/export` + the env config (§3).
- One 5-min smoke per client (Claude Code header confirm; Copilot + OpenCode already confirmed).

**Deferred (§4 — full OAuth AS): +2–4 days**, most of it fighting the per-client OAuth bugs
in §7. Do it only if devs reject the one-time paste step, and only after those upstream
bugs settle.

Suggested order: (1) `author` stamping, (2) token page + middleware behind `PALAZZO_AUTH=email`,
(3) smoke each client, (4) flip it on for nxp.
