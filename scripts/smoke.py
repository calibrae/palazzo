#!/usr/bin/env python3
"""end-to-end smoke test for palazzo against a throwaway Qdrant collection.

Spins up the binary with COLLECTION=palazzo-test, speaks MCP JSON-RPC over
stdio, round-trips store/find/recall/status/check_duplicate, then drops the
collection. Fails loudly on any non-200 / non-match.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

QDRANT = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
COLLECTION = "palazzo-test"
BIN = os.path.join(os.path.dirname(__file__), "..", "target", "release", "palazzo")


def http(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    if os.environ.get("QDRANT_API_KEY", "").strip():
        headers["api-key"] = os.environ["QDRANT_API_KEY"].strip()
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def create_collection():
    status, _ = http("DELETE", f"{QDRANT}/collections/{COLLECTION}")
    status, body = http(
        "PUT",
        f"{QDRANT}/collections/{COLLECTION}",
        {"vectors": {"size": 768, "distance": "Cosine"}},
    )
    assert status == 200, f"create: {status} {body}"
    print(f"created {COLLECTION}")


def drop_collection():
    http("DELETE", f"{QDRANT}/collections/{COLLECTION}")
    print(f"dropped {COLLECTION}")


class MCPClient:
    def __init__(self, proc):
        self.proc = proc
        self._id = 0

    def _send(self, method, params=None):
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line.encode())
        self.proc.stdin.flush()
        resp_line = self.proc.stdout.readline()
        if not resp_line:
            stderr = self.proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"no response to {method}. stderr:\n{stderr}")
        return json.loads(resp_line)

    def _notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line.encode())
        self.proc.stdin.flush()

    def initialize(self):
        r = self._send(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "palazzo-smoke", "version": "0.1"},
            },
        )
        assert "result" in r, f"initialize: {r}"
        self._notify("notifications/initialized")
        return r["result"]

    def list_tools(self):
        r = self._send("tools/list")
        return r["result"]["tools"]

    def call(self, name, args):
        r = self._send("tools/call", {"name": name, "arguments": args})
        assert "result" in r, f"{name} failed: {r}"
        content = r["result"]["content"]
        assert content and content[0]["type"] == "text", f"{name}: bad content {content}"
        return json.loads(content[0]["text"])


def main():
    if not os.path.isfile(BIN):
        sys.exit(f"binary not found: {BIN}. Run `cargo build --release` first.")

    create_collection()

    env = os.environ.copy()
    env["COLLECTION"] = COLLECTION
    env["QDRANT_URL"] = QDRANT
    env["OLLAMA_URL"] = OLLAMA
    env["RUST_LOG"] = "palazzo=info"

    proc = subprocess.Popen(
        [BIN],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        client = MCPClient(proc)
        info = client.initialize()
        print("initialized:", info.get("serverInfo"))

        tools = client.list_tools()
        names = sorted(t["name"] for t in tools)
        print("tools:", names)
        expected = {
            "palace_store",
            "palace_find",
            "palace_recall",
            "palace_status",
            "palace_taxonomy",
            "palace_check_duplicate",
        }
        assert expected.issubset(set(names)), f"missing tools. got {names}"

        # 1. Store a memory
        stored = client.call(
            "palace_store",
            {
                "text": "palazzo smoke test — the MCP server boots and round-trips correctly.",
                "category": "project-memory",
                "wing": "projects",
                "room": "palazzo",
                "hall": "events",
                "session": "smoke-test",
            },
        )
        print("stored:", stored)
        new_id = stored["id"]
        assert new_id >= 1_000_000_000, f"id below floor: {new_id}"

        # 2. Find it back
        hits = client.call("palace_find", {"query": "smoke test mcp round trip", "limit": 3})
        print("found:", [(h["id"], round(h["score"], 3)) for h in hits])
        assert any(h["id"] == new_id for h in hits), "stored memory not retrievable"

        # 3. Recall by ID
        recalled = client.call("palace_recall", {"ids": [new_id]})
        assert len(recalled) == 1 and recalled[0]["id"] == new_id, f"recall: {recalled}"
        assert recalled[0]["text"].startswith("palazzo smoke test")
        print("recalled ok")

        # 4. Status
        status = client.call("palace_status", {})
        print("status:", status)
        assert status["total"] >= 1

        # 5. Check duplicate (exact match -> should flag)
        dup = client.call(
            "palace_check_duplicate",
            {
                "text": "palazzo smoke test — the MCP server boots and round-trips correctly."
            },
        )
        print("dup check:", dup)
        assert dup["is_duplicate"], "exact duplicate not detected"

        # 6. Store the exact same text -> should return existing id
        stored_again = client.call(
            "palace_store",
            {
                "text": "palazzo smoke test — the MCP server boots and round-trips correctly.",
                "category": "project-memory",
                "wing": "projects",
                "room": "palazzo",
                "hall": "events",
            },
        )
        print("stored again:", stored_again)
        assert stored_again["id"] == new_id, "duplicate store should return existing id"
        assert stored_again["duplicate_of"] == new_id

        # 7. Filter: wing=projects room=palazzo
        filtered = client.call(
            "palace_find",
            {
                "query": "smoke test",
                "wing": "projects",
                "room": "palazzo",
                "hall": "events",
                "limit": 5,
            },
        )
        assert any(h["id"] == new_id for h in filtered), "filter miss"
        print("filtered ok")

        # 8. since/until filter — future window should return nothing
        future = client.call(
            "palace_find",
            {"query": "smoke test", "since": "2100-01-01T00:00:00Z", "limit": 5},
        )
        assert future == [], f"future window should be empty, got {future}"
        # …and a since that includes our timestamp should still hit
        included = client.call(
            "palace_find",
            {"query": "smoke test", "since": "2020-01-01T00:00:00Z", "limit": 5},
        )
        assert any(h["id"] == new_id for h in included), "since filter excluded stored memory"
        print("since/until ok")

        # 9. recency boost — should still return the hit, scores should change
        boosted = client.call(
            "palace_find",
            {"query": "smoke test", "recency_half_life_days": 1, "limit": 5},
        )
        assert any(h["id"] == new_id for h in boosted), "recency boost dropped stored memory"
        print("recency boost ok")

        # 10. bad timestamp → error
        try:
            client.call("palace_find", {"query": "smoke", "since": "not-a-date"})
        except AssertionError as e:
            assert "RFC3339" in str(e), f"unexpected error: {e}"
            print("bad timestamp rejected ok")
        else:
            raise AssertionError("bad since should have failed")

        # 11. palace_supersede replaces the stored point with a corrected version
        superseded = client.call(
            "palace_supersede",
            {
                "supersedes": [new_id],
                "text": "palazzo smoke test — corrected entry, v0.5 supersede path works.",
                "category": "project-memory",
                "wing": "projects",
                "room": "palazzo",
                "hall": "events",
                "session": "smoke-test",
                "reason": "superseded by smoke test v0.5",
            },
        )
        print("superseded:", superseded)
        new_point_id = superseded["id"]
        assert new_point_id != new_id, "supersede should mint a new id"
        assert superseded["supersedes"] == [new_id]
        assert superseded["marked"][0]["id"] == new_id
        assert superseded["marked"][0]["ok"] is True

        # 12. palace_find default excludes superseded points
        after_supersede = client.call(
            "palace_find",
            {"query": "smoke test mcp round trip", "limit": 10},
        )
        ids_after = [h["id"] for h in after_supersede]
        assert new_id not in ids_after, "default find should hide superseded point"
        assert new_point_id in ids_after, "default find should surface the replacement"
        print("default hides superseded ok")

        # 13. palace_find with include_superseded=true surfaces history
        with_superseded = client.call(
            "palace_find",
            {"query": "smoke test mcp round trip", "limit": 10, "include_superseded": True},
        )
        ids_with = [h["id"] for h in with_superseded]
        assert new_id in ids_with, "include_superseded should surface the old point"
        print("include_superseded surfaces history ok")

        # 14. palace_recall exposes valid_until + superseded_by on the old point
        recalled_old = client.call("palace_recall", {"ids": [new_id]})
        assert recalled_old[0]["valid_until"] is not None, "old point missing valid_until"
        assert recalled_old[0]["superseded_by"] == new_point_id
        assert "superseded by smoke test v0.5" in recalled_old[0]["superseded_reason"]
        print("recall exposes temporal metadata ok")

        # 15. palace_store_batch round-trips a batch with mixed shapes:
        #     [0] new memory, [1] new memory, [2] EXACT duplicate of one we
        #     just stored above (palace_supersede already wrote palazzo-test
        #     content). Verify per-item statuses + counts.
        batch_resp = client.call(
            "palace_store_batch",
            {
                "items": [
                    {
                        "text": "palazzo smoke batch item alpha",
                        "category": "project-memory",
                        "wing": "projects",
                        "room": "palazzo",
                        "hall": "events",
                        "session": "smoke-test",
                    },
                    {
                        "text": "palazzo smoke batch item beta",
                        "category": "project-memory",
                        "wing": "projects",
                        "room": "palazzo",
                        "hall": "events",
                        "session": "smoke-test",
                    },
                    {
                        # Same text as the supersede new point — should dedup.
                        "text": "palazzo smoke test — corrected entry, v0.5 supersede path works.",
                        "category": "project-memory",
                        "wing": "projects",
                        "room": "palazzo",
                        "hall": "events",
                    },
                ]
            },
        )
        print("batch:", batch_resp["counts"])
        items = batch_resp["items"]
        assert len(items) == 3, f"expected 3 items, got {len(items)}"
        assert items[0]["ok"] is True and items[0].get("duplicate_of") is None
        assert items[1]["ok"] is True and items[1].get("duplicate_of") is None
        assert items[2]["ok"] is True and items[2].get("duplicate_of") == new_point_id
        assert batch_resp["counts"]["stored"] == 2
        assert batch_resp["counts"]["duplicates_returned"] == 1
        # Newly-stored points should now be discoverable.
        alpha_id = items[0]["id"]
        recalled_alpha = client.call("palace_recall", {"ids": [alpha_id]})
        assert recalled_alpha[0]["text"] == "palazzo smoke batch item alpha"
        print("palace_store_batch ok")

        # 16. batch with empty items array → error
        try:
            client.call("palace_store_batch", {"items": []})
        except AssertionError as e:
            assert "empty" in str(e).lower(), f"unexpected error: {e}"
            print("empty batch rejected ok")
        else:
            raise AssertionError("empty batch should have failed")

        # 17. `palazzo ingest` CLI subcommand: same backend, no MCP transcript.
        # Two fresh items + one exact-duplicate of #15's alpha. Verify counts
        # via --json, then recall the new items through the live MCP session.
        ingest_path = os.path.join("/tmp", f"palazzo-ingest-smoke-{COLLECTION}.jsonl")
        with open(ingest_path, "w") as f:
            f.write(json.dumps({
                "text": "palazzo ingest CLI gamma — smoke item",
                "category": "project-memory", "wing": "projects",
                "room": "palazzo", "hall": "events",
            }) + "\n")
            f.write(json.dumps({
                "text": "palazzo ingest CLI delta — smoke item",
                "category": "project-memory", "wing": "projects",
                "room": "palazzo", "hall": "events",
            }) + "\n")
            f.write(json.dumps({
                # exact duplicate of step-15 batch alpha
                "text": "palazzo smoke batch item alpha",
                "category": "project-memory", "wing": "projects",
                "room": "palazzo", "hall": "events",
            }) + "\n")
        try:
            ing_env = env.copy()
            ing_env["RUST_LOG"] = "palazzo=warn"
            ing = subprocess.run(
                [BIN, "ingest", "--file", ingest_path, "--json"],
                env=ing_env, capture_output=True, check=True,
            )
            ing_resp = json.loads(ing.stdout)
            print("ingest:", ing_resp["counts"])
            assert ing_resp["counts"]["stored"] == 2, ing_resp["counts"]
            assert ing_resp["counts"]["duplicates_returned"] == 1, ing_resp["counts"]
            assert len(ing_resp["items"]) == 3
            new_ids = [it["id"] for it in ing_resp["items"][:2]]
            recalled_ingest = client.call("palace_recall", {"ids": new_ids})
            assert {r["text"] for r in recalled_ingest} == {
                "palazzo ingest CLI gamma — smoke item",
                "palazzo ingest CLI delta — smoke item",
            }, recalled_ingest
            print("palazzo ingest CLI ok")
        finally:
            try:
                os.remove(ingest_path)
            except FileNotFoundError:
                pass

        print("\n✅ all smoke checks passed")
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        stderr = proc.stderr.read().decode(errors="replace")
        if stderr.strip():
            print("\n--- server stderr ---")
            print(stderr)
        drop_collection()


if __name__ == "__main__":
    main()
