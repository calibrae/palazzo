#!/usr/bin/env python3
"""One-shot replay: read every point in SOURCE collection, re-embed each point's
`text` with fastembed's nomic-embed-text-v1.5 (ONNX), upsert into TARGET
collection preserving the original ID and full payload.

Intended for the A/B migration from Ollama-embedded `claude-memory` to an
identical fastembed-embedded `claude-memory-fastembed`. Idempotent — safe to
re-run; Qdrant upsert with existing IDs just overwrites.

Usage:
    python replay_to_fastembed.py \\
        --qdrant http://127.0.0.1:6333 \\
        --source claude-memory \\
        --target claude-memory-fastembed \\
        [--batch 32] [--dry-run]

Requires: fastembed (pip install fastembed), urllib (stdlib).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from fastembed import TextEmbedding


def http(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def scroll_all(qdrant: str, collection: str, batch: int = 256) -> list[dict]:
    url = f"{qdrant}/collections/{collection}/points/scroll"
    points: list[dict] = []
    next_page: Any = None
    while True:
        body: dict[str, Any] = {
            "limit": batch,
            "with_payload": True,
            "with_vector": False,  # we don't need source vectors; we're re-embedding
        }
        if next_page is not None:
            body["offset"] = next_page
        status, resp = http("POST", url, body)
        if status != 200:
            raise SystemExit(f"scroll failed: {status} {resp}")
        chunk = resp["result"]["points"]
        points.extend(chunk)
        next_page = resp["result"].get("next_page_offset")
        if not next_page:
            break
    return points


def upsert(qdrant: str, collection: str, batch: list[dict]) -> None:
    url = f"{qdrant}/collections/{collection}/points?wait=true"
    body = {"points": batch}
    status, resp = http("PUT", url, body)
    if status != 200:
        raise SystemExit(f"upsert failed: {status} {resp}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdrant", required=True, help="Qdrant base URL")
    ap.add_argument("--source", required=True, help="Source collection (reads text)")
    ap.add_argument("--target", required=True, help="Target collection (writes new vectors)")
    ap.add_argument("--batch", type=int, default=32, help="Embed+upsert batch size (default 32)")
    ap.add_argument("--dry-run", action="store_true", help="Skip writes; print counts only")
    args = ap.parse_args()

    print(f"source: {args.qdrant}/{args.source}")
    print(f"target: {args.qdrant}/{args.target}")

    # Confirm target exists
    status, resp = http("GET", f"{args.qdrant}/collections/{args.target}")
    if status != 200:
        raise SystemExit(f"target collection missing: {status} {resp}")
    target_size = resp["result"]["config"]["params"]["vectors"]["size"]
    if target_size != 768:
        raise SystemExit(f"target dim is {target_size}, expected 768")

    print("loading fastembed NomicEmbedTextV15 (first run downloads ~137 MB)…")
    t0 = time.time()
    model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
    print(f"model ready in {time.time() - t0:.1f}s")

    print(f"scrolling source…")
    t0 = time.time()
    points = scroll_all(args.qdrant, args.source)
    print(f"{len(points)} points in {time.time() - t0:.1f}s")

    skipped = 0
    replayed = 0
    for i in range(0, len(points), args.batch):
        chunk = points[i : i + args.batch]
        ids = [p["id"] for p in chunk]
        payloads = [p.get("payload") or {} for p in chunk]
        texts = [pl.get("text") for pl in payloads]
        # Skip points missing text (shouldn't happen in our schema, but defend).
        valid = [(i, t) for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
        if len(valid) != len(texts):
            skipped += len(texts) - len(valid)
        if not valid:
            continue
        batch_texts = [t for _, t in valid]
        t0 = time.time()
        vectors = list(model.embed(batch_texts))
        embed_ms = (time.time() - t0) * 1000
        out = []
        for j, (idx, _) in enumerate(valid):
            out.append(
                {
                    "id": ids[idx],
                    "vector": vectors[j].tolist(),
                    "payload": payloads[idx],
                }
            )
        if not args.dry_run:
            upsert(args.qdrant, args.target, out)
        replayed += len(out)
        print(
            f"  batch {i // args.batch + 1}: {len(out)} points embedded in {embed_ms:.0f}ms"
        )

    print(f"\ndone. replayed={replayed} skipped={skipped} total={len(points)}")
    if args.dry_run:
        print("DRY RUN — nothing was written.")


if __name__ == "__main__":
    main()
