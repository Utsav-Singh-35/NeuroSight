"""Embed the knowledge base and persist the retrieval index.

Writes ``knowledge/index.npz`` (float32 embedding matrix) and
``knowledge/index_meta.json`` (model name, dimension, and the chunk IDs the
vectors correspond to, so a stale index can be detected rather than silently
returning wrong evidence).

The embedding model is downloaded on first run (~80 MB) and cached by
``sentence-transformers`` afterwards. Embedding 28 chunks takes a couple of
seconds on CPU; this is a one-time offline cost, not a per-request one.

Usage
-----
Run from ``backend/fastapi``::

    python scripts/build_index.py
    python scripts/build_index.py --model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.knowledge import get_knowledge_base  # noqa: E402
from app.services.retrieval import (  # noqa: E402
    DEFAULT_MODEL,
    INDEX_FILENAME,
    EmbeddingRetriever,
    save_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--module", default=None, help="Restrict to one module")
    parser.add_argument("--knowledge-root", default=None)
    args = parser.parse_args()

    kb = get_knowledge_base(args.knowledge_root)
    if not kb.available:
        print("ERROR: knowledge base is empty or missing.", file=sys.stderr)
        return 2

    chunks = kb.chunks(args.module)
    root = Path(args.knowledge_root) if args.knowledge_root else Path(kb.root)

    print(f"Knowledge root : {root}")
    print(f"Chunks         : {len(chunks)}")
    print(f"Model          : {args.model}")

    words = [len(c["text"].split()) for c in chunks]
    print(
        f"Chunk words    : min={min(words)} max={max(words)} "
        f"mean={sum(words) / len(words):.0f}"
    )

    retriever = EmbeddingRetriever(chunks, model_name=args.model)

    print("\nEmbedding (first run downloads the model)...", flush=True)
    started = time.time()
    try:
        vectors = retriever.build()
    except Exception as exc:  # noqa: BLE001
        print(
            f"\nERROR: could not build embeddings ({type(exc).__name__}: {exc})\n"
            "Retrieval will fall back to the TF-IDF backend, which needs no "
            "model download. Re-run this script when a network connection is "
            "available.",
            file=sys.stderr,
        )
        return 3

    elapsed = time.time() - started
    print(f"  embedded {len(chunks)} chunks in {elapsed:.1f}s")
    print(f"  matrix shape {vectors.shape}, dtype {vectors.dtype}")

    # Vectors are L2-normalised, so these should all be ~1.0.
    norms = np.linalg.norm(vectors, axis=1)
    print(f"  norms: min={norms.min():.6f} max={norms.max():.6f} (expect ~1.0)")

    index_path = root / INDEX_FILENAME
    save_index(index_path, vectors, chunks, args.model)

    size_kb = index_path.stat().st_size / 1024
    print(f"\nWritten: {index_path}  ({size_kb:.1f} KB)")
    print(f"Written: {root / 'index_meta.json'}")

    # A quick self-check: every chunk should retrieve itself first when its own
    # text is the query. If that fails, the index and metadata are misaligned.
    # Self-retrieval check. A chunk queried with its own text should rank
    # itself first. Two failure modes look alike and must be told apart:
    # a genuinely misaligned index, versus near-duplicate content across
    # classes (e.g. the "investigations" sections read very similarly for
    # glioma and meningioma). Only the former is a bug.
    print("\nSelf-retrieval sanity check...")
    exact = 0
    near_duplicate = 0
    misaligned = 0

    for chunk in chunks:
        hits = retriever.search(chunk["text"][:400], k=1)
        if not hits:
            misaligned += 1
            print(f"  NO HIT: {chunk['chunk_id']}", file=sys.stderr)
            continue

        top = hits[0]
        if top.chunk_id == chunk["chunk_id"]:
            exact += 1
        elif top.section == chunk["section"]:
            near_duplicate += 1
            print(
                f"  near-duplicate: {chunk['chunk_id']} -> {top.chunk_id} "
                f"(same '{chunk['section']}' section, different class)"
            )
        else:
            misaligned += 1
            print(
                f"  MISALIGNED: {chunk['chunk_id']} -> {top.chunk_id}",
                file=sys.stderr,
            )

    print(
        f"  exact self-retrieval {exact}/{len(chunks)}, "
        f"cross-class near-duplicates {near_duplicate}, "
        f"misaligned {misaligned}"
    )
    if misaligned:
        print(
            "  WARNING: misaligned results indicate a real index problem.",
            file=sys.stderr,
        )
    elif near_duplicate:
        print(
            "  No index problem. Near-duplicates are expected: the same section\n"
            "  reads similarly across classes. Retrieval for report generation\n"
            "  filters by class, so this does not affect narrative grounding."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
