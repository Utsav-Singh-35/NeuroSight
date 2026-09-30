"""Evaluate evidence retrieval against the hand-labelled query set.

Compares the semantic (MiniLM embedding) and lexical (TF-IDF) backends on the
same 30 queries in ``knowledge/retrieval_eval.json``, so the choice of retriever
rests on a measurement rather than a preference.

Metrics
-------
``Hit@1`` / ``Hit@3``
    Fraction of queries with at least one relevant chunk in the top 1 / top 3.
    The most practically meaningful numbers: report generation uses a handful of
    chunks, so what matters is whether something relevant appears near the top.

``Recall@5``
    Fraction of each query's relevant chunks that appear in the top 5, averaged.

``MRR``
    Mean reciprocal rank of the first relevant hit. Rewards ranking relevant
    evidence higher rather than merely including it.

``Precision@5``
    Reported for completeness only. Relevant sets here contain 1-4 chunks, so
    precision@5 is mathematically capped below 1.0 and a "low" value does not
    indicate poor retrieval. ``Recall@5`` is the honest counterpart.

Results are also broken down by whether the query was deliberately paraphrased
away from the corpus wording, which is where semantic and lexical retrieval are
expected to diverge.

Usage
-----
Run from ``backend/fastapi``::

    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --k 5 --show-failures
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.knowledge import get_knowledge_base  # noqa: E402
from app.services.retrieval import (  # noqa: E402
    INDEX_FILENAME,
    EmbeddingRetriever,
    TfidfRetriever,
    load_index,
)


def evaluate(retriever, queries: list[dict], k: int) -> dict:
    """Score one retriever over the query set."""
    per_query = []

    for item in queries:
        relevant = set(item["relevant"])
        hits = retriever.search(item["query"], k=k, module="brain_mri")
        retrieved = [h.chunk_id for h in hits]

        found = [rank for rank, cid in enumerate(retrieved, 1) if cid in relevant]
        first_rank = found[0] if found else None

        per_query.append(
            {
                "id": item["id"],
                "query": item["query"],
                "paraphrase": item.get("paraphrase", False),
                "relevant": sorted(relevant),
                "retrieved": retrieved,
                "hit_at_1": bool(retrieved[:1]) and retrieved[0] in relevant,
                "hit_at_3": any(cid in relevant for cid in retrieved[:3]),
                "recall_at_k": len(relevant & set(retrieved)) / len(relevant),
                "precision_at_k": len(relevant & set(retrieved)) / max(1, len(retrieved)),
                "reciprocal_rank": (1.0 / first_rank) if first_rank else 0.0,
                "first_relevant_rank": first_rank,
            }
        )

    def mean(key: str, subset: list[dict]) -> float:
        if not subset:
            return float("nan")
        return sum(float(row[key]) for row in subset) / len(subset)

    paraphrased = [r for r in per_query if r["paraphrase"]]
    literal = [r for r in per_query if not r["paraphrase"]]

    return {
        "backend": retriever.backend,
        "k": k,
        "n_queries": len(per_query),
        "overall": {
            "hit_at_1": round(mean("hit_at_1", per_query), 4),
            "hit_at_3": round(mean("hit_at_3", per_query), 4),
            "recall_at_k": round(mean("recall_at_k", per_query), 4),
            "precision_at_k": round(mean("precision_at_k", per_query), 4),
            "mrr": round(mean("reciprocal_rank", per_query), 4),
        },
        "paraphrased_only": {
            "n": len(paraphrased),
            "hit_at_1": round(mean("hit_at_1", paraphrased), 4),
            "hit_at_3": round(mean("hit_at_3", paraphrased), 4),
            "recall_at_k": round(mean("recall_at_k", paraphrased), 4),
            "mrr": round(mean("reciprocal_rank", paraphrased), 4),
        },
        "literal_only": {
            "n": len(literal),
            "hit_at_1": round(mean("hit_at_1", literal), 4),
            "hit_at_3": round(mean("hit_at_3", literal), 4),
            "recall_at_k": round(mean("recall_at_k", literal), 4),
            "mrr": round(mean("reciprocal_rank", literal), 4),
        },
        "per_query": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--knowledge-root", default=None)
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()

    kb = get_knowledge_base(args.knowledge_root)
    root = Path(args.knowledge_root) if args.knowledge_root else Path(kb.root)
    chunks = kb.chunks("brain_mri")

    eval_path = root / "retrieval_eval.json"
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found.", file=sys.stderr)
        return 2

    with open(eval_path, "r", encoding="utf-8") as handle:
        eval_doc = json.load(handle)
    queries = eval_doc["queries"]

    # Every labelled chunk ID must exist, or the metrics are meaningless.
    known = {c["chunk_id"] for c in chunks}
    bad = {
        item["id"]: [c for c in item["relevant"] if c not in known]
        for item in queries
        if any(c not in known for c in item["relevant"])
    }
    if bad:
        print(f"ERROR: labelled chunk IDs not in the knowledge base: {bad}", file=sys.stderr)
        return 2

    print(f"Chunks    : {len(chunks)}")
    print(f"Queries   : {len(queries)}  (paraphrased: "
          f"{sum(1 for q in queries if q.get('paraphrase'))})")
    print(f"k         : {args.k}")

    results: dict[str, dict] = {}

    # TF-IDF baseline: always available.
    results["tfidf"] = evaluate(TfidfRetriever(chunks), queries, args.k)

    # Embedding backend: needs a built index.
    vectors, meta = load_index(root / INDEX_FILENAME)
    if vectors is None:
        print(
            "\nNOTE: no embedding index found. Run scripts/build_index.py to "
            "include the semantic backend in this comparison.",
            file=sys.stderr,
        )
    elif meta.get("chunk_ids") != [c["chunk_id"] for c in chunks]:
        print(
            "\nNOTE: embedding index is stale relative to the knowledge base. "
            "Re-run scripts/build_index.py.",
            file=sys.stderr,
        )
    else:
        results["embedding"] = evaluate(
            EmbeddingRetriever(chunks, vectors, meta.get("model", "")), queries, args.k
        )

    # --- report -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("RETRIEVAL COMPARISON")
    print("=" * 78)
    header = f"  {'backend':11s} {'Hit@1':>7s} {'Hit@3':>7s} {'Recall@'+str(args.k):>9s} {'MRR':>7s} {'P@'+str(args.k):>7s}"
    print(header)
    for name, res in results.items():
        o = res["overall"]
        print(
            f"  {name:11s} {o['hit_at_1']:>7.4f} {o['hit_at_3']:>7.4f} "
            f"{o['recall_at_k']:>9.4f} {o['mrr']:>7.4f} {o['precision_at_k']:>7.4f}"
        )

    print("\n--- paraphrased queries only (semantic advantage expected here) ---")
    print(f"  {'backend':11s} {'n':>4s} {'Hit@1':>7s} {'Hit@3':>7s} {'Recall':>8s} {'MRR':>7s}")
    for name, res in results.items():
        p = res["paraphrased_only"]
        print(
            f"  {name:11s} {p['n']:>4d} {p['hit_at_1']:>7.4f} {p['hit_at_3']:>7.4f} "
            f"{p['recall_at_k']:>8.4f} {p['mrr']:>7.4f}"
        )

    print("\n--- literal queries only ---")
    print(f"  {'backend':11s} {'n':>4s} {'Hit@1':>7s} {'Hit@3':>7s} {'Recall':>8s} {'MRR':>7s}")
    for name, res in results.items():
        l = res["literal_only"]
        print(
            f"  {name:11s} {l['n']:>4d} {l['hit_at_1']:>7.4f} {l['hit_at_3']:>7.4f} "
            f"{l['recall_at_k']:>8.4f} {l['mrr']:>7.4f}"
        )

    best = max(results, key=lambda n: results[n]["overall"]["mrr"]) if results else None
    if best:
        print(f"\nBest by MRR: {best}")

    if args.show_failures:
        for name, res in results.items():
            misses = [r for r in res["per_query"] if r["reciprocal_rank"] == 0.0]
            print(f"\n--- {name}: {len(misses)} complete misses ---")
            for row in misses:
                print(f"  [{row['id']}] {row['query']}")
                print(f"      wanted : {row['relevant']}")
                print(f"      got    : {row['retrieved'][:3]}")

    out = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "k": args.k,
        "n_chunks": len(chunks),
        "n_queries": len(queries),
        "eval_set": "knowledge/retrieval_eval.json",
        "best_by_mrr": best,
        "metric_notes": {
            "precision_at_k": (
                "Capped below 1.0 by construction: relevant sets contain 1-4 "
                "chunks while k=5. Use recall_at_k and MRR to judge quality."
            )
        },
        "results": {
            name: {k: v for k, v in res.items() if k != "per_query"}
            for name, res in results.items()
        },
        "per_query": {name: res["per_query"] for name, res in results.items()},
    }

    out_path = root / "retrieval_results.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nWritten: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
