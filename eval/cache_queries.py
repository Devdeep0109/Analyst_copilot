"""
eval/cache_queries.py

Embeds the 127 gold questions once and writes them to
.cache/embeddings/_queries__<model>.npz

Run once (takes a few seconds):

    python eval/cache_queries.py

WHY
---
Chunk embeddings were already cached, but query embeddings still required
loading the model -- which means torch, which means the full ML stack must be
present for any dense experiment.

With the queries cached too, the entire Day-4 fusion problem becomes pure numpy
over cached vectors: no model, no torch, no 45-minute runs. Hybrid weight
sweeps that would otherwise need a GPU-ish machine become instant, which is the
difference between tuning fusion properly and guessing at it.

This caches BOTH the boilerplate-stripped and raw forms of each question, so a
retriever configured either way can be evaluated from the same file.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.gold import load_gold                              # noqa: E402
from retrieval.dense import DEFAULT_MODEL, DenseRetriever    # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    gold = load_gold()
    questions = [q.question for q in gold]
    print(f"gold questions : {len(questions)}")
    print(f"model          : {args.model}")

    r = DenseRetriever(model_name=args.model, strip_query_boilerplate=True)
    n_new = r.save_query_cache(questions)

    path = r._query_cache_path()
    size_kb = path.stat().st_size / 1024 if path.exists() else 0
    print(f"newly embedded : {n_new}")
    print(f"written        : {path.name}  ({size_kb:.0f} KB)")
    print("\nDense retrieval can now run without loading the model.")


if __name__ == "__main__":
    main()
