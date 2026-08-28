"""
eval/diagnose_dense.py

Dense lost badly on the first run -- 0.283 vs BM25's 0.480 at k=10 -- and lost
WORST on reasoning questions (0.154 vs 0.423), which is the exact opposite of
the stated prediction. That prediction was: dense gains on reasoning, ties on
extractive.

A falsified prediction has exactly two honest explanations, and they must be
separated before drawing any conclusion:

  (a) the hypothesis is wrong  -- embeddings really don't help on filings, or
  (b) the experiment is wrong  -- dense was handicapped by the setup.

Two specific setup faults were identified after the fact, both of which would
depress dense scores without saying anything about embeddings:

  1. AGGREGATION. page_sum was chosen for parity with BM25. But BM25 scores
     are sparse (non-matching chunk = exactly 0) while cosine scores are dense
     (every chunk scores ~0.0-0.4 regardless of relevance). Summing dense
     scores accumulates noise with page length, so it ranks long pages, not
     relevant ones. page_max is the correct analogue.

  2. QUERY BOILERPLATE. BM25 strips "Assume that you are a public equities
     analyst..." and gained 0.024 from it. Dense was never given the same
     treatment, and dense is far more sensitive to it -- an embedding averages
     over every token, and it has no IDF to discount ubiquitous phrases.

This script re-runs dense under every combination, using CACHED embeddings, so
it costs seconds rather than 45 minutes. Only then is the (a)-vs-(b) question
answerable.

Run:  python eval/diagnose_dense.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.corpus import load_corpus                     # noqa: E402
from eval.gold import load_gold                         # noqa: E402
from eval.harness import evaluate_retrieval             # noqa: E402
from retrieval.bm25 import PageSumBM25                  # noqa: E402
from retrieval.dense import (DEFAULT_MODEL, DenseRetriever,  # noqa: E402
                             strip_boilerplate)

BEST_K1, BEST_B = 0.6, 0.0


def show_boilerplate_effect(gold) -> None:
    print("=" * 78)
    print("WHAT BOILERPLATE STRIPPING ACTUALLY DOES TO A QUERY")
    print("=" * 78)
    longest = sorted(gold, key=lambda q: -len(q.question))[:3]
    for q in longest:
        before = q.question
        after = strip_boilerplate(before)
        print(f"  before ({len(before.split()):2d}w): {before[:150]}")
        print(f"  after  ({len(after.split()):2d}w): {after[:150]}")
        print()


def main() -> None:
    gold = load_gold()
    corpus = load_corpus({q.doc_name for q in gold})
    print(f"corpus: {len(corpus)} filings, {sum(len(v) for v in corpus.values())} chunks")
    print(f"gold  : {len(gold)} questions")
    print("(uses cached embeddings -- no re-encoding)\n")

    show_boilerplate_effect(gold)

    bm25 = PageSumBM25(k1=BEST_K1, b=BEST_B, name="BM25 tuned")

    configs = [
        # (aggregate, strip, topn, label)
        ("page_sum", False, 0, "page_sum   + boilerplate  (ORIGINAL RUN)"),
        ("page_sum", True, 0, "page_sum   + stripped"),
        ("page_sum", True, 3, "page_sum@3 + stripped"),
        ("page_max", False, 0, "page_max   + boilerplate"),
        ("page_max", True, 0, "page_max   + stripped     (NEW DEFAULT)"),
        ("chunk", True, 0, "chunk-level + stripped"),
    ]

    print("=" * 78)
    print("DENSE UNDER EVERY CONFIGURATION (k=10)")
    print("=" * 78)
    print(f"  {'config':<42} {'page':>6} {'evid':>6} {'allpg':>6} {'mrr':>6}")

    rep_bm = evaluate_retrieval(bm25, gold=gold, k=10, corpus=corpus)
    print(f"  {'BM25 tuned (the line to beat)':<42} {rep_bm.page_recall:>6.3f} "
          f"{rep_bm.evidence_recall:>6.3f} {rep_bm.all_pages_recall:>6.3f} {rep_bm.mrr:>6.3f}")
    print()

    best = None
    reps = {}
    for agg, strip, topn, label in configs:
        r = DenseRetriever(aggregate=agg, strip_query_boilerplate=strip,
                           topn_pool=topn, name=label)
        rep = evaluate_retrieval(r, gold=gold, k=10, corpus=corpus)
        reps[label] = rep
        print(f"  {label:<42} {rep.page_recall:>6.3f} {rep.evidence_recall:>6.3f} "
              f"{rep.all_pages_recall:>6.3f} {rep.mrr:>6.3f}")
        if best is None or rep.page_recall > best[1].page_recall:
            best = (label, rep, r)

    print(f"\n  best dense config: {best[0]}  page_recall@10={best[1].page_recall:.3f}")
    print(f"  BM25 tuned       : {rep_bm.page_recall:.3f}")

    print("\n" + "=" * 78)
    print("SPLIT BY QUESTION TYPE -- the prediction under test (k=10)")
    print("=" * 78)
    for label, rep in (("BM25 tuned", rep_bm), (best[0], best[1])):
        print(f"  {label}")
        for name, subset in rep.split().items():
            if subset:
                pr = sum(x.page_hit for x in subset) / len(subset)
                print(f"      {name:<12} n={len(subset):<4} page_recall={pr:.3f}")
    print(
        "\n  The prediction was: dense gains on 'reasoning'. If dense is STILL\n"
        "  behind on reasoning after these fixes, the hypothesis is wrong --\n"
        "  not the experiment -- and that is a real finding worth reporting."
    )

    print("\n" + "=" * 78)
    print("COMPLEMENTARITY WITH BEST DENSE CONFIG (k=10)")
    print("=" * 78)
    b = {r.qid: r.page_hit for r in rep_bm.results}
    d = {r.qid: r.page_hit for r in best[1].results}
    both = sum(1 for q in b if b[q] and d.get(q))
    only_b = sum(1 for q in b if b[q] and not d.get(q))
    only_d = sum(1 for q in b if not b[q] and d.get(q))
    neither = sum(1 for q in b if not b[q] and not d.get(q))
    n = len(b)
    union = (both + only_b + only_d) / n
    print(f"  both      : {both:4d} ({both/n:.1%})")
    print(f"  BM25 only : {only_b:4d} ({only_b/n:.1%})")
    print(f"  dense only: {only_d:4d} ({only_d/n:.1%})   <- what hybrid would add")
    print(f"  neither   : {neither:4d} ({neither/n:.1%})   <- retrieval's real ceiling problem")
    print(f"\n  union ceiling      : {union:.3f}")
    print(f"  best single method : {max(rep_bm.page_recall, best[1].page_recall):.3f}")
    print(f"  headroom for hybrid: {union - max(rep_bm.page_recall, best[1].page_recall):+.3f}")


if __name__ == "__main__":
    main()
