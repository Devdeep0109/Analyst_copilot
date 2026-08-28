"""
eval/run_hybrid.py

Day 4: fuse BM25 + dense, and decide whether fusion earns its complexity.

Runs entirely from cached chunk and query embeddings -- no model load, seconds
not minutes. If it complains about missing query vectors, run:

    python eval/cache_queries.py

Usage:
    python eval/run_hybrid.py             # headline table + type split
    python eval/run_hybrid.py --sweep     # weights, rrf_k, candidate pool
    python eval/run_hybrid.py --routing   # the per-type weight conflict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.corpus import load_corpus                  # noqa: E402
from eval.gold import load_gold                      # noqa: E402
from eval.harness import evaluate_retrieval          # noqa: E402
from retrieval.bm25 import PageSumBM25               # noqa: E402
from retrieval.dense import DenseRetriever           # noqa: E402
from retrieval.hybrid import HybridRetriever, OracleUnion  # noqa: E402

BEST_K1, BEST_B = 0.6, 0.0

# Tuned below. Weighted min-max fusion, equal weights, pool of 50 candidates
# per retriever. Notably the weight barely matters (0.520-0.559 across a 8x
# range), which is a good sign -- fusion here is robust rather than balanced on
# a knife edge.
FUSION_METHOD = "weighted"
FUSION_POOL = 50


def _fmt(name, rep):
    return (f"  {name:<44} page={rep.page_recall:.3f}  evid={rep.evidence_recall:.3f}  "
            f"allpg={rep.all_pages_recall:.3f}  mrr={rep.mrr:.3f}")


def _types(rep):
    out = {}
    for nm, sub in rep.split().items():
        if sub:
            out[nm] = sum(x.page_hit for x in sub) / len(sub)
    return out


def headline(gold, corpus):
    rs = [
        PageSumBM25(k1=BEST_K1, b=BEST_B, name="BM25 tuned"),
        DenseRetriever(name="Dense (page_max, stripped)"),
        HybridRetriever(method="rrf", name="Hybrid RRF"),
        HybridRetriever(method=FUSION_METHOD, candidate_pool=FUSION_POOL,
                        name="Hybrid weighted  <-- WINNER"),
        OracleUnion(),
    ]
    for k in (1, 3, 5, 10, 20):
        print(f"k={k}")
        for r in rs:
            print(_fmt(r.name, evaluate_retrieval(r, gold=gold, k=k, corpus=corpus)))
        print()

    print("=" * 78)
    print("SPLIT BY QUESTION TYPE (k=10) -- READ THIS BEFORE CELEBRATING")
    print("=" * 78)
    for r in rs[:4]:
        t = _types(evaluate_retrieval(r, gold=gold, k=10, corpus=corpus))
        print(f"  {r.name:<44}" + "  ".join(f"{k}={v:.3f}" for k, v in t.items()))
    print(
        "\n  Hybrid raises the average while LOWERING reasoning below either\n"
        "  parent. That is not a rounding artifact -- see --routing."
    )


def sweep(gold, corpus):
    print("=" * 78)
    print("WEIGHT SWEEP (k=10)")
    print("=" * 78)
    print(f"  {'w_bm25':>7} {'RRF':>8} {'weighted':>10}")
    for wb in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        a = evaluate_retrieval(HybridRetriever(method="rrf", w_bm25=wb),
                               gold=gold, k=10, corpus=corpus)
        b = evaluate_retrieval(HybridRetriever(method="weighted", w_bm25=wb,
                                               candidate_pool=FUSION_POOL),
                               gold=gold, k=10, corpus=corpus)
        print(f"  {wb:>7} {a.page_recall:>8.3f} {b.page_recall:>10.3f}")
    print("\n  Flat across an 8x weight range -> fusion is robust here, not\n"
          "  precariously tuned. Equal weights are a defensible default.\n")

    print("=" * 78)
    print("CANDIDATE POOL (weighted, k=10)")
    print("=" * 78)
    for cp in (10, 20, 50, 100, 0):
        rep = evaluate_retrieval(HybridRetriever(method="weighted", candidate_pool=cp),
                                 gold=gold, k=10, corpus=corpus)
        print(f"  pool={cp if cp else 'all':<5} page={rep.page_recall:.3f} "
              f"evid={rep.evidence_recall:.3f} mrr={rep.mrr:.3f}")
    print("\n  Too small starves fusion of candidates; unbounded lets each\n"
          "  retriever vote in the depth where its ordering is noise.\n")


def routing(gold, corpus):
    print("=" * 78)
    print("THE PER-TYPE WEIGHT CONFLICT")
    print("=" * 78)
    print(f"  {'w_bm25':>7} {'overall':>8} {'extractive':>11} {'reasoning':>10} {'multipage':>10}")
    for wb in (0.25, 1.0, 2.0, 4.0, 8.0):
        rep = evaluate_retrieval(HybridRetriever(method="weighted", w_bm25=wb,
                                                 candidate_pool=FUSION_POOL),
                                 gold=gold, k=10, corpus=corpus)
        t = _types(rep)
        print(f"  {wb:>7} {rep.page_recall:>8.3f} {t.get('extractive',0):>11.3f} "
              f"{t.get('reasoning',0):>10.3f} {t.get('multi-page',0):>10.3f}")
    print(
        "\n  Extractive peaks at LOW w_bm25 (dense-led); reasoning peaks at HIGH\n"
        "  w_bm25 (lexical-led). No single weight serves both -- the optimum for\n"
        "  one is near the worst case for the other.\n"
    )

    print("=" * 78)
    print("WOULD ROUTING BY QUESTION TYPE ACTUALLY PAY? (oracle labels)")
    print("=" * 78)
    ext = [q for q in gold if not q.is_reasoning]
    rea = [q for q in gold if q.is_reasoning]
    for k in (5, 10, 20):
        a = evaluate_retrieval(HybridRetriever(method="weighted", w_bm25=0.25,
                                               candidate_pool=FUSION_POOL),
                               gold=ext, k=k, corpus=corpus)
        b = evaluate_retrieval(HybridRetriever(method="weighted", w_bm25=8.0,
                                               candidate_pool=FUSION_POOL),
                               gold=rea, k=k, corpus=corpus)
        single = evaluate_retrieval(HybridRetriever(method="weighted",
                                                    candidate_pool=FUSION_POOL),
                                    gold=gold, k=k, corpus=corpus)
        n = len(gold)
        routed = (a.page_recall * len(ext) + b.page_recall * len(rea)) / n
        print(f"  k={k:<3} single={single.page_recall:.3f}  oracle-routed={routed:.3f}  "
              f"({routed - single.page_recall:+.3f})")
    print(
        "\n  HONEST READ: the aggregate gain is small and inconsistent, because\n"
        "  reasoning is only 26 of 127 questions -- routing fixes a lot of a\n"
        "  small slice. And this uses ORACLE labels; a real classifier is worse.\n"
        "\n  But page_recall understates it. Reasoning questions are the ones that\n"
        "  produce CONFIDENTLY WRONG multi-fact answers, which score -1 on the\n"
        "  rubric while a miss on an extractive question usually becomes a 0\n"
        "  abstain. The decision belongs to Day 5, judged on expected SCORE\n"
        "  rather than on recall."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--routing", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    gold = load_gold()
    corpus = load_corpus({q.doc_name for q in gold})
    print(f"corpus: {len(corpus)} filings, {sum(len(v) for v in corpus.values())} chunks")
    print(f"gold  : {len(gold)} questions\n")

    headline(gold, corpus)
    if args.sweep or args.all:
        sweep(gold, corpus)
    if args.routing or args.all:
        routing(gold, corpus)


if __name__ == "__main__":
    main()
