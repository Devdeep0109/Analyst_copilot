"""
eval/run_rerank.py

Day 4, rung 4: does a cross-encoder reranker convert the headroom?

THE OPPORTUNITY, RESTATED
-------------------------
Hybrid page_recall by depth:  top-10 0.559 | top-50 0.890 | top-200 0.976
The right page is usually retrieved and then ranked too low. Reranking the
top-50 into a top-10 therefore has a ceiling of 0.890 vs today's 0.559.

The numbers to watch, in order of importance:

  page_recall@5 and @10  -- did the reranker actually pull gold pages up?
  mrr                    -- did it put them at the TOP, not merely in the list?
                            This is what lets K shrink, and a smaller K means
                            fewer tokens and less surface for a bad citation.
  reasoning split        -- hybrid currently scores 0.269 there, BELOW both
                            parents. A reranker reads query and passage
                            together, so this is where it should help most.
                            If it doesn't, the reasoning problem is not a
                            ranking problem and Day 5 has to route instead.

RUN ORDER
---------
    python eval/run_rerank.py --smoke          # 3 filings, ~1 min, do this first
    python eval/run_rerank.py                  # ms-marco, depth sweep
    python eval/run_rerank.py --model bge      # the bigger model
    python eval/run_rerank.py --both           # everything (slowest)

The depth sweep is cheap after the first pass: scores are cached per
(model, query, chunk), so depth 20 and 50 reuse what depth 100 already scored.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.corpus import load_corpus                       # noqa: E402
from eval.gold import load_gold                           # noqa: E402
from eval.harness import evaluate_retrieval               # noqa: E402
from retrieval.hybrid import HybridRetriever              # noqa: E402
from retrieval.rerank import (BGE_BASE, MODEL_NOTES, MS_MARCO,  # noqa: E402
                              RerankRetriever, cache_stats)

DEPTHS = (20, 50, 100)
KS = (1, 3, 5, 10)


def _fmt(name, rep):
    return (f"  {name:<40} page={rep.page_recall:.3f}  evid={rep.evidence_recall:.3f}  "
            f"allpg={rep.all_pages_recall:.3f}  mrr={rep.mrr:.3f}")


def _types(rep):
    return {nm: sum(x.page_hit for x in sub) / len(sub)
            for nm, sub in rep.split().items() if sub}


def smoke(model_name):
    print("=" * 78)
    print(f"SMOKE TEST -- 3 filings  ({MODEL_NOTES.get(model_name, '')})")
    print("=" * 78)
    gold_all = load_gold()
    docs = sorted({q.doc_name for q in gold_all})[:3]
    gold = [q for q in gold_all if q.doc_name in docs]
    corpus = load_corpus(docs)
    print(f"questions: {len(gold)}   filings: {docs}\n")

    base = HybridRetriever(method="weighted", candidate_pool=50, name="hybrid (no rerank)")
    print(_fmt(base.name, evaluate_retrieval(base, gold=gold, k=5, corpus=corpus)))

    print(f"\nloading {model_name} (first run downloads from Hugging Face) ...")
    r = RerankRetriever(model_name=model_name, depth=50, show_progress=True)
    t0 = time.time()
    rep = evaluate_retrieval(r, gold=gold, k=5, corpus=corpus)
    dt = time.time() - t0
    r.save_cache()
    print(_fmt(r.name, rep))
    pairs = len(gold) * 50
    print(f"\n{dt:.1f}s for ~{pairs} pairs  (~{pairs/dt:.0f} pairs/sec)")
    print(f"Full run (127 q x 100 depth = 12,700 pairs) would take "
          f"~{12700/(pairs/dt)/60:.1f} min, once. Cached after.")


def full(model_name, gold, corpus):
    print("=" * 78)
    print(f"RERANKER: {model_name}")
    print(f"  {MODEL_NOTES.get(model_name, '')}")
    print("=" * 78)

    base = HybridRetriever(method="weighted", candidate_pool=50,
                           name="hybrid (baseline, no rerank)")
    base_reps = {k: evaluate_retrieval(base, gold=gold, k=k, corpus=corpus) for k in KS}

    # Run the deepest first: it populates the score cache that the shallower
    # depths then reuse for free.
    reranker = {}
    for depth in sorted(DEPTHS, reverse=True):
        r = RerankRetriever(model_name=model_name, depth=depth, show_progress=True)
        t0 = time.time()
        reranker[depth] = {k: evaluate_retrieval(r, gold=gold, k=k, corpus=corpus)
                           for k in KS}
        r.save_cache()
        print(f"  depth={depth} done in {time.time()-t0:.0f}s   [{cache_stats(model_name)}]")

    print()
    for k in KS:
        print(f"k={k}")
        print(_fmt(base.name, base_reps[k]))
        for depth in DEPTHS:
            print(_fmt(f"  + rerank depth={depth}", reranker[depth][k]))
        print()

    print("=" * 78)
    print("CEILING CHECK")
    print("=" * 78)
    ceil50 = evaluate_retrieval(base, gold=gold, k=50, corpus=corpus).page_recall
    best = max(reranker[d][10].page_recall for d in DEPTHS)
    print(f"  hybrid page_recall@50 (what depth-50 rerank could reach) : {ceil50:.3f}")
    print(f"  hybrid page_recall@10 (starting point)                   : {base_reps[10].page_recall:.3f}")
    print(f"  best reranked page_recall@10                             : {best:.3f}")
    span = ceil50 - base_reps[10].page_recall
    got = best - base_reps[10].page_recall
    if span > 0:
        print(f"\n  captured {got/span:.0%} of the available headroom "
              f"({got:+.3f} of a possible {span:+.3f})")

    print("\n" + "=" * 78)
    print("SPLIT BY QUESTION TYPE (k=10) -- does it fix reasoning?")
    print("=" * 78)
    print(f"  {'hybrid (no rerank)':<40}" +
          "  ".join(f"{a}={b:.3f}" for a, b in _types(base_reps[10]).items()))
    for depth in DEPTHS:
        print(f"  {'+ rerank depth=' + str(depth):<40}" +
              "  ".join(f"{a}={b:.3f}" for a, b in _types(reranker[depth][10]).items()))
    print(
        "\n  Hybrid scored 0.269 on reasoning, below both its parents. If the\n"
        "  reranker lifts that materially, ranking was the problem. If it does\n"
        "  not, reasoning needs routing or a different evidence strategy on\n"
        "  Day 5 -- and that is worth knowing now, not on Day 6."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="msmarco", choices=["msmarco", "bge"])
    ap.add_argument("--both", action="store_true")
    args = ap.parse_args()

    chosen = MS_MARCO if args.model == "msmarco" else BGE_BASE
    if args.smoke:
        smoke(chosen)
        return

    gold = load_gold()
    corpus = load_corpus({q.doc_name for q in gold})
    print(f"corpus: {len(corpus)} filings, {sum(len(v) for v in corpus.values())} chunks")
    print(f"gold  : {len(gold)} questions\n")

    for m in ([MS_MARCO, BGE_BASE] if args.both else [chosen]):
        full(m, gold, corpus)
        print()


if __name__ == "__main__":
    main()
