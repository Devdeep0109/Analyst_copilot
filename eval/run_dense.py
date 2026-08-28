"""
eval/run_dense.py

Day 3, rung 2: dense retrieval, measured against the tuned BM25 line.

RUN THIS FIRST -- it takes about a minute and catches setup problems before
you commit to the full embedding pass:

    python eval/run_dense.py --smoke

Then the real thing (first run embeds ~48,600 chunks; 10-25 min on CPU, cached
after that):

    python eval/run_dense.py

Useful extras:

    python eval/run_dense.py --model BAAI/bge-small-en-v1.5
    python eval/run_dense.py --aggregate page_max
    python eval/run_dense.py --status        # how much is already embedded

THE PREDICTION WE ARE TESTING
-----------------------------
BM25 failed on questions whose wording shares no vocabulary with the answering
page ("is 3M capital-intensive?"). So dense should gain most on REASONING
questions and roughly tie on EXTRACTIVE ones, where the question often quotes
the line item almost verbatim.

A uniform win across both splits is a red flag, not a triumph -- it usually
means the two retrievers aren't being compared on equal terms.
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
from retrieval.bm25 import PageSumBM25                    # noqa: E402
from retrieval.dense import (DEFAULT_MODEL, DenseRetriever,  # noqa: E402
                             cache_status)

BEST_K1, BEST_B = 0.6, 0.0          # from eval/run_bm25.py --sweep
KS = (1, 3, 5, 10, 20)


def _line(label: str, rep) -> str:
    return (f"  {label:<34} k={rep.k:<3} page={rep.page_recall:.3f}  "
            f"evid={rep.evidence_recall:.3f}  all_pages={rep.all_pages_recall:.3f}  "
            f"mrr={rep.mrr:.3f}")


def _split_table(name: str, rep) -> str:
    rows = [f"  {name}  (k={rep.k})"]
    for label, subset in rep.split().items():
        if not subset:
            continue
        pr = sum(r.page_hit for r in subset) / len(subset)
        rows.append(f"      {label:<12} n={len(subset):<4} page_recall={pr:.3f}")
    return "\n".join(rows)


def smoke(model_name: str) -> None:
    """Three filings, a handful of questions. Proves the model loads, encodes,
    caches, and ranks -- in about a minute instead of half an hour."""
    print("=" * 78)
    print("SMOKE TEST -- 3 filings only")
    print("=" * 78)

    gold_all = load_gold()
    docs = sorted({q.doc_name for q in gold_all})[:3]
    gold = [q for q in gold_all if q.doc_name in docs]
    corpus = load_corpus(docs)
    n_chunks = sum(len(v) for v in corpus.values())
    print(f"filings   : {docs}")
    print(f"chunks    : {n_chunks}")
    print(f"questions : {len(gold)}\n")

    print(f"loading model {model_name} ...")
    print("(first time only: ~90 MB downloads from Hugging Face to "
          "~/.cache/huggingface/)\n")

    t0 = time.time()
    dense = DenseRetriever(model_name=model_name, show_progress=True)
    _ = dense.model
    print(f"model loaded in {time.time() - t0:.1f}s\n")

    t0 = time.time()
    rep = evaluate_retrieval(dense, gold=gold, k=5, corpus=corpus)
    dt = time.time() - t0
    rate = n_chunks / dt if dt else 0
    print(f"\nembedded + searched in {dt:.1f}s  (~{rate:.0f} chunks/sec)")
    print(_line(dense.name, rep))

    bm25 = PageSumBM25(k1=BEST_K1, b=BEST_B)
    print(_line("BM25 tuned (same 3 filings)",
                evaluate_retrieval(bm25, gold=gold, k=5, corpus=corpus)))

    full = 48596
    print(f"\nAt that rate the full corpus ({full} chunks) would take "
          f"~{full / rate / 60:.0f} min, once.")
    print("If this looks right, run without --smoke.")


def full(model_name: str, aggregate: str) -> None:
    gold = load_gold()
    corpus = load_corpus({q.doc_name for q in gold})
    docs = sorted(corpus)
    print(f"corpus  : {len(corpus)} filings, {sum(len(v) for v in corpus.values())} chunks")
    print(f"gold    : {len(gold)} questions")
    print(f"model   : {model_name}")
    print(f"embed   : {cache_status(docs, model_name)}\n")

    bm25 = PageSumBM25(k1=BEST_K1, b=BEST_B, name="BM25 tuned")
    dense = DenseRetriever(model_name=model_name, aggregate=aggregate, show_progress=True)

    print("=" * 78)
    print("HEAD TO HEAD")
    print("=" * 78)
    reps = {}
    for r in (bm25, dense):
        t0 = time.time()
        for k in KS:
            rep = evaluate_retrieval(r, gold=gold, k=k, corpus=corpus)
            reps[(r.name, k)] = rep
            print(_line(r.name, rep))
        print(f"  ({time.time() - t0:.1f}s total)\n")

    print("=" * 78)
    print("THE ACTUAL QUESTION: WHERE DOES DENSE WIN? (k=10)")
    print("=" * 78)
    print(_split_table(bm25.name, reps[(bm25.name, 10)]))
    print(_split_table(dense.name, reps[(dense.name, 10)]))
    print(
        "\n  Expected: dense gains on 'reasoning', roughly ties on 'extractive'.\n"
        "  If dense wins everywhere by the same margin, check the comparison\n"
        "  before believing it."
    )

    print("\n" + "=" * 78)
    print("COMPLEMENTARITY -- the case for hybrid (k=10)")
    print("=" * 78)
    b = {r.qid: r.page_hit for r in reps[(bm25.name, 10)].results}
    d = {r.qid: r.page_hit for r in reps[(dense.name, 10)].results}
    both = sum(1 for q in b if b[q] and d.get(q))
    only_b = sum(1 for q in b if b[q] and not d.get(q))
    only_d = sum(1 for q in b if not b[q] and d.get(q))
    neither = sum(1 for q in b if not b[q] and not d.get(q))
    union = both + only_b + only_d
    n = len(b)
    print(f"  both found it      : {both:4d}  ({both/n:.1%})")
    print(f"  BM25 only          : {only_b:4d}  ({only_b/n:.1%})")
    print(f"  dense only         : {only_d:4d}  ({only_d/n:.1%})")
    print(f"  neither            : {neither:4d}  ({neither/n:.1%})")
    print(f"\n  union ceiling      : {union/n:.3f}  <- what a PERFECT hybrid could reach")
    print(f"  best single method : {max(reps[(bm25.name,10)].page_recall, reps[(dense.name,10)].page_recall):.3f}")
    print(
        "\n  The gap between those two lines is the entire prize for Day 4.\n"
        "  If it's small, hybrid isn't worth building and a reranker on the\n"
        "  better single retriever is the smarter spend."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 filings, ~1 min")
    ap.add_argument("--status", action="store_true", help="embedding cache status")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--aggregate", default="page_sum",
                    choices=["page_sum", "page_max", "chunk"])
    args = ap.parse_args()

    if args.status:
        gold = load_gold()
        docs = sorted({q.doc_name for q in gold})
        print(cache_status(docs, args.model))
        return
    if args.smoke:
        smoke(args.model)
        return
    full(args.model, args.aggregate)


if __name__ == "__main__":
    main()
