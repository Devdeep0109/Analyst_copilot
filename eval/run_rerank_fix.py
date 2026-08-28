"""
eval/run_rerank_fix.py

Measures the reranker fix: score ALL chunks on each candidate page instead of
one representative.

NO API QUOTA IS USED. The cross-encoder runs locally on your CPU. This costs
time, not tokens -- which is the whole reason to do it before spending any
Groq allowance on answering.

THE PROBLEM BEING FIXED
-----------------------
The hybrid returns one chunk per page. Reranking only that chunk means the
cross-encoder answers "is this page relevant?" while reading text that, 33% of
the time, does not contain the evidence (gold pages average 4.2 chunks). The
model was fine -- separation d=0.86 -- it was being fed the wrong input.

WHAT TO EXPECT
--------------
~4x more pairs to score. Offset by cutting the cross-encoder's max_chars from
1800 to 600, which is free: its window is 512 TOKENS, so 1800 chars were being
truncated by the tokenizer regardless.

Watch `all_pages` -- that is the metric gating calculation questions, and the
one that was 0.583 and needs to be higher.

Run:  python eval/run_rerank_fix.py           # full comparison, slow
      python eval/run_rerank_fix.py --quick   # 20 filings, faster signal
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.corpus import load_corpus                   # noqa: E402
from eval.gold import load_gold                       # noqa: E402
from eval.harness import evaluate_retrieval           # noqa: E402
from pipeline.classify import classify, evidence_k    # noqa: E402
from retrieval.hybrid import HybridRetriever          # noqa: E402
from retrieval.rerank import MS_MARCO, RerankRetriever  # noqa: E402


class KAware:
    """Wraps a retriever so the harness's fixed k is replaced by the
    classifier's per-question k -- matching what the pipeline actually does."""

    def __init__(self, inner, name):
        self.inner = inner
        self.name = name

    def index(self, doc_name, chunks):
        self.inner.index(doc_name, chunks)

    def search(self, doc_name, query, k):
        return self.inner.search(doc_name, query,
                                 evidence_k(classify(query), k))


def build(all_chunks: bool, depth: int = 20):
    base = HybridRetriever(method="weighted", candidate_pool=50)
    return RerankRetriever(model_name=MS_MARCO, depth=depth, base=base,
                           all_chunks_per_page=all_chunks,
                           show_progress=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20 filings only")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    gold = load_gold()
    if args.quick:
        docs = sorted({q.doc_name for q in gold})[:20]
        gold = [q for q in gold if q.doc_name in docs]
    corpus = load_corpus({q.doc_name for q in gold})

    print(f"questions: {len(gold)}   filings: {len(corpus)}")
    print("cross-encoder runs LOCALLY -- no API quota used\n")

    results = {}
    for all_chunks, label in ((False, "OLD  one chunk per page"),
                              (True, "NEW  all chunks per page")):
        r = build(all_chunks)
        kr = KAware(r, label)
        print(f"running {label} ...", flush=True)
        t0 = time.time()
        rep = evaluate_retrieval(kr, gold=gold, k=args.k, corpus=corpus)
        r.save_cache()
        dt = time.time() - t0
        results[label] = rep
        print(f"  {dt/60:.1f} min")
        print(f"  page_recall={rep.page_recall:.3f}  "
              f"ALL_PAGES={rep.all_pages_recall:.3f}  "
              f"evidence={rep.evidence_recall:.3f}  mrr={rep.mrr:.3f}\n")

    print("=" * 74)
    print("COMPARISON")
    print("=" * 74)
    old = results["OLD  one chunk per page"]
    new = results["NEW  all chunks per page"]
    for metric in ("page_recall", "all_pages_recall", "evidence_recall", "mrr"):
        o, n = getattr(old, metric), getattr(new, metric)
        flag = "better" if n > o else ("worse" if n < o else "same")
        print(f"  {metric:20s} {o:.3f} -> {n:.3f}   ({n-o:+.3f}, {flag})")

    print("\n  Split by question type (all_pages):")
    for label, rep in (("OLD", old), ("NEW", new)):
        parts = []
        for nm, sub in rep.split().items():
            if sub:
                parts.append(f"{nm}={sum(x.all_pages_hit for x in sub)/len(sub):.3f}")
        print(f"    {label}: " + "  ".join(parts))

    print("\n" + "=" * 74)
    if new.all_pages_recall > old.all_pages_recall + 0.02:
        print("  WORTH KEEPING -- all_pages improved materially.")
        print("  Next: spend quota on a 20-30 question answering batch.")
    elif new.all_pages_recall < old.all_pages_recall:
        print("  REVERT -- the fix made retrieval worse. Do not spend quota.")
    else:
        print("  MARGINAL -- gain is inside noise at this sample size.")
        print("  Judge it on the split above rather than the headline number.")


if __name__ == "__main__":
    main()
