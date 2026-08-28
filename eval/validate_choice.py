"""
eval/validate_choice.py

Before locking "hybrid + ms-marco rerank" as THE retrieval config, check
whether that decision actually survives scrutiny. Three specific worries:

1. IS THE GAIN REAL, OR NOISE?
   0.559 -> 0.606 on 127 questions is six questions. With n=127 the standard
   error on a proportion near 0.5 is about 0.044, which is the same size as
   the effect. A paired test is required, not a comparison of two averages --
   the same questions are being scored twice, so the pairing carries most of
   the information.

2. IS IT OVERFIT?
   Every parameter -- k1, b, fusion method, weights, candidate pool, reranker
   model, depth -- was selected on these same 127 questions. There is no
   held-out set. That is textbook selection-on-the-test-set, and the honest
   check is whether the choice still wins on data it was not tuned on.

3. HOW MANY CHOICES DID WE MAKE?
   Roughly 40 configurations were evaluated across the sweeps. At that many
   comparisons, the best-looking one is expected to be flattered by chance
   even if every option were identical. The winner's margin has to be read
   with that in mind.

Run:  python eval/validate_choice.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.corpus import load_corpus                  # noqa: E402
from eval.gold import load_gold                      # noqa: E402
from eval.harness import evaluate_retrieval          # noqa: E402
from retrieval.bm25 import PageSumBM25               # noqa: E402
from retrieval.dense import DenseRetriever           # noqa: E402
from retrieval.hybrid import HybridRetriever         # noqa: E402
from retrieval.rerank import MS_MARCO, RerankRetriever  # noqa: E402

K = 10


def hits(rep) -> dict[str, bool]:
    return {r.qid: r.page_hit for r in rep.results}


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    """Exact binomial McNemar on the discordant pairs.

    Only questions where the two systems DISAGREE carry information about
    which is better. Questions both get right, or both get wrong, tell us
    nothing about the difference.
    """
    b_only = sum(1 for q in a if b.get(q) and not a[q])   # rerank wins
    a_only = sum(1 for q in a if a[q] and not b.get(q))   # baseline wins
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    # Two-sided exact binomial p under H0: p(win) = 0.5
    from math import comb
    k = min(a_only, b_only)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return a_only, b_only, min(1.0, p)


def bootstrap_ci(a: dict[str, bool], b: dict[str, bool], iters: int = 20000,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap on the PAIRED difference in page_recall."""
    rng = random.Random(seed)
    qids = list(a)
    n = len(qids)
    diffs = []
    for _ in range(iters):
        s = [qids[rng.randrange(n)] for _ in range(n)]
        da = sum(a[q] for q in s) / n
        db = sum(b[q] for q in s) / n
        diffs.append(db - da)
    diffs.sort()
    obs = sum(b[q] for q in qids) / n - sum(a[q] for q in qids) / n
    return obs, diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def split_half_check(gold, corpus, seed: int = 0) -> None:
    """Does the winner still win on halves it was not tuned on?

    A config that only wins on the full set but flips on random halves is
    fitted to noise. One that wins on both halves is at least stable.
    """
    print("=" * 78)
    print("SPLIT-HALF STABILITY (10 random splits)")
    print("=" * 78)
    print("  Does hybrid+rerank beat plain hybrid on BOTH halves each time?")
    print()

    base = HybridRetriever(method="weighted", candidate_pool=50)
    rr = RerankRetriever(model_name=MS_MARCO, depth=20)
    hb = hits(evaluate_retrieval(base, gold=gold, k=K, corpus=corpus))
    hr = hits(evaluate_retrieval(rr, gold=gold, k=K, corpus=corpus))

    qids = list(hb)
    rng = random.Random(seed)
    both_win = 0
    for i in range(10):
        rng.shuffle(qids)
        h1, h2 = qids[: len(qids) // 2], qids[len(qids) // 2:]
        d1 = sum(hr[q] for q in h1) / len(h1) - sum(hb[q] for q in h1) / len(h1)
        d2 = sum(hr[q] for q in h2) / len(h2) - sum(hb[q] for q in h2) / len(h2)
        ok = d1 > 0 and d2 > 0
        both_win += ok
        print(f"    split {i+1:2d}: half A {d1:+.3f}   half B {d2:+.3f}   "
              f"{'both positive' if ok else 'NOT consistent'}")
    print(f"\n  consistent on both halves: {both_win}/10")


def main() -> None:
    gold = load_gold()
    corpus = load_corpus({q.doc_name for q in gold})
    print(f"gold: {len(gold)} questions, k={K}\n")

    systems = {
        "BM25 tuned": PageSumBM25(k1=0.6, b=0.0),
        "Dense page_max": DenseRetriever(),
        "Hybrid weighted": HybridRetriever(method="weighted", candidate_pool=50),
        "Hybrid + rerank(20)": RerankRetriever(model_name=MS_MARCO, depth=20),
    }
    H = {}
    print("=" * 78)
    print("POINT ESTIMATES")
    print("=" * 78)
    for name, r in systems.items():
        rep = evaluate_retrieval(r, gold=gold, k=K, corpus=corpus)
        H[name] = hits(rep)
        n = len(rep.results)
        p = rep.page_recall
        se = (p * (1 - p) / n) ** 0.5
        print(f"  {name:<22} page_recall={p:.3f}  +/-{1.96*se:.3f} (95% CI, unpaired)")
    print("\n  Those unpaired intervals all overlap heavily. That is exactly why\n"
          "  the paired tests below are the ones that matter.\n")

    print("=" * 78)
    print("PAIRED COMPARISONS (McNemar, exact)")
    print("=" * 78)
    pairs = [
        ("Hybrid weighted", "Hybrid + rerank(20)"),
        ("BM25 tuned", "Hybrid weighted"),
        ("Dense page_max", "Hybrid weighted"),
        ("BM25 tuned", "Hybrid + rerank(20)"),
    ]
    for a_name, b_name in pairs:
        a_only, b_only, p = mcnemar(H[a_name], H[b_name])
        obs, lo, hi = bootstrap_ci(H[a_name], H[b_name])
        verdict = ("significant" if p < 0.05
                   else "NOT significant" if p > 0.10 else "borderline")
        print(f"  {b_name} vs {a_name}")
        print(f"    wins {b_only:3d} / losses {a_only:3d}   p={p:.3f}  ({verdict})")
        print(f"    diff {obs:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
        if lo < 0 < hi:
            print("    CI includes zero -- cannot rule out no real difference")
        print()

    split_half_check(gold, corpus)

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print(
        "  ~40 configurations were compared while tuning. With that many looks,\n"
        "  a p-value near 0.05 on the winner is weak evidence -- the winner was\n"
        "  SELECTED for looking good on this data.\n\n"
        "  A choice is safe to lock in when it is (a) not significantly worse\n"
        "  than alternatives, and (b) stable across splits. Demanding that it be\n"
        "  significantly BETTER on 127 questions is a bar almost nothing clears,\n"
        "  and waiting for it would stall the project."
    )


if __name__ == "__main__":
    main()
