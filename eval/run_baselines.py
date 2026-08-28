"""
eval/run_baselines.py

Day-2 smoke test: run the baseline retrievers through the harness and confirm
the metrics separate them. Also self-tests the scoring rubric.

Run:  python eval/run_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.corpus import load_corpus                                    # noqa: E402
from eval.gold import load_gold, summarize                             # noqa: E402
from eval.harness import evaluate_retrieval                            # noqa: E402
from eval.scoring import (Outcome, Report, SystemAnswer, answers_match,  # noqa: E402
                          grade)
from retrieval.baselines import (FirstKRetriever, KeywordOverlapRetriever,  # noqa: E402
                                 RandomRetriever)


def test_scoring() -> None:
    print("=" * 70)
    print("SCORING RUBRIC SELF-TEST")
    print("=" * 70)

    cases = [
        # (predicted, gold, expected answers_match)
        ("$1577.00", "$1577.00", True),
        ("1,577", "$1577.00", True),
        ("$1.577 billion", "$1577.00 million", True),
        ("1577", "1578", False),          # off-by-one is WRONG, not "within tolerance"
        ("$8.70", "$8.70", True),
        ("(1,577)", "-1577", True),       # accounting negative
        ("8.7012", "$8.70", True),        # rounds to the gold's stated precision
        ("8.70", "8.75", False),          # same precision, different number
        # Gold "$8.70" comes from a question that says "Answer in USD billions" --
        # the unit lives in the QUESTION, not the answer. So a prediction that
        # spells out the scale must still match. This is why the comparator has a
        # scale-agnostic fallback, and why it is correct rather than sloppy.
        ("8.7 billion", "$8.70", True),
        # Yes/no verdicts must agree even when the numbers line up.
        ("Yes, debt decreased by $229 million.",
         "No. Verizon's debt decreased by $229 million.", False),
        ("$66.56 per share", "They could receive $66.56 per share.", True),
    ]
    ok = True
    for pred, gold, expect in cases:
        got = answers_match(pred, gold)
        flag = "ok " if got == expect else "FAIL"
        if got != expect:
            ok = False
        print(f"  {flag} answers_match({pred!r:22}, {gold!r:20}) = {got}  expected {expect}")

    prose = ("No, the company is managing its CAPEX and Fixed Assets pretty "
             "efficiently, which is evident from below key metrics: CAPEX/Revenue "
             "Ratio: 5.1%")
    got = answers_match("No", prose)
    print(f"  {'ok ' if got is None else 'FAIL'} prose gold -> {got} (None = defer to LLM judge)")
    if got is not None:
        ok = False

    print("\n  --- rubric outcomes ---")
    checks = [
        (SystemAnswer(answer_text="$1577.00", cited_page=60), "$1577.00", {60}, Outcome.CORRECT_LOCATED),
        (SystemAnswer(answer_text="$1577.00", cited_page=12), "$1577.00", {60}, Outcome.CORRECT_WRONG_PAGE),
        (SystemAnswer(abstained=True), "$1577.00", {60}, Outcome.ABSTAINED),
        (SystemAnswer(answer_text="$9999", cited_page=60), "$1577.00", {60}, Outcome.CONFIDENTLY_WRONG),
    ]
    rep = Report()
    for sysans, gold_ans, pages, expect in checks:
        got_o = grade(sysans, gold_ans, pages)
        rep.add(got_o)
        flag = "ok " if got_o is expect else "FAIL"
        if got_o is not expect:
            ok = False
        print(f"  {flag} {got_o.value:<20} score={got_o.score:+.1f}  expected {expect.value}")

    print(f"\n  total score for those 4: {rep.total_score:+.1f} (expect +0.5)")
    print(f"\n  scoring self-test: {'PASS' if ok else 'FAIL'}\n")


def main() -> None:
    gold = load_gold()
    print("=" * 70)
    print("GOLD SET")
    print("=" * 70)
    print(summarize(gold))
    print()

    test_scoring()

    print("=" * 70)
    print("BASELINE RETRIEVERS")
    print("=" * 70)
    corpus = load_corpus({q.doc_name for q in gold})
    print(f"corpus: {len(corpus)} filings, {sum(len(v) for v in corpus.values())} chunks\n")

    retrievers = [RandomRetriever(seed=0), FirstKRetriever(), KeywordOverlapRetriever()]
    for r in retrievers:
        for k in (5, 10):
            print(evaluate_retrieval(r, gold=gold, k=k, corpus=corpus).render())
            print()

    print("=" * 70)
    print("READ THIS")
    print("=" * 70)
    print(
        "The harness is trustworthy only if the numbers above are ORDERED:\n"
        "  keyword overlap > random, and both far below 1.0.\n"
        "If random scores well, k is too large relative to document length and\n"
        "page_recall is not discriminating -- fix that before Day 3.\n"
        "Whatever BM25 scores tomorrow means nothing except as a margin over\n"
        "these lines."
    )


if __name__ == "__main__":
    main()
