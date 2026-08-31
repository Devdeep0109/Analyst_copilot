"""
eval/tune_gate.py

Verifier #1 currently blocks on ANY "insufficient" verdict. Of the 23 answers
it blocks, 7 would have been confidently wrong (-7) and 5 would have been
right (+4) -- a net gain of +3, but a blunt one.

This asks whether a SELECTIVE gate does better: block only when we have an
additional reason to distrust the answer, and let the rest through.

Signals available for free (no API calls -- everything is already cached):

  v1              Verifier #1's sufficiency verdict
  cite_passed     did the quote verify against the cited page?
  quote_len       a 3-word quote proves less than a 20-word one
  page_corrected  did the model cite the wrong page and we fixed it?
  is_reasoning    judgement questions fail more often than lookups
  multi_page      questions needing figures from several pages
  n_pages         how many distinct pages the evidence spans

The point is not to invent a clever rule. It is to check whether the blunt
gate is leaving anything on the table, using data we already paid for.

Run:  python eval/tune_gate.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.corpus import load_corpus                      # noqa: E402
from eval.gold import load_gold                          # noqa: E402
from eval.harness import answerable_from                 # noqa: E402
from eval.scoring import Outcome, answers_match          # noqa: E402
from pipeline.answer import PROMPT as AP, Answer         # noqa: E402
from pipeline.classify import classify, evidence_k       # noqa: E402
from pipeline.verify import PROMPT as VP, Verdict, format_evidence  # noqa: E402
from pipeline.verify2 import check_arithmetic, check_citation  # noqa: E402
from eval.run_pipeline import build_retriever, score_row  # noqa: E402


def load_rows():
    a = json.loads((PROJECT_ROOT / ".cache/answers/answers__faef90e3.json")
                   .read_text(encoding="utf-8"))
    v = json.loads((PROJECT_ROOT / ".cache/verdicts/verdicts__faef90e3.json")
                   .read_text(encoding="utf-8"))
    gold = load_gold()
    corpus = load_corpus({q.doc_name for q in gold})
    r = build_retriever()
    rows = []
    for q in gold:
        ch = corpus.get(q.doc_name)
        if not ch:
            continue
        r.index(q.doc_name, ch)
        hits = r.search(q.doc_name, q.question, evidence_k(classify(q.question)))
        if not hits:
            continue
        ev = format_evidence(hits, 1000)
        ka = hashlib.md5(f"{AP}||{q.question}||{ev}".encode()).hexdigest()
        if ka not in a:
            continue
        d = a[ka]
        ans = Answer(d.get("answer"), d.get("quote", ""), d.get("page"),
                     d.get("working", ""), retried=d.get("retried", False))
        vd = v.get(hashlib.md5(f"{VP}||{q.question}||{ev}".encode()).hexdigest(),
                   {"sufficient": True})
        ok, why = answerable_from(q, hits)
        rows.append({
            "q": q, "hits": hits, "ans": ans, "answerable": ok, "why": why,
            "v1": Verdict(vd["sufficient"], vd.get("reason", "")),
            "cite": check_citation(ans.quote, ans.page, hits),
            "arith": check_arithmetic(str(ans.answer or ""), ans.working),
            "cls": classify(q.question),
        })
    return rows


def score_with(rows, gate_fn) -> tuple[float, dict]:
    """gate_fn(row) -> True to BLOCK the answer."""
    from collections import Counter
    outs = []
    for r in rows:
        if r["ans"].abstained:
            outs.append(Outcome.ABSTAINED)
            continue
        if gate_fn(r):
            outs.append(Outcome.ABSTAINED)
            continue
        outs.append(score_row(r, False, False, False))
    return sum(o.score for o in outs), Counter(o.value for o in outs)


def main() -> None:
    rows = load_rows()
    print(f"rows: {len(rows)}  (all from cache, no API calls)\n")

    strategies = [
        ("no gate (answer everything)",
         lambda r: False),
        ("CURRENT: block on any V1 'insufficient'",
         lambda r: not r["v1"].sufficient),
        ("block only if V1 no AND citation failed",
         lambda r: not r["v1"].sufficient and not r["cite"].passed),
        ("block only if V1 no AND reasoning question",
         lambda r: not r["v1"].sufficient and r["cls"].is_reasoning),
        ("block only if V1 no AND quote is short (<8 tokens)",
         lambda r: not r["v1"].sufficient and r["cite"].quote_len < 8),
        ("block only if V1 no AND multi-page question",
         lambda r: not r["v1"].sufficient and r["cls"].multi_page),
        ("block if V1 no OR citation failed",
         lambda r: not r["v1"].sufficient or not r["cite"].passed),
        ("block only if citation failed",
         lambda r: not r["cite"].passed),
        ("block if V1 no AND (reasoning OR multi-page)",
         lambda r: not r["v1"].sufficient
                   and (r["cls"].is_reasoning or r["cls"].multi_page)),
    ]

    print(f"  {'strategy':52s}{'score':>8}{'+1':>5}{'0':>5}{'-1':>5}")
    results = []
    for name, fn in strategies:
        s, c = score_with(rows, fn)
        results.append((s, name, c))
        print(f"  {name:52s}{s:>8.1f}"
              f"{c.get('correct+located',0):>5}"
              f"{c.get('abstained',0):>5}"
              f"{c.get('confidently-wrong',0):>5}")

    best = max(results)
    cur = next(r for r in results if r[1].startswith("CURRENT"))
    print(f"\n  best    : {best[1]}  ->  {best[0]:+.1f}")
    print(f"  current : {cur[0]:+.1f}   ({best[0]-cur[0]:+.1f})")

    if best[0] - cur[0] < 1.0:
        print("\n  The blunt gate is already near-optimal. Any 'improvement' of")
        print("  under a point here is fitted to 126 questions and would not")
        print("  survive a different sample -- not worth shipping.")


if __name__ == "__main__":
    main()
