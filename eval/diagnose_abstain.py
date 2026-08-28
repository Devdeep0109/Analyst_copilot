"""
eval/diagnose_abstain.py

The model declines 40% of questions. Prompt changes have not moved it (34% ->
40% after an explicit "ANSWER BY DEFAULT" rewrite). So the question is whether
those declines are TIMIDITY or CORRECT.

Two possibilities, needing opposite fixes:

  (a) the evidence is there and the model is being cautious
        -> prompt / model problem
  (b) the evidence genuinely is not there
        -> retrieval problem: raise K, or improve ranking

This distinguishes them with ZERO API calls, by asking of each declined
question: are the numbers needed to answer it actually present in the chunks
we showed the model?

The test uses the GOLD ANSWER's own figures plus the gold evidence text -- if
neither appears in what the model was shown, no amount of prompting could have
produced a correct answer, and declining was right.

Run:  python eval/diagnose_abstain.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.corpus import load_corpus                       # noqa: E402
from eval.gold import load_gold                           # noqa: E402
from eval.harness import answerable_from                  # noqa: E402
from eval.run_pipeline import build_retriever             # noqa: E402
from eval.scoring import extract_numbers                  # noqa: E402
from parser.page_parser import _normalize                 # noqa: E402
from pipeline.answer import PROMPT as APROMPT             # noqa: E402
from pipeline.classify import classify, evidence_k        # noqa: E402
from pipeline.verify import format_evidence               # noqa: E402

CACHE = PROJECT_ROOT / ".cache" / "answers"


def main() -> None:
    files = sorted(CACHE.glob("*.json"))
    if not files:
        print("no answer cache yet")
        return
    answers: dict = {}
    for f in files:
        try:
            answers.update(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass

    gold = load_gold()
    corpus = load_corpus({q.doc_name for q in gold})
    retriever = build_retriever()

    declined, answered = [], []
    for q in gold:
        chunks = corpus.get(q.doc_name)
        if not chunks:
            continue
        retriever.index(q.doc_name, chunks)
        hits = retriever.search(q.doc_name, q.question,
                                evidence_k(classify(q.question)))
        if not hits:
            continue
        shown = format_evidence(hits, 1000)
        key = hashlib.md5(f"{APROMPT}||{q.question}||{shown}".encode()).hexdigest()
        if key not in answers:
            continue
        rec = answers[key]
        (declined if not str(rec.get("answer") or "").strip()
         else answered).append((q, hits, shown))

    print(f"answered: {len(answered)}   declined: {len(declined)}\n")
    if not declined:
        print("no declines in cache")
        return

    print("=" * 78)
    print("WAS THE ANSWER ACTUALLY THERE? (per declined question)")
    print("=" * 78)

    had_it = 0
    rows = []
    for q, hits, shown in declined:
        blob = _normalize(shown)
        tight = blob.replace(" ", "")

        # SIGNAL 1 -- REMOVED, it was worse than useless.
        #
        # It searched for the gold answer's digits in the evidence, e.g. "24"
        # from gold "24.26". Two fatal flaws:
        #   * "24" matches almost any financial page as a substring, so it
        #     returned True constantly -- a false positive machine.
        #   * for CALCULATION questions the gold answer is a COMPUTED result
        #     (24.26 = revenue / average PP&E). It never appears in the filing
        #     at all, so looking for it tests nothing.
        # Together these produced a confident "75% of declines had the
        # evidence -- it's a prompt problem" verdict that was simply wrong.
        # Checked by hand: the Activision fixed-asset-turnover question was
        # missing its revenue figure entirely from all 10 excerpts.
        #
        # The honest test for a multi-fact question is whether ALL its gold
        # evidence pages were retrieved -- one of three inputs is not enough.
        num_hit = False

        # signal 2: does the gold proving text appear?
        lines = []
        for e in q.evidence:
            ls = [l.strip() for l in e.text.split("\n") if len(l.strip()) >= 20]
            lines.extend(sorted(ls, key=len, reverse=True)[:4])
        text_hit = any(_normalize(l) in blob for l in lines)

        # signal 3: our own retrieval-level answerability check
        ans_ok, why = answerable_from(q, hits)

        # signal 4 (the strict one): were ALL gold evidence pages retrieved?
        # A question needing income statement + balance sheet is unanswerable
        # with only one of them, however good that one is.
        pages = {c.page_num for c in hits}
        all_pages = q.gold_pages.issubset(pages)

        present = all_pages or (text_hit and len(q.gold_pages) == 1)
        had_it += present
        rows.append((q, present, num_hit, text_hit, ans_ok, why))

    print(f"  declined questions where the evidence WAS shown : "
          f"{had_it}/{len(declined)} ({had_it/len(declined):.0%})")
    print(f"  declined where it genuinely was NOT             : "
          f"{len(declined)-had_it}/{len(declined)}")

    print("\n" + "=" * 78)
    print("TIMID -- evidence was shown, model declined anyway")
    print("=" * 78)
    n = 0
    for q, present, num_hit, text_hit, ans_ok, why in rows:
        if present and n < 8:
            n += 1
            print(f"  Q: {q.question[:88]}")
            print(f"     gold={q.answer[:44]!r}  gold pages={sorted(q.gold_pages)} "
                  f"all retrieved={present}")

    print("\n" + "=" * 78)
    print("CORRECT -- evidence really was missing, declining was right")
    print("=" * 78)
    n = 0
    for q, present, num_hit, text_hit, ans_ok, why in rows:
        if not present and n < 8:
            n += 1
            print(f"  Q: {q.question[:88]}")
            print(f"     gold={q.answer[:44]!r}  retrieval-answerable={ans_ok} ({why})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    frac = had_it / len(declined)
    if frac >= 0.6:
        print(f"  {frac:.0%} of declines had the evidence in front of them.")
        print("  This is a PROMPT/MODEL problem -- the answering step is the")
        print("  bottleneck, not retrieval.")
    elif frac <= 0.35:
        print(f"  Only {frac:.0%} of declines had the evidence available.")
        print("  This is a RETRIEVAL problem -- the model is declining correctly.")
        print("  Raising K or improving ranking is the fix; prompting will not help.")
    else:
        print(f"  Mixed: {frac:.0%} had evidence, {1-frac:.0%} did not.")
        print("  Both levers matter; neither alone explains the abstention rate.")


if __name__ == "__main__":
    main()
