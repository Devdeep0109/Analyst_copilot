"""
eval/report_verdicts.py

Print the Verifier #1 report from CACHED verdicts only. Makes ZERO API calls.

WHY THIS EXISTS
---------------
Measuring and calling the API got tangled together: to see any numbers you had
to sit through a full sweep, and a sweep that hit rate limits produced nothing
at all despite having judged 60+ questions successfully. Verdicts were already
being written to disk every 25 questions -- there was just no way to read them.

This separates the two. Run it any time, including while a sweep is still
going in another terminal, and it reports on whatever has been judged so far.

Run:  python eval/report_verdicts.py
      python eval/report_verdicts.py --model openai/gpt-oss-120b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.corpus import load_corpus                     # noqa: E402
from eval.gold import load_gold                         # noqa: E402
from eval.harness import answerable_from                # noqa: E402
from eval.run_verifier import build_retriever, report   # noqa: E402
from pipeline.classify import classify, evidence_k      # noqa: E402
from pipeline.verify import PROMPT, Verdict, format_evidence  # noqa: E402

VERDICT_CACHE = PROJECT_ROOT / ".cache" / "verdicts"


def load_cache(client_name: str) -> dict:
    tag = hashlib.md5(client_name.encode()).hexdigest()[:8]
    p = VERDICT_CACHE / f"verdicts__{tag}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def list_caches() -> None:
    if not VERDICT_CACHE.exists():
        print("no verdict cache yet")
        return
    print("cached verdict files:")
    for p in sorted(VERDICT_CACHE.glob("*.json")):
        try:
            n = len(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            n = 0
        print(f"  {p.name}  {n} verdicts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_caches()
        return

    client_name = f"{args.provider}:{args.model}"
    cache = load_cache(client_name)
    print(f"cache for {client_name}: {len(cache)} verdicts\n")
    if not cache:
        list_caches()
        print("\nNo verdicts for that model. Use --list to see what exists,")
        print("then pass the matching --model / --provider.")
        return

    gold = load_gold()
    corpus = load_corpus({q.doc_name for q in gold})
    retriever = build_retriever()

    rows, missing = [], 0
    for q in gold:
        chunks = corpus.get(q.doc_name)
        if not chunks:
            continue
        retriever.index(q.doc_name, chunks)
        hits = retriever.search(q.doc_name, q.question,
                                evidence_k(classify(q.question)))
        if not hits:
            continue
        evidence = format_evidence(hits, args.max_chars)
        key = hashlib.md5(f"{PROMPT}||{q.question}||{evidence}".encode()).hexdigest()

        if key not in cache:
            missing += 1
            continue
        d = cache[key]
        answerable, why = answerable_from(q, hits)
        rows.append({
            "q": q, "answerable": answerable, "why": why,
            "verdict": Verdict(d["sufficient"], d.get("reason", ""),
                               d.get("which_excerpt"), cached=True),
            "pages": sorted({c.page_num for c in hits}), "n_chunks": len(hits),
        })

    print(f"questions with a cached verdict : {len(rows)}")
    print(f"questions still unjudged        : {missing}")
    if missing:
        print(f"  (run the sweep again to fill them -- cached ones are skipped)")
    if not rows:
        print("\nNo verdicts matched. The prompt or --max-chars may have changed\n"
              "since those verdicts were written, which invalidates the keys.")
        return

    report(rows, 0.0, client_name + "  [from cache]")


if __name__ == "__main__":
    main()
