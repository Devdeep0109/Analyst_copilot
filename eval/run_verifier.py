"""
eval/run_verifier.py

Day 6: does Verifier #1 actually separate answerable from unanswerable?

Ground truth is mechanical: a question is ANSWERABLE from the retrieved
evidence if and only if a gold page is among the retrieved pages. Retrieval
currently achieves that for ~63% of questions, so ~37% arrive at the answering
stage with no possible correct answer. Those are the -1s the gate exists to
prevent.

The asymmetry that drives everything here:

    missing a winnable question   costs 1 point  (+1 -> 0)
    answering an unanswerable one costs 2 points ( 0 -> -1)

So RECALL ON "INSUFFICIENT" matters roughly twice as much as recall on
"sufficient", and a gate tuned for balanced accuracy is tuned wrong.

Run:
    python eval/run_verifier.py --smoke     # 15 questions, ~30s
    python eval/run_verifier.py             # all 127
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.corpus import load_corpus                    # noqa: E402
from eval.gold import load_gold                        # noqa: E402
from eval.harness import answerable_from               # noqa: E402
from pipeline.classify import classify, evidence_k     # noqa: E402
from pipeline.llm import get_client                    # noqa: E402
from pipeline.verify import Verifier, expected_score   # noqa: E402
from retrieval.hybrid import HybridRetriever           # noqa: E402
from retrieval.rerank import MS_MARCO, RerankRetriever  # noqa: E402


def build_retriever():
    return RerankRetriever(
        model_name=MS_MARCO, depth=20,
        base=HybridRetriever(method="weighted", candidate_pool=50))


def sample(gold, n, seed=0):
    """A RANDOM sample, not the first n.

    gold[:15] looked like a smoke test but was actually "every question about
    3M and Activision" -- load_gold returns questions in file order, which is
    alphabetical by company. That sample shares filings, filing style, and
    even the same balance sheet across questions, so its error rate says
    little about the other 70 companies.

    It also produced a badly skewed class balance: 14 answerable, 1 not. The
    reported 'unanswerable correctly caught: 1.000' rested on a single example.
    """
    import random
    qs = list(gold)
    random.Random(seed).shuffle(qs)
    return qs[:n]


def run(gold, corpus, verifier, retriever, limit=None):
    rows = []
    qs = sample(gold, limit) if limit else gold
    t0 = time.time()
    for i, q in enumerate(qs, 1):
        chunks = corpus.get(q.doc_name)
        if not chunks:
            continue
        retriever.index(q.doc_name, chunks)
        k = evidence_k(classify(q.question))
        hits = retriever.search(q.doc_name, q.question, k)
        pages = {c.page_num for c in hits}
        # NOT `pages & q.gold_pages`. See answerable_from() -- the strict page
        # test called 22% of genuinely answerable questions unanswerable, and
        # graded the verifier as wrong when it was right.
        answerable, why = answerable_from(q, hits)

        v = verifier.judge(q.question, hits)
        rows.append({"q": q, "answerable": answerable, "why": why, "verdict": v,
                     "pages": sorted(pages), "n_chunks": len(hits)})
        if i % 5 == 0:
            done = time.time() - t0
            err = f", {verifier.n_errors} err" if verifier.n_errors else ""
            per = done / max(1, verifier.n_calls)
            print(f"    {i}/{len(qs)}  {done:.0f}s  "
                  f"({verifier.n_calls} calls, {per:.1f}s each{err})"
                  f"  -> full sweep ~{127*per/60:.0f} min")
        # Save periodically: a sweep interrupted at question 100 should not
        # throw away 100 verdicts.
        if i % 25 == 0:
            verifier.save()
    verifier.save()
    return rows, time.time() - t0


def report(rows, elapsed, model_name):
    # CRITICAL: questions the model never actually judged (API errors, rate
    # limits, unparseable output) must be EXCLUDED from the confusion matrix.
    #
    # The first full sweep counted them as "insufficient", which is right for
    # the runtime pipeline -- an error should abstain, not guess. But it is
    # wrong for MEASUREMENT: 65 of 127 questions were rate-limit failures, and
    # scoring them as verdicts made the verifier look like it rejected
    # everything. "answerable correctly kept = 0.296" was mostly HTTP 429.
    failed = [r for r in rows if r["verdict"].parse_failed]
    judged = [r for r in rows if not r["verdict"].parse_failed]

    tp = sum(1 for r in judged if r["answerable"] and r["verdict"].sufficient)
    fn = sum(1 for r in judged if r["answerable"] and not r["verdict"].sufficient)
    fp = sum(1 for r in judged if not r["answerable"] and r["verdict"].sufficient)
    tn = sum(1 for r in judged if not r["answerable"] and not r["verdict"].sufficient)
    n = len(judged)
    if n == 0:
        print("\n  NO QUESTIONS WERE ACTUALLY JUDGED -- every call failed.")
        print(f"  {len(failed)} failures. Check rate limits / API key.")
        return
    parse_fail = len(failed)
    print("\n" + "=" * 78)
    print(f"COVERAGE: {n}/{len(rows)} questions judged, "
          f"{parse_fail} failed (excluded from the matrix below)")
    if parse_fail:
        from collections import Counter
        kinds = Counter("rate limit (429)" if "429" in r["verdict"].reason
                        else "api error" if "llm error" in r["verdict"].reason
                        else "unparseable" for r in failed)
        print(f"  failure types: {dict(kinds)}")
        if parse_fail > 0.2 * len(rows):
            print("  WARNING: >20% of questions never got a verdict. "
                  "These numbers are provisional.")

    print("\n" + "=" * 78)
    print(f"VERIFIER #1 -- {model_name}   ({n} questions, {elapsed:.0f}s)")
    print("=" * 78)
    from collections import Counter
    why = Counter(r["why"] for r in rows if r["answerable"])
    print(f"  evidence present : {tp+fn}/{n} ({(tp+fn)/n:.1%})   <- retrieval ceiling")
    print(f"     via {dict(why)}")
    print(f"  evidence absent  : {fp+tn}/{n} ({(fp+tn)/n:.1%})   <- must be abstained")
    if parse_fail:
        print(f"  unparseable output: {parse_fail} (counted as insufficient)")

    print(f"\n  {'':<26}{'verifier: SUFFICIENT':>22}{'INSUFFICIENT':>16}")
    print(f"  {'evidence present':<26}{tp:>22}{fn:>16}")
    print(f"  {'evidence absent':<26}{fp:>22}{tn:>16}")

    acc = (tp + tn) / n
    caught = tn / (fp + tn) if fp + tn else 0
    kept = tp / (tp + fn) if tp + fn else 0
    print(f"\n  accuracy                    : {acc:.3f}")
    print(f"  unanswerable correctly caught: {caught:.3f}  "
          f"<- the -1 defence  (n={fp+tn})")
    print(f"  answerable correctly kept    : {kept:.3f}  "
          f"<- opportunity retained (n={tp+fn})")
    if fp + tn < 20:
        print(f"\n  CAUTION: only {fp+tn} unanswerable questions in this sample. "
              f"That rate\n  is not measurable at this n -- run the full set "
              f"before trusting it.")

    print("\n" + "=" * 78)
    print("EXPECTED RUBRIC SCORE (the number that actually matters)")
    print("=" * 78)
    print("  assuming attempts with good evidence are answered correctly at rate a:\n")
    print(f"  {'a':>6}{'no gate':>12}{'with gate':>12}{'gain':>10}")
    for a in (0.5, 0.7, 0.9, 1.0):
        no_gate = (tp + fn) * a - (fp + tn)      # answer everything
        with_gate = expected_score(tp, fn, fp, tn, a)
        print(f"  {a:>6.1f}{no_gate:>12.1f}{with_gate:>12.1f}{with_gate-no_gate:>+10.1f}")
    print("\n  'no gate' = answer every question. Negative totals there are not a\n"
          "  bug: with 37% of questions unanswerable and -1 apiece, answering\n"
          "  everything is worse than answering nothing.")

    print("\n" + "=" * 78)
    print("SAMPLE ERRORS")
    print("=" * 78)
    bad = [r for r in rows if not r["answerable"] and r["verdict"].sufficient]
    print(f"  FALSE CONFIDENCE ({len(bad)}) -- would produce -1:")
    for r in bad[:4]:
        print(f"    Q: {r['q'].question[:88]}")
        print(f"       gold={sorted(r['q'].gold_pages)} got={r['pages'][:6]}")
        print(f"       verifier said: {r['verdict'].reason[:90]}")
    missed = [r for r in rows if r["answerable"] and not r["verdict"].sufficient]
    print(f"\n  OVER-CAUTIOUS ({len(missed)}) -- gave up a winnable point:")
    for r in missed[:4]:
        print(f"    Q: {r['q'].question[:88]}")
        print(f"       gold={sorted(r['q'].gold_pages)} got={r['pages'][:6]}")
        print(f"       verifier said: {r['verdict'].reason[:90]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="15 questions")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-chars", type=int, default=1000,
                    help="chars of each chunk shown to the verifier")
    ap.add_argument("--fast", action="store_true",
                    help="use gpt-oss-20b instead of the 120b model")
    ap.add_argument("--provider", default=None,
                    help="groq | gemini | ollama (inferred from --model if omitted)")
    ap.add_argument("--rate", type=float, default=3.0,
                    help="min seconds between API calls (free tiers limit "
                         "tokens/min, and ~3k-token prompts exhaust it fast)")
    args = ap.parse_args()
    if args.fast and not args.model:
        args.model = "openai/gpt-oss-20b"

    gold = load_gold()
    corpus = load_corpus({q.doc_name for q in gold})
    # Infer the provider from the model name. Passing --model gemini-flash-latest
    # while GROQ_API_KEY is set would otherwise ask Groq for a Gemini model and
    # fail with a confusing 404.
    provider = args.provider
    if provider is None and args.model:
        m = args.model.lower()
        if m.startswith("gemini"):
            provider = "gemini"
        elif m.startswith(("openai/", "groq/", "qwen/", "llama", "meta-llama/")):
            provider = "groq"
        elif ":" in m:
            provider = "ollama"
    client = get_client(provider=provider, model=args.model)
    if hasattr(client, "min_interval_s"):
        client.min_interval_s = args.rate
    # Print the RESOLVED provider and model, not what was requested. Auto-
    # selection and 404-redirects can both change the model underneath, and a
    # result attributed to the wrong model is worse than no result.
    print("=" * 78)
    print(f"  PROVIDER : {type(client).__name__.replace('Client', '').upper()}")
    print(f"  MODEL    : {getattr(client, 'model', '?')}")
    print(f"  client   : {client.name}")
    print(f"  evidence : {args.max_chars} chars/chunk")
    print("=" * 78)
    print(f"gold : {len(gold)} questions\n")

    verifier = Verifier(client=client, use_cache=not args.no_cache,
                        max_chars=args.max_chars)
    retriever = build_retriever()
    limit = 15 if args.smoke else None

    print("running ...")
    rows, elapsed = run(gold, corpus, verifier, retriever, limit=limit)
    report(rows, elapsed, client.name)

    if args.smoke:
        per = elapsed / max(1, verifier.n_calls)
        print(f"\n  {verifier.n_calls} live calls, {per:.1f}s each")
        print(f"  full 127-question sweep ~= {127*per/60:.1f} min")


if __name__ == "__main__":
    main()
