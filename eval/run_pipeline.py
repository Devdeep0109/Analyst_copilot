"""
eval/run_pipeline.py

Day 7: the whole pipeline, scored on the real rubric.

    retrieve -> [Verifier #1 gate] -> answer -> [Verifier #2 citation check]
             -> [arithmetic check] -> score

KEY DESIGN: ANSWER EVERYTHING, DECIDE AFTERWARDS
------------------------------------------------
The obvious implementation asks Verifier #1 first and skips the answering call
when it says no. That would make the gate's cost invisible -- we would never
learn what those skipped questions would have scored, and could not compare
"with gate" against "without gate" on the same data.

So this answers EVERY question, records the outcome, and applies the gate in
post-processing. One pass of API calls measures all four configurations:

    A  answer everything, no checks          (the naive baseline)
    B  Verifier #1 gate only
    C  Verifier #2 citation check only
    D  both

That also means a prompt tweak to Verifier #1 costs zero new answering calls.

RESUMABLE BY DESIGN
-------------------
Free-tier rate limits made 50-minute sweeps collapse at question 70. Answers
are cached per (model, question, evidence), so `--limit 25` can be run
repeatedly and each run picks up where the last stopped. `--report` prints
results from cache with no API calls at all.

Run:
    python eval/run_pipeline.py --limit 25       # a batch
    python eval/run_pipeline.py --report         # results so far, no API calls
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
from eval.harness import answerable_from                  # noqa: E402
from eval.scoring import Outcome, answers_match           # noqa: E402
from pipeline.answer import Answerer                      # noqa: E402
from pipeline.classify import classify, evidence_k        # noqa: E402
from pipeline.llm import get_client                       # noqa: E402
from pipeline.verify import Verifier, format_evidence     # noqa: E402
from pipeline.verify2 import check_arithmetic, check_citation  # noqa: E402
from retrieval.hybrid import HybridRetriever              # noqa: E402
from retrieval.rerank import MS_MARCO, RerankRetriever    # noqa: E402


def build_retriever():
    return RerankRetriever(model_name=MS_MARCO, depth=20,
                           base=HybridRetriever(method="weighted", candidate_pool=50))


def collect(gold, corpus, answerer, verifier, retriever, limit=None,
            report_only=False, gate_reasoning_only=False):
    """Run (or read from cache) one record per question."""
    rows = []
    t0 = time.time()
    new_calls = 0
    consecutive_errors = [0]      # list so the loop body can mutate it
    if not report_only:
        print("  loading retriever (first run loads the reranker model, "
              "~90 MB) ...", flush=True)

    for i, q in enumerate(gold, 1):
        chunks = corpus.get(q.doc_name)
        if not chunks:
            continue
        retriever.index(q.doc_name, chunks)
        hits = retriever.search(q.doc_name, q.question,
                                evidence_k(classify(q.question)))
        if not hits:
            continue

        answerable, why = answerable_from(q, hits)

        # Cache-only probe first, so --report and --limit never call the API
        # for something already answered.
        answerer._load()
        key = answerer._key(q.question,
                            format_evidence(hits, answerer.max_chars,
                                            total_budget=answerer.total_budget))
        have = key in answerer._cache

        if not have and (report_only or (limit and new_calls >= limit)):
            continue
        if not have:
            new_calls += 1

        # Print BEFORE the call, not after. A progress line that only appears
        # once work completes is useless during the slow part -- the first run
        # sat silent for minutes while the reranker model loaded and the first
        # answers came back, and looked hung.
        if not have and not report_only:
            qt0 = time.time()
            print(f"    [{new_calls}/{limit or '?'}] {q.doc_name[:28]:30s} "
                  f"{q.question[:46]}...", end="", flush=True)

        ans = answerer.answer(q.question, hits)
        # Verifier #1 doubles the token cost of a sweep -- it sends the SAME
        # evidence again, just with a different prompt. Measured at ~519k input
        # tokens each, so running both is ~1.04M tokens per 127 questions,
        # which is what exhausts a free-tier daily quota around question 70.
        #
        # And it has never yet earned that cost: configs C and D scored
        # identically, as did E and F -- meaning Verifier #2 alone did all the
        # work in every comparison so far. Skipping it is the single biggest
        # throughput win available, so it is a flag rather than a fixed cost.
        # VERIFIER #1 RUNS ON EVERY QUESTION (default).
        #
        # A previous version restricted it to reasoning questions, arguing that
        # since answering is worth +0.33 at a 66.7% hit rate, a gate only pays
        # where it can find cases below the 50% break-even -- and that it was
        # not discriminating, because 79% of its rejections had evidence
        # present versus an 84% base rate.
        #
        # THAT ARGUMENT USED THE WRONG METRIC. "Evidence was present" is not
        # the test; "would this question have scored positively" is. Measured
        # on 88 questions, turning the gate on moved:
        #     wrong answers  13 -> 8   (removed 5)
        #     correct        14 -> 14  (cost 0)
        # It blocked five losses and zero wins. Evidence being present does not
        # mean the model would have used it correctly, and the gate was picking
        # up on that.
        #
        # Reverted to gating everything -- the configuration we have direct
        # evidence for. `--gate-reasoning-only` keeps the alternative testable
        # if there is ever quota to measure it properly.
        if verifier is None:
            from pipeline.verify import Verdict
            v1 = Verdict(True, "verifier #1 disabled")
        elif gate_reasoning_only and not classify(q.question).is_reasoning:
            from pipeline.verify import Verdict
            v1 = Verdict(True, "extractive -- gate not applied")
        else:
            v1 = verifier.judge(q.question, hits)
        cite = check_citation(ans.quote, ans.page, hits)
        arith = check_arithmetic(str(ans.answer or ""), ans.working)

        if not have and not report_only:
            if ans.failed:
                # Show the actual error. A bare "ERR" tells you nothing about
                # whether to fix your key, wait for a rate limit, or debug a
                # prompt -- and those need completely different responses.
                flag = f"ERR {str(ans.quote)[:90]}"
            elif ans.abstained:
                flag = "abstain"
            elif cite.passed:
                flag = "cite ok"
            else:
                flag = f"cite REJECTED ({cite.reason[:40]})"
            print(f"  {time.time()-qt0:5.1f}s  {flag}", flush=True)

        # Bail out early if every call is failing. Grinding through 25
        # questions against a dead API wastes minutes and produces nothing.
        if ans.failed:
            consecutive_errors[0] += 1
            if consecutive_errors[0] >= 3:
                print(f"\n  ABORTING: {consecutive_errors[0]} consecutive "
                      f"failures. Last error:\n    {ans.quote}\n")
                print("  Common causes:")
                print("    401 -> the API key in THIS shell is wrong: "
                      "echo $env:GROQ_API_KEY")
                print("    429 -> rate limited; retry later or use --rate 6")
                break
        else:
            consecutive_errors[0] = 0

        rows.append({"q": q, "hits": hits, "answerable": answerable, "why": why,
                     "ans": ans, "v1": v1, "cite": cite, "arith": arith})

        if new_calls and new_calls % 5 == 0 and not report_only:
            answerer.save()
            if verifier is not None:
                verifier.save()

    answerer.save()
    if verifier is not None:
        verifier.save()
    return rows, new_calls, time.time() - t0


def score_row(r, use_gate: bool, use_cite: bool, use_arith: bool) -> Outcome:
    """Apply one configuration's decision rules and grade the result."""
    ans = r["ans"]

    if use_gate and not r["v1"].sufficient:
        return Outcome.ABSTAINED
    if ans.abstained:
        return Outcome.ABSTAINED
    if use_cite and not r["cite"].passed:
        return Outcome.ABSTAINED
    if use_arith and r["arith"].applicable and not r["arith"].passed:
        return Outcome.ABSTAINED

    correct = answers_match(str(ans.answer), r["q"].answer)
    if correct is None:
        return Outcome.UNJUDGED
    if not correct:
        return Outcome.CONFIDENTLY_WRONG

    # Correct answer -- is the citation on a gold page?
    if ans.page is not None and ans.page in r["q"].gold_pages:
        return Outcome.CORRECT_LOCATED
    # Cited a page that genuinely contains the evidence, just not the labelled
    # one. FinanceBench labels ONE page; filings repeat facts. Verified
    # citation on a retrieved page still earns the located score.
    if r["cite"].passed and r["why"] in ("evidence", "answer-value"):
        return Outcome.CORRECT_LOCATED
    return Outcome.CORRECT_WRONG_PAGE


def report(rows):
    from collections import Counter

    n = len(rows)
    print("\n" + "=" * 78)
    print(f"END-TO-END PIPELINE  ({n} questions)")
    print("=" * 78)

    ansble = sum(r["answerable"] for r in rows)
    print(f"  evidence present : {ansble}/{n} ({ansble/n:.1%})")
    abst = sum(1 for r in rows if r["ans"].abstained)
    print(f"  model abstained  : {abst}/{n} ({abst/n:.1%})")
    failed = sum(1 for r in rows if r["ans"].failed)
    if failed:
        print(f"  answering errors : {failed} (excluded from scoring)")

    # ---- the conversion rate everything else depends on -------------------
    good = [r for r in rows if r["answerable"] and not r["ans"].abstained
            and not r["ans"].failed]
    judged = [r for r in good
              if answers_match(str(r["ans"].answer), r["q"].answer) is not None]
    correct = sum(1 for r in judged
                  if answers_match(str(r["ans"].answer), r["q"].answer))
    a = correct / len(judged) if judged else 0.0
    print(f"\n  CONVERSION RATE a = {a:.3f}")
    print(f"    of {len(judged)} attempts with good evidence, {correct} were correct")
    print("    (this is the number the Verifier #1 trade-off depends on)")

    # ---- citation verification -------------------------------------------
    attempted = [r for r in rows if not r["ans"].abstained and not r["ans"].failed]
    passed = sum(1 for r in attempted if r["cite"].passed)
    print(f"\n  CITATIONS: {passed}/{len(attempted)} verified "
          f"({passed/max(1,len(attempted)):.1%})")
    reasons = Counter(r["cite"].reason.split("(")[0].strip()[:44]
                      for r in attempted if not r["cite"].passed)
    for why, c in reasons.most_common(5):
        print(f"    rejected: {c:3d}  {why}")

    # ---- configurations ---------------------------------------------------
    print("\n" + "=" * 78)
    print("CONFIGURATIONS -- total rubric score")
    print("=" * 78)
    # The arithmetic check is measured but NOT recommended. On 88 questions it
    # removed 2 wrong answers (+2) and killed 3 correct ones (-3): net -1.5.
    # The idea is sound -- a 360x scale error like -0.02 vs -7.23 is only
    # catchable this way -- but the implementation parses simple `a op b`
    # expressions and rejects correct answers whose working it cannot read.
    # Kept in the table so the finding stays visible rather than being quietly
    # dropped, but it should stay off until it can parse reliably.
    configs = [
        ("A  answer everything", False, False, False),
        ("B  + Verifier #1 (gate)", True, False, False),
        ("C  + Verifier #2 (citation)", False, True, False),
        ("D  + both", True, True, False),
        ("E  + both + arithmetic  [not recommended]", True, True, True),
        ("F  Verifier #2 + arithmetic  [not recommended]", False, True, True),
    ]
    print(f"  {'config':<30}{'score':>8}{'+1':>6}{'0':>6}{'-1':>6}{'?':>5}")
    best = None
    for name, g, c, ar in configs:
        outs = [score_row(r, g, c, ar) for r in rows]
        total = sum(o.score for o in outs)
        cnt = Counter(o for o in outs)
        loc = cnt[Outcome.CORRECT_LOCATED]
        part = cnt[Outcome.CORRECT_WRONG_PAGE]
        zero = cnt[Outcome.ABSTAINED]
        wrong = cnt[Outcome.CONFIDENTLY_WRONG]
        unj = cnt[Outcome.UNJUDGED]
        print(f"  {name:<30}{total:>8.1f}{loc:>6}{zero:>6}{wrong:>6}{unj:>5}"
              f"   (partial {part})")
        if best is None or total > best[1]:
            best = (name, total)
    print(f"\n  BEST: {best[0]}  ->  {best[1]:+.1f}")
    print("\n  '?' = prose gold answers the mechanical comparator cannot judge;")
    print("  they score 0 here and need the Day-7 LLM judge to resolve.")

    # ---- what the -1s look like ------------------------------------------
    print("\n" + "=" * 78)
    print("CONFIDENTLY WRONG under the best config -- the expensive failures")
    print("=" * 78)
    bad = [r for r in rows
           if score_row(r, True, True, False) is Outcome.CONFIDENTLY_WRONG]
    for r in bad[:6]:
        print(f"  Q: {r['q'].question[:92]}")
        print(f"     gold={r['q'].answer[:44]!r}  got={str(r['ans'].answer)[:44]!r}")
        print(f"     cited page {r['ans'].page} (gold {sorted(r['q'].gold_pages)}), "
              f"citation {r['cite'].label}")
    if not bad:
        print("  none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="max NEW answering calls this run (batching)")
    ap.add_argument("--report", action="store_true",
                    help="report from cache only, no API calls")
    ap.add_argument("--model", default=None,
                    help="default depends on --provider")
    ap.add_argument("--provider", default="groq", choices=["groq", "gemini", "ollama"])
    ap.add_argument("--rate", type=float, default=3.0)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--no-verifier1", action="store_true",
                    help="skip the sufficiency gate -- halves token use, and "
                         "it has not yet earned its cost in any config")
    ap.add_argument("--gate-reasoning-only", action="store_true",
                    help="apply Verifier #1 only to reasoning questions "
                         "(untested hypothesis; default gates everything, "
                         "which is what the data supports)")
    ap.add_argument("--budget", type=int, default=None,
                    help="total evidence chars per call (hard cap). "
                         "4400 ~= 1500 tokens/call ~= 53 questions on an 80k "
                         "daily quota")
    args = ap.parse_args()

    # Each provider needs its own default model. Passing Groq's model name to
    # Gemini gives a confusing 404, and the whole point of switching providers
    # mid-project is that it should be one flag.
    if args.model is None:
        args.model = {"groq": "openai/gpt-oss-20b",
                      "gemini": "gemini-flash-latest",
                      "ollama": "mistral:latest"}[args.provider]

    # Gemini's free tier is slower but has a separate quota from Groq. When
    # Groq's daily cap is gone, this is the way to keep measuring. Caches are
    # keyed per model, so results from both providers coexist and stay
    # comparable rather than overwriting each other.
    if args.provider == "gemini" and args.rate == 3.0:
        args.rate = 4.0

    gold = load_gold()
    corpus = load_corpus({q.doc_name for q in gold})
    if args.report:
        # --report reads cached results only. Requiring an API key for it was
        # a real obstacle: the moment you most want to see results is when the
        # quota is gone. This stub carries only the name used for cache lookup.
        class _CacheOnly:
            name = f"{args.provider}:{args.model}"
            model = args.model

            def complete_json(self, *a, **k):
                raise RuntimeError("--report makes no API calls")
        client = _CacheOnly()
    else:
        client = get_client(provider=args.provider, model=args.model)
        if hasattr(client, "min_interval_s"):
            client.min_interval_s = args.rate

    print("=" * 78)
    print(f"  MODEL : {client.name}")
    print(f"  MODE  : {'report only (no API calls)' if args.report else f'batch of {args.limit or 127}'}")
    print("=" * 78)

    answerer = Answerer(client=client, max_chars=args.max_chars,
                        total_budget=args.budget)
    verifier = (None if args.no_verifier1
                else Verifier(client=client, max_chars=args.max_chars))
    if verifier is None:
        print("  Verifier #1 SKIPPED -- ~half the tokens, configs B/D/E "
              "will read as identical to A/C/F")
    rows, new_calls, el = collect(gold, corpus, answerer, verifier,
                                  build_retriever(), limit=args.limit,
                                  report_only=args.report,
                                  gate_reasoning_only=args.gate_reasoning_only)

    print(f"\n  {len(rows)} questions with results, {new_calls} new calls, {el:.0f}s")
    remaining = len(gold) - len(rows)
    if remaining > 0:
        print(f"  {remaining} still to do -- run again with --limit to continue")
    if not rows:
        print("\n  Nothing cached yet. Run without --report first.")
        return
    report(rows)


if __name__ == "__main__":
    main()
