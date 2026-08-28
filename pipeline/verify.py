"""
pipeline/verify.py

Verifier #1 -- the sufficiency gate. Decides whether to attempt an answer at all.

WHY THIS IS THE HIGHEST-LEVERAGE COMPONENT IN THE SYSTEM
--------------------------------------------------------
Retrieval puts a gold page in the top-K for 63% of questions. For the other
37%, the evidence needed simply is not in front of the model. Under the rubric:

    attempt without the evidence  ->  confidently wrong  ->  -1
    abstain                       ->  "not found"        ->   0

So the 37% is not a missed opportunity, it is a LIABILITY. A system that
answers everything scores worse than one that answers nothing. Verifier #1's
job is to tell those two populations apart.

HOW IT IS EVALUATED (this is the important part)
------------------------------------------------
The ground truth is not "did a human think this was answerable" -- it is
mechanical and exact: IS A GOLD PAGE PRESENT IN THE RETRIEVED EVIDENCE?

    gold page retrieved + verifier says SUFFICIENT   -> correct: attempt
    gold page retrieved + verifier says INSUFFICIENT -> lost a winnable point
    gold page absent    + verifier says SUFFICIENT   -> the -1 case, the worst
    gold page absent    + verifier says INSUFFICIENT -> correct: saved a point

The two error types are NOT symmetric. Missing a winnable question costs 1
point (+1 becomes 0). Answering an unanswerable one costs 2 (0 becomes -1).
So the verifier should be biased toward abstaining, and the prompt says so
explicitly rather than hoping the model infers it.

DESIGN NOTES
------------
* The verifier never sees the gold answer or gold page -- only the question and
  the retrieved text, exactly as at runtime.
* It is asked to judge SUFFICIENCY, not to answer. Asking a model to answer and
  then inferring confidence from the answer conflates two different failures.
* Verdicts are cached on (model, question, evidence-hash), so revising the
  prompt re-runs only what changed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parser.chunker import Chunk          # noqa: E402
from pipeline.llm import get_client       # noqa: E402

VERDICT_CACHE = PROJECT_ROOT / ".cache" / "verdicts"

SYSTEM = (
    "You are a meticulous financial analyst's assistant. You judge whether a "
    "set of excerpts from an SEC filing is SUFFICIENT to answer a question. "
    "You do not answer the question yourself."
)

PROMPT = """\
A question was asked about an SEC filing. Below are excerpts retrieved from that \
filing. Decide whether these excerpts CONTAIN THE FACTS NEEDED to answer it.

QUESTION:
{question}

RETRIEVED EXCERPTS:
{evidence}

How to judge:

1. FILINGS DO NOT USE THE QUESTION'S WORDING. Match on meaning, not on labels.
   A question about "net PPNE" is answered by "Property, plant and equipment - net".
   "Capital expenditure" appears as "Purchases of property, plant and equipment (PP&E)".
   "Revenue" appears as "Net sales". If the underlying figure is present under any
   reasonable synonym, that is SUFFICIENT.

2. TABLE ROWS COUNT AS EVIDENCE. Financial statements are rendered as pipe-separated
   rows, e.g. "Property, plant and equipment - net | 8,738 | 8,866". A row like that
   fully answers a question about that line item. Columns are usually the most recent
   fiscal year first.

3. FOR CALCULATIONS, check that every input is present somewhere in the excerpts --
   they may be on different pages. The arithmetic itself is not your concern.

4. Use only the excerpts. Do not rely on your own knowledge of this company.

5. Answer INSUFFICIENT when the figure genuinely is not there: only a different
   fiscal year is shown, only narrative discussion with no number, or a needed
   input to a calculation is missing. Do not answer INSUFFICIENT merely because
   the wording differs from the question, or because you would like more context.

Most retrieved sets DO contain the answer. Say INSUFFICIENT when the evidence is
actually absent, not when you are merely unsure.

Respond with JSON only:
{{"sufficient": true or false, "reason": "<one short sentence>"}}
"""


@dataclass
class Verdict:
    sufficient: bool
    reason: str
    which_excerpt: int | None = None
    raw: str = ""
    cached: bool = False
    parse_failed: bool = False


def _relevant_window(text: str, query: str, cap: int) -> str:
    """Keep the part of the chunk that matches the QUERY, not the first N chars.

    Head-truncation assumes the answer is at the top of the chunk. It usually
    isn't: a balance-sheet chunk starts with "Total current assets" and the
    PP&E row the question asks about sits 700 characters in. Measured, a flat
    budget cut visible gold evidence from 69.3% to 55.1% -- and every one of
    those losses was text we HAD retrieved and then threw away.

    This slides a window over the chunk and keeps the position where the most
    query terms appear, so the same number of characters carries far more of
    the answer. Cost is a few string operations; no model, no extra tokens.
    """
    if len(text) <= cap:
        return text
    import re as _re
    terms = {t for t in _re.findall(r"[a-z0-9]{4,}", query.lower())}
    if not terms:
        return text[:cap]

    step = max(64, cap // 4)
    best_start, best_score = 0, -1
    for start in range(0, max(1, len(text) - cap + step), step):
        window = text[start:start + cap].lower()
        score = sum(1 for t in terms if t in window)
        # Numbers are the payload in financial evidence -- weight windows that
        # actually contain figures over ones that are all prose.
        score += 0.5 * len(_re.findall(r"\d[\d,]{2,}", window))
        if score > best_score:
            best_start, best_score = start, score

    out = text[best_start:best_start + cap]
    return out if best_start == 0 else "..." + out


def format_evidence(chunks: list[Chunk], max_chars: int = 1200,
                    total_budget: int | None = None,
                    query: str | None = None) -> str:
    """Render evidence for the model.

    The header format is load-bearing. It used to be "[1] (page 60)", and the
    answering prompt said "cite the page number shown in brackets" -- but the
    BRACKETS hold the excerpt index and the PAGE is in parentheses. The model
    followed the instruction literally and returned 1, 2, 3, 4 as page numbers.
    Verifier #2 then correctly rejected them as pages that were never
    retrieved, which sank the citation pass rate to 34.6% and looked like
    rampant hallucination. It was an ambiguous label.

    "EXCERPT 1 | PAGE 60" removes the ambiguity: the word PAGE sits directly
    against the number we want back.
    """
    # TOTAL BUDGET, allocated by rank.
    #
    # A per-chunk cap does not bound the call: K varies 10-20 by question type,
    # so the same max_chars gives wildly different totals. On a hard token
    # quota that is unusable -- you cannot plan how many questions fit.
    #
    # Chunks arrive in rank order, so the budget is weighted toward the top:
    # rank 1 gets roughly twice the room of rank 10. If the reranker is any
    # good, the early chunks deserve the space, and truncating rank 15 to a
    # stub costs little.
    if total_budget:
        n = len(chunks)
        weights = [1.0 / (0.5 + i * 0.07) for i in range(n)]
        wsum = sum(weights)
        # ~34 chars of header per chunk
        usable = max(200, total_budget - 34 * n)
        caps = [max(120, int(usable * w / wsum)) for w in weights]
    else:
        caps = [max_chars] * len(chunks)

    # NO EXCERPT NUMBERS. Twice now the model has returned the excerpt index
    # where a page number was wanted -- first with "[1] (page 60)", then again
    # with "--- EXCERPT 3 | PAGE 60 ---" after the prompt was compressed and
    # the four-line warning about it shrank to half a line. Citations fell from
    # 74% to 50%, with rejections reading "page 5 / 7 / 8 / 9 not among
    # retrieved pages" in filings whose pages run 50-150.
    #
    # Instructions did not hold. Removing the competing number does: if the
    # header contains exactly one integer, there is nothing to pick wrongly.
    # Structure beats instruction -- the model cannot confuse two things when
    # only one is present.
    parts = []
    for c, cap in zip(chunks, caps):
        text = (_relevant_window(c.text, query, cap) if query
                else c.text[:cap])
        parts.append(f"--- PAGE {c.page_num} ---\n{text}")
    return "\n\n".join(parts)


class Verifier:
    # 1000 is a measured compromise, not a guess. 1200 gave ~5k input tokens
    # per call and 26s per question. 700 halved that -- but the FY2018 net PPNE
    # answer sits at character 688 of its chunk, i.e. 12 characters inside the
    # window. That is not a margin, that is luck. Balance-sheet chunks put the
    # PP&E and equity rows in exactly that region, so 700 was one layout change
    # away from silently hiding the answer.
    def __init__(self, client=None, max_chars: int = 1000, use_cache: bool = True):
        self.client = client or get_client()
        self.max_chars = max_chars
        self.use_cache = use_cache
        self._cache: dict | None = None
        self._dirty = False
        self.n_calls = 0
        self.n_errors = 0

    # ---------------------------------------------------------------- cache --

    def _cache_path(self) -> Path:
        tag = hashlib.md5(getattr(self.client, "name", "?").encode()).hexdigest()[:8]
        return VERDICT_CACHE / f"verdicts__{tag}.json"

    def _load(self):
        if self._cache is not None:
            return
        p = self._cache_path()
        if p.exists():
            try:
                self._cache = json.loads(p.read_text(encoding="utf-8"))
                return
            except json.JSONDecodeError:
                pass
        self._cache = {}

    def save(self):
        if self._cache is None or not self._dirty:
            return
        VERDICT_CACHE.mkdir(parents=True, exist_ok=True)
        self._cache_path().write_text(json.dumps(self._cache), encoding="utf-8")
        self._dirty = False

    def _key(self, question: str, evidence: str) -> str:
        # Prompt text is part of the key: revising the prompt must invalidate
        # old verdicts, or a "fix" would be silently evaluated on stale output.
        blob = f"{PROMPT}||{question}||{evidence}"
        return hashlib.md5(blob.encode()).hexdigest()

    # ---------------------------------------------------------------- judge --

    def judge(self, question: str, chunks: list[Chunk]) -> Verdict:
        if not chunks:
            return Verdict(False, "no evidence retrieved")

        evidence = format_evidence(chunks, self.max_chars)
        key = self._key(question, evidence)

        if self.use_cache:
            self._load()
            if key in self._cache:
                d = self._cache[key]
                return Verdict(d["sufficient"], d.get("reason", ""),
                               d.get("which_excerpt"), cached=True)

        # 600 rather than 200: reasoning models spend part of the budget
        # deliberating before emitting anything, and an exhausted budget looks
        # like a prompt failure rather than what it is.
        try:
            resp = self.client.complete_json(
                PROMPT.format(question=question, evidence=evidence),
                system=SYSTEM, max_tokens=600)
        except Exception as e:
            # A single bad question must not abort a 127-question sweep. Treat
            # an API failure as insufficient -- the safe direction -- record it,
            # and do NOT cache it, so a retry can still resolve it properly.
            self.n_errors += 1
            return Verdict(False, f"llm error: {str(e)[:120]}", parse_failed=True)

        self.n_calls += 1
        data = resp.json()

        if data is None or "sufficient" not in data:
            # Unparseable output must NOT be read as "sufficient". Defaulting
            # to attempt would convert a formatting glitch into a -1.
            v = Verdict(False, "verifier output unparseable", raw=resp.text,
                        parse_failed=True)
        else:
            v = Verdict(bool(data.get("sufficient")),
                        str(data.get("reason", ""))[:200],
                        data.get("which_excerpt"), raw=resp.text)

        if self.use_cache:
            self._cache[key] = {"sufficient": v.sufficient, "reason": v.reason,
                                "which_excerpt": v.which_excerpt}
            self._dirty = True
        return v


def expected_score(tp: int, fn: int, fp: int, tn: int, answer_accuracy: float = 1.0) -> float:
    """Expected rubric points under a given confusion matrix.

    tp = evidence present, attempted     -> +1 * answer_accuracy
    fn = evidence present, abstained     ->  0   (a point left on the table)
    fp = evidence absent,  attempted     -> -1   (the case that must be avoided)
    tn = evidence absent,  abstained     ->  0

    `answer_accuracy` is the share of attempts with good evidence that actually
    produce a correct answer. It is not 1.0 in reality -- Day 7 measures it --
    so this is an optimistic upper bound on the gate's value.
    """
    return tp * answer_accuracy - fp
