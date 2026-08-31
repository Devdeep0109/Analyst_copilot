"""
pipeline/answer.py

The answering stage. Produces an answer, an EXACT quote, and a page number.

WHY THE QUOTE IS MANDATORY
--------------------------
Verifier #2 is a plain substring check: does the quoted text actually exist on
the cited page? That check is only possible if the model is forced to quote
VERBATIM rather than paraphrase. So the quote is not decoration for the user --
it is the hook that makes mechanical verification possible at all.

This is the difference between "the model says page 60" and "we confirmed the
text it cited is really on page 60". The first is a claim; the second is a fact
we can check in code, with no second LLM and no judgement involved.

WHAT THIS STAGE MUST NOT DO
---------------------------
It must not use its own knowledge of the company. A model that knows 3M's
revenue can produce a correct-looking answer with a fabricated citation -- which
under this rubric scores worse than declining, because it looks trustworthy
while being unverifiable. The prompt says this explicitly and Verifier #2
enforces it.

ABSTAINING IS A FIRST-CLASS OUTPUT
----------------------------------
The model can return {"answer": null} when the excerpts do not support an
answer. That is worth 0, versus -1 for a confident guess. Making abstention an
explicit, easy option in the output schema matters more than telling the model
to be careful -- an escape hatch it can actually take beats an instruction.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parser.chunker import Chunk           # noqa: E402
from pipeline.llm import get_client        # noqa: E402
from pipeline.verify import format_evidence  # noqa: E402

ANSWER_CACHE = PROJECT_ROOT / ".cache" / "answers"

SYSTEM = (
    "You are a financial analyst assistant. You answer strictly from the "
    "excerpts you are given, and you always quote the exact text that proves "
    "your answer. You never rely on outside knowledge."
)

# COMPACT PROMPT. The verbose version was 726 tokens of fixed overhead --
# ~45% of the entire per-call budget on an 80k/day quota. Every instruction
# below earned its place by fixing an observed failure; nothing is here for
# completeness. Kept: verbatim quoting, PAGE-not-excerpt-number, synonyms,
# calculate-don't-decline, show working, narrow abstain rule.
PROMPT = """\
Answer using ONLY these excerpts from one SEC filing.

Q: {question}

{evidence}

Rules:
1. Only these excerpts. Ignore what you know about the company.
2. "quote" must be copied CHARACTER-FOR-CHARACTER from one excerpt. It is
   checked automatically. Do not reword or reformat.
3. "page" = the number in the `--- PAGE n ---` header above the text you quoted.
   Copy it exactly. It is usually two or three digits.
4. Synonyms: "Net sales"=revenue. "Purchases of property, plant and equipment
   (PP&E)"=capex. "Property, plant and equipment - net"=net PP&E/PPNE. Table
   rows use | separators, latest fiscal year first.
5. CALCULATE when asked. Ratios, growth, averages, margins across two or three
   excerpts are normal work, not a reason to decline.
6. Put the actual expression in "working", e.g. "177866/135987-1 = 0.308 =
   30.8%". Re-check that "answer" equals what it evaluates to. Watch the scale:
   a ratio near 0.01 is not 7.2 -- confirm the magnitude is plausible.
7. FIND A QUOTE BEFORE YOU DECIDE. Search every excerpt for the most relevant
   line, write it into "quote", and only then judge whether you can answer.
   If you have a relevant quote, you have evidence -- use it.
   answer=null is for one case only: you searched and found NO relevant quote.
   It is NOT for:
     - the calculation takes several steps
     - the filing words it differently from the question
     - you would prefer more surrounding context
     - you are merely unsure
   An answer you flag as approximate is more useful than a refusal.
8. YES/NO QUESTIONS ("Does X have...", "Is X...", "Did X report...", "Has X...")
   can almost always be decided from the figures present. Answer yes or no, and
   quote the line you based it on.
9. Use the units the question asks for.

FILL THE FIELDS IN THIS ORDER. Find the evidence BEFORE deciding whether you can
answer -- write "quote" and "page" first, then "working", and only then "answer".

JSON only, keys in this exact order:
{{"quote":"<verbatim text from an excerpt>","page":<the PAGE number>,"working":"<expression, or empty>","answer":"<answer or null>"}}
"""


# SECOND PASS for questions the model declined.
#
# Four prompt rewrites failed to move the abstention rate (38.6% -> 28.6%, and
# the last three changes moved it by ~1 point between them). But 26 of the 36
# declines HAD the evidence in front of them, and at a conversion rate of 0.685
# each is worth about +0.37 in expectation -- roughly +10 points left on the
# table.
#
# So instead of arguing with the model in the first prompt, ask again with the
# escape hatch removed. There is no `null` in this schema. The safety net is
# unchanged: Verifier #1 and Verifier #2 still gate whatever comes back, and a
# forced answer that cannot be cited is caught exactly as any other would be.
RETRY_PROMPT = """\
You previously declined this question. The excerpts below DO contain relevant
figures. Give your best answer using them.

Q: {question}

{evidence}

Rules:
1. You MUST answer. Estimate from the figures present rather than declining.
2. "quote" = text copied CHARACTER-FOR-CHARACTER from one excerpt.
3. "page" = the number in the `--- PAGE n ---` header above the text you quoted.
4. Synonyms: "Net sales"=revenue. "Purchases of property, plant and equipment
   (PP&E)"=capex. "Property, plant and equipment - net"=net PP&E/PPNE.
5. If the exact figure is missing, use the closest relevant one and say what you
   assumed in "working". A stated approximation beats a refusal.
6. For yes/no questions, decide yes or no from the figures and cite the line.

JSON only:
{{"quote":"<verbatim>","page":<the PAGE number>,"working":"<expression or assumption>","answer":"<your best answer>"}}
"""


import re as _re

# A refusal announced at the START of the answer. Anchored on purpose -- see
# Answer.abstained. "Not meaningful", "Insufficient data", "Cannot be
# calculated" are declines; a sentence that mentions "not available" halfway
# through is usually a real answer with a caveat.
_REFUSAL_RE = _re.compile(
    r"^\s*(insufficient\b"
    r"|not\s+(calculable|determinable|available|provided|disclosed|possible|"
    r"enough)\b"
    r"|cannot\s+be\s+(calculated|determined|computed|answered)\b"
    r"|unable\s+to\b"
    r"|no\s+(data|information|figures?)\s+(is\s+)?(available|provided|given)\b"
    r"|the\s+(filing|excerpts?|document)\s+does\s+not\s+"
    r"(provide|contain|disclose|state)\b"
    r")",
    _re.I,
)


@dataclass
class Answer:
    answer: str | None
    quote: str = ""
    page: int | None = None
    working: str = ""
    raw: str = ""
    cached: bool = False
    failed: bool = False
    retried: bool = False   # produced by the second, forced pass

    @property
    def abstained(self) -> bool:
        """Did the model decline -- in ANY form?

        It does not only return null. It also refuses in prose:
            "Insufficient data to calculate the FY2019 cash conversion cycle"
            "Not calculable - the filing does not provide cost of goods sold"
            "Conventional inventory turnover cannot be calculated from ..."
        Those are honest refusals worth 0, but they were being scored as
        CONFIDENTLY WRONG (-1) because the field was non-empty. Each one cost a
        point it should not have.

        The pattern is deliberately anchored to the START of the answer. A
        refusal announces itself up front; a real answer that happens to
        mention "not available" later ("Revenue was $5m; the segment split is
        not available") is a genuine answer and must not be caught.
        """
        raw = str(self.answer or "").strip()
        if not raw or raw.lower() in {"null", "none", "n/a", "not found",
                                      "not available", "unknown"}:
            return True
        return bool(_REFUSAL_RE.match(raw))


class Answerer:
    def __init__(self, client=None, max_chars: int = 1000,
                 use_cache: bool = True, total_budget: int | None = None,
                 retry_on_abstain: bool = False):
        self.retry_on_abstain = retry_on_abstain
        self.client = client or get_client()
        self.max_chars = max_chars
        self.total_budget = total_budget
        self.use_cache = use_cache
        self._cache: dict | None = None
        self._dirty = False
        self.n_calls = 0
        self.n_errors = 0

    def _cache_path(self) -> Path:
        tag = hashlib.md5(getattr(self.client, "name", "?").encode()).hexdigest()[:8]
        return ANSWER_CACHE / f"answers__{tag}.json"

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
        ANSWER_CACHE.mkdir(parents=True, exist_ok=True)
        self._cache_path().write_text(json.dumps(self._cache), encoding="utf-8")
        self._dirty = False

    def _key(self, question: str, evidence: str, prompt: str = PROMPT) -> str:
        return hashlib.md5(f"{prompt}||{question}||{evidence}".encode()).hexdigest()

    def _ask(self, prompt_template: str, question: str, evidence: str,
             retried: bool) -> Answer:
        """One LLM call against a given prompt template."""
        try:
            resp = self.client.complete_json(
                prompt_template.format(question=question, evidence=evidence),
                system=SYSTEM, max_tokens=900)
        except Exception as e:
            self.n_errors += 1
            return Answer(None, f"llm error: {str(e)[:120]}", failed=True,
                          retried=retried)

        self.n_calls += 1
        data = resp.json()
        if data is None:
            self.n_errors += 1
            return Answer(None, "unparseable answer", raw=resp.text,
                          failed=True, retried=retried)

        page = data.get("page")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None

        return Answer(
            answer=data.get("answer"),
            quote=str(data.get("quote", ""))[:1500],
            page=page,
            working=str(data.get("working", ""))[:500],
            raw=resp.text,
            retried=retried,
        )

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        if not chunks:
            return Answer(None, "no evidence retrieved")

        evidence = format_evidence(chunks, self.max_chars,
                                   total_budget=self.total_budget)
        key = self._key(question, evidence)

        a = None
        if self.use_cache:
            self._load()
            if key in self._cache:
                d = self._cache[key]
                a = Answer(d.get("answer"), d.get("quote", ""), d.get("page"),
                           d.get("working", ""), cached=True,
                           retried=d.get("retried", False))
                # A cached ABSTAIN is still eligible for the retry pass, so a
                # first run and a later --retry run compose rather than the
                # cache locking in the refusal.
                if not (self.retry_on_abstain and a.abstained):
                    return a

        # Only pay for the first call if we have not already got one cached.
        # Without this guard, --retry-abstain re-ran the FIRST prompt for every
        # cached abstention as well as the retry -- two calls where one was
        # needed, and 36 of them silently.
        if a is None:
            a = self._ask(PROMPT, question, evidence, retried=False)

        # SECOND PASS. Only fires on a genuine refusal -- never on an API error
        # (retrying a dead quota just burns two calls instead of one).
        if self.retry_on_abstain and a.abstained and not a.failed:
            rkey = self._key(question, evidence, RETRY_PROMPT)
            if self.use_cache and rkey in self._cache:
                d = self._cache[rkey]
                a = Answer(d.get("answer"), d.get("quote", ""), d.get("page"),
                           d.get("working", ""), cached=True, retried=True)
            else:
                forced = self._ask(RETRY_PROMPT, question, evidence, retried=True)
                if not forced.failed and not forced.abstained:
                    if self.use_cache:
                        self._cache[rkey] = {
                            "answer": forced.answer, "quote": forced.quote,
                            "page": forced.page, "working": forced.working,
                            "retried": True}
                        self._dirty = True
                    a = forced

        if self.use_cache and not a.retried:
            self._cache[key] = {"answer": a.answer, "quote": a.quote,
                                "page": a.page, "working": a.working,
                                "retried": False}
            self._dirty = True
        return a
