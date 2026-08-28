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
7. ANSWER BY DEFAULT. Your best supported answer is far more useful than a
   refusal. Give your best reading of the figures even if you are not certain,
   as long as the numbers are on the page.
   Use answer=null ONLY when a required figure is genuinely absent -- the
   question asks FY2019 and only FY2021 appears, or the line item is nowhere in
   any excerpt. Do NOT return null because:
     - the calculation takes several steps
     - the filing words it differently from the question
     - you would prefer more surrounding context
     - you are merely unsure
   If a figure is on the page and it plausibly answers the question, answer.
8. Use the units the question asks for.

JSON only:
{{"answer":"<answer or null>","quote":"<verbatim>","page":<the PAGE number>,"working":"<expression>"}}
"""


@dataclass
class Answer:
    answer: str | None
    quote: str = ""
    page: int | None = None
    working: str = ""
    raw: str = ""
    cached: bool = False
    failed: bool = False

    @property
    def abstained(self) -> bool:
        return self.answer is None or not str(self.answer).strip() \
            or str(self.answer).strip().lower() in {"null", "none", "n/a", "not found"}


class Answerer:
    def __init__(self, client=None, max_chars: int = 1000,
                 use_cache: bool = True, total_budget: int | None = None):
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

    def _key(self, question: str, evidence: str) -> str:
        return hashlib.md5(f"{PROMPT}||{question}||{evidence}".encode()).hexdigest()

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        if not chunks:
            return Answer(None, "no evidence retrieved")

        evidence = format_evidence(chunks, self.max_chars,
                                   total_budget=self.total_budget)
        key = self._key(question, evidence)

        if self.use_cache:
            self._load()
            if key in self._cache:
                d = self._cache[key]
                return Answer(d.get("answer"), d.get("quote", ""), d.get("page"),
                              d.get("working", ""), cached=True)

        try:
            resp = self.client.complete_json(
                PROMPT.format(question=question, evidence=evidence),
                system=SYSTEM, max_tokens=900)
        except Exception as e:
            self.n_errors += 1
            # An API failure must abstain, never guess. 0 beats -1.
            return Answer(None, f"llm error: {str(e)[:120]}", failed=True)

        self.n_calls += 1
        data = resp.json()
        if data is None:
            self.n_errors += 1
            return Answer(None, "unparseable answer", raw=resp.text, failed=True)

        page = data.get("page")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None

        a = Answer(
            answer=data.get("answer"),
            quote=str(data.get("quote", ""))[:1500],
            page=page,
            working=str(data.get("working", ""))[:500],
            raw=resp.text,
        )
        if self.use_cache:
            self._cache[key] = {"answer": a.answer, "quote": a.quote,
                                "page": a.page, "working": a.working}
            self._dirty = True
        return a
