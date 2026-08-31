"""
pipeline/decompose.py

Two-stage answering for questions the single-shot prompt refuses.

WHY
---
Abstention concentrates in one question shape:

    Logical reasoning (based on numerical)   57% abstain
    Numerical reasoning                      23%
    Information extraction                   21%

"Is 3M capital-intensive based on FY2022 data?" needs capex, revenue and total
assets found, two ratios computed, and a judgement formed -- all in one call,
while also producing a VERBATIM QUOTE that proves the conclusion. No such quote
exists: a synthesised verdict is not written on any single row. The model has
nothing valid to put in `quote`, and takes the exit.

Four prompt rewrites and a forced retry (schema with no `null` at all) failed to
move this. So this changes the SHAPE of the task rather than its wording:

    STAGE 1  "extract these figures from the excerpts"
             -> pure extraction, which the model does well (21% abstain)
             -> the quote comes from here, where a real row exists to cite

    STAGE 2  "given capex=X, revenue=Y, assets=Z, is this capital-intensive?"
             -> pure reasoning over numbers already in hand
             -> no retrieval, no quoting burden, no excerpt wall to search

HONEST EXPECTATION
------------------
This is the fifth attempt at the abstention problem; four failed. It targets the
actual mechanism rather than arguing with the model, which is why it is worth
trying -- not because it is likely to work. Costs 2 calls per question and is
applied ONLY to questions already refused, so the downside is bounded.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parser.chunker import Chunk            # noqa: E402
from pipeline.answer import Answer          # noqa: E402
from pipeline.llm import get_client         # noqa: E402
from pipeline.verify import format_evidence  # noqa: E402

DECOMP_CACHE = PROJECT_ROOT / ".cache" / "decompose"

EXTRACT_SYSTEM = (
    "You extract figures from SEC filing excerpts. You do not answer questions "
    "or draw conclusions -- you only report what numbers are present."
)

EXTRACT_PROMPT = """\
A question will be answered from these excerpts. Your ONLY job right now is to
find the figures it needs. Do not answer the question.

QUESTION (for context only):
{question}

{evidence}

List every figure that is relevant, with the exact line it came from.

Rules:
1. Report ONLY numbers that appear in the excerpts. Never estimate.
2. Financial statements use synonyms: "Net sales"=revenue, "Purchases of
   property, plant and equipment (PP&E)"=capex, "Property, plant and equipment
   - net"=net PP&E.
3. Table rows use | separators, most recent fiscal year first.
4. If a figure genuinely is not present, omit it. An incomplete list is useful.

JSON only:
{{"figures":[{{"name":"<what it is>","value":"<the number>","page":<PAGE number>,
"quote":"<the verbatim row it came from>"}}]}}
"""

REASON_SYSTEM = (
    "You answer financial questions from figures that have already been "
    "extracted for you. The extraction is trustworthy; use it."
)

REASON_PROMPT = """\
Answer this question using the figures below. They were extracted from the
filing and are reliable.

QUESTION:
{question}

FIGURES FOUND IN THE FILING:
{figures}

Rules:
1. Use these figures. You do not need to search anything -- the work of finding
   them is done.
2. Compute what the question asks for. Show the expression in "working".
3. For a yes/no or judgement question, decide and say why in one sentence.
4. Answer even if you would prefer more figures. State any assumption in
   "working". Only answer null if the figures list is empty or contains nothing
   relevant at all.
5. Use the units the question asks for.

JSON only:
{{"working":"<expression or reasoning>","answer":"<your answer>"}}
"""


@dataclass
class Decomposed:
    answer: Answer
    n_figures: int = 0
    stage: str = ""          # where it stopped: "extract" | "reason" | "ok"


class Decomposer:
    def __init__(self, client=None, max_chars: int = 1000, use_cache: bool = True):
        self.client = client or get_client()
        self.max_chars = max_chars
        self.use_cache = use_cache
        self._cache: dict | None = None
        self._dirty = False
        self.n_calls = 0

    # ---------------------------------------------------------------- cache --

    def _cache_path(self) -> Path:
        tag = hashlib.md5(getattr(self.client, "name", "?").encode()).hexdigest()[:8]
        return DECOMP_CACHE / f"decompose__{tag}.json"

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
        DECOMP_CACHE.mkdir(parents=True, exist_ok=True)
        self._cache_path().write_text(json.dumps(self._cache), encoding="utf-8")
        self._dirty = False

    # ----------------------------------------------------------------- run --

    def answer(self, question: str, chunks: list[Chunk]) -> Decomposed:
        if not chunks:
            return Decomposed(Answer(None, "no evidence"), stage="extract")

        evidence = format_evidence(chunks, self.max_chars)
        key = hashlib.md5(
            f"{EXTRACT_PROMPT}{REASON_PROMPT}||{question}||{evidence}".encode()
        ).hexdigest()

        if self.use_cache:
            self._load()
            if key in self._cache:
                d = self._cache[key]
                return Decomposed(
                    Answer(d.get("answer"), d.get("quote", ""), d.get("page"),
                           d.get("working", ""), cached=True, retried=True),
                    d.get("n_figures", 0), d.get("stage", "ok"))

        # ---- stage 1: extract -------------------------------------------
        try:
            r1 = self.client.complete_json(
                EXTRACT_PROMPT.format(question=question, evidence=evidence),
                system=EXTRACT_SYSTEM, max_tokens=900)
            self.n_calls += 1
        except Exception as e:
            return Decomposed(Answer(None, f"llm error: {str(e)[:100]}",
                                     failed=True), stage="extract")

        d1 = r1.json() or {}
        figures = d1.get("figures") or []
        if not isinstance(figures, list) or not figures:
            return Decomposed(Answer(None, "no figures extracted"),
                              0, stage="extract")

        # The quote and page come from stage 1, where a real row exists to
        # cite. This is the point of the split: the citation belongs to an
        # extracted FIGURE, not to a synthesised verdict.
        best = figures[0]
        quote = str(best.get("quote", ""))[:1500]
        try:
            page = int(best.get("page")) if best.get("page") is not None else None
        except (TypeError, ValueError):
            page = None

        rendered = "\n".join(
            f"- {f.get('name','?')}: {f.get('value','?')}  (page {f.get('page','?')})"
            for f in figures[:12])

        # ---- stage 2: reason --------------------------------------------
        try:
            r2 = self.client.complete_json(
                REASON_PROMPT.format(question=question, figures=rendered),
                system=REASON_SYSTEM, max_tokens=700)
            self.n_calls += 1
        except Exception as e:
            return Decomposed(Answer(None, f"llm error: {str(e)[:100]}",
                                     failed=True), len(figures), stage="reason")

        d2 = r2.json() or {}
        a = Answer(answer=d2.get("answer"), quote=quote, page=page,
                   working=str(d2.get("working", ""))[:500], retried=True)

        if self.use_cache:
            self._cache[key] = {"answer": a.answer, "quote": a.quote,
                                "page": a.page, "working": a.working,
                                "n_figures": len(figures), "stage": "ok"}
            self._dirty = True
        return Decomposed(a, len(figures), stage="ok")
