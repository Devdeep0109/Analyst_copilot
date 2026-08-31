"""
pipeline/judge.py

An LLM judge for the answers our mechanical comparator cannot grade.

WHY THIS EXISTS
---------------
20 of 127 gold answers are prose arguments, not facts:

    "No, the company is managing its CAPEX and Fixed Assets pretty
     efficiently, which is evident from below key metrics: CAPEX/Revenue
     Ratio: 5.1% ..."

`answers_match` returns None for those -- correctly, because string matching
cannot tell whether a paragraph agrees with another paragraph. They then score
0 whether the system was right or wrong, which UNDERSTATES the result. Roughly
half are probably correct.

This is a MEASUREMENT component, not a pipeline one. It changes nothing about
how the system answers; it only grades answers already produced. That
distinction matters: it is the same thing a human marker would do, automated.

DESIGN CONSTRAINTS
------------------
* The judge sees ONLY the gold answer and the predicted answer. Not the
  filing, not the evidence, not whether the citation verified. Its single
  question is "do these say the same thing?" -- anything else invites it to
  re-answer the question itself and mark its own homework.
* It must be able to say UNCLEAR. A judge forced to choose will guess, and a
  guessed +1 is worse than an honest 0.
* Verdicts are cached like everything else, so re-running costs nothing.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.llm import get_client   # noqa: E402

JUDGE_CACHE = PROJECT_ROOT / ".cache" / "judge"

SYSTEM = (
    "You grade answers to financial questions. You compare a candidate answer "
    "against a reference answer and decide whether they agree. You never "
    "answer the question yourself."
)

PROMPT = """\
Do these two answers to the same question say the same thing?

QUESTION:
{question}

REFERENCE ANSWER (known correct):
{gold}

CANDIDATE ANSWER (to grade):
{predicted}

Rules:
1. Judge AGREEMENT, not completeness. A shorter answer that states the same
   conclusion is correct. The candidate need not repeat the reference's
   supporting detail.
2. The KEY FIGURE or VERDICT must match. If the reference says margins declined
   and the candidate says they improved, that is wrong. If the reference says
   5.1% and the candidate says 5.1%, that is right even if the wording differs.
3. Numbers may be rounded differently (101.5% vs 101.6%) or scaled differently
   ($1,577 million vs $1.577 billion). Those still match.
4. If the candidate declines to answer, or gives no conclusion, that is NOT a
   match -- return "no".
5. If you genuinely cannot tell whether they agree, return "unclear". Do not
   guess.

JSON only:
{{"verdict":"yes" or "no" or "unclear","why":"<one short sentence>"}}
"""


@dataclass
class Judgement:
    verdict: str          # "yes" | "no" | "unclear"
    why: str = ""
    cached: bool = False
    failed: bool = False

    @property
    def correct(self) -> bool | None:
        """None means unclear -- keeps the same contract as answers_match."""
        return {"yes": True, "no": False}.get(self.verdict)


class Judge:
    def __init__(self, client=None, use_cache: bool = True):
        self.client = client or get_client()
        self.use_cache = use_cache
        self._cache: dict | None = None
        self._dirty = False
        self.n_calls = 0
        self.n_errors = 0

    def _cache_path(self) -> Path:
        tag = hashlib.md5(getattr(self.client, "name", "?").encode()).hexdigest()[:8]
        return JUDGE_CACHE / f"judge__{tag}.json"

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
        JUDGE_CACHE.mkdir(parents=True, exist_ok=True)
        self._cache_path().write_text(json.dumps(self._cache), encoding="utf-8")
        self._dirty = False

    def judge(self, question: str, gold: str, predicted: str) -> Judgement:
        if not str(predicted or "").strip():
            return Judgement("no", "no answer given")

        key = hashlib.md5(
            f"{PROMPT}||{question}||{gold}||{predicted}".encode()).hexdigest()
        if self.use_cache:
            self._load()
            if key in self._cache:
                d = self._cache[key]
                return Judgement(d["verdict"], d.get("why", ""), cached=True)

        try:
            resp = self.client.complete_json(
                PROMPT.format(question=question[:600], gold=gold[:800],
                              predicted=str(predicted)[:800]),
                system=SYSTEM, max_tokens=300)
        except Exception as e:
            self.n_errors += 1
            # An unreachable judge must not silently mark things correct.
            return Judgement("unclear", f"judge error: {str(e)[:100]}",
                             failed=True)

        self.n_calls += 1
        data = resp.json()
        if data is None or data.get("verdict") not in {"yes", "no", "unclear"}:
            self.n_errors += 1
            return Judgement("unclear", "unparseable judge output", failed=True)

        j = Judgement(data["verdict"], str(data.get("why", ""))[:150])
        if self.use_cache:
            self._cache[key] = {"verdict": j.verdict, "why": j.why}
            self._dirty = True
        return j
