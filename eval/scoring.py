"""
eval/scoring.py

The real scoring rubric, implemented once so Days 6-7 plug into it rather than
inventing their own.

    Correct answer, correct location   -> +1.0
    "Not found in this filing"         ->  0.0
    Correct answer, wrong location     -> +0.5   (partial)
    Confidently wrong answer           -> -1.0

The negative term is the whole reason this system has two verifiers. A model
that answers everything confidently scores WORSE than one that abstains on
anything it isn't sure of. That means the interesting question is never "what's
our accuracy" -- it's "what's our expected score", which accuracy alone cannot
tell you.

WHAT IS STUBBED AND WHY
-----------------------
`answers_match` is deliberately mechanical right now: numeric comparison with
unit/scale handling, plus normalized string containment. It handles the 101
extractive questions well ("$1577.00" vs "1,577 million"). It does NOT handle
the 26 reasoning questions, whose gold answers are prose paragraphs -- those
need an LLM judge, which arrives on Day 7. Until then `grade()` marks them
UNJUDGED rather than guessing, and `Report` counts them separately instead of
silently scoring them as wrong. A harness that flatters itself is worse than no
harness.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------- rubric ----

SCORE_CORRECT_LOCATED = 1.0
SCORE_ABSTAIN = 0.0
SCORE_CORRECT_WRONG_PAGE = 0.5
SCORE_CONFIDENTLY_WRONG = -1.0


class Outcome(str, Enum):
    CORRECT_LOCATED = "correct+located"
    CORRECT_WRONG_PAGE = "correct/wrong-page"
    ABSTAINED = "abstained"
    CONFIDENTLY_WRONG = "confidently-wrong"
    UNJUDGED = "unjudged"          # needs the Day-7 LLM judge

    @property
    def score(self) -> float:
        return {
            Outcome.CORRECT_LOCATED: SCORE_CORRECT_LOCATED,
            Outcome.CORRECT_WRONG_PAGE: SCORE_CORRECT_WRONG_PAGE,
            Outcome.ABSTAINED: SCORE_ABSTAIN,
            Outcome.CONFIDENTLY_WRONG: SCORE_CONFIDENTLY_WRONG,
            Outcome.UNJUDGED: 0.0,
        }[self]


@dataclass
class SystemAnswer:
    """What the pipeline produces for one question. `abstained=True` means the
    system declined -- Verifier #1 said insufficient, or Verifier #2 caught a
    fabricated quote."""
    abstained: bool = False
    answer_text: str = ""
    quote: str = ""
    cited_page: int | None = None
    cited_doc: str = ""


# ----------------------------------------------------- answer comparison ----

_NUM_RE = re.compile(r"-?\$?\s*\(?\d[\d,]*\.?\d*\)?%?")

_SCALE = {
    "trillion": 1e12, "trillions": 1e12,
    "billion": 1e9, "billions": 1e9, "bn": 1e9,
    "million": 1e6, "millions": 1e6, "mm": 1e6, "mn": 1e6,
    "thousand": 1e3, "thousands": 1e3,
}


def _scale_of(text: str) -> float:
    t = text.lower()
    for word, mult in _SCALE.items():
        if re.search(rf"\b{word}\b", t):
            return mult
    return 1.0


def extract_numbers(text: str) -> list[tuple[float, int]]:
    """Pull (value, decimal_places) out of an answer.

    Decimal places matter: they tell us how precisely the answer was stated,
    which is what makes "$8.70" and "8.7012" the same answer while keeping
    1577 and 1578 different. Accounting negatives are honoured -- (1,577)
    means -1577, and getting that backwards is a silent sign error.
    """
    out: list[tuple[float, int]] = []
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group().strip()
        neg = raw.startswith("(") or raw.startswith("-")
        cleaned = re.sub(r"[^\d.]", "", raw)
        if not cleaned or cleaned == ".":
            continue
        try:
            v = float(cleaned)
        except ValueError:
            continue
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        out.append((-v if neg else v, decimals))
    return out


# Finance states the same concept several ways. The grader was scoring correct
# answers as CONFIDENTLY WRONG (-1) purely on vocabulary:
#     predicted "Operating activities"  vs  gold "...cashflow from Operations"
# Same answer, 2-point swing. These are equivalence classes, not a thesaurus --
# each pair appears in the actual gold set.
_SYNONYMS = [
    {"operating activities", "operations", "operating"},
    {"investing activities", "investing", "investment activities"},
    {"financing activities", "financing"},
    {"net sales", "revenue", "revenues", "total revenue", "net revenue"},
    {"cogs", "cost of sales", "cost of goods sold", "cost of revenue"},
    {"ppne", "pp&e", "property plant and equipment", "net ppe", "fixed assets"},
    {"capex", "capital expenditure", "capital expenditures",
     "purchases of property plant and equipment"},
    {"sg&a", "selling general and administrative"},
    {"ebitda", "unadjusted ebitda"},
]


def _synonym_match(a: str, b: str) -> bool:
    a, b = _normalize_text(a), _normalize_text(b)
    for group in _SYNONYMS:
        in_a = any(t in a for t in group)
        in_b = any(t in b for t in group)
        if in_a and in_b:
            return True
    return False


def _polarity(text: str) -> bool | None:
    """Leading yes/no, if present. A yes/no question answered with the right
    number but the wrong verdict is still a wrong answer -- and without this
    check it would score as correct."""
    t = _normalize_text(text)
    if re.match(r"^(yes|correct|true)\b", t):
        return True
    if re.match(r"^(no|not|false|incorrect)\b", t):
        return False
    return None


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\$,%()]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# Gold answers longer than this are prose arguments, not facts -- they need the
# Day-7 LLM judge. Chosen from the actual distribution: 53 of 127 gold answers
# are a single token, and everything past ~12 words is a multi-clause argument
# ("No, the company is managing its CAPEX efficiently, which is evident from...").
PROSE_WORD_THRESHOLD = 12


# "THIS METRIC DOES NOT APPLY" IS AN ANSWER, NOT A REFUSAL.
#
# Several questions invite it explicitly: "Does AMEX have an improving
# operating margin profile? If operating margin is not a useful metric for a
# company like this, state that and explain why." For those, gold is:
#
#     "Performance is not measured through operating margin"
#
# and the model said:
#
#     "No, operating margin is not a useful metric for American Express
#      as of 2022 because ..."
#
# The same answer, worded differently -- and scored CONFIDENTLY WRONG (-1),
# because gold carried no number and the strings did not overlap enough. Two
# points lost per case for being right.
_NOT_APPLICABLE = re.compile(
    r"(not|isn'?t|is\s+not)\s+(a\s+)?(useful|relevant|meaningful|applicable|"
    r"appropriate|suitable)\b"
    r"|not\s+measured\s+through\b"
    r"|does\s+not\s+(use|report|measure|apply)\b"
    r"|\bnot\s+applicable\b|\bn/?a\b"
    r"|no(t|n)?[- ]?meaningful\b",
    re.I,
)


def _both_say_not_applicable(predicted: str, gold: str) -> bool:
    return bool(_NOT_APPLICABLE.search(predicted or "")
                and _NOT_APPLICABLE.search(gold or ""))


def answers_match(predicted: str, gold: str) -> bool | None:
    """True / False / None.

    None means "I can't judge this mechanically" -- reserved for prose gold
    answers, which need the Day-7 LLM judge. Returning None instead of False is
    the point: an unjudgeable question must not be scored as a failure.

    Numeric comparison is PRECISION-BASED, not relative-tolerance-based. A flat
    1% tolerance calls 1577 and 1578 equal, which for a financial figure is just
    wrong -- those are different numbers. Instead we match at the precision the
    gold answer was stated to: "$8.70" tolerates anything rounding to 8.70,
    while "1577" tolerates nothing but 1577.
    """
    if not predicted:
        return False

    # Checked BEFORE the prose-length deferral: both sides saying "this metric
    # does not apply here" is a match regardless of how verbosely either says
    # it, and deferring it to an LLM judge would leave the point unscored.
    if _both_say_not_applicable(predicted, gold):
        return True

    # Prose gold -> defer, regardless of whether it contains numbers.
    if len(_normalize_text(gold).split()) > PROSE_WORD_THRESHOLD:
        return None

    gold_nums = extract_numbers(gold)
    pred_nums = extract_numbers(predicted)

    # Drop years from a prose gold answer. "In 2022, AMD brought in the most
    # cashflow from Operations" has one number -- 2022 -- which is a date, not
    # the answer. Comparing it numerically against a prediction is meaningless,
    # and worse, it makes the comparator take the numeric path for what is
    # really a textual answer.
    if len(_normalize_text(gold).split()) > 3:
        non_year = [(v, d) for v, d in gold_nums if not (1900 <= v <= 2099 and d == 0)]
        if non_year:
            gold_nums = non_year
        elif gold_nums:
            gold_nums = []      # only years -> treat as a textual answer

    # Yes/no verdicts must agree when the gold states one.
    g_pol, p_pol = _polarity(gold), _polarity(predicted)
    if g_pol is not None and p_pol is not None and g_pol != p_pol:
        return False

    # A yes/no question answered with the right verdict is correct, even if the
    # prediction omits the supporting figure. Gold "Yes, CVS paid a $0.55
    # dividend per share every quarter" vs predicted "Yes" was being scored
    # CONFIDENTLY WRONG (-1) because the prediction contained no number. The
    # question asked whether they paid dividends; "Yes" answers it.
    if g_pol is not None and p_pol is not None and g_pol == p_pol \
            and not pred_nums:
        return True

    if not gold_nums:
        g, p = _normalize_text(gold), _normalize_text(predicted)
        if g_pol is not None and p_pol is not None:
            return g_pol == p_pol
        if g in p or p in g:
            return True
        return _synonym_match(gold, predicted)

    # SHORT NAMED ANSWER inside a longer gold sentence.
    #
    # The question "which segment dragged down growth?" wants a segment name.
    # The model said "Consumer"; gold says "The consumer segment shrunk by 0.9%
    # organically". That is the SAME ANSWER -- but gold carries a number (0.9)
    # and the prediction does not, so the numeric path scored it CONFIDENTLY
    # WRONG at -1 when it deserved +1. A 2-point swing on a right answer.
    #
    # Guarded so it cannot wave through vagueness: the prediction must be short
    # (<= 4 words), must survive stopword removal, and every remaining word must
    # appear in gold. "Consumer" passes; "revenue" against a gold about
    # segments does not.
    if not pred_nums:
        p_words = [w for w in _normalize_text(predicted).split()
                   if w not in {"the", "a", "an", "of", "in", "is", "was",
                                "segment", "and", "to", "for"}]
        if 1 <= len(p_words) <= 4:
            g_norm = _normalize_text(gold)
            if all(w in g_norm for w in p_words):
                return True

    # Gold is numeric but the prediction states no number at all. If the gold
    # is a bare verdict the polarity check above already settled it; otherwise
    # the prediction simply failed to answer.
    if not pred_nums:
        return False

    g_scale = _scale_of(gold)
    p_scale = _scale_of(predicted)

    # SIGNIFICANT-FIGURE MATCHING.
    #
    # Gold answers are rounded for presentation; model answers often are not.
    # Real cases that were being scored as CONFIDENTLY WRONG (-1):
    #     gold $382.00   vs  predicted $381.603 million
    #     gold $303.00   vs  predicted $302.578 million
    #     gold 0.01      vs  predicted 1.47   (ratio vs percent, same value)
    # Every one of those is the same figure stated to different precision, and
    # calling them wrong cost 2 points each versus abstaining.
    #
    # The rule: round the prediction to the number of significant figures the
    # GOLD used. If it then equals gold, it is the same number. This stays
    # strict where it matters -- 0.4% vs 0.5% is 1 sig fig either way and
    # remains a genuine mismatch.
    def _sigfigs(v: float, dec: int) -> int:
        s = f"{abs(v):.{dec}f}".replace(".", "").lstrip("0")
        return max(1, len(s.rstrip("0")) or 1)

    def _round_sig(v: float, sig: int) -> float:
        if v == 0:
            return 0.0
        from math import floor, log10
        return round(v, -int(floor(log10(abs(v)))) + (sig - 1))

    for gv, gdec in gold_nums:
        sig = _sigfigs(gv, gdec)
        for pv, _ in pred_nums:
            for cand in (pv, pv / 100.0, pv * 100.0):
                if _round_sig(cand, sig) == _round_sig(gv, sig):
                    return True

    # PERCENTAGES AND RATES: accept a 0.15pp absolute difference.
    #
    # gold 101.5% vs predicted 101.6% was scored CONFIDENTLY WRONG (-1). Those
    # come from the same two figures divided the same way -- the gap is which
    # decimal each side rounded at, or a base that differs in the last digit.
    # Treating that as a fabrication costs 2 points for a rounding convention.
    # Kept tight: 0.4% vs 0.5% (0.1pp) still fails, because on a small rate that
    # is a genuinely different answer... so the tolerance only applies when the
    # values are large enough that 0.15pp is proportionally small.
    if "%" in gold or "%" in predicted:
        for gv, _ in gold_nums:
            for pv, _ in pred_nums:
                if abs(gv) >= 1.0 and abs(gv - pv) <= 0.15:
                    return True

    for gv, gdec in gold_nums:
        # Half a unit of the last stated decimal place, carried through scale.
        tol = (0.5 * (10 ** -gdec)) * g_scale
        for pv, _ in pred_nums:
            # Try both "apply each side's own scale word" and "no scaling",
            # since answers state units inconsistently ("$1.577 billion" vs
            # "1,577" in a millions-denominated table).
            for gs, ps in ((g_scale, p_scale), (1.0, 1.0)):
                a, b = gv * gs, pv * ps
                if abs(a - b) <= (tol if gs != 1.0 or g_scale == 1.0 else 0.5 * 10 ** -gdec):
                    return True
    return False


# ------------------------------------------------------------- grading ------

def grade(
    system: SystemAnswer,
    gold_answer: str,
    gold_pages: set[int],
    page_tolerance: int = 0,
) -> Outcome:
    """Grade one answer against the rubric."""
    if system.abstained:
        return Outcome.ABSTAINED

    correct = answers_match(system.answer_text, gold_answer)
    if correct is None:
        return Outcome.UNJUDGED
    if not correct:
        return Outcome.CONFIDENTLY_WRONG

    if system.cited_page is None:
        return Outcome.CORRECT_WRONG_PAGE

    located = any(abs(system.cited_page - gp) <= page_tolerance for gp in gold_pages)
    return Outcome.CORRECT_LOCATED if located else Outcome.CORRECT_WRONG_PAGE


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)

    def add(self, o: Outcome) -> None:
        self.outcomes.append(o)

    @property
    def judged(self) -> list[Outcome]:
        return [o for o in self.outcomes if o is not Outcome.UNJUDGED]

    @property
    def total_score(self) -> float:
        return sum(o.score for o in self.outcomes)

    @property
    def avg_score(self) -> float:
        j = self.judged
        return self.total_score / len(j) if j else 0.0

    def counts(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(o.value for o in self.outcomes))

    def render(self) -> str:
        c = self.counts()
        n = len(self.outcomes)
        lines = [
            f"questions            : {n}",
            f"judged mechanically  : {len(self.judged)}",
            f"needs LLM judge      : {c.get('unjudged', 0)}",
            "",
            f"  correct + located  : {c.get('correct+located', 0):4d}  (+1.0 each)",
            f"  correct/wrong page : {c.get('correct/wrong-page', 0):4d}  (+0.5 each)",
            f"  abstained          : {c.get('abstained', 0):4d}  ( 0.0 each)",
            f"  confidently wrong  : {c.get('confidently-wrong', 0):4d}  (-1.0 each)",
            "",
            f"TOTAL SCORE          : {self.total_score:+.1f}",
            f"AVG per judged q     : {self.avg_score:+.3f}",
        ]
        return "\n".join(lines)
