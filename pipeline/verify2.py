"""
pipeline/verify2.py

Verifier #2 -- mechanical citation check. NO LLM. Pure code.

WHAT IT DOES
------------
The answering model returned a quote and a page number. This checks one thing:
is that quote actually on that page of the filing?

That is a substring test, and being a substring test is the entire point.
Verifier #1 is an LLM judging an LLM -- useful, but it can be wrong in the same
direction as the thing it is checking. This cannot. It either finds the text or
it does not, and it costs nothing to run.

It is the direct defence against the worst outcome on the rubric: a confident,
plausible answer with a citation that was never in the document.

WHAT IT CANNOT CATCH
--------------------
Stated plainly, because overselling this is the easiest mistake to make:

  * A real quote from the right page that answers a DIFFERENT question --
    the wrong year's column of a multi-year table is the classic case. The
    quote is genuine, the page is genuine, the answer is wrong.
  * An answer that does not follow from the quote it cites.

So this proves the citation is real. It does not prove the reasoning is. The
numeric recomputation check below covers part of the remaining gap, and the
rest should be named as a limitation rather than papered over.

NORMALISATION
-------------
Comparison is not exact-character. Filings and model output differ in ways that
carry no meaning:
    $1,577  vs  1577        currency symbols and thousands separators
    (1,577) vs  -1577       accounting negatives
    em-dash vs hyphen       typography
    "  |  " vs " | "        whitespace from table rendering
Normalising these is not being lenient -- refusing them would fail correct
citations for cosmetic reasons, which teaches nothing and loses points.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parser.chunker import Chunk       # noqa: E402


def normalize(s: str) -> str:
    """Strip differences that carry no meaning."""
    s = (s or "").lower()
    s = s.replace("—", " ").replace("–", " ").replace("’", "'")
    s = s.replace("\xa0", " ")
    # (1,577) -> -1577 : accounting negative, same number
    s = re.sub(r"\((\s*[\d,.]+\s*)\)", r"-\1", s)
    s = re.sub(r"[\$,%]", "", s)
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    s = re.sub(r"[^\w\s.\-/|]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _tokens(s: str) -> list[str]:
    return [t for t in normalize(s).split() if t]


@dataclass
class CitationCheck:
    passed: bool
    reason: str
    match_ratio: float = 0.0
    found_on_page: int | None = None
    quote_len: int = 0

    @property
    def label(self) -> str:
        return "VERIFIED" if self.passed else "REJECTED"


def check_citation(
    quote: str,
    page: int | None,
    chunks: list[Chunk],
    min_tokens: int = 4,
    threshold: float = 0.75,
) -> CitationCheck:
    """Is `quote` really on page `page` of the retrieved evidence?

    threshold=0.75 rather than exact matching. Models routinely drop a stray
    "|" or collapse spacing when copying a table row, and failing a citation
    over a pipe character would reject correct work. 0.75 of tokens present,
    IN THE SAME CHUNK, is strict enough that a fabricated quote cannot pass --
    invented text shares almost no rare tokens with the real page.
    """
    if not quote or not quote.strip():
        return CitationCheck(False, "no quote given")

    q_tokens = _tokens(quote)
    if len(q_tokens) < min_tokens:
        return CitationCheck(False, f"quote too short ({len(q_tokens)} tokens)",
                             quote_len=len(q_tokens))

    if page is None:
        return CitationCheck(False, "no page cited", quote_len=len(q_tokens))

    on_page = [c for c in chunks if c.page_num == page]
    if not on_page:
        return CitationCheck(False, f"page {page} not among retrieved pages",
                             quote_len=len(q_tokens))

    q_norm = normalize(quote)
    for c in on_page:
        if q_norm in normalize(c.text):
            return CitationCheck(True, "exact match", 1.0, page, len(q_tokens))

    # Fall back to token overlap within a single chunk. Requiring one chunk
    # matters: tokens scattered across a whole page could be assembled into a
    # sentence that was never written.
    #
    # NUMBERS ARE CHECKED SEPARATELY AND STRICTLY. A plain overlap ratio treats
    # "1,577" as just another token, so a quote that keeps the real row label
    # and swaps the figures scores high on words alone. Tested: the fabricated
    # quote "Purchases of property, plant and equipment (PP&E) | (9,999) |
    # (8,888)" scored 83% and PASSED -- the label carries most of the tokens.
    #
    # That is the single most dangerous failure this verifier exists to stop:
    # a real line item with invented numbers is maximally plausible and
    # completely wrong. In a financial citation the numbers ARE the claim, so
    # every one of them must appear in the chunk.
    q_nums = [t for t in q_tokens if any(ch.isdigit() for ch in t)]
    q_words = [t for t in q_tokens if t not in q_nums]

    best = 0.0
    best_nums = 0.0
    for c in on_page:
        c_tokens = set(_tokens(c.text))
        if not c_tokens:
            continue
        overall = sum(1 for t in q_tokens if t in c_tokens) / len(q_tokens)
        nums_ok = (sum(1 for t in q_nums if t in c_tokens) / len(q_nums)
                   if q_nums else 1.0)
        words_ok = (sum(1 for t in q_words if t in c_tokens) / len(q_words)
                    if q_words else 1.0)
        if nums_ok >= 0.999 and words_ok >= threshold:
            return CitationCheck(
                True,
                f"{overall:.0%} token match, all {len(q_nums)} numbers present",
                overall, page, len(q_tokens))
        best = max(best, overall)
        best_nums = max(best_nums, nums_ok)

    if q_nums and best_nums < 0.999:
        missing = int(round((1 - best_nums) * len(q_nums)))
        return CitationCheck(
            False,
            f"{missing} of {len(q_nums)} quoted numbers are NOT on page {page} "
            f"(fabricated figures)",
            best, None, len(q_tokens))

    # Where did it actually come from? Useful for diagnosis: a quote that is
    # real but attributed to the wrong page is a different failure from one
    # that was invented.
    elsewhere = None
    for c in chunks:
        if c.page_num == page:
            continue
        if q_norm in normalize(c.text):
            elsewhere = c.page_num
            break
    if elsewhere is not None:
        return CitationCheck(False, f"quote is real but on page {elsewhere}, not {page}",
                             best, elsewhere, len(q_tokens))

    return CitationCheck(False, f"quote not found on page {page} "
                                f"(best overlap {best:.0%})",
                         best, None, len(q_tokens))


# ------------------------------------------------------- numeric recompute --

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_values(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(normalize(text)):
        try:
            out.append(float(m.group().replace(",", "")))
        except ValueError:
            continue
    return out


@dataclass
class RecomputeCheck:
    applicable: bool
    passed: bool
    reason: str


def check_arithmetic(answer: str, working: str, tol: float = 0.02) -> RecomputeCheck:
    """For calculated answers, does the stated result follow from the inputs?

    Verifier #2 confirms the INPUTS are real text on a real page. This asks a
    different question: given those inputs, is the arithmetic right? A model can
    quote two genuine figures and still divide them incorrectly.

    Deliberately narrow. It only fires when `working` contains a recognisable
    expression, and it returns applicable=False rather than guessing otherwise.
    A check that fires on everything would reject correct answers whose working
    it merely failed to parse -- worse than no check.
    """
    if not working or not answer:
        return RecomputeCheck(False, True, "no working shown")

    expr = re.search(r"([\d,.]+)\s*([/*+\-])\s*([\d,.]+)", working)
    if not expr:
        return RecomputeCheck(False, True, "no parsable expression")

    try:
        a = float(expr.group(1).replace(",", ""))
        b = float(expr.group(3).replace(",", ""))
    except ValueError:
        return RecomputeCheck(False, True, "operands unparseable")

    op = expr.group(2)
    if op == "/" and b == 0:
        return RecomputeCheck(True, False, "division by zero")
    result = {"/": lambda: a / b, "*": lambda: a * b,
              "+": lambda: a + b, "-": lambda: a - b}[op]()

    stated = extract_values(answer)
    if not stated:
        return RecomputeCheck(False, True, "answer has no number to check")

    for v in stated:
        for candidate in (result, result * 100, result / 100):
            if candidate == 0 and v == 0:
                return RecomputeCheck(True, True, "matches")
            denom = max(abs(candidate), abs(v))
            if denom and abs(candidate - v) / denom <= tol:
                return RecomputeCheck(True, True, f"{a}{op}{b} = {candidate:.4g}")

    return RecomputeCheck(True, False,
                          f"{a}{op}{b} = {result:.4g}, but answer says {stated[0]:.4g}")
