"""
pipeline/classify.py

Classifies a question as EXTRACTIVE or REASONING, using only the question text.

WHY THIS EXISTS
---------------
Two measured findings make this load-bearing rather than decorative:

1. Retrieval wants different weighting per type. Extractive questions peak at
   w_bm25=0.25 (dense-led, because they are paraphrase problems); reasoning
   peaks at w_bm25=8.0 (lexical-led, because rare literal tokens pin down the
   specific statement pages). The optimum for one is near the worst case for
   the other -- 0.634/0.269 versus 0.535/0.423 on the two splits.

2. They need different evidence-set sizes. Reasoning questions average more
   gold pages, so a K tuned for extractive starves them.

WHY RULES AND NOT A TRAINED MODEL
---------------------------------
26 positives in 127 examples. Any model with real capacity memorizes that, and
we would not find out until Day 10 -- the same trap the retrieval sweeps nearly
fell into. Weighted keyword rules have essentially no capacity to memorize, are
inspectable when they misfire, and are cross-validated below rather than
scored on the data that motivated them.

The features come from actual frequency analysis of the two classes, not
intuition:

    EXTRACTIVE markers  "statement", "income", "balance sheet", "round to",
                        "decimal", "in USD", "as shown in"
        -> these questions name the statement and the format they want back.
           They are lookups with a specified answer shape.

    REASONING markers   "explain", "why", "is <company> a", "based on",
                        "if ... then", "state whether", "do you", "would you"
        -> these ask for a judgment, and the answer is synthesized from facts
           that live in several places.

IMPORTANT: this never sees FinanceBench's `question_type` or `question_reasoning`
fields. Those are the LABELS. Using them as features would be evaluating the
classifier against its own input.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# (weight, pattern). Positive weight = reasoning, negative = extractive.
# Weights are coarse on purpose -- fine-grained values fitted to 127 examples
# would be noise dressed up as precision.
RULES: list[tuple[float, str]] = [
    # --- reasoning signals ---
    (2.0, r"\bexplain\b"),
    (2.0, r"\bwhy\b"),
    (1.5, r"\bstate\s+(whether|if)\b"),
    (1.5, r"\bdo you\b|\bwould you\b|\bdoes it\b"),
    (1.5, r"\bis\s+(this|the)?\s*\w+\s+(a|an)\s+\w+[- ]?(intensive|heavy|efficient)"),
    (1.5, r"\bhealthy\b|\bstable\b|\bimproving\b|\bworsening\b|\breasonable\b"),
    (1.2, r"\bbased (on|off)\b"),
    (1.2, r"\bif\b.{0,40}\bthen\b"),
    (1.0, r"\bshould\b|\bcould\b"),
    (1.0, r"\bjudg(e|ment)\b|\bassess\b|\bevaluate\b"),
    (1.0, r"\bplausible\b|\breasonably\b|\bapproximation\b"),
    (0.8, r"\bnot\s+relevant\b|\bnot\s+meaningful\b"),
    (0.8, r"\bconcern\b|\brisk\b|\bdrag(ged)?\b"),
    (0.8, r"\btrend\b|\bprofile\b|\bposition\b"),

    # --- extractive signals ---
    (-2.0, r"\bround(ed)? to\b"),
    (-2.0, r"\bdecimal place"),
    # Must name the STATEMENT, not merely the concept. An earlier version was
    # r"\b(income|cash ?flow|balance) ?(statement|sheet)?\b", where the optional
    # suffix let a bare "cash flow" count as "names the cash flow statement".
    # That misfired on "Does Adobe have an improving Free cashflow conversion?"
    # -- a judgment question -- and pushed it to extractive.
    (-1.5, r"\b(income statement|cash ?flow statement|balance sheet|"
           r"statement of (income|operations|cash ?flows))\b"),
    (-1.5, r"\bas (shown|seen|stated|reported) in\b"),
    (-1.2, r"\bin (usd|units of)\b"),
    (-1.2, r"\bwhat is the\b"),
    (-1.0, r"\banswer in\b"),
    (-1.0, r"\bstatement of\b"),
    (-0.8, r"\bline item\b"),
]

COMPILED = [(w, re.compile(p, re.I)) for w, p in RULES]

# Above this score -> reasoning. Chosen by cross-validated sweep, not by eye.
DEFAULT_THRESHOLD = 0.5


# A question that DEFINES A FORMULA needs figures from several statements, no
# matter how it opens. This is a separate axis from extractive-vs-reasoning and
# it is the one that decides the evidence budget.
#
# THE BUG THIS FIXES
# ------------------
# "What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed
# asset turnover is defined as: FY2019 revenue / (average PP&E between FY2018
# and FY2019)" was classified EXTRACTIVE -- "what is the" is a strong
# extractive marker -- and therefore got K=10. It needs revenue from the income
# statement plus PP&E from two years of the balance sheet. Revenue was never
# retrieved, so the model correctly declined.
#
# 28 of the 34 multi-page questions were mislabelled this way. That single
# mismatch produced most of the 40% abstention rate: classifier says simple ->
# K too small -> pages missing -> model declines -> looks like timidity.
FORMULA_PATTERNS = [
    r"\bdefined as\b", r"\bcalculated as\b", r"\bcompute[d]?\s+(as|using)\b",
    r"\bratio\b", r"\bturnover\b", r"\bmargin\b", r"\bcagr\b",
    r"\baverage of\b", r"\baverage\s+\w+\s+between\b",
    r"\b\d\s*year\s+average\b", r"\bgrowth rate\b", r"\byear[- ]over[- ]year\b",
    r"\bdays\s+(payable|sales|inventory)\b", r"\bconversion cycle\b",
    r"\bfy\d{4}\s*[-–]\s*fy\d{4}\b",          # a span of fiscal years
    r"\bfy\d{4}\b.*\bfy\d{4}\b",              # two fiscal years mentioned
    r"[/*]\s*\(?\s*(average|total|net)\b",     # an actual expression
]
COMPILED_FORMULA = [re.compile(p, re.I) for p in FORMULA_PATTERNS]


def needs_multi_page(question: str) -> bool:
    """Does answering this require figures from more than one place?"""
    return any(rx.search(question) for rx in COMPILED_FORMULA)


@dataclass
class Classification:
    is_reasoning: bool
    score: float
    fired: list[str]
    multi_page: bool = False

    @property
    def label(self) -> str:
        return "reasoning" if self.is_reasoning else "extractive"

    @property
    def confidence(self) -> float:
        """Distance from the threshold, squashed to 0-1. Used later to decide
        when NOT to trust the routing decision and fall back to the balanced
        config -- a wrong route is worse than no route."""
        return min(1.0, abs(self.score - DEFAULT_THRESHOLD) / 3.0)


def classify(question: str, threshold: float = DEFAULT_THRESHOLD) -> Classification:
    score = 0.0
    fired: list[str] = []
    for w, rx in COMPILED:
        if rx.search(question):
            score += w
            fired.append(f"{'+' if w > 0 else ''}{w:g} {rx.pattern[:34]}")

    # Length is a weak but real signal: multi-clause questions that define a
    # formula ("CCC is defined as: DIO + DSO - DPO...") are computations over
    # several facts. Deliberately small so it can only break near-ties.
    n_words = len(question.split())
    if n_words > 45:
        score += 0.5
        fired.append("+0.5 long question (>45 words)")

    mp = needs_multi_page(question)
    if mp:
        fired.append("multi-page (formula detected)")

    return Classification(is_reasoning=score > threshold, score=score,
                          fired=fired, multi_page=mp)


K_EXTRACTIVE = 10
# 16, not 20. Swept against retrieval quality AND token cost, because on a
# fixed daily quota K decides how many questions can be measured at all:
#
#   K_calc   all_pages   tokens/question   questions per 200k/day
#     20       0.583          8,556                 23
#     16       0.575          7,454                 27      <- chosen
#     14       0.551          6,902                 29
#
# 20 -> 16 gives up 0.008 of all_pages and buys 4 more measured questions.
# Below 16 the recall loss accelerates sharply, so this is the knee.
K_REASONING = 16


def evidence_k(c: Classification, base_k: int = K_EXTRACTIVE) -> int:
    """How many pages to retrieve for this question type.

    MEASURED (k=10 baseline, with reranker):

        fixed k=10           page=0.606  reasoning=0.385  avg 10.0 chunks
        ext=10, rea=15       page=0.614  reasoning=0.423  avg 10.9 chunks
        ext=10, rea=20       page=0.630  reasoning=0.462  avg 12.0 chunks
        ext=5,  rea=15       page=0.480  reasoning=0.423  avg  7.1 chunks

    Reasoning questions need income statement AND balance sheet AND cash flow,
    so they need the bigger budget. Shrinking the extractive side to save
    tokens backfires badly (0.480) -- the savings are not there to take.

    The cost of rea=20 is ~2 extra chunks on average, since only ~20% of
    questions are classified reasoning. That is a cheap way to buy +0.077 on
    the split that produces the -1 answers.
    """
    # Either axis earns the larger budget. A formula question is multi-page
    # whether or not it reads as "reasoning" -- "What is the FY2019 fixed asset
    # turnover ratio" opens like a lookup and needs three figures.
    return K_REASONING if (c.is_reasoning or c.multi_page) else base_k


def bm25_weight(c: Classification) -> float:
    """Fusion weight. Returns 1.0 always -- deliberately.

    Before the reranker existed, the two question types wanted opposite fusion
    weights: extractive peaked at w_bm25=0.25 (0.634/0.269) and reasoning at
    w_bm25=8.0 (0.535/0.423). That conflict was the main argument for routing.

    The reranker dissolved it. Re-measured WITH rerank(depth=20) at k=10:

        w_bm25   overall  extractive  reasoning  multipage
          0.25     0.598      0.673      0.308      0.588
          1.00     0.606      0.663      0.385      0.647   <- best everywhere
          2.00     0.591      0.653      0.346      0.618
          8.00     0.567      0.634      0.308      0.588

    Equal weights are now simultaneously best overall, best on reasoning, and
    best on multi-page. There is no tradeoff left to route around, and routing
    on PREDICTED labels actively hurt (0.591 vs 0.606) because the classifier's
    mistakes cost more than its correct calls gained.

    Kept as a function rather than deleted so the finding stays visible: the
    reranker did the job routing was meant to do.
    """
    return 1.0


if __name__ == "__main__":
    samples = [
        "What is the FY2018 capital expenditure amount (in USD millions) for 3M? "
        "Give a response by relying on the details shown in the cash flow statement.",
        "Is 3M a capital-intensive business based on FY2022 data?",
        "Does Adobe have an improving Free cashflow conversion as of FY2022?",
        "What is Amazon's year-over-year change in revenue from FY2016 to FY2017 "
        "(in units of percents and round to one decimal place)?",
        "Does 3M maintain a stable trend of dividend distribution?",
    ]
    for s in samples:
        c = classify(s)
        print(f"[{c.label:<10}] score={c.score:+.1f} conf={c.confidence:.2f} "
              f"K={evidence_k(c)} w_bm25={bm25_weight(c)}")
        print(f"   {s[:95]}")
        for f in c.fired:
            print(f"      {f}")
        print()
