"""
eval/gold.py

Loads practice-questions.jsonl into a typed gold set, and -- the important part
-- translates FinanceBench's `evidence_page_num` into OUR parser's page
numbering.

WHY THE OFFSET EXISTS
---------------------
FinanceBench's page numbers come from the original PDF and are 0-indexed. Our
parser splits on `page-break-after:always` and numbers pages 1..N. Measured on
unique-match snippets only (see eval/diagnose_offset.py), the offset is +1 for
81.5% of datapoints and is stable within each filing (32 of 35 filings with data
agree on +1).

So: our_page = gold_page + PAGE_OFFSET, with PAGE_OFFSET = 1.

Known exception: 3M_2023Q2_10Q shows mixed offsets and is on a watchlist.
Rather than pretend the offset is exact everywhere, every metric that consumes
this module supports a `tolerance` (default 0) so we can report both strict
page accuracy and +/-1 "near miss" accuracy honestly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILINGS_DIR = PROJECT_ROOT / "data" / "filings"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "practice-questions.jsonl"

# Derived empirically -- see eval/diagnose_offset.py. Do not change without
# re-running that script.
PAGE_OFFSET = 1

# Filings where the offset was not internally consistent. Excluded from strict
# page scoring by default so one bad filing doesn't silently distort retrieval
# numbers -- but reported separately, never dropped quietly.
OFFSET_WATCHLIST = {"3M_2023Q2_10Q"}


@dataclass
class GoldEvidence:
    """One evidence block: the passage that proves the answer, and where it lives."""
    text: str                 # the exact proving passage
    full_page_text: str       # the whole page it came from
    gold_page_pdf: int        # FinanceBench's number (0-indexed, from the PDF)

    @property
    def page(self) -> int:
        """Page number in OUR parser's 1-indexed numbering."""
        return self.gold_page_pdf + PAGE_OFFSET


@dataclass
class GoldQuestion:
    qid: str
    question: str
    answer: str
    doc_name: str
    company: str
    doc_type: str             # "10k" | "10q" | "8k"
    doc_period: int
    question_type: str        # FinanceBench's own label
    reasoning: str            # "Information extraction" | "Logical reasoning" | ...
    justification: str
    evidence: list[GoldEvidence] = field(default_factory=list)

    # ---- derived helpers -------------------------------------------------

    @property
    def gold_pages(self) -> set[int]:
        """All pages (our numbering) that count as a correct location."""
        return {ev.page for ev in self.evidence}

    @property
    def is_multi_evidence(self) -> bool:
        """Needs facts from more than one page -- income statement + balance
        sheet + cash flow, etc. These are the questions that need K>=3."""
        return len(self.gold_pages) > 1

    @property
    def is_reasoning(self) -> bool:
        """Our own extractive-vs-reasoning split, derived from FinanceBench's
        fields. Reasoning = the answer is synthesized/computed, not copied.

        Note this is a LABEL for evaluation, not the runtime classifier -- the
        runtime classifier (Day 5) only sees the question text, never these
        fields. Keeping them separate is what stops us from accidentally
        evaluating a classifier against its own inputs.
        """
        if self.question_type == "metrics-generated":
            return False
        return "logical reasoning" in self.reasoning.lower() or self.is_multi_evidence

    @property
    def filing_path(self) -> Path:
        return FILINGS_DIR / f"{self.doc_name}.htm"

    @property
    def filing_exists(self) -> bool:
        return self.filing_path.exists()

    @property
    def on_watchlist(self) -> bool:
        return self.doc_name in OFFSET_WATCHLIST


def load_gold(
    path: Path | str = QUESTIONS_PATH,
    doc_types: tuple[str, ...] | None = ("10k", "10q"),
    require_filing: bool = True,
    exclude_watchlist: bool = False,
) -> list[GoldQuestion]:
    """Load the gold set.

    doc_types        -- restrict to these form types. Default excludes 8-Ks,
                        whose two incompatible file formats are a known open
                        issue (6 of 78 filings). Pass None for everything.
    require_filing   -- drop questions whose .htm isn't on disk.
    exclude_watchlist-- drop filings with unstable page offsets.
    """
    out: list[GoldQuestion] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            if doc_types and raw.get("doc_type") not in doc_types:
                continue

            ev = [
                GoldEvidence(
                    text=e.get("evidence_text", ""),
                    full_page_text=e.get("evidence_text_full_page", ""),
                    gold_page_pdf=e.get("evidence_page_num", -1),
                )
                for e in raw.get("evidence", [])
                if e.get("evidence_page_num") is not None
            ]

            q = GoldQuestion(
                qid=raw["financebench_id"],
                question=raw["question"],
                answer=raw["answer"],
                doc_name=raw["doc_name"],
                company=raw.get("company", ""),
                doc_type=raw.get("doc_type", ""),
                doc_period=raw.get("doc_period", 0),
                question_type=raw.get("question_type", ""),
                reasoning=raw.get("question_reasoning") or "",
                justification=raw.get("justification", ""),
                evidence=ev,
            )

            if not q.evidence:
                continue
            if require_filing and not q.filing_exists:
                continue
            if exclude_watchlist and q.on_watchlist:
                continue
            out.append(q)
    return out


def summarize(gold: list[GoldQuestion]) -> str:
    from collections import Counter

    docs = {q.doc_name for q in gold}
    reasoning = sum(1 for q in gold if q.is_reasoning)
    multi = sum(1 for q in gold if q.is_multi_evidence)
    watch = sum(1 for q in gold if q.on_watchlist)
    by_type = Counter(q.doc_type for q in gold)
    ev_counts = Counter(len(q.evidence) for q in gold)

    lines = [
        f"questions          : {len(gold)}",
        f"distinct filings   : {len(docs)}",
        f"by form type       : {dict(sorted(by_type.items()))}",
        f"extractive         : {len(gold) - reasoning}",
        f"reasoning          : {reasoning}",
        f"multi-page evidence: {multi}",
        f"on offset watchlist: {watch}",
        f"evidence blocks/q  : {dict(sorted(ev_counts.items()))}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    gold = load_gold()
    print(summarize(gold))
    print("\n--- sample ---")
    q = gold[0]
    print(f"qid       : {q.qid}")
    print(f"question  : {q.question[:100]}...")
    print(f"answer    : {q.answer}")
    print(f"doc       : {q.doc_name}  ({q.doc_type} {q.doc_period})")
    print(f"gold pages: {sorted(q.gold_pages)}  (pdf: {[e.gold_page_pdf for e in q.evidence]})")
    print(f"reasoning?: {q.is_reasoning}")
