"""
eval/harness.py

The measurement layer everything downstream plugs into.

Two things live here:

  1. `Retriever` -- the interface Days 3-4 implement four times (BM25, dense,
     hybrid, reranked). Index a filing's chunks, return ranked chunks for a
     question. Nothing else.

  2. `evaluate_retrieval` -- scores a retriever against the gold set.

WHY RETRIEVAL IS SCORED SEPARATELY FROM ANSWERS
-----------------------------------------------
If the right page never enters the top-K, no verifier or answering prompt can
recover it -- the ceiling is already lost, and every later metric is measuring
something else. So retrieval gets its own number: page_recall@k is the hard
upper bound on how well the finished system can possibly score.

METRICS AND THEIR TRAPS
-----------------------
  page_recall@k   -- did ANY retrieved chunk land on a gold page? The headline
                     number. For multi-evidence questions this is the lenient
                     reading: one of three needed pages counts.
  all_pages@k     -- did we retrieve EVERY gold page? The honest number for
                     reasoning questions, which need income statement AND
                     balance sheet AND cash flow. Optimising page_recall alone
                     will quietly wreck this, and reasoning questions are the
                     ones with multi-part answers.
  evidence_recall -- did a retrieved chunk actually CONTAIN the proving text?
                     Stricter than page_recall: right page, wrong chunk still
                     fails. This is what the answering LLM actually sees.
  mrr             -- 1/rank of the first gold-page hit. Tells you whether K can
                     be shrunk, which directly cuts tokens and hallucination
                     surface.

Retrieval is scoped per-filing: the question names its document, so we never
search across filings. That is a real product assumption, not a shortcut --
"Add Filing" means the user asks about a filing they chose.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parser.chunker import Chunk                       # noqa: E402
from parser.page_parser import _normalize              # noqa: E402
from eval.corpus import load_corpus                    # noqa: E402
from eval.gold import GoldQuestion, load_gold          # noqa: E402


# ------------------------------------------------------------ interface ----

class Retriever(Protocol):
    """Implement this four times on Days 3-4. Keep it this small."""

    name: str

    def index(self, doc_name: str, chunks: list[Chunk]) -> None:
        """Build whatever structure this retriever needs for one filing."""
        ...

    def search(self, doc_name: str, query: str, k: int) -> list[Chunk]:
        """Return up to k chunks, best first."""
        ...


# -------------------------------------------------------------- metrics ----

@dataclass
class QuestionResult:
    qid: str
    doc_name: str
    is_reasoning: bool
    gold_pages: set[int]
    retrieved_pages: list[int]
    page_hit: bool
    all_pages_hit: bool
    evidence_hit: bool
    first_hit_rank: int | None

    @property
    def rr(self) -> float:
        return 1.0 / self.first_hit_rank if self.first_hit_rank else 0.0


@dataclass
class RetrievalReport:
    retriever: str
    k: int
    results: list[QuestionResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    def _mean(self, attr: str, subset=None) -> float:
        rows = subset if subset is not None else self.results
        if not rows:
            return 0.0
        return sum(float(getattr(r, attr)) for r in rows) / len(rows)

    @property
    def page_recall(self) -> float:
        return self._mean("page_hit")

    @property
    def evidence_recall(self) -> float:
        return self._mean("evidence_hit")

    @property
    def all_pages_recall(self) -> float:
        return self._mean("all_pages_hit")

    @property
    def mrr(self) -> float:
        return self._mean("rr")

    def split(self) -> dict[str, list[QuestionResult]]:
        return {
            "extractive": [r for r in self.results if not r.is_reasoning],
            "reasoning": [r for r in self.results if r.is_reasoning],
            "multi-page": [r for r in self.results if len(r.gold_pages) > 1],
        }

    def render(self) -> str:
        lines = [
            f"{self.retriever}  (k={self.k}, n={len(self.results)}, {self.elapsed_s:.1f}s)",
            f"  page_recall@{self.k}      : {self.page_recall:.3f}   <- ceiling for the whole system",
            f"  evidence_recall@{self.k}  : {self.evidence_recall:.3f}   <- proving text actually in a chunk",
            f"  all_pages@{self.k}        : {self.all_pages_recall:.3f}   <- every gold page retrieved",
            f"  mrr                      : {self.mrr:.3f}",
        ]
        for label, rows in self.split().items():
            if rows:
                pr = sum(r.page_hit for r in rows) / len(rows)
                ap = sum(r.all_pages_hit for r in rows) / len(rows)
                lines.append(f"    {label:<11} n={len(rows):<4} page_recall={pr:.3f}  all_pages={ap:.3f}")
        return "\n".join(lines)

    def failures(self, limit: int = 10) -> list[QuestionResult]:
        return [r for r in self.results if not r.page_hit][:limit]


# ------------------------------------------------------------- the loop ----

def _evidence_in_chunks(question: GoldQuestion, chunks: list[Chunk]) -> bool:
    """Did any retrieved chunk contain a gold proving passage?

    We test the longest distinctive LINE of the evidence rather than the whole
    block: gold `evidence_text` often spans a page's worth of table, which no
    single chunk should be expected to hold. The line-level test asks the
    question that matters -- is the actual proving fact in front of the model.
    """
    blob = _normalize(" ".join(c.text for c in chunks))
    for ev in question.evidence:
        lines = [l.strip() for l in ev.text.split("\n") if len(l.strip()) >= 15]
        if not lines:
            continue
        for line in sorted(lines, key=len, reverse=True)[:5]:
            if _normalize(line) in blob:
                return True
    return False


def answerable_from(question: GoldQuestion, chunks: list[Chunk]) -> tuple[bool, str]:
    """Can this question be answered from these chunks? Returns (bool, why).

    WHY page_hit IS NOT THE RIGHT TEST
    ----------------------------------
    FinanceBench labels ONE gold page per evidence block, but a filing states
    the same fact in several places. For "What is 3M's FY2018 capex?", gold is
    page 60 (the cash flow statement) -- yet the identical line, "Purchases of
    property, plant and equipment (PP&E) | $ | (1,577)", also appears on page
    46, and the figure appears again on page 39. Citing page 46 is a correct,
    verifiable answer.

    Scoring that as a miss punished the system for being right. Measured across
    all 127 questions, the strict page test says 63.0% answerable; allowing any
    of three independent signals says 85.0%. That is a 22-point understatement,
    and it was distorting both the retrieval numbers and Verifier #1's grade.

    THE THREE SIGNALS
        page     -- a gold page is among the retrieved pages (strictest)
        evidence -- gold proving TEXT appears in a retrieved chunk
        answer   -- the answer's own value appears in a retrieved chunk

    CAVEAT, stated plainly: the third signal is the loosest. A number can
    coincide by chance, so it is restricted to values >= 100 and should be read
    as a proxy, not proof. The union of three noisy indicators may overcount
    slightly -- but it is far closer to the truth than a test that calls a
    correct, citable answer a failure.
    """
    from eval.scoring import extract_numbers

    pages = {c.page_num for c in chunks}
    if pages & question.gold_pages:
        return True, "page"
    if _evidence_in_chunks(question, chunks):
        return True, "evidence"
    blob = _normalize(" ".join(c.text for c in chunks)).replace(" ", "")
    for v, _ in extract_numbers(question.answer):
        if abs(v) >= 100 and str(int(abs(v))) in blob:
            return True, "answer-value"
    return False, "none"


def evaluate_retrieval(
    retriever: Retriever,
    gold: list[GoldQuestion] | None = None,
    k: int = 5,
    corpus: dict[str, list[Chunk]] | None = None,
    verbose: bool = False,
) -> RetrievalReport:
    gold = gold if gold is not None else load_gold()
    if corpus is None:
        corpus = load_corpus({q.doc_name for q in gold})

    by_doc: dict[str, list[GoldQuestion]] = defaultdict(list)
    for q in gold:
        if q.doc_name in corpus:
            by_doc[q.doc_name].append(q)

    report = RetrievalReport(retriever=retriever.name, k=k)
    t0 = time.time()

    for doc_name, qs in by_doc.items():
        chunks = corpus[doc_name]
        retriever.index(doc_name, chunks)

        for q in qs:
            hits = retriever.search(doc_name, q.question, k)
            pages = [c.page_num for c in hits]
            gold_pages = q.gold_pages

            first_rank = None
            for i, p in enumerate(pages, 1):
                if p in gold_pages:
                    first_rank = i
                    break

            report.results.append(QuestionResult(
                qid=q.qid,
                doc_name=doc_name,
                is_reasoning=q.is_reasoning,
                gold_pages=gold_pages,
                retrieved_pages=pages,
                page_hit=bool(gold_pages & set(pages)),
                all_pages_hit=gold_pages.issubset(set(pages)),
                evidence_hit=_evidence_in_chunks(q, hits),
                first_hit_rank=first_rank,
            ))

        if verbose:
            print(f"  {doc_name}: {len(qs)} q, {len(chunks)} chunks")

    report.elapsed_s = time.time() - t0
    return report


def compare(retrievers: list[Retriever], ks=(1, 3, 5, 10), gold=None) -> None:
    """Run the ladder. This is the Day 3-4 driver -- results, not opinions."""
    gold = gold if gold is not None else load_gold()
    corpus = load_corpus({q.doc_name for q in gold})
    print(f"corpus: {len(corpus)} filings, {sum(len(v) for v in corpus.values())} chunks")
    print(f"gold  : {len(gold)} questions\n")
    for r in retrievers:
        for k in ks:
            print(evaluate_retrieval(r, gold=gold, k=k, corpus=corpus).render())
            print()
