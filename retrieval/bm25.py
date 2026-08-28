"""
retrieval/bm25.py

BM25 over a filing's chunks.

ON THE IMPLEMENTATION -- AND A CORRECTION
-----------------------------------------
This file originally claimed the builtin and `rank_bm25` implement "the same
scoring formula" and defaulted to rank_bm25 when installed. That was WRONG, and
it produced two different numbers for the same experiment on two machines:
page_recall@10 of 0.480 with rank_bm25 versus 0.496 with the builtin.

They differ in how IDF handles very common terms:

    builtin      idf = ln(1 + (N - n + 0.5) / (n + 0.5))
    rank_bm25    idf = ln(N - n + 0.5) - ln(n + 0.5), and where that goes
                 NEGATIVE (any term in more than ~half the documents) it is
                 replaced by epsilon * average_idf, with epsilon = 0.25.

On ordinary prose the difference is negligible, because few terms appear in
half the documents. On SEC filings it is not: within a single filing, terms
like "total", "net", "december", the company's own name, and the fiscal year
appear on most pages. That is exactly the region where the two formulas
disagree, so the choice is load-bearing here in a way it usually isn't.

The builtin is now the DEFAULT (`use_external=False`) for one reason:
reproducibility. An eval harness whose numbers depend on which packages happen
to be installed is not an eval harness. rank_bm25 remains available via
`use_external=True` for comparison, and `test_bm25_equivalence()` now reports
the gap honestly instead of asserting it away.

WHY BM25 SHOULD DO WELL HERE (AND WHERE IT WON'T)
-------------------------------------------------
Financial questions often quote the line item nearly verbatim -- "capital
expenditure" is literally "Purchases of property, plant and equipment (PP&E)"
on the cash flow statement, and the question usually names the company, the
fiscal year, and the statement. That is a lexical match problem, and IDF is
exactly the right tool: "expenditure" is rare and discriminating, "total" is
not.

Where it must fail is vocabulary mismatch. "Is 3M capital-intensive?" shares
almost no terms with the three statements needed to answer it. No amount of
BM25 tuning fixes that -- it is a semantics problem, and it is the specific
gap dense retrieval has to close on the next rung of the ladder.

PARAMETERS
----------
k1 controls term-frequency saturation, b controls length normalization.
Defaults (1.5, 0.75) are the standard starting point. `sweep_params()` in
eval/run_bm25.py tunes them against the harness rather than trusting folklore.
Note b matters unusually much here: chunks range from one-line table rows to
1200-character prose windows, so length normalization is doing real work.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parser.chunker import Chunk                              # noqa: E402
from retrieval.tokenize import tokenize, tokenize_query       # noqa: E402

try:
    from rank_bm25 import BM25Okapi as _ExternalBM25
    HAVE_RANK_BM25 = True
except ImportError:                                            # pragma: no cover
    _ExternalBM25 = None
    HAVE_RANK_BM25 = False


class SimpleBM25:
    """BM25Okapi. Same formula rank_bm25 uses, no dependency.

        score(q, d) = sum over terms t in q of
            IDF(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d|/avgdl))

        IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

    The +1 inside the log is what keeps IDF non-negative for terms appearing in
    more than half the documents -- without it, common terms get negative
    weight and can actively push the right chunk *down* the ranking.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size else 0.0

        self.doc_freqs: list[Counter] = []
        df: Counter = Counter()
        for doc in corpus:
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            df.update(freqs.keys())

        self.idf: dict[str, float] = {
            term: math.log(1 + (self.corpus_size - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        if not self.avgdl:
            return scores
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores


class BM25Retriever:
    """Per-filing BM25. Indexes lazily and caches, so re-running the harness
    across many k values doesn't rebuild the same index each time."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        drop_question_boilerplate: bool = True,
        use_external: bool = False,   # see module docstring: reproducibility
        name: str | None = None,
    ):
        self.k1 = k1
        self.b = b
        self.drop_question_boilerplate = drop_question_boilerplate
        self.use_external = use_external and HAVE_RANK_BM25
        backend = "rank_bm25" if self.use_external else "builtin"
        self.name = name or f"BM25 ({backend}, k1={k1}, b={b})"
        self._chunks: dict[str, list[Chunk]] = {}
        self._index: dict[str, object] = {}

    def index(self, doc_name: str, chunks: list[Chunk]) -> None:
        if doc_name in self._index:
            return
        self._chunks[doc_name] = chunks

        # A filing can legitimately yield zero chunks -- a truncated download,
        # an 8-K in the raw-SGML wrapper format, an unparseable upload. Guard
        # here rather than letting it propagate: rank_bm25 raises
        # ZeroDivisionError on an empty corpus (it computes num_doc /
        # corpus_size unguarded), which would take down a whole eval run over
        # one bad file. In the product this must degrade to "no results for
        # that filing", never a crash.
        if not chunks:
            self._index[doc_name] = None
            return

        tokenized = [tokenize(c.text) for c in chunks]
        if self.use_external:
            self._index[doc_name] = _ExternalBM25(tokenized, k1=self.k1, b=self.b)
        else:
            self._index[doc_name] = SimpleBM25(tokenized, k1=self.k1, b=self.b)

    def search(self, doc_name: str, query: str, k: int) -> list[Chunk]:
        chunks = self._chunks.get(doc_name, [])
        idx = self._index.get(doc_name)
        if not chunks or idx is None:
            return []
        q = tokenize_query(query, drop_question_boilerplate=self.drop_question_boilerplate)
        if not q:
            return []
        scores = idx.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [chunks[i] for i in ranked[:k] if scores[i] > 0]


class PageDiversifiedBM25(BM25Retriever):
    """BM25, but the top-k are k DISTINCT pages rather than k chunks.

    THE PROBLEM THIS FIXES
    ----------------------
    A filing averages ~675 chunks, and a financial statement page can be 20+
    table chunks that all share the same vocabulary. Plain top-5 therefore
    often spends all five slots on five chunks of ONE page. If that page is
    wrong, the query gets no second guess -- and page_recall@5 is effectively
    page_recall@1.

    Taking the best chunk from each of the top k distinct pages spends the same
    budget on five independent hypotheses. Since the product cites a PAGE and
    the rubric scores a PAGE, page-level diversity is the unit that matters.

    Same k, same scores, same tokens -- only the selection policy differs, so
    any gain is attributable to that alone.
    """

    def __init__(self, *args, **kwargs):
        explicit_name = kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        self.name = explicit_name or f"BM25 page-diversified (k1={self.k1}, b={self.b})"

    def search(self, doc_name: str, query: str, k: int) -> list[Chunk]:
        chunks = self._chunks.get(doc_name, [])
        idx = self._index.get(doc_name)
        if not chunks or idx is None:
            return []
        q = tokenize_query(query, drop_question_boilerplate=self.drop_question_boilerplate)
        if not q:
            return []
        scores = idx.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))

        out: list[Chunk] = []
        seen_pages: set[int] = set()
        for i in ranked:
            if scores[i] <= 0:
                break
            page = chunks[i].page_num
            if page in seen_pages:
                continue
            seen_pages.add(page)
            out.append(chunks[i])
            if len(out) >= k:
                break
        return out


class PageSumBM25(BM25Retriever):
    """Score PAGES by summing their chunks' scores, then return the top-k pages.

    Different hypothesis from PageDiversifiedBM25: there, a page is as good as
    its single best chunk. Here, a page that matches the query weakly across
    many chunks outranks one with a single strong hit.

    Which is right is an empirical question -- a balance sheet page spreads the
    query's terms over many rows (favouring sum), while a specific line item
    lives in one row (favouring max). Both are measured; neither is assumed.

    `length_penalty` divides by sqrt(number of chunks on the page), so long
    pages don't win purely by having more chunks to accumulate over.
    """

    def __init__(self, *args, length_penalty: bool = True, **kwargs):
        explicit_name = kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        self.length_penalty = length_penalty
        self.name = explicit_name or (
            f"BM25 page-sum{'/norm' if length_penalty else ''} "
            f"(k1={self.k1}, b={self.b})")

    def search(self, doc_name: str, query: str, k: int) -> list[Chunk]:
        chunks = self._chunks.get(doc_name, [])
        idx = self._index.get(doc_name)
        if not chunks or idx is None:
            return []
        q = tokenize_query(query, drop_question_boilerplate=self.drop_question_boilerplate)
        if not q:
            return []
        scores = idx.get_scores(q)

        page_score: dict[int, float] = {}
        page_count: dict[int, int] = {}
        page_best: dict[int, tuple[float, int]] = {}
        for i, c in enumerate(chunks):
            p = c.page_num
            page_score[p] = page_score.get(p, 0.0) + scores[i]
            page_count[p] = page_count.get(p, 0) + 1
            if scores[i] > page_best.get(p, (-1.0, -1))[0]:
                page_best[p] = (scores[i], i)

        def final(p: int) -> float:
            s = page_score[p]
            return s / math.sqrt(page_count[p]) if self.length_penalty else s

        top_pages = sorted(page_score, key=lambda p: (-final(p), p))[:k]
        return [chunks[page_best[p][1]] for p in top_pages if page_score[p] > 0]


def test_bm25_equivalence(n_docs: int = 200, k: int = 10) -> bool:
    """Report how far the builtin and rank_bm25 diverge on real chunks.

    This does NOT assert they agree -- they don't, and pretending otherwise is
    what let a 0.016 discrepancy in page_recall go unnoticed across two
    machines. It exists to make the size of the gap visible.
    """
    if not HAVE_RANK_BM25:
        print("rank_bm25 not installed -- builtin in use (which is the default anyway).")
        return True

    from eval.corpus import load_chunks

    chunks = load_chunks("3M_2018_10K")[:n_docs]
    tokenized = [tokenize(c.text) for c in chunks]
    q = tokenize_query("What is the FY2018 capital expenditure amount for 3M?")

    a = _ExternalBM25(tokenized, k1=1.5, b=0.75).get_scores(q)
    b = SimpleBM25(tokenized, k1=1.5, b=0.75).get_scores(q)

    rank_a = sorted(range(len(a)), key=lambda i: (-a[i], i))[:k]
    rank_b = sorted(range(len(b)), key=lambda i: (-b[i], i))[:k]
    max_delta = max((abs(x - y) for x, y in zip(a, b)), default=0.0)

    same = rank_a == rank_b
    print(f"top-{k} ranking identical : {same}")
    print(f"max score delta          : {max_delta:.6f}")
    if not same:
        print(f"  rank_bm25: {rank_a}")
        print(f"  builtin  : {rank_b}")
    return same


if __name__ == "__main__":
    print(f"rank_bm25 installed: {HAVE_RANK_BM25}\n")
    test_bm25_equivalence()
