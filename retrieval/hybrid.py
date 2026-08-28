"""
retrieval/hybrid.py

Fuses BM25 and dense retrieval.

WHY THIS IS WORTH BUILDING (measured, not assumed)
--------------------------------------------------
At k=10 on the 127-question gold set, with both retrievers individually tuned:

    both found it   43 (33.9%)
    BM25 only       18 (14.2%)
    dense only      24 (18.9%)
    neither         42 (33.1%)

    best single method   0.528
    union ceiling        0.669   -> +0.142 headroom

The two methods fail on genuinely different questions, and the split by
question type says why:

    extractive (n=101)   BM25 0.495   dense 0.584
    reasoning  (n=26)    BM25 0.423   dense 0.308

Extractive questions are paraphrase problems -- "capital expenditure" must find
"Purchases of property, plant and equipment", which is what embeddings do well.
Reasoning questions need several specific statement pages, and what pins those
down is rare literal tokens (company name, fiscal year, "Total assets"), which
is what IDF does well. So the complementarity is structural, not noise, and
that is the precondition for fusion being worth the complexity.

TWO FUSION STRATEGIES
---------------------
RRF (Reciprocal Rank Fusion)
    score(page) = sum over retrievers of weight / (rank_constant + rank)
    Uses only RANKS, never raw scores. That matters because BM25 scores are
    unbounded sums of IDF terms while cosine sits in [0,1] -- they are not
    comparable quantities, and any attempt to add them directly is comparing
    apples to a different fruit entirely. RRF sidesteps the problem instead of
    pretending to solve it, and has one interpretable knob.

Weighted score fusion
    Min-max normalizes each retriever's scores per query, then blends. Keeps
    magnitude information that RRF discards -- a page BM25 scores far above
    everything else stays far above after normalization, where RRF flattens it
    to "rank 1". The cost is fragility: per-query min-max is unstable when one
    retriever returns a near-flat score distribution, which happens whenever a
    query has no strong lexical match at all.

Both are implemented and measured. RRF is the default only if it wins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parser.chunker import Chunk                               # noqa: E402
from retrieval.bm25 import PageSumBM25                         # noqa: E402
from retrieval.dense import DEFAULT_MODEL, DenseRetriever      # noqa: E402


def _minmax(d: dict[int, float]) -> dict[int, float]:
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


class HybridRetriever:
    """BM25 + dense, fused at the PAGE level.

    Fusing pages rather than chunks is deliberate: the two retrievers pick
    different best-chunks for the same page, so chunk-level fusion would treat
    one page as two separate candidates and split its own vote.
    """

    def __init__(
        self,
        method: str = "rrf",              # "rrf" | "weighted"
        w_bm25: float = 1.0,
        w_dense: float = 1.0,
        rrf_k: int = 60,
        candidate_pool: int = 50,
        k1: float = 0.6,
        b: float = 0.0,
        model_name: str = DEFAULT_MODEL,
        name: str | None = None,
    ):
        self.method = method
        self.w_bm25 = w_bm25
        self.w_dense = w_dense
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

        self.bm25 = PageSumBM25(k1=k1, b=b)
        self.dense = DenseRetriever(model_name=model_name, aggregate="page_max")

        if name:
            self.name = name
        elif method == "rrf":
            self.name = f"Hybrid RRF (w={w_bm25}/{w_dense}, rrf_k={rrf_k})"
        else:
            self.name = f"Hybrid weighted (w={w_bm25}/{w_dense})"

        self._chunks: dict[str, list[Chunk]] = {}

    def index(self, doc_name: str, chunks: list[Chunk]) -> None:
        self._chunks[doc_name] = chunks
        self.bm25.index(doc_name, chunks)
        self.dense.index(doc_name, chunks)

    # ------------------------------------------------------ page scoring --

    def _bm25_pages(self, doc_name: str, query: str) -> tuple[dict[int, float], dict[int, int]]:
        from retrieval.tokenize import tokenize_query
        chunks = self._chunks[doc_name]
        idx = self.bm25._index.get(doc_name)
        if idx is None or not chunks:
            return {}, {}
        q = tokenize_query(query, drop_question_boilerplate=True)
        if not q:
            return {}, {}
        scores = idx.get_scores(q)

        page_total: dict[int, float] = {}
        page_count: dict[int, int] = {}
        page_best: dict[int, tuple[float, int]] = {}
        for i, c in enumerate(chunks):
            p = c.page_num
            page_total[p] = page_total.get(p, 0.0) + scores[i]
            page_count[p] = page_count.get(p, 0) + 1
            if scores[i] > page_best.get(p, (-1.0, -1))[0]:
                page_best[p] = (scores[i], i)

        final = {p: page_total[p] / (page_count[p] ** 0.5) for p in page_total
                 if page_total[p] > 0}
        return final, {p: page_best[p][1] for p in final}

    def _dense_pages(self, doc_name: str, query: str) -> tuple[dict[int, float], dict[int, int]]:
        chunks = self._chunks[doc_name]
        emb = self.dense._emb.get(doc_name)
        if emb is None or emb.shape[0] == 0 or not chunks:
            return {}, {}
        qv = self.dense._embed_query(query)
        sims = np.clip(emb @ qv, 0.0, None)

        page_score: dict[int, float] = {}
        page_best: dict[int, int] = {}
        for i, c in enumerate(chunks):
            s = float(sims[i])
            if s > page_score.get(c.page_num, -1.0):
                page_score[c.page_num] = s
                page_best[c.page_num] = i
        return page_score, page_best

    # ------------------------------------------------------------ search --

    def search(self, doc_name: str, query: str, k: int) -> list[Chunk]:
        chunks = self._chunks.get(doc_name, [])
        if not chunks:
            return []

        b_scores, b_best = self._bm25_pages(doc_name, query)
        d_scores, d_best = self._dense_pages(doc_name, query)
        if not b_scores and not d_scores:
            return []

        # Truncate each side to its own top-N before fusing. Beyond that depth
        # a retriever's ordering is noise, and letting it vote there just adds
        # random pages that the other retriever must outrank.
        def topn(d: dict[int, float]) -> dict[int, float]:
            if not self.candidate_pool or len(d) <= self.candidate_pool:
                return d
            keep = sorted(d, key=lambda p: -d[p])[: self.candidate_pool]
            return {p: d[p] for p in keep}

        b_scores, d_scores = topn(b_scores), topn(d_scores)

        fused: dict[int, float] = {}
        if self.method == "rrf":
            for scores, w in ((b_scores, self.w_bm25), (d_scores, self.w_dense)):
                for rank, p in enumerate(sorted(scores, key=lambda x: -scores[x]), 1):
                    fused[p] = fused.get(p, 0.0) + w / (self.rrf_k + rank)
        else:
            bn, dn = _minmax(b_scores), _minmax(d_scores)
            for p in set(bn) | set(dn):
                fused[p] = self.w_bm25 * bn.get(p, 0.0) + self.w_dense * dn.get(p, 0.0)

        top = sorted(fused, key=lambda p: (-fused[p], p))[:k]

        # Represent each page by the better of the two retrievers' chosen
        # chunks -- prefer whichever side ranked that page higher.
        out: list[Chunk] = []
        for p in top:
            if p in d_best and (p not in b_best or d_scores.get(p, 0) >= b_scores.get(p, 0)):
                out.append(chunks[d_best[p]])
            elif p in b_best:
                out.append(chunks[b_best[p]])
            elif p in d_best:
                out.append(chunks[d_best[p]])
        return out


class OracleUnion:
    """Upper bound: counts a hit if EITHER retriever found it.

    Not a real retriever -- it is the union ceiling made runnable, so the
    harness reports the same number the complementarity table predicts. Any
    real fusion must land between the best single method and this line, and
    seeing both bounds in the same table keeps 'hybrid helped' honest.
    """

    def __init__(self, k1: float = 0.6, b: float = 0.0, model_name: str = DEFAULT_MODEL):
        self.name = "ORACLE union (upper bound, not a real retriever)"
        self.bm25 = PageSumBM25(k1=k1, b=b)
        self.dense = DenseRetriever(model_name=model_name, aggregate="page_max")

    def index(self, doc_name: str, chunks: list[Chunk]) -> None:
        self.bm25.index(doc_name, chunks)
        self.dense.index(doc_name, chunks)

    def search(self, doc_name: str, query: str, k: int) -> list[Chunk]:
        a = self.bm25.search(doc_name, query, k)
        b = self.dense.search(doc_name, query, k)
        seen, out = set(), []
        for c in a + b:
            if c.page_num not in seen:
                seen.add(c.page_num)
                out.append(c)
        return out
