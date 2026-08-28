"""
retrieval/rerank.py

Cross-encoder reranking on top of the hybrid retriever.

WHY THIS IS THE BIGGEST REMAINING LEVER
---------------------------------------
Hybrid page_recall by depth, on the 127-question gold set:

    top-10    0.559     <- what we would ship without reranking
    top-20    0.701
    top-50    0.890
    top-100   0.921
    top-200   0.976

The right page is already in the top 50 for 89% of questions. Retrieval is not
failing to FIND it -- it is failing to RANK it. So reordering the top-50 down
to a top-10 has a ceiling of 0.890 against today's 0.559: roughly +0.33 of
available gain, versus the +0.03 that tuning fusion weights bought.

WHY A CROSS-ENCODER AND NOT A BETTER EMBEDDING
----------------------------------------------
Bi-encoders (what dense.py uses) embed the query and the chunk separately and
compare vectors. The chunk's vector is computed without ever seeing the
question, so it must summarize everything the chunk might be asked about into
one point. A cross-encoder reads query and chunk TOGETHER in one forward pass
and outputs a relevance score directly, so it can attend to the specific thing
being asked -- "which column is FY2018" is answerable jointly and not
separately.

The cost is that it cannot be precomputed: every (query, chunk) pair needs its
own forward pass. That is exactly why it is unusable over 49,582 chunks and
ideal over 50 candidates.

CACHING
-------
Scores are cached to disk keyed on (model, doc, query, chunk_id). The sweep
over depths 20/50/100 reuses everything computed at the largest depth, so the
whole sweep costs one pass at depth 100 rather than three separate passes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parser.chunker import Chunk                            # noqa: E402
from retrieval.hybrid import HybridRetriever                # noqa: E402

RERANK_CACHE = PROJECT_ROOT / ".cache" / "rerank"

MS_MARCO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_BASE = "BAAI/bge-reranker-base"

MODEL_NOTES = {
    MS_MARCO: "~90 MB. WINNER: d=0.86 separation, page@10 0.606 at depth 20",
    BGE_BASE: "~1.1 GB. WORSE despite 12x the size: d=0.60, page@10 0.567",
}

# MEASURED, not assumed -- the bigger model lost.
#
#   model                 size     effect size d   top-1   page@10 (depth 20)
#   ms-marco-MiniLM-L-6   ~90 MB       0.86        19.7%       0.606
#   bge-reranker-base     ~1.1 GB      0.60        10.3%       0.567
#
# bge-reranker-base is built on XLM-RoBERTa, a multilingual model whose 250k
# vocabulary is most of its weight and none of its usefulness on English SEC
# filings. ms-marco-MiniLM is English-only and trained directly on passage
# ranking, which is precisely this task. Bigger is not better when the extra
# capacity is spent on languages the corpus doesn't contain.
#
# DEPTH: 20 beats 50 and 100 consistently. Deeper reranking feeds the
# cross-encoder more candidates than its precision can sort, so noise from
# ranks 20-100 outweighs the extra recall it could recover.
BEST_MODEL = MS_MARCO
BEST_DEPTH = 20


def _load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "sentence-transformers is required for reranking.\n"
            "  pip install sentence-transformers\n"
            "(No extra install needed beyond what dense retrieval already uses.)"
        ) from e
    return CrossEncoder(model_name, max_length=512)


class RerankRetriever:
    """Hybrid retrieval, then cross-encoder rescoring of the top-N pages.

    `depth` is how many candidate pages get rescored. `search(k)` returns the
    best k after rescoring. depth >= k always; depth == k means no reranking
    benefit is possible, which makes it a useful control.
    """

    def __init__(
        self,
        model_name: str = MS_MARCO,
        depth: int = 20,          # measured optimum; see BEST_DEPTH above
        base: HybridRetriever | None = None,
        batch_size: int = 32,
        max_chars: int = 1800,
        name: str | None = None,
        show_progress: bool = False,
        # REVERTED TO FALSE -- MEASURED WORSE.
        #
        # The theory was sound: gold pages average 4.2 chunks and for 33% of
        # them the evidence sits in a chunk other than the representative, so
        # the cross-encoder was judging pages by the wrong text.
        #
        # Measured on 45 questions / 20 filings it lost on every metric:
        #     page_recall     0.689 -> 0.667
        #     all_pages       0.600 -> 0.578
        #     evidence_recall 0.756 -> 0.711
        #
        # The likely reason is a multiple-comparisons effect. Scoring a page as
        # "its best chunk" means a page with 10 chunks gets 10 draws at a high
        # score while a 2-chunk page gets 2. Long pages then win on luck rather
        # than relevance -- structurally the same trap as dense page_sum on
        # Day 3, where accumulating over more chunks rewarded length instead of
        # relevance.
        #
        # Fixing it properly needs length-normalised page scoring, not just
        # more chunks. Left switchable so that can be tried, but off by
        # default: the data says the simple version is better.
        #
        # CAVEAT ON THAT TEST: it also changed max_chars 1800 -> 600, so the
        # two effects are confounded. If this is revisited, change one at a
        # time.
        all_chunks_per_page: bool = False,
    ):
        self.model_name = model_name
        self.depth = depth
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.all_chunks_per_page = all_chunks_per_page
        self.show_progress = show_progress
        self.base = base or HybridRetriever(method="weighted", candidate_pool=50)
        short = model_name.split("/")[-1]
        self.name = name or f"Rerank {short} (depth={depth})"

        self._model = None
        self._chunks: dict[str, list[Chunk]] = {}
        self._cache: dict[str, float] | None = None
        self._dirty = False

    # --------------------------------------------------------------- model --

    @property
    def model(self):
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name)
        return self._model

    # --------------------------------------------------------------- cache --

    def _cache_path(self) -> Path:
        tag = hashlib.md5(self.model_name.encode()).hexdigest()[:8]
        return RERANK_CACHE / f"scores__{tag}.json"

    def _load_cache(self) -> None:
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

    def save_cache(self) -> None:
        if not self._dirty or self._cache is None:
            return
        RERANK_CACHE.mkdir(parents=True, exist_ok=True)
        self._cache_path().write_text(json.dumps(self._cache), encoding="utf-8")
        self._dirty = False

    @staticmethod
    def _key(query: str, chunk_id: str) -> str:
        return hashlib.md5(f"{query}||{chunk_id}".encode()).hexdigest()

    # ------------------------------------------------------------ indexing --

    def index(self, doc_name: str, chunks: list[Chunk]) -> None:
        self._chunks[doc_name] = chunks
        self.base.index(doc_name, chunks)

    # ----------------------------------------------------------- searching --

    def _expand_to_page_chunks(self, doc_name: str, candidates: list[Chunk]
                               ) -> list[Chunk]:
        """Replace each candidate with EVERY chunk on its page.

        THE BUG THIS FIXES
        ------------------
        The hybrid returns one representative chunk per page. Reranking only
        that chunk means the cross-encoder judges "is this page relevant?"
        while reading a piece of the page that may not contain the answer.
        Measured: gold pages average 4.2 chunks, and for 33% of them the
        proving evidence is in a DIFFERENT chunk from the representative.

        So a third of the time the reranker was scoring the wrong text and
        rejecting pages that did hold the answer. The model itself was fine --
        separation d=0.86 -- it was being shown the wrong input.

        Cost: ~4x more pairs to score. That is local CPU, not API quota.
        """
        chunks = self._chunks.get(doc_name, [])
        if not chunks:
            return candidates
        by_page: dict[int, list[Chunk]] = {}
        for c in chunks:
            by_page.setdefault(c.page_num, []).append(c)

        out: list[Chunk] = []
        seen: set[int] = set()
        for cand in candidates:
            if cand.page_num in seen:
                continue
            seen.add(cand.page_num)
            out.extend(by_page.get(cand.page_num, [cand]))
        return out

    def search(self, doc_name: str, query: str, k: int) -> list[Chunk]:
        depth = max(self.depth, k)
        candidates = self.base.search(doc_name, query, depth)
        if not candidates:
            return []
        if len(candidates) <= k and self.depth <= k:
            return candidates[:k]

        if self.all_chunks_per_page:
            candidates = self._expand_to_page_chunks(doc_name, candidates)

        self._load_cache()

        # Truncating to max_chars matters: the cross-encoder has a 512-token
        # window shared between query and passage. A 1200-char table chunk plus
        # a 100-word question can overflow it, and the model silently truncates
        # from the END -- dropping exactly the table rows that often hold the
        # answer. Truncating deliberately at least makes the behaviour known.
        pending_pairs, pending_keys = [], []
        for c in candidates:
            key = self._key(query, c.chunk_id)
            if key not in self._cache:
                pending_pairs.append([query, c.text[: self.max_chars]])
                pending_keys.append(key)

        if pending_pairs:
            scores = self.model.predict(
                pending_pairs,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress,
            )
            for key, s in zip(pending_keys, scores):
                self._cache[key] = float(s)
            self._dirty = True

        scored = [(self._cache[self._key(query, c.chunk_id)], i, c)
                  for i, c in enumerate(candidates)]
        # Tie-break on the base retriever's order, so the reranker can only
        # help or leave things alone -- never scramble equal scores randomly.
        scored.sort(key=lambda t: (-t[0], t[1]))

        if not self.all_chunks_per_page:
            return [c for _, _, c in scored[:k]]

        # A PAGE is worth its BEST chunk. Without this, one strong page would
        # occupy several of the k slots with its own chunks and crowd out other
        # pages -- the same mistake plain top-k chunk retrieval made on Day 3,
        # which cost 0.236 vs 0.315. Return the winning chunk per page so the
        # k slots hold k independent pages.
        out: list[Chunk] = []
        seen: set[int] = set()
        for _, _, c in scored:
            if c.page_num in seen:
                continue
            seen.add(c.page_num)
            out.append(c)
            if len(out) >= k:
                break
        return out


def cache_stats(model_name: str = MS_MARCO) -> str:
    r = RerankRetriever(model_name=model_name)
    r._load_cache()
    p = r._cache_path()
    size = p.stat().st_size / 1e6 if p.exists() else 0
    return f"{len(r._cache)} cached pair-scores ({size:.1f} MB)"
