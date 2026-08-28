"""
retrieval/dense.py

Dense (embedding) retrieval over a filing's chunks.

WHAT THIS IS FOR
----------------
BM25's failures on Day 3 were not tuning failures, they were vocabulary
failures. "Is 3M a capital-intensive business?" shares essentially no terms
with the pages that answer it -- those pages say "Purchases of property, plant
and equipment", "Total assets", "Net sales". No lexical method reaches them.

Dense retrieval embeds text into a vector space where meaning, not wording,
determines closeness. That is the specific gap it exists to close, which gives
us a falsifiable prediction to check rather than a vague hope:

    dense should beat BM25 on REASONING questions and roughly tie on
    EXTRACTIVE ones.

If dense wins uniformly across both, be suspicious -- that pattern usually
means the comparison is unfair somewhere, not that embeddings are magic.

DESIGN NOTES
------------
* Aggregation mirrors BM25's winner exactly (page-sum with sqrt normalization),
  so the comparison isolates lexical-vs-semantic and nothing else. Changing two
  things at once would make the result uninterpretable.
* Embeddings are cached to .cache/embeddings/ keyed on model + source file
  mtime. The first full run costs 10-25 min on CPU; every run after is seconds.
* Similarity is a plain dot product because embeddings are L2-normalized at
  encode time, which makes dot product identical to cosine similarity and
  avoids recomputing norms on every query.
* BGE models were trained with an asymmetric query prefix. Omitting it costs
  real accuracy, and it is a silent failure -- the model still returns
  plausible-looking results. `QUERY_PREFIXES` handles it per model family.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parser.chunker import Chunk                    # noqa: E402
from retrieval.tokenize import QUESTION_BOILERPLATE  # noqa: E402


def strip_boilerplate(query: str) -> str:
    """Remove FinanceBench's question framing before embedding.

    A typical question is 40 words of which ~25 are "Assume that you are a
    public equities analyst. Answer the following question by primarily using
    information that is shown in the balance sheet." An embedding averages over
    all of it, so the framing -- identical across most questions -- dominates
    the vector and pushes every query toward the same point in space. BM25
    barely cares (IDF already discounts terms that appear everywhere); dense
    care a great deal, because it has no IDF.
    """
    import re as _re
    words = _re.findall(r"[\w$%.,()-]+", query)
    kept = [w for w in words if w.strip(".,()").lower() not in QUESTION_BOILERPLATE]
    out = " ".join(kept).strip()
    return out if len(out.split()) >= 3 else query

EMBED_CACHE = PROJECT_ROOT / ".cache" / "embeddings"
FILINGS_DIR = PROJECT_ROOT / "data" / "filings"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Models trained with asymmetric query/passage encoding. Getting this wrong
# degrades results quietly rather than erroring.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-small-v2": "query: ",
    "intfloat/e5-base-v2": "query: ",
}
PASSAGE_PREFIXES = {
    "intfloat/e5-small-v2": "passage: ",
    "intfloat/e5-base-v2": "passage: ",
}


def _load_model(model_name: str):
    """Imported lazily so the rest of the project runs without torch installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "sentence-transformers is not installed.\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install sentence-transformers\n"
            "See SETUP.md."
        ) from e
    return SentenceTransformer(model_name)


def _cache_path(doc_name: str, model_name: str) -> Path:
    tag = hashlib.md5(model_name.encode()).hexdigest()[:8]
    return EMBED_CACHE / f"{doc_name}__{tag}.npz"


def _source_key(doc_name: str) -> str:
    src = FILINGS_DIR / f"{doc_name}.htm"
    if not src.exists():
        return "nosrc"
    st = src.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


class DenseRetriever:
    """Embedding retrieval with the same page-sum policy BM25 settled on."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        aggregate: str = "page_max",     # "page_sum" | "page_max" | "chunk"
        length_penalty: bool = True,
        batch_size: int = 64,
        name: str | None = None,
        show_progress: bool = False,
        strip_query_boilerplate: bool = True,
        topn_pool: int = 0,
    ):
        self.model_name = model_name
        self.aggregate = aggregate
        self.length_penalty = length_penalty
        self.batch_size = batch_size
        self.show_progress = show_progress
        # BM25 strips FinanceBench's question framing and measurably gained from
        # it. Dense was not given the same treatment in the first run, which
        # made that comparison unfair to dense -- the embedding of a 40-word
        # question that is mostly "Assume that you are a public equities
        # analyst..." is dominated by the framing, not the actual question.
        self.strip_query_boilerplate = strip_query_boilerplate
        # Sum only over the top-N chunks of a page rather than all of them.
        # 0 = all. See the comment in `search` for why this matters for cosine.
        self.topn_pool = topn_pool
        short = model_name.split("/")[-1]
        suffix = aggregate
        if aggregate == "page_sum":
            if topn_pool:
                suffix += f"-top{topn_pool}"
            if length_penalty:
                suffix += "/norm"
        self.name = name or f"Dense {short} ({suffix})"

        self._model = None
        self._chunks: dict[str, list[Chunk]] = {}
        self._emb: dict[str, np.ndarray] = {}
        self._query_cache: dict[str, np.ndarray] = {}
        self._disk_queries: dict[str, np.ndarray] | None = None

    # -------------------------------------------------------------- model --

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    def _encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        prefix = ""
        if is_query:
            prefix = QUERY_PREFIXES.get(self.model_name, "")
        else:
            prefix = PASSAGE_PREFIXES.get(self.model_name, "")
        if prefix:
            texts = [prefix + t for t in texts]
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,       # makes dot product == cosine
            show_progress_bar=self.show_progress,
        ).astype(np.float32)

    # ------------------------------------------------------------ indexing --

    def index(self, doc_name: str, chunks: list[Chunk]) -> None:
        if doc_name in self._emb:
            return
        self._chunks[doc_name] = chunks

        # Same guard as BM25: a filing that parsed to nothing must degrade to
        # "no results", not crash the run. sentence-transformers on an empty
        # list returns a 0-d array that breaks the matmul downstream.
        if not chunks:
            self._emb[doc_name] = np.zeros((0, 1), dtype=np.float32)
            return

        cp = _cache_path(doc_name, self.model_name)
        key = f"{_source_key(doc_name)}-{len(chunks)}"

        if cp.exists():
            try:
                blob = np.load(cp, allow_pickle=False)
                if str(blob["key"]) == key and blob["emb"].shape[0] == len(chunks):
                    self._emb[doc_name] = blob["emb"]
                    return
            except Exception:
                pass  # stale or corrupt -> re-embed

        emb = self._encode([c.text for c in chunks], is_query=False)
        EMBED_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, emb=emb, key=np.array(key))
        self._emb[doc_name] = emb

    # ----------------------------------------------------------- searching --

    # ------------------------------------------------- query vector cache --
    #
    # Query embeddings are cached to DISK, not just in memory. That sounds like
    # a micro-optimization; it isn't. Loading the model requires torch, so any
    # experiment touching dense retrieval otherwise requires the full ML stack.
    # With the 127 gold questions cached as a small .npz, every fusion and
    # tuning experiment on Day 4 becomes pure numpy over cached vectors -- it
    # runs anywhere, in seconds, with no model load at all.

    def _query_cache_path(self) -> Path:
        tag = hashlib.md5(self.model_name.encode()).hexdigest()[:8]
        return EMBED_CACHE / f"_queries__{tag}.npz"

    def _load_query_cache(self) -> None:
        if self._disk_queries is not None:
            return
        self._disk_queries = {}
        p = self._query_cache_path()
        if p.exists():
            try:
                blob = np.load(p, allow_pickle=False)
                keys = [str(k) for k in blob["keys"]]
                vecs = blob["vecs"]
                self._disk_queries = {k: vecs[i] for i, k in enumerate(keys)}
            except Exception:
                self._disk_queries = {}

    def save_query_cache(self, queries: list[str]) -> int:
        """Embed and persist these queries (raw text; prefixing/stripping is
        applied internally exactly as at search time)."""
        self._load_query_cache()
        todo = []
        for q in queries:
            key = strip_boilerplate(q) if self.strip_query_boilerplate else q
            if key not in self._disk_queries:
                todo.append(key)
        # Also store the un-stripped form, so a retriever configured either way
        # can be evaluated from the same cache file.
        for q in queries:
            if q not in self._disk_queries and q not in todo:
                todo.append(q)
        if todo:
            vecs = self._encode(todo, is_query=True)
            for k, v in zip(todo, vecs):
                self._disk_queries[k] = v
        EMBED_CACHE.mkdir(parents=True, exist_ok=True)
        keys = list(self._disk_queries)
        np.savez_compressed(
            self._query_cache_path(),
            keys=np.array(keys),
            vecs=np.stack([self._disk_queries[k] for k in keys]).astype(np.float32),
        )
        return len(todo)

    def _embed_query(self, query: str) -> np.ndarray:
        if self.strip_query_boilerplate:
            query = strip_boilerplate(query)
        if query in self._query_cache:
            return self._query_cache[query]

        self._load_query_cache()
        if query in self._disk_queries:
            v = self._disk_queries[query]
        else:
            v = self._encode([query], is_query=True)[0]
        self._query_cache[query] = v
        return v

    def search(self, doc_name: str, query: str, k: int) -> list[Chunk]:
        chunks = self._chunks.get(doc_name, [])
        emb = self._emb.get(doc_name)
        if not chunks or emb is None or len(chunks) == 0:
            return []

        qv = self._embed_query(query)
        scores = emb @ qv          # cosine, since both sides are normalized

        if self.aggregate == "chunk":
            order = np.argsort(-scores)[:k]
            return [chunks[i] for i in order]

        # WHY page_sum IS THE WRONG DEFAULT FOR COSINE (learned the hard way)
        # -------------------------------------------------------------------
        # BM25 scores are SPARSE: a chunk sharing no query terms scores exactly
        # 0, so summing a page adds up only the chunks that actually matched.
        # Cosine similarity is DENSE: every chunk scores something positive,
        # typically 0.0-0.4 even when completely irrelevant. Summing therefore
        # accumulates noise proportional to page length, and dividing by
        # sqrt(n) only half-corrects it -- the result ranks pages mostly by
        # "long page of vaguely financial text", which describes every page of
        # a 10-K.
        #
        # This is why the first dense run scored 0.283 at k=10 against BM25's
        # 0.480. The aggregation was chosen for parity with BM25, and parity
        # was exactly the mistake: the same policy means different things when
        # applied to sparse vs dense score distributions.
        #
        # page_max ("a page is as good as its single best chunk") is the
        # appropriate analogue, and is now the default.
        pos = np.clip(scores, 0.0, None)

        page_idx: dict[int, list[int]] = {}
        for i, c in enumerate(chunks):
            page_idx.setdefault(c.page_num, []).append(i)

        page_score: dict[int, float] = {}
        page_best: dict[int, int] = {}
        for p, idxs in page_idx.items():
            vals = [(float(pos[i]), i) for i in idxs]
            vals.sort(key=lambda t: -t[0])
            page_best[p] = vals[0][1]

            if self.aggregate == "page_max":
                page_score[p] = vals[0][0]
            else:  # page_sum
                pool = vals[: self.topn_pool] if self.topn_pool else vals
                s = sum(v for v, _ in pool)
                if self.length_penalty:
                    s /= len(pool) ** 0.5
                page_score[p] = s

        top = sorted(page_score, key=lambda p: (-page_score[p], p))[:k]
        return [chunks[page_best[p]] for p in top if page_score[p] > 0]


def cache_status(doc_names: list[str], model_name: str = DEFAULT_MODEL) -> str:
    """How much of the embedding work is already done."""
    done = sum(1 for d in doc_names if _cache_path(d, model_name).exists())
    size_mb = sum(
        _cache_path(d, model_name).stat().st_size
        for d in doc_names
        if _cache_path(d, model_name).exists()
    ) / 1e6
    return f"{done}/{len(doc_names)} filings embedded ({size_mb:.0f} MB cached)"
