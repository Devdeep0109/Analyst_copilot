"""
retrieval/baselines.py

Two deliberately dumb retrievers whose only job is to prove the harness works.

A metric you have never seen move is not a metric, it's a decoration. Before
trusting any number from BM25 on Day 3, we need to know the harness can tell
good from bad. So:

  RandomRetriever  -- seeded random chunks. This is the floor. Whatever BM25
                      scores on Day 3 is only meaningful as a distance above
                      this line.
  FirstKRetriever  -- always the first k chunks of the document. Sounds silly,
                      but it catches a specific bug class: if a "smart"
                      retriever ever ties this, it isn't ranking at all.

If a real retriever cannot clearly beat both, the bug is in the retriever --
or in the harness, and better to learn that now than on Day 4.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parser.chunker import Chunk  # noqa: E402


class RandomRetriever:
    """Uniformly random chunks. Seeded, so runs are reproducible."""

    def __init__(self, seed: int = 0):
        self.name = "random (floor)"
        self._rng = random.Random(seed)
        self._chunks: dict[str, list[Chunk]] = {}

    def index(self, doc_name: str, chunks: list[Chunk]) -> None:
        self._chunks[doc_name] = chunks

    def search(self, doc_name: str, query: str, k: int) -> list[Chunk]:
        chunks = self._chunks.get(doc_name, [])
        if not chunks:
            return []
        return self._rng.sample(chunks, min(k, len(chunks)))


class FirstKRetriever:
    """The first k chunks, ignoring the query entirely."""

    def __init__(self):
        self.name = "first-k (no ranking)"
        self._chunks: dict[str, list[Chunk]] = {}

    def index(self, doc_name: str, chunks: list[Chunk]) -> None:
        self._chunks[doc_name] = chunks

    def search(self, doc_name: str, query: str, k: int) -> list[Chunk]:
        return self._chunks.get(doc_name, [])[:k]


class KeywordOverlapRetriever:
    """Crude bag-of-words overlap -- no IDF, no length normalization.

    This is the sanity rung between "random" and real BM25. It should beat
    random clearly. If BM25 on Day 3 does NOT beat this by a wide margin,
    something is wrong with the BM25 implementation, because the entire value
    of BM25 over this is IDF weighting and length normalization.
    """

    def __init__(self):
        self.name = "keyword overlap (naive)"
        self._chunks: dict[str, list[Chunk]] = {}
        self._toks: dict[str, list[set[str]]] = {}

    @staticmethod
    def _tokenize(s: str) -> set[str]:
        import re
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    def index(self, doc_name: str, chunks: list[Chunk]) -> None:
        if doc_name in self._chunks:
            return
        self._chunks[doc_name] = chunks
        self._toks[doc_name] = [self._tokenize(c.text) for c in chunks]

    def search(self, doc_name: str, query: str, k: int) -> list[Chunk]:
        chunks = self._chunks.get(doc_name, [])
        if not chunks:
            return []
        q = self._tokenize(query)
        scored = [(len(q & t), i) for i, t in enumerate(self._toks[doc_name])]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [chunks[i] for score, i in scored[:k] if score > 0]
