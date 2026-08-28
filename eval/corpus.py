"""
eval/corpus.py

Parses + chunks filings once, caches to disk, and hands back Chunk objects.

Why this exists: parsing 72 filings takes minutes. Days 3-4 involve running the
eval loop over and over while swapping retrievers (BM25 vs dense vs hybrid vs
reranked). If every run re-parses, the iteration loop is too slow to actually
experiment, and slow experiments turn into guessed conclusions -- exactly what
the retrieval ladder is meant to avoid.

Cache lives in .cache/chunks/<doc_name>.json and is keyed on the source file's
mtime+size, so editing the parser or swapping a filing invalidates it
automatically.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parser.page_parser import parse_filing            # noqa: E402
from parser.chunker import Chunk, chunk_filing         # noqa: E402

FILINGS_DIR = PROJECT_ROOT / "data" / "filings"
CACHE_DIR = PROJECT_ROOT / ".cache" / "chunks"


def _cache_key(src: Path) -> str:
    st = src.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def _cache_path(doc_name: str) -> Path:
    return CACHE_DIR / f"{doc_name}.json"


def load_chunks(doc_name: str, use_cache: bool = True) -> list[Chunk]:
    """Return chunks for one filing, parsing only if the cache is cold/stale."""
    src = FILINGS_DIR / f"{doc_name}.htm"
    if not src.exists():
        raise FileNotFoundError(src)

    cp = _cache_path(doc_name)
    key = _cache_key(src)

    if use_cache and cp.exists():
        try:
            blob = json.loads(cp.read_text(encoding="utf-8"))
            if blob.get("key") == key:
                return [Chunk(**c) for c in blob["chunks"]]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # corrupt cache -> just rebuild

    filing = parse_filing(str(src), doc_name=doc_name)
    chunks = chunk_filing(filing)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp.write_text(
        json.dumps({"key": key, "chunks": [asdict(c) for c in chunks]}),
        encoding="utf-8",
    )
    return chunks


def load_corpus(
    doc_names,
    use_cache: bool = True,
    verbose: bool = False,
    warn_empty: bool = True,
) -> dict[str, list[Chunk]]:
    """doc_name -> chunks, for many filings.

    Warns loudly about filings that produce zero chunks. That happened once for
    real -- a filing was truncated during copy (11 MB -> 20 KB) and parsed to
    nothing. Nothing complained until rank_bm25 hit a division by zero deep in
    an eval run, which is the worst place to learn about it. A retrieval score
    silently computed over 71 filings instead of 72 is a corrupted measurement,
    so this must be visible.
    """
    out: dict[str, list[Chunk]] = {}
    names = sorted(set(doc_names))
    empty: list[str] = []
    for i, name in enumerate(names, 1):
        try:
            out[name] = load_chunks(name, use_cache=use_cache)
        except FileNotFoundError:
            if verbose:
                print(f"  [{i}/{len(names)}] MISSING {name}")
            continue
        if not out[name]:
            empty.append(name)
        if verbose:
            print(f"  [{i}/{len(names)}] {name}: {len(out[name])} chunks")

    if warn_empty and empty:
        print(f"  WARNING: {len(empty)} filing(s) produced ZERO chunks: {empty}")
        print("           Check the source file is complete "
              "(a truncated download parses to nothing).")
    return out


def warm_cache(doc_names=None) -> None:
    if doc_names is None:
        doc_names = [p.stem for p in FILINGS_DIR.glob("*.htm")]
    corpus = load_corpus(doc_names, verbose=True)
    total = sum(len(v) for v in corpus.values())
    print(f"\ncached {len(corpus)} filings, {total} chunks -> {CACHE_DIR}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="warm cache for every filing on disk")
    ap.add_argument("--doc", help="warm cache for one filing")
    args = ap.parse_args()

    if args.doc:
        warm_cache([args.doc])
    elif args.all:
        warm_cache()
    else:
        from eval.gold import load_gold
        warm_cache([q.doc_name for q in load_gold()])
