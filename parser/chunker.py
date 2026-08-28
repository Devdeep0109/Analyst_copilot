"""
parser/chunker.py

Turns a ParsedFiling (from page_parser.py) into retrieval-ready Chunks.

Design decisions:
  - Every table on a page becomes its OWN chunk, rendered via ParsedTable.to_text()
    (pipe-delimited grid: "Net sales | $34,229 | $35,355 | $32,184"). This is the
    single most important decision in this file: it's what keeps a number
    attached to its line-item label AND its year/period column when the chunk
    later gets embedded/searched independently of the rest of the page. Splitting
    a table across chunk boundaries (or losing column headers) is the #1 way a
    naive chunker produces a plausible-looking but wrong number.
  - Prose (non-table) text is split into sentence-aware windows, not fixed-size
    character windows, to avoid cutting a sentence (and therefore a fact) in half
    at an arbitrary boundary.
  - Every chunk carries doc_name + page_num, since that's what a citation needs.
  - Table text is intentionally ALSO present inside the page's flattened prose
    (see page_parser.py) which means table content can appear twice, once as a
    structured table chunk and once folded into a prose chunk. This is a known
    v1 tradeoff, not an oversight: redundancy costs a bit of extra index size but
    gives retrieval two different representations to match against (a raw number
    search may hit the flattened prose form; a "row label + value" style query
    may hit the structured table form). Revisit only if eval numbers show it's
    actually hurting precision.
"""

from __future__ import annotations
from dataclasses import dataclass
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from parser.page_parser import ParsedFiling, Page, ParsedTable


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    page_num: int          # OUR page numbering (see page_parser docstring re: +1 offset vs PDF)
    chunk_type: str        # "table" or "prose"
    text: str

    def contains(self, snippet: str) -> bool:
        from parser.page_parser import _normalize
        return _normalize(snippet) in _normalize(self.text)


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_prose(text: str, max_chars: int = 1200, overlap_sentences: int = 1) -> list[str]:
    """Split prose into windows of up to max_chars, breaking only at sentence
    boundaries. Keeps a small sentence overlap between consecutive chunks so a
    fact split across a boundary sentence still has a chance of appearing whole
    in at least one chunk."""
    if not text.strip():
        return []

    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    chunks = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > max_chars and current:
            chunks.append(" ".join(current))
            # start next window with overlap from the tail of the previous one
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_page(page: Page, doc_name: str, max_chars: int = 1200) -> list[Chunk]:
    chunks: list[Chunk] = []

    for i, table in enumerate(page.tables):
        table_text = table.to_text()
        if table_text.strip():
            chunks.append(Chunk(
                chunk_id=f"{doc_name}_p{page.page_num}_table{i}",
                doc_name=doc_name,
                page_num=page.page_num,
                chunk_type="table",
                text=table_text,
            ))

    prose_windows = _split_prose(page.text, max_chars=max_chars)
    for j, t in enumerate(prose_windows):
        chunks.append(Chunk(
            chunk_id=f"{doc_name}_p{page.page_num}_prose{j}",
            doc_name=doc_name,
            page_num=page.page_num,
            chunk_type="prose",
            text=t,
        ))

    return chunks


def chunk_filing(filing: ParsedFiling, max_chars: int = 1200) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for page in filing.pages:
        all_chunks.extend(chunk_page(page, filing.doc_name, max_chars=max_chars))
    return all_chunks


if __name__ == "__main__":
    import sys as _sys
    from parser.page_parser import parse_filing

    path = _sys.argv[1] if len(_sys.argv) > 1 else \
        str(PROJECT_ROOT / "data" / "filings" / "3M_2018_10K.htm")
    filing = parse_filing(path)
    chunks = chunk_filing(filing)

    n_table = sum(1 for c in chunks if c.chunk_type == "table")
    n_prose = sum(1 for c in chunks if c.chunk_type == "prose")
    lens = [len(c.text) for c in chunks]

    print(f"{filing.doc_name}: {len(filing.pages)} pages -> {len(chunks)} chunks "
          f"({n_table} table, {n_prose} prose)")
    if lens:
        print(f"chunk length: min={min(lens)} max={max(lens)} avg={sum(lens)/len(lens):.0f}")
    print("\nSample table chunk:")
    for c in chunks:
        if c.chunk_type == "table":
            print(f"  [{c.chunk_id}]")
            print("  " + c.text[:300].replace("\n", "\n  "))
            break
