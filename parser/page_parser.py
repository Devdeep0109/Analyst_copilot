"""
parser/page_parser.py

Splits a raw SEC EDGAR .htm filing into pages, using the
`<hr style="page-break-after:always">` convention we validated by hand.

Key lessons baked in from manual validation (see /tests/validate_pages.py history):
  1. These filings are inline XBRL (iXBRL). There is exactly one <ix:header> and
     one <ix:hidden> block per document — both are pure machine-readable metadata
     that never renders visually. They MUST be stripped before extracting page
     text, or a naive text search can match a hidden/unrelated fact instead of
     the actual visible number on the page.
  2. Page boundaries are marked by self-closing <hr style="page-break-after:always"/>
     tags. This holds for 10-Ks and 10-Qs. It does NOT hold for 8-Ks ~(0 markers
     found in every 8-K tested) — 8-Ks need a fallback (treat whole doc as 1 page).
  3. Even after stripping iXBRL noise, a full-document text search for a short
     phrase can still hit multiple genuine occurrences (e.g. a balance sheet line
     item repeated in footnote rollforward tables for different quarter-end dates).
     So downstream consumers should search WITHIN a specific page's text, not the
     whole document, whenever they know which page they expect the answer on.
  4. Practice-question `evidence_page_num` values come from the ORIGINAL PDF.
     Our own page-break-derived numbering tends to run +1 relative to that in
     10-Ks/10-Qs we've checked so far (unconfirmed as a universal constant —
     validate per-document where possible, don't hardcode blindly).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
import re
import warnings
from bs4 import XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

PAGE_BREAK_STYLE_RE = re.compile(r"page-break-after\s*:\s*always", re.I)


def _pick_backend() -> str:
    """Prefer lxml, fall back to the stdlib parser if it isn't installed.

    Verified equivalent on 5 filings (389 pages): identical page counts, and
    the only text difference anywhere was EDGAR submission-wrapper noise on
    page 1 ("HTML document created with Merrill Bridge...") which html.parser
    drops and lxml keeps -- html.parser is arguably the better behaviour there,
    since that text isn't part of the filing.

    lxml is ~1.3x faster, so it stays the default, but it is NOT required for
    correct results and nothing should hard-fail without it.
    """
    try:
        import lxml  # noqa: F401
        return "lxml"
    except ImportError:
        return "html.parser"


HTML_BACKEND = _pick_backend()


@dataclass
class TableCell:
    text: str
    is_header: bool = False


@dataclass
class ParsedTable:
    rows: list[list[TableCell]]

    def to_text(self) -> str:
        """Render as a pipe-delimited grid so row/column relationships survive
        into plain text (critical: this is what keeps a number attached to its
        line-item label and year/period column instead of floating free)."""
        lines = []
        for row in self.rows:
            cells = [c.text.strip() for c in row if c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
        return "\n".join(lines)


@dataclass
class Page:
    page_num: int          # 1-indexed, in OUR parser's counting
    text: str               # plain text of the page (tables rendered as grids)
    tables: list[ParsedTable] = field(default_factory=list)

    def contains(self, snippet: str) -> bool:
        return _normalize(snippet) in _normalize(self.text)


@dataclass
class ParsedFiling:
    doc_name: str
    pages: list[Page]
    has_page_breaks: bool   # False for 8-Ks etc. where we fell back to 1 page
    raw_break_count: int

    def find_snippet(self, snippet: str) -> list[int]:
        """Return page numbers (1-indexed, our numbering) where snippet appears."""
        return [p.page_num for p in self.pages if p.contains(snippet)]


def _normalize(s: str) -> str:
    s = s.lower()
    s = s.replace("\xa0", " ")
    s = s.replace("—", " ").replace("–", " ").replace("-", " ")
    s = re.sub(r"[\$,()]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _strip_ixbrl_noise(soup: BeautifulSoup) -> None:
    """Remove elements that are never rendered visually: XBRL header metadata
    and hidden fact blocks. Modifies soup in place."""
    for tagname in ("ix:header", "ix:hidden"):
        for el in soup.find_all(tagname):
            el.decompose()
    # Some filings namespace these differently depending on parser; also try
    # without the ix: prefix in case lxml normalized it away.
    for tagname in ("header", "hidden"):
        for el in soup.find_all(tagname, attrs={"xmlns": True}):
            el.decompose()


def _table_to_parsed(table_tag) -> ParsedTable:
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = []
        for cell in tr.find_all(["td", "th"]):
            cells.append(TableCell(
                text=cell.get_text(separator=" ", strip=True),
                is_header=(cell.name == "th"),
            ))
        if cells:
            rows.append(cells)
    return ParsedTable(rows=rows)


def parse_filing(html_path: str, doc_name: str | None = None) -> ParsedFiling:
    """
    Extraction strategy (rewritten to fix an inline-XBRL bug):

    We walk every node under <body> in true document order via `.descendants`
    (a single flat depth-first traversal that visits Tags and NavigableStrings
    interleaved exactly as they appear in the source). For each leaf text node
    we append its stripped text to the current page buffer.

    This matters because numbers are often tagged inline, mid-sentence, e.g.:
        "the amount would be payable over <ix:nonNumeric>10</ix:nonNumeric> years"
    An earlier version of this parser collected "this element's own direct
    text" tag-by-tag, which separated a nested inline tag's text from its
    surrounding sentence and re-appended it out of order — silently detaching
    numbers from their sentences ("payable over years" with "10" appearing
    elsewhere on the page). Walking .descendants directly avoids this because
    every leaf string is visited in its true, original position — nesting
    depth doesn't matter, only document order does.

    Tables are the one deliberate exception: we still special-case <table>
    elements so we can (a) preserve row/column structure via ParsedTable, and
    (b) avoid appending the *same* table content twice (once as a table, once
    again as loose leaf text from its descendants). We precompute every
    descendant id under each table and skip those nodes during the main walk.
    """
    if doc_name is None:
        doc_name = html_path.split("/")[-1].replace(".htm", "")

    with open(html_path, encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    soup = BeautifulSoup(raw, HTML_BACKEND)
    _strip_ixbrl_noise(soup)

    body = soup.body or soup

    # Precompute break markers.
    break_elements = body.find_all(style=PAGE_BREAK_STYLE_RE)
    break_ids = set(id(e) for e in break_elements)
    raw_break_count = len(break_elements)

    # Precompute which node ids belong "inside" a table, so the leaf-text walk
    # can skip them (they're handled separately, as structured ParsedTables).
    skip_ids = set()
    for t in body.find_all("table"):
        for d in t.descendants:
            skip_ids.add(id(d))

    SKIP_PARENT_TAGS = {"script", "style"}

    from bs4 import NavigableString, Tag

    pages: list[Page] = []
    page_num = 1
    buf_parts: list[str] = []
    seg_tables: list[ParsedTable] = []

    for node in body.descendants:
        node_id = id(node)

        if node_id in skip_ids:
            continue  # already-consumed table content

        if isinstance(node, Tag):
            if node.name == "table":
                seg_tables.append(_table_to_parsed(node))
                txt = node.get_text(separator=" ", strip=True)
                if txt:
                    buf_parts.append(txt)
                continue

            if node_id in break_ids:
                pages.append(Page(page_num=page_num,
                                   text=_join_parts(buf_parts),
                                   tables=seg_tables))
                page_num += 1
                buf_parts = []
                seg_tables = []
                continue

            # other tags contribute nothing directly; their text arrives via
            # their own NavigableString descendants later in this same walk
            continue

        elif isinstance(node, NavigableString):
            parent = node.parent
            if parent is not None and getattr(parent, "name", None) in SKIP_PARENT_TAGS:
                continue
            txt = str(node).strip()
            if txt:
                buf_parts.append(txt)

    if buf_parts or seg_tables:
        pages.append(Page(page_num=page_num, text=_join_parts(buf_parts), tables=seg_tables))

    has_page_breaks = raw_break_count > 0
    return ParsedFiling(doc_name=doc_name, pages=pages,
                         has_page_breaks=has_page_breaks, raw_break_count=raw_break_count)


def _join_parts(parts: list[str]) -> str:
    """Join leaf text fragments with a single space, then collapse any
    resulting double-spacing. Using a real separator (vs "".join) is what
    keeps 'over' + '10' + 'years' from gluing into 'over10years'."""
    text = " ".join(parts)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parent.parent
    path = sys.argv[1] if len(sys.argv) > 1 else \
        str(_root / "data" / "filings" / "3M_2018_10K.htm")
    filing = parse_filing(path)
    print(f"{filing.doc_name}: {filing.raw_break_count} breaks -> {len(filing.pages)} pages "
          f"(has_page_breaks={filing.has_page_breaks})")
    print(f"Page 1 preview: {filing.pages[0].text[:200]}")
