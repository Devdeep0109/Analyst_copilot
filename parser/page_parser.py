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
    """TWO page numbers, because two different consumers need two conventions.

    page_num      sequential position in the document, 1-indexed. This is the
                  PDF sheet number, and it is what FinanceBench's
                  `evidence_page_num` refers to (0-based, hence the +1 in
                  eval/gold.py). EVALUATION SCORES AGAINST THIS.

    printed_page  the number actually printed at the foot of the page. A 10-K
                  puts ~14 pages of cover and table of contents before printed
                  page 1, so sheet 60 is commonly printed "46". THE UI SHOWS
                  THIS, because it is what an analyst turns to.

    Collapsing these into one field forces a choice between a correct product
    and a correct score. Keeping both means calibration can improve what the
    user sees without silently making every cited page count as "wrong
    location" against the answer key.
    """
    page_num: int
    text: str               # plain text of the page (tables rendered as grids)
    tables: list[ParsedTable] = field(default_factory=list)
    printed_page: int | None = None

    @property
    def display_page(self) -> int:
        """What to show a human. Falls back to the sheet number when the
        filing's own numbering could not be determined."""
        return self.printed_page if self.printed_page is not None else self.page_num

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

    if has_page_breaks:
        _calibrate_page_nums(pages)

    return ParsedFiling(doc_name=doc_name, pages=pages,
                         has_page_breaks=has_page_breaks, raw_break_count=raw_break_count)


# ------------- TOC-based page-number calibration ----------------------------

# Each tuple: (regex matching "Section Title  NN" in the TOC, substring to
# search for in the body page).  These section titles appear in virtually every
# SEC 10-K/10-Q and are distinctive enough to land on only one body page.
_TOC_ANCHORS = [
    (re.compile(r"Legal Proceedings\s+(\d+)"),
     "Legal Proceedings"),
    (re.compile(r"Risk Factors\s+(\d+)"),
     "Risk Factors"),
    (re.compile(r"Controls and Procedures\s+(\d+)"),
     "Controls and Procedures"),
    (re.compile(r"Unregistered Sales\s+(\d+)"),
     "Unregistered Sales"),
    (re.compile(r"Mine Safety Disclosures\s+(\d+)"),
     "Mine Safety Disclosures"),
    (re.compile(r"Quantitative and Qualitative Disclosures\s+(\d+)"),
     "Quantitative and Qualitative Disclosures"),
    (re.compile(r"Unresolved Staff Comments\s+(\d+)"),
     "Unresolved Staff Comments"),
    (re.compile(r"Properties\s+(\d+)"),
     "Properties"),
]


def _calibrate_page_nums(pages: list[Page]) -> None:
    """Detect the offset between raw parser page numbers and the filing's own
    numbering (as printed in its Table of Contents), and shift all page numbers
    so they match the filing's convention.

    Strategy: find the TOC page(s), extract section→page mappings using known
    anchor patterns, locate those same sections in the body, and compute the
    dominant offset.
    """
    if len(pages) < 4:
        return  # too few pages to calibrate

    # 1. Find TOC page(s) — always in the first ~8 pages.
    toc_text = ""
    toc_page_nums: set[int] = set()
    for p in pages[:8]:
        if "TABLE OF CONTENTS" in p.text.upper():
            toc_text += " " + p.text
            toc_page_nums.add(p.page_num)

    if not toc_text:
        return  # no TOC found — cannot calibrate

    # 2. Cross-reference: for each TOC anchor, find the stated page, then
    #    locate that section title in body pages and record the offset.
    offsets: list[int] = []
    for pat, label in _TOC_ANCHORS:
        m = pat.search(toc_text)
        if not m:
            continue
        toc_stated_page = int(m.group(1))
        label_lower = label.lower()
        for p in pages:
            if p.page_num in toc_page_nums:
                continue  # skip the TOC page itself
            # Check near top of page (first 600 chars) case-insensitively
            if label_lower in p.text[:600].lower():
                offsets.append(p.page_num - toc_stated_page)
                break

    if not offsets:
        return

    # 3. Take the dominant offset.
    from collections import Counter
    counts = Counter(offsets)
    dominant_offset, support = counts.most_common(1)[0]

    # ---- SANITY CHECKS -------------------------------------------------
    # Without these, one bad TOC match silently corrupts every page number in
    # the filing. 3M_2023Q2_10Q produced a shift of +45 -- the anchor matched
    # the wrong section, and every page the UI showed for that document was
    # wrong. A calibration you cannot trust is worse than none, because it is
    # invisible.

    # (a) A NEGATIVE offset means the body page came BEFORE the page the TOC
    #     says it is on. That is impossible in a real document; it means the
    #     anchor matched a mention in the TOC region or a cross-reference.
    if dominant_offset < 0:
        return

    # (b) Front matter is a cover page, a TOC, maybe a forward-looking
    #     statement. More than 25 pages of it is not front matter, it is a
    #     mismatch.
    if dominant_offset > 25:
        return

    # (c) One anchor agreeing with itself is not evidence. Require either two
    #     anchors agreeing, or a single anchor with no competing answer.
    if support < 2 and len(counts) > 1:
        return

    if dominant_offset == 0:
        for p in pages:
            p.printed_page = p.page_num
        return

    # 4. RECORD the printed number. Do NOT overwrite page_num -- that is the
    #    sheet index the evaluation scores against.
    for p in pages:
        printed = p.page_num - dominant_offset
        # Front matter has no printed number (or uses roman numerals we do not
        # parse). Leave it None so display_page falls back to the sheet number
        # rather than showing a fabricated "1" for the first fifteen pages.
        p.printed_page = printed if printed >= 1 else None


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
