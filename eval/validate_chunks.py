"""
tests/validate_chunks.py

For a sample of practice questions, checks whether the gold evidence text
actually appears whole inside at least one chunk on the (offset-corrected)
expected page. This is the property retrieval will depend on -- a page can be
"right" while still failing if the specific fact got cut across two chunks.
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from parser.page_parser import parse_filing, _normalize
from parser.chunker import chunk_filing

FILINGS_DIR = str(PROJECT_ROOT / "data" / "filings")
QUESTIONS = str(PROJECT_ROOT / "data" / "practice-questions.jsonl")

BOILERPLATE = re.compile(
    r"^(table of contents|consolidated (statement|balance)|years ended|"
    r"at december|the accompanying notes|assets|liabilities|current assets)$",
    re.I,
)


def pick_distinctive_snippet(evidence_text: str) -> str | None:
    lines = [l.strip() for l in evidence_text.split("\n") if l.strip()]
    candidates = []
    for l in lines:
        if BOILERPLATE.match(l):
            continue
        if not re.search(r"\d", l):
            continue
        if not (10 <= len(l) <= 100):
            continue
        if l.count(" ") < len(l) / 25:
            continue
        candidates.append(l)
    if not candidates:
        return None
    return max(candidates, key=len)


def main():
    with open(QUESTIONS) as f:
        all_q = [json.loads(l) for l in f]

    tenk_tenq = [q for q in all_q if q["doc_type"] in ("10k", "10q")]
    doc_names = sorted(set(q["doc_name"] for q in tenk_tenq))

    found_in_chunk = 0
    found_on_page_not_chunk = 0   # page-level match exists but no single chunk has it whole
    not_found_at_all = 0
    tested = 0
    split_examples = []

    for doc_name in doc_names:
        html_path = f"{FILINGS_DIR}/{doc_name}.htm"
        try:
            filing = parse_filing(html_path, doc_name=doc_name)
        except FileNotFoundError:
            continue
        chunks = chunk_filing(filing)

        qs = [q for q in tenk_tenq if q["doc_name"] == doc_name]
        for q in qs:
            for ev in q["evidence"]:
                gold_page = ev["evidence_page_num"]
                snippet = pick_distinctive_snippet(ev["evidence_text"])
                if not snippet:
                    continue
                tested += 1

                # search window: gold_page, gold_page+1 (our confirmed offset), gold_page-1
                candidate_pages = {gold_page, gold_page + 1, gold_page - 1}
                page_has_it = any(
                    _normalize(snippet) in _normalize(p.text)
                    for p in filing.pages if p.page_num in candidate_pages
                )
                chunk_has_it = any(
                    c.contains(snippet)
                    for c in chunks if c.page_num in candidate_pages
                )

                if chunk_has_it:
                    found_in_chunk += 1
                elif page_has_it:
                    found_on_page_not_chunk += 1
                    if len(split_examples) < 6:
                        split_examples.append((doc_name, gold_page, snippet))
                else:
                    not_found_at_all += 1

    print(f"Tested: {tested} evidence snippets across {len(doc_names)} 10-K/10-Q filings\n")
    print(f"Found whole inside a single chunk:      {found_in_chunk} "
          f"({100*found_in_chunk/tested:.1f}%)")
    print(f"On the right page but SPLIT across chunks: {found_on_page_not_chunk} "
          f"({100*found_on_page_not_chunk/tested:.1f}%)")
    print(f"Not found at all (page/snippet mismatch):  {not_found_at_all} "
          f"({100*not_found_at_all/tested:.1f}%)")

    if split_examples:
        print("\n=== Examples where evidence was split across chunk boundaries ===")
        for doc_name, gold_page, snippet in split_examples:
            print(f"  [{doc_name}] gold_page={gold_page}  snippet={snippet!r}")


if __name__ == "__main__":
    main()
