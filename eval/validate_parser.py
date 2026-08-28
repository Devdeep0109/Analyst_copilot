"""
tests/validate_parser.py

Runs the real parser (parser/page_parser.py) against every 10-K/10-Q question
in practice-questions.jsonl, and checks: does the evidence snippet actually
appear on the page our parser assigns, at gold_page + offset, for some
consistent offset?

This replaces the earlier throwaway validate_pages.py script now that the
parser itself handles iXBRL stripping properly.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from parser.page_parser import parse_filing, _normalize

FILINGS_DIR = str(PROJECT_ROOT / "data" / "filings")
QUESTIONS = str(PROJECT_ROOT / "data" / "practice-questions.jsonl")

BOILERPLATE = re.compile(
    r"^(table of contents|consolidated (statement|balance)|years ended|"
    r"at december|the accompanying notes|assets|liabilities|current assets)$",
    re.I,
)


def pick_distinctive_snippet(evidence_text: str) -> str | None:
    """Pick a line from evidence_text that is specific enough to be a fair
    single-page test: has a number, reasonable length, not a boilerplate
    header, and not glued-together words (a source-data artifact we found
    earlier, e.g. 'Excessconsiderationpaid...')."""
    lines = [l.strip() for l in evidence_text.split("\n") if l.strip()]
    candidates = []
    for l in lines:
        if BOILERPLATE.match(l):
            continue
        if not re.search(r"\d", l):
            continue
        if not (15 <= len(l) <= 100):
            continue
        # crude check against glued-word artifacts: a normal line should have
        # a healthy ratio of spaces to length
        if l.count(" ") < len(l) / 25:
            continue
        candidates.append(l)
    if not candidates:
        return None
    return max(candidates, key=len)  # prefer the most specific (longest) line


def main():
    with open(QUESTIONS) as f:
        all_q = [json.loads(l) for l in f]

    tenk_tenq = [q for q in all_q if q["doc_type"] in ("10k", "10q")]
    doc_names = sorted(set(q["doc_name"] for q in tenk_tenq))
    print(f"Testing {len(doc_names)} 10-K/10-Q filings, {len(tenk_tenq)} questions total\n")

    offset_counts = Counter()
    closest_offset_counts = Counter()   # NEW: one entry per question, nearest match only
    match_count_per_q = []              # NEW: how many pages did the snippet hit?
    not_found = 0
    not_testable = 0
    tested = 0
    not_found_examples = []
    noisy_examples = []                 # snippets matching too many pages

    for doc_name in doc_names:
        html_path = f"{FILINGS_DIR}/{doc_name}.htm"
        try:
            filing = parse_filing(html_path, doc_name=doc_name)
        except FileNotFoundError:
            continue
        if not filing.has_page_breaks:
            continue

        qs = [q for q in tenk_tenq if q["doc_name"] == doc_name]
        for q in qs:
            for ev in q["evidence"]:
                gold_page = ev["evidence_page_num"]
                snippet = pick_distinctive_snippet(ev["evidence_text"])
                if not snippet:
                    not_testable += 1
                    continue
                tested += 1
                found_pages = filing.find_snippet(snippet)
                if not found_pages:
                    not_found += 1
                    if len(not_found_examples) < 8:
                        not_found_examples.append((doc_name, gold_page, snippet))
                    continue

                match_count_per_q.append(len(found_pages))
                if len(found_pages) > 4:
                    if len(noisy_examples) < 8:
                        noisy_examples.append((doc_name, gold_page, snippet, found_pages))

                for fp in found_pages:
                    offset_counts[fp - gold_page] += 1

                # closest single match to gold page -> this is the fair per-question signal
                closest = min(found_pages, key=lambda fp: abs(fp - gold_page))
                closest_offset_counts[closest - gold_page] += 1

    print("=== PER-QUESTION CLOSEST-MATCH OFFSET (fair signal, one vote per question) ===")
    for offset, count in closest_offset_counts.most_common(15):
        print(f"  offset {offset:+d}: {count} questions")
    total_q_with_match = sum(closest_offset_counts.values())
    if total_q_with_match:
        top_offset, top_count = closest_offset_counts.most_common(1)[0]
        print(f"\nMost common closest-offset: {top_offset:+d} "
              f"({100*top_count/total_q_with_match:.1f}% of {total_q_with_match} questions with a match)")

    within_1 = sum(c for o, c in closest_offset_counts.items() if abs(o) <= 1)
    print(f"Within +/-1 page of gold: {within_1}/{total_q_with_match} "
          f"({100*within_1/total_q_with_match:.1f}%)")

    print(f"\nAvg pages matched per question (snippet distinctiveness check): "
          f"{sum(match_count_per_q)/len(match_count_per_q):.1f}" if match_count_per_q else "n/a")
    print(f"Questions where snippet matched >4 pages (too generic): "
          f"{sum(1 for c in match_count_per_q if c > 4)}/{len(match_count_per_q)}")

    print(f"\nQuestions tested (had a usable snippet): {tested}")
    print(f"Not testable (no distinctive snippet found): {not_testable}")
    print(f"NOT FOUND anywhere in doc: {not_found}")

    print("\n=== SAMPLE NOISY CASES (snippet matched >4 pages -> not distinctive) ===")
    for doc_name, gold_page, snippet, pages in noisy_examples:
        print(f"  [{doc_name}] gold={gold_page} matched={pages} snippet={snippet!r}")

    print("\n=== SAMPLE NOT-FOUND CASES ===")
    for doc_name, gold_page, snippet in not_found_examples:
        print(f"  [{doc_name}] gold_page={gold_page}  snippet={snippet!r}")


if __name__ == "__main__":
    main()
