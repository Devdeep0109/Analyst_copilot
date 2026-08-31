"""
eval/check_printed_pages.py

Verifies the two-page-number scheme:

  page_num      sequential sheet index -- what the evaluation scores against
  printed_page  the number printed at the foot of the page -- what the UI shows

Two things must hold, and both were broken before:

  1. page_num must be strictly sequential for EVERY filing. Calibration used to
     overwrite it, which meant a correct citation could be scored as "wrong
     location" against the answer key on ~36% of filings.

  2. printed_page, when set, must match what the page actually prints. A
     mis-detected TOC anchor gave 3M_2023Q2_10Q a 45-page shift -- every page
     the UI showed for that filing was wrong, invisibly.

This reads the trailing number off each page and compares. No API calls.

Run:  python eval/check_printed_pages.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.gold import load_gold                 # noqa: E402
from parser.page_parser import parse_filing     # noqa: E402

# A page footer is a small number at the very end of the text.
TRAILING_NUM = re.compile(r"(\d{1,3})\s*\$?\s*$")


def printed_at_foot(text: str) -> int | None:
    m = TRAILING_NUM.search(text.strip()[-60:])
    return int(m.group(1)) if m else None


def main() -> None:
    gold = load_gold()
    docs = sorted({q.doc_name for q in gold})[:30]

    seq_bad = []
    calibrated = 0
    agree = disagree = unknown = 0
    examples = []

    for d in docs:
        q = next(x for x in gold if x.doc_name == d)
        try:
            f = parse_filing(str(q.filing_path), doc_name=d)
        except Exception:
            continue
        if len(f.pages) < 20:
            continue

        # (1) sequential integrity
        if not all(p.page_num == i for i, p in enumerate(f.pages, 1)):
            seq_bad.append(d)

        has_printed = any(p.printed_page is not None for p in f.pages)
        if has_printed:
            calibrated += 1

        # (2) does printed_page match the footer?
        for p in f.pages[10:]:
            foot = printed_at_foot(p.text)
            if foot is None or not (1 <= foot <= len(f.pages) + 5):
                continue
            if p.printed_page is None:
                unknown += 1
            elif p.printed_page == foot:
                agree += 1
            else:
                disagree += 1
                if len(examples) < 8:
                    examples.append((d, p.page_num, p.printed_page, foot))

    print("=" * 70)
    print("1. IS page_num STILL SEQUENTIAL? (required for correct scoring)")
    print("=" * 70)
    if seq_bad:
        print(f"  BROKEN on {len(seq_bad)} filings: {seq_bad[:5]}")
    else:
        print(f"  OK -- sequential on all {len(docs)} filings checked")

    print("\n" + "=" * 70)
    print("2. DOES printed_page MATCH THE PAGE FOOTER?")
    print("=" * 70)
    total = agree + disagree
    print(f"  filings where calibration fired : {calibrated}/{len(docs)}")
    if total:
        print(f"  footer comparisons              : {total}")
        print(f"    agree                         : {agree} ({agree/total:.0%})")
        print(f"    disagree                      : {disagree} ({disagree/total:.0%})")
    print(f"  pages with no printed_page set    : {unknown} "
          f"(UI falls back to sheet number)")

    if examples:
        print("\n  mismatches:")
        for d, sheet, printed, foot in examples:
            print(f"    {d:26s} sheet {sheet:3d}  we say {printed}  footer says {foot}")

    print("\n" + "=" * 70)
    if seq_bad:
        print("  FAIL -- scoring is compromised. page_num must not be shifted.")
    elif total and disagree / total > 0.3:
        print("  printed_page is unreliable. Better to show the sheet number than")
        print("  a confidently wrong page: consider disabling calibration.")
    else:
        print("  OK -- eval scores against sheet numbers, UI shows printed ones,")
        print("  and the two no longer fight each other.")


if __name__ == "__main__":
    main()
