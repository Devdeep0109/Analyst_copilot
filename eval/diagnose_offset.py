"""
eval/diagnose_offset.py

Answers one question: is the parser-page -> gold-page offset a single global
constant, a per-filing constant, or genuinely inconsistent?

Why this script exists: validate_parser.py picks the *closest* matching page to
gold when a snippet hits many pages. That biases the measured offset toward 0
(if the snippet hits pages 48-53 and gold is 51, "closest" is 51, offset +0 --
regardless of the truth). With an average of 14 pages matched per snippet, that
confound is large enough to invent a bimodal distribution that isn't there.

So here we only trust UNIQUE matches: snippets that appear on exactly one page
of the filing. Those carry no ambiguity, and the offset they report is real.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from parser.page_parser import parse_filing

FILINGS_DIR = PROJECT_ROOT / "data" / "filings"
QUESTIONS = PROJECT_ROOT / "data" / "practice-questions.jsonl"

BOILERPLATE = re.compile(
    r"^(table of contents|consolidated (statement|balance)|years ended|"
    r"at december|the accompanying notes|assets|liabilities|current assets)$",
    re.I,
)


def candidate_snippets(evidence_text: str, limit: int = 6) -> list[str]:
    """Return several candidate lines, longest first -- we want as many shots
    as possible at finding one that is unique within the document."""
    out = []
    for l in (x.strip() for x in evidence_text.split("\n")):
        if not l or BOILERPLATE.match(l):
            continue
        if not re.search(r"\d", l):
            continue
        if not (15 <= len(l) <= 120):
            continue
        if l.count(" ") < len(l) / 25:
            continue
        out.append(l)
    out.sort(key=len, reverse=True)
    return out[:limit]


def main():
    with open(QUESTIONS, encoding="utf-8") as f:
        all_q = [json.loads(l) for l in f]

    qs = [q for q in all_q if q["doc_type"] in ("10k", "10q")]
    doc_names = sorted(set(q["doc_name"] for q in qs))

    per_doc_offsets = defaultdict(list)   # doc -> [offsets from unique matches]
    global_offsets = Counter()
    n_unique = n_ambiguous = n_notfound = 0

    for doc_name in doc_names:
        path = FILINGS_DIR / f"{doc_name}.htm"
        if not path.exists():
            continue
        try:
            filing = parse_filing(str(path), doc_name=doc_name)
        except Exception as e:
            print(f"  !! parse failed {doc_name}: {e}")
            continue
        if not filing.has_page_breaks:
            continue

        for q in (x for x in qs if x["doc_name"] == doc_name):
            for ev in q["evidence"]:
                gold = ev["evidence_page_num"]
                got_unique = False
                for snip in candidate_snippets(ev["evidence_text"]):
                    pages = filing.find_snippet(snip)
                    if len(pages) == 1:
                        off = pages[0] - gold
                        per_doc_offsets[doc_name].append(off)
                        global_offsets[off] += 1
                        n_unique += 1
                        got_unique = True
                        break
                    elif len(pages) > 1:
                        continue
                if not got_unique:
                    n_ambiguous += 1

    print("=" * 68)
    print("OFFSET FROM UNIQUE-MATCH SNIPPETS ONLY (no closest-match bias)")
    print("=" * 68)
    total = sum(global_offsets.values())
    for off, c in sorted(global_offsets.items()):
        bar = "#" * int(40 * c / max(global_offsets.values()))
        print(f"  offset {off:+3d}: {c:4d}  {bar}")
    if total:
        top, topc = global_offsets.most_common(1)[0]
        print(f"\n  dominant offset {top:+d} = {100*topc/total:.1f}% of {total} unique-match datapoints")

    print(f"\n  evidence blocks with a unique snippet : {n_unique}")
    print(f"  evidence blocks with none (ambiguous)  : {n_ambiguous}")

    print("\n" + "=" * 68)
    print("IS THE OFFSET CONSISTENT WITHIN EACH FILING?")
    print("=" * 68)
    consistent = inconsistent = single = 0
    bad_docs = []
    for doc, offs in sorted(per_doc_offsets.items()):
        uniq = set(offs)
        if len(offs) == 1:
            single += 1
        elif len(uniq) == 1:
            consistent += 1
        else:
            inconsistent += 1
            bad_docs.append((doc, Counter(offs)))

    print(f"  filings with a single consistent offset : {consistent}")
    print(f"  filings with only one datapoint         : {single}")
    print(f"  filings with MIXED offsets              : {inconsistent}")

    if bad_docs:
        print("\n  --- mixed-offset filings ---")
        for doc, c in bad_docs:
            print(f"    {doc:38s} {dict(sorted(c.items()))}")

    print("\n" + "=" * 68)
    print("PER-FILING DOMINANT OFFSET (what a per-doc calibration would use)")
    print("=" * 68)
    doc_mode = Counter()
    for doc, offs in sorted(per_doc_offsets.items()):
        mode = Counter(offs).most_common(1)[0][0]
        doc_mode[mode] += 1
    for off, c in sorted(doc_mode.items()):
        print(f"  offset {off:+3d}: {c} filings")


if __name__ == "__main__":
    main()
