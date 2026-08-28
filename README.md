# The Analyst Copilot

Answers analyst questions about SEC filings **with a verified page citation**,
and declines when the evidence isn't there.

Built against a rubric that scores `+1` for a correct answer with the correct
page, `0` for an honest "not found", and `−1` for a confident wrong answer. That
asymmetry drives every design decision here: a system that answers everything
scores worse than one that answers carefully.

## Results

Measured on 72 of the 127 practice questions (10-K / 10-Q):

| metric | value |
|---|---|
| **precision when it answers** | **77%** |
| **citations mechanically verified** | **94.3%** |
| rubric score | **+10.0** (vs +6.5 answering everything) |
| retrieval — evidence present | 87.5% |
| conversion rate (right answer given good evidence) | 0.692 |
| declined | 51% |

The verifiers are the difference between +6.5 and +10.0: they remove wrong
answers roughly 7× faster than they remove correct ones.

## Pipeline

```
filing.htm
  ├─ page-aware parser      split on page-break markers, strip iXBRL noise
  ├─ table-aware chunker    tables never split; prose in sentence windows
  ├─ hybrid retrieval       BM25 + dense embeddings, fused at page level
  ├─ cross-encoder rerank   ms-marco-MiniLM, depth 20
  ├─ Verifier #1  (LLM)     is this evidence sufficient? → abstain if not
  ├─ answering LLM          answer + VERBATIM quote + page number
  └─ Verifier #2  (code)    is that quote really on that page? → abstain if not
```

Verifier #2 uses no LLM. It is a normalized substring check, and it is the
component that makes "verified citation" a fact rather than a claim.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export GROQ_API_KEY=...            # Windows: $env:GROQ_API_KEY='...'
```

**Data is not in this repo** (336 MB, and not ours to redistribute). Restore it
to:

```
data/filings/*.htm
data/practice-questions.jsonl
```

## Running

```bash
python eval/run_pipeline.py --rate 34 --limit 30   # a batch, resumable
python eval/run_pipeline.py --report               # results, no API calls
```

Everything caches to `.cache/`, so batches resume where they stopped and
`--report` is always free.

### Component evaluations

```bash
python eval/validate_parser.py     # page splitting vs the answer key
python eval/validate_chunks.py     # does evidence survive chunking?
python eval/run_bm25.py --all      # lexical retrieval + tuning sweeps
python eval/run_dense.py           # embedding retrieval
python eval/run_hybrid.py --all    # fusion, and the case for it
python eval/run_rerank.py          # cross-encoder comparison
python eval/run_classifier.py      # question typing + evidence sizing
python eval/diagnose_abstain.py    # why the model declines (no API calls)
python eval/validate_choice.py     # significance tests on the retrieval config
```

## Findings worth knowing

**Page-level beats chunk-level retrieval.** Top-k *chunks* often spend every
slot on one page. Scoring pages and taking the top-k *pages* gives k
independent guesses: 0.236 → 0.315.

**The 90 MB reranker beat the 1.1 GB one.** `ms-marco-MiniLM` (d=0.86) against
`bge-reranker-base` (d=0.60). The larger model is multilingual and spends its
capacity on languages no 10-K contains.

**Dense retrieval needs `page_max`, not `page_sum`.** BM25 scores are sparse
(irrelevant chunk = 0); cosine scores are dense (everything scores something).
Summing rewards long pages: 0.283 → 0.528 on that change alone.

**Structure beats instruction.** The model twice returned the excerpt index
where a page number was wanted, through two rounds of increasingly emphatic
prompting. Removing the excerpt number from the header fixed it permanently:
citations 50% → 94%.

**Half the abstentions were correct.** Calculation questions were classified
"extractive" (they open with "What is the...") and given too small an evidence
budget, so required figures were genuinely missing. Detecting formula language
and raising K cut abstention 40% → 26%.

**The arithmetic check costs more than it saves.** It removed 2 wrong answers
and killed 3 correct ones. Kept in the config table, off by default.

## Known limitations

- **8-K filings are excluded.** The supplied 8-K files are the wrong documents —
  two PepsiCo files are byte-identical, and gold evidence appears in none of
  them. Declining is the correct behaviour, not a parser gap.
- **13 of 72 questions have prose gold answers** the mechanical comparator
  cannot grade. They score 0 regardless of correctness and need an LLM judge.
- **Verifier #2 cannot catch a real quote that answers the wrong question** —
  the wrong year's column of a multi-year table is genuine, cited correctly,
  and wrong. This is the largest residual risk.
- Sample sizes are small (n=72 of 127); most individual comparisons are not
  statistically significant on their own. See `eval/validate_choice.py`.
