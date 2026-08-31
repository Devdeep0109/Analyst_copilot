# The Analyst Copilot

A chatbot that answers analyst questions about SEC filings **with a page
citation it can mechanically prove** — and says "not found in this filing" when
it cannot.

Built against a rubric that scores `+1` for a correct answer in the correct
place, `0` for an honest decline, and `−1` for a confident wrong answer. That
asymmetry drove every design decision here: a system that guesses finishes below
zero, so the engineering problem is not "answer more" — it is knowing the
difference between an answer you can prove and one you cannot.

---

## Quick start

```bash
git clone <repo> && cd analyst_hackathon
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu   # ~200 MB, not the 2.5 GB CUDA build
pip install -r requirements.txt

$env:GROQ_API_KEY = 'gsk_...'          # free key: console.groq.com/keys
streamlit run app.py                   # opens http://localhost:8501
```

**The data pack is not in this repo** (336 MB of SEC filings, not ours to
redistribute). Restore it to:

```
data/filings/*.htm
data/practice-questions.jsonl
```

**First question is slow** (~40 s): it downloads MiniLM (~90 MB) and the
cross-encoder (~90 MB) from Hugging Face, then embeds that filing. Everything
caches to `.cache/`; later questions take a few seconds plus LLM latency.

**Launch from the activated venv.** If `where streamlit` points outside
`.venv\Scripts\`, it is the wrong interpreter and `sentence-transformers` will
fail to import.

---

## What the product does

| requirement | how it is met |
|---|---|
| **Add filing** | Sidebar uploader accepts SEC inline-XBRL `.htm`. Parses, chunks and embeds on the spot with staged progress. |
| **within 10 minutes** | Measured **~40 s** for a 10 MB 10-K (parse 4 s, chunk 0.01 s, embed ~35 s). |
| **Chat box** | Plain-English questions against the selected filing. |
| **Evidence on every answer** | Document, page number, the exact quoted row, and a `CITATION VERIFIED` badge. |
| **Ability to decline** | Three distinct decline states, so the user knows *which* check stopped us. |

The three decline states are deliberate — "the filing doesn't say" and "I found
something but couldn't verify the quote" are different facts, and an analyst can
act on the difference.

---

## Architecture

```
filing.htm
   │
   ├─ page-aware parser      split on page-break markers; strip inline-XBRL
   │                         noise; record both sheet number and printed page
   ├─ table-aware chunker    tables never split; prose in sentence windows
   │
   ├─ hybrid retrieval       BM25 + dense embeddings, fused at PAGE level
   ├─ cross-encoder rerank   ms-marco-MiniLM, depth 20
   ├─ question classifier    sizes the evidence set (K=10 lookup, K=16 formula)
   │
   ├─ Verifier #1  (LLM)     is this evidence sufficient?     → decline if not
   ├─ answering LLM          answer + VERBATIM quote + page
   └─ Verifier #2  (code)    is that quote really on that page? → decline if not
```

**Verifier #2 uses no LLM.** It is a normalised substring check with strict
handling of numbers: every figure in the quote must appear in the cited chunk.
That is what makes "verified citation" a fact rather than a claim, and it cannot
be talked out of its answer.

---

## Results

127 practice questions (10-K / 10-Q), `openai/gpt-oss-20b` via Groq, temperature 0.

| metric | value |
|---|---|
| **rubric score** | **+13.5** (vs `0` always-abstain, `+8.5` answer-everything) |
| citations mechanically verified | **93.3%** |
| answered / declined | 71 / 56 |
| correct when it answers | ~66% |
| evidence retrieved successfully | 85.8% |

**Reported honestly:** the score was +15.5 while 20 prose answers sat ungraded.
Building an LLM judge to grade them revealed 15 of 31 were wrong, and the score
*fell* to +13.5. The lower number is the complete measurement; the higher one was
flattered by questions we had not scored.

The answer-matching threshold (12 words, above which grading is deferred to the
LLM judge) is worth ±2 points either way — the honest range is **+11 to +14**.

---

## Approach: what we measured, and what it changed

Every component was chosen by measurement against the answer key, not by
reputation. The findings that mattered:

**Page-level retrieval beats chunk-level.** Top-*k* chunks often spend every slot
on one page. Scoring pages and taking the top *k* pages gives *k* independent
guesses: **0.236 → 0.315**.

**Dense retrieval needed `page_max`, not `page_sum`.** BM25 scores are sparse
(an irrelevant chunk scores exactly 0); cosine scores are dense (everything
scores something). Summing rewards long pages. Fixing this alone: **0.283 →
0.528**.

**The 90 MB reranker beat the 1.1 GB one.** `ms-marco-MiniLM` (d=0.86) against
`bge-reranker-base` (d=0.60). The larger model is multilingual and spends its
capacity on languages no 10-K contains.

**Structure beats instruction.** The model twice returned the *excerpt index*
where a page number was wanted, through two rounds of increasingly emphatic
prompting. Removing the competing number from the header fixed it permanently:
citations **50% → 93%**.

**Half the abstentions were correct.** Calculation questions were classified
"extractive" (they open with "What is the...") and given too small an evidence
budget, so required figures genuinely were missing. Detecting formula language
and raising K cut abstention **40% → 27%**.

**The two verifiers do different jobs, and only one is strong.** Verifier #2
(mechanical) proves citations and cannot be fooled. Verifier #1 (LLM) separates
a 66%-accurate population from a 43%-accurate one — a weak filter worth about
**+3 points**, not the hallucination defence the architecture diagram implies.
Both are kept; only one is claimed.

**Five approaches to the abstention problem, four failed.** Prompt rewrites
(×4), a forced retry with `null` removed from the schema entirely (34 of 36 still
declined), gate-strategy sweeps (9 variants, all worse than the blunt gate),
two-stage decomposition, and a different evidence budget. Only the last of these
moved the number. The model cannot answer ~27% of these questions, and no
prompting changes that.

**The arithmetic recomputation check costs more than it saves.** It removed 2
wrong answers and killed 3 correct ones. Implemented, measured, and switched off
— kept in the results table so the finding stays visible.

---

## Running the evaluation

```bash
python eval/run_pipeline.py --rate 34 --limit 30   # a batch (resumable)
python eval/run_pipeline.py --report               # results, zero API calls
python eval/run_pipeline.py --judge                # grade prose answers too
```

Everything caches by content hash, so batches resume where they stopped and
`--report` is always free. Free-tier rate limits are handled with adaptive
backoff; a daily-quota exhaustion aborts cleanly rather than retrying for 30
minutes.

### Component evaluations

```bash
python eval/validate_parser.py       # page splitting vs the answer key
python eval/validate_chunks.py       # does evidence survive chunking? (0 splits)
python eval/check_printed_pages.py   # sheet number vs printed page (97% agree)
python eval/run_bm25.py --all        # lexical retrieval + tuning sweeps
python eval/run_dense.py             # embedding retrieval
python eval/run_hybrid.py --all      # fusion, and the case for it
python eval/run_rerank.py            # cross-encoder comparison
python eval/run_classifier.py        # question typing + evidence sizing
python eval/diagnose_abstain.py      # why the model declines (no API calls)
python eval/tune_gate.py             # 9 gating strategies compared
python eval/validate_choice.py       # significance tests on the retrieval config
```

---

## Known limitations

**8-K filings are excluded.** The supplied 8-K files are the wrong documents —
two PepsiCo files are byte-identical, and gold evidence appears in none of them.
Declining is correct behaviour, not a parser gap.

**Verifier #2 cannot catch a real quote that answers the wrong question.** The
wrong year's column of a multi-year table is genuine, correctly cited, and wrong.
This is the largest residual risk and no mechanical check addresses it.

**The judge is the same model grading itself.** `gpt-oss-20b` judging
`gpt-oss-20b` shares its blind spots. It graded strictly rather than leniently,
which is reassuring, but a second opinion from a different model family was never
run.

**Sample size is 127.** Most individual comparisons are not statistically
significant on their own — `eval/validate_choice.py` runs the paired tests and
says so. The full retrieval stack versus plain BM25 does clear significance
(p=0.02); the individual rungs do not.

**Scoring includes one judgement call.** A verified citation on a page that
contains the evidence — but is not the single page the answer key labels — is
scored as correct. Filings repeat facts across pages (3M's capex row appears on
46, 49 and 60; gold says 60). A stricter grader comparing page numbers alone
would score these +0.5 rather than +1.

---

## Repository layout

```
app.py                  Streamlit chatbot
parser/                 page-aware parser, table-aware chunker
retrieval/              BM25, dense, hybrid fusion, cross-encoder reranker
pipeline/               LLM clients, classifier, verifiers, judge, decomposition
eval/                   answer key loader, harness, scoring, diagnostics
data/                   filings + practice questions (not in repo)
.cache/                 chunks, embeddings, rerank scores, LLM responses
```
