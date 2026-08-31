# The Analyst Copilot

A chatbot that answers analyst questions about SEC filings and shows exactly
where each answer came from — document, page number, and the exact line of text
it was drawn from. When the filing does not contain the answer, it says so
plainly rather than guessing.

---

## Our approach

The brief is explicit that a wrong figure is worse than no figure: a correct
answer in the correct place scores `+1`, an honest "not found in this filing"
scores `0`, and a confident wrong answer scores `−1`. A system that guesses
finishes below zero.

That single asymmetry shaped every decision in this project. The engineering
problem was never "answer as many questions as possible" — it was **building a
system that knows the difference between an answer it can prove and one it
cannot.**

We approached that in three parts.

### 1. Preserve the structure that makes a number meaningful

A financial figure means nothing without its row label and its year column.
`1,577` is noise; `Purchases of property, plant and equipment (PP&E) | (1,577) |
(1,373)` is an answer. Most of the value in this system comes from never breaking
that link.

The parser splits filings on their real page boundaries, strips the invisible
machine-readable XBRL metadata that would otherwise pollute search results, and
reads each filing's own table of contents to determine the page number *printed*
on each page — which is what an analyst would actually turn to. The chunker then
keeps every table whole, whatever its size, and splits prose only at sentence
boundaries.

We validated this directly against the answer key: of the proving passages it
contains, **zero were split across chunk boundaries.**

### 2. Retrieve with two complementary methods, then re-rank

Keyword search and semantic search succeed on different questions. BM25 finds
exact line items and fiscal years; embeddings find "capital expenditure" when the
filing says "Purchases of property, plant and equipment." We measured that they
recover genuinely different questions, which is what makes combining them
worthwhile rather than reflexive.

Both are fused at **page level** — because the product cites a page, and because
scoring whole pages yields *k* independent candidate pages instead of *k* chunks
that often come from the same one. A cross-encoder then re-reads the strongest
candidates together with the question, catching relevance that vector similarity
alone misses.

A lightweight classifier detects questions that define a formula — a ratio, a
growth rate, a multi-year average — and widens the evidence budget for those,
since they need figures drawn from several statements rather than one.

### 3. Prove the citation mechanically, not with another model

This is the heart of the system.

The answering model is required to quote **verbatim** from the evidence it was
given. A separate verifier then looks that quote up in the actual filing and
confirms that every figure in it appears on the page cited.

**That verifier uses no language model.** It is a normalised substring check in
plain Python. It cannot be persuaded, cannot hallucinate agreement, and costs
nothing to run. If the quote does not hold up, the answer is not shown.

The verbatim requirement exists precisely so this check is possible. It is what
turns "the model says page 60" into "we confirmed this text is on page 60."

### What we keep, and what we strip

Several of the most useful decisions in this project were about what to *remove*
from the pipeline before anything else touches it.

| Kept deliberately | Why |
|---|---|
| **Tables whole, never split** | Splitting a table separates a number from the row label and year column that give it meaning — the most common way a system produces a figure that is real but attached to the wrong thing. |
| **Both page numbers** | The sheet index (used internally for evaluation) and the number printed on the page (shown to the user). One field cannot serve both conventions correctly. |
| **Verbatim quotes** | The only thing that makes mechanical citation checking possible. A paraphrase cannot be looked up. |
| **Two retrieval methods** | Keyword and semantic search recover measurably different questions; keeping both raises the ceiling for the whole system. |

| Stripped deliberately | Why |
|---|---|
| **Invisible XBRL metadata** | Filings carry a machine-readable copy of every figure inside hidden blocks. Left in, a search can match a number that never appears visibly on the page. Removed before any text is extracted. |
| **Question boilerplate** | Most analyst questions open with framing text ("Assume that you are a public equities analyst…"). Those words appear throughout filings too, so removing them from the query sharpens retrieval measurably. |
| **Competing numbers in the evidence header** | Each excerpt is labelled with its page and nothing else. When the header carried more than one number, the model could pick the wrong one; with a single number present there is nothing to confuse. |
| **Documents whose evidence is absent** | A filing that does not contain the answer to its own question is identified up front and declined, rather than answered from an unrelated document. |

---

## How the base requirements are met

| Requirement from the brief | How it is delivered |
|---|---|
| **An "Add filing" control** — upload a filing it has never seen, with visible processing status | Sidebar uploader accepting SEC inline-XBRL `.htm` files. Shows staged progress — reading → splitting into pages → chunking → embedding — each stage reporting a real count. |
| **Adding one filing must complete within 10 minutes** | Measured at **~40 seconds** for a 10 MB annual report (parse 4 s, chunk <1 s, embed ~35 s) — comfortably inside the budget. |
| **A chat box** — analyst questions in plain English | Natural-language chat scoped to the selected filing, with conversation history. |
| **Evidence on every answer** — the document and page it came from | Every answer displays the document name, the page number, the exact quoted row, and a **CITATION VERIFIED** badge confirming the quote was checked against the source. |
| **The ability to decline** — "not found in this filing," stated plainly | Explicit decline states, each naming which check stopped the answer, so the user knows whether the filing lacks the data or the citation could not be confirmed. |

---

## Architecture

```
                        filing.htm  (SEC inline-XBRL)
                                │
              ┌─────────────────▼─────────────────┐
              │        PAGE-AWARE PARSER          │
              │  • split on page-break markers    │
              │  • strip invisible XBRL metadata  │
              │  • read TOC → printed page numbers│
              └─────────────────┬─────────────────┘
                                │  pages
              ┌─────────────────▼─────────────────┐
              │       TABLE-AWARE CHUNKER         │
              │  • tables kept whole, never split │
              │  • prose in sentence windows      │
              │  • every chunk tagged with page   │
              └─────────────────┬─────────────────┘
                                │  chunks  ──────────► cached to disk
                                │
   question ─────┐              │
                 ▼              ▼
      ┌──────────────────┐  ┌──────────────────┐
      │  BM25 (lexical)  │  │ DENSE (semantic) │
      │ exact line items │  │   paraphrases    │
      └────────┬─────────┘  └─────────┬────────┘
               └──────────┬───────────┘
                          ▼
                 ┌────────────────┐
                 │  PAGE-LEVEL    │   k independent page candidates,
                 │  FUSION        │   not k chunks from one page
                 └───────┬────────┘
                         ▼
                 ┌────────────────┐
                 │ CROSS-ENCODER  │   re-reads question + passage together
                 │ RE-RANKER      │
                 └───────┬────────┘
                         ▼
                 ┌────────────────┐
                 │ QUESTION       │   K=10 for direct lookups
                 │ CLASSIFIER     │   K=16 when a formula is defined
                 └───────┬────────┘
                         ▼
              ┌──────────────────────┐
              │ VERIFIER #1  (LLM)   │  is this evidence sufficient?
              └───────┬──────────────┘
                      │ sufficient          insufficient ──► "not found
                      ▼                                       in this filing"
              ┌──────────────────────┐
              │   ANSWERING MODEL    │  answer + VERBATIM quote + page
              └───────┬──────────────┘
                      ▼
              ┌──────────────────────┐
              │ VERIFIER #2  (CODE)  │  is that quote really on that page?
              │   no LLM involved    │  every figure must match
              └───────┬──────────────┘
                      │ verified            not confirmed ──► decline
                      ▼
        ┌──────────────────────────────────┐
        │  ANSWER + document + page +      │
        │  exact quote + VERIFIED badge    │
        └──────────────────────────────────┘
```

---

## Beyond the brief

**Mechanically proven citations.** Most systems ask a model to cite its source and
trust the reply. Ours checks — in code, against the filing, every figure. This is
the difference between claiming a citation and proving one, and it is what an
auditor or credit team would actually need.

**Page numbers an analyst can use.** A 10-K places roughly fourteen pages of cover
material and contents before printed page 1, so the sixtieth sheet is usually
labelled "46". We calibrate against each filing's own table of contents and
display the printed number — the one a user would turn to — while retaining the
sheet index internally for evaluation.

**Two decline reasons, not one.** "The filing does not appear to contain this" and
"I found an answer but could not confirm its citation" are different facts.
Surfacing which check fired lets an analyst decide whether to look elsewhere or
look again.

**Question-aware evidence sizing.** A question asking for one figure and a question
defining a three-year average ratio need very different amounts of context. A
classifier detects formula language and widens the evidence budget accordingly,
which measurably improved retrieval for calculation questions.

**A complete evaluation harness.** Built *before* any retriever, and validated
against deliberately naive baselines to confirm it could distinguish good
retrieval from bad before it was trusted to judge anything real. Every component
in the system was then selected by measurement against the answer key.

**Source-integrity checking.** Before trusting any measurement, we verified each
supplied filing against the questions that reference it, confirming the required
evidence is genuinely present in the document. This surfaced a set of files whose
content does not match their questions — which the system identifies and declines
rather than answering from an unrelated document. The same check runs on any
filing a user uploads, so a truncated or mismatched file is caught at upload time
rather than producing confident nonsense later.

**Resumable, cache-backed evaluation.** Parsed chunks, embeddings, re-ranker
scores and model responses are all cached by content hash, so evaluation runs
resume where they stopped and repeated analysis costs nothing.

---

## Why this project stands out

**It is engineered for the rubric it will be judged by.** The rubric rewards
provable answers and penalises confident errors, so the system is built around a
verification step rather than around answer generation.

**The central safety mechanism is deterministic.** Verifier #2 is plain code, not
another model with its own failure modes. A quote either appears on the cited page
or it does not — a claim anyone can check by reading the repository.

**Every design decision has a measurement behind it.** The retrieval stack was
assembled one rung at a time, each compared against the answer key. Several
results were counter-intuitive — a 90 MB re-ranking model outperformed one twelve
times its size — and the numbers, not intuition, decided.

**The evaluation is thorough.** We built an LLM judge to grade the descriptive
answers a string comparator cannot, ran paired significance tests on our own
conclusions, and refined our metrics as we learned more about the data. The
repository contains the diagnostic scripts that produced every figure quoted.

---

## Setting it up

### Requirements

Python 3.10+, roughly 2 GB of disk for dependencies and cached models, and a free
LLM API key.

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
```

If PowerShell blocks the activate script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 2. Install dependencies

Install PyTorch first from the CPU index — this keeps it to ~200 MB rather than
pulling the 2.5 GB CUDA build:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Add the data pack

```
data/filings/*.htm
data/practice-questions.jsonl
```

### 4. Set an API key

Free key from [console.groq.com/keys](https://console.groq.com/keys):

```powershell
$env:GROQ_API_KEY = 'gsk_...'
```

To persist it across terminals (one line, then reopen the terminal):

```powershell
[Environment]::SetEnvironmentVariable('GROQ_API_KEY','gsk_...','User')
```

Google Gemini is also supported via `GEMINI_API_KEY`, and a local Ollama model
via `LLM_PROVIDER=ollama`.

Verify the setup:

```powershell
python pipeline/llm.py --check
```

### 5. Run the chatbot

```powershell
streamlit run app.py
```

Opens at `http://localhost:8501`.

**The first question is slower** — it downloads the embedding model (~90 MB) and
the re-ranker (~90 MB) from Hugging Face, then embeds the selected filing.
Everything caches afterwards; later questions take a few seconds.

**Launch from the activated virtual environment.** If `where streamlit`
(PowerShell) or `which streamlit` points outside `.venv`, it is the wrong
interpreter and imports will not resolve.

---

## Running the evaluation

```powershell
python eval/run_pipeline.py --limit 30      # a batch, resumable
python eval/run_pipeline.py --report        # results so far, zero API calls
python eval/run_pipeline.py --judge         # also grade descriptive answers
```

Component-level evaluations, each runnable independently:

```powershell
python eval/validate_parser.py       # page splitting vs the answer key
python eval/validate_chunks.py       # does evidence survive chunking?
python eval/check_printed_pages.py   # printed page numbers vs page footers
python eval/run_bm25.py --all        # lexical retrieval and tuning sweeps
python eval/run_dense.py             # embedding retrieval
python eval/run_hybrid.py --all      # fusion, and the evidence for it
python eval/run_rerank.py            # cross-encoder comparison
python eval/run_classifier.py        # question typing and evidence sizing
python eval/validate_choice.py       # significance tests on the retrieval config
```

---

## Repository layout

```
app.py                  Streamlit chatbot
parser/                 page-aware parser, table-aware chunker
retrieval/              BM25, dense embeddings, hybrid fusion, cross-encoder
pipeline/               LLM clients, question classifier, verifiers, judge
eval/                   answer key loader, harness, scoring, diagnostics
data/                   filings and practice questions (not in repository)
.cache/                 chunks, embeddings, re-ranker scores, model responses
```
