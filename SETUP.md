# Setup — The Analyst Copilot

Everything runs from this folder. Two separate downloads are involved, and it
helps to keep them straight:

| What | From | How |
|---|---|---|
| Python packages (`rank-bm25`, `sentence-transformers`, `torch`) | **PyPI** (pypi.org) | `pip install` |
| Embedding model weights (`all-MiniLM-L6-v2` etc.) | **Hugging Face** (huggingface.co) | downloaded **automatically** on first use — you never visit the site |

You do not manually download anything from Hugging Face. You name a model in
code, and `sentence-transformers` fetches and caches it for you.

---

## 1. Create a virtualenv

Open a terminal in this folder (`analyst_hackathon`).

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks the activate script, run this once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

In VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter" → pick `.venv`.

---

## 2. Install torch first (Windows/CPU only — this step saves ~2 GB)

`sentence-transformers` depends on PyTorch. If you let pip resolve it normally
on Windows it may pull the CUDA build (~2.5 GB) even without an NVIDIA GPU.
Installing the CPU wheel first avoids that:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Skip this if you have an NVIDIA GPU and want CUDA, or if you're on macOS.

---

## 3. Install the rest

```bash
pip install -r requirements.txt
```

Rough download sizes: `rank-bm25` is trivial (<50 KB), `torch` CPU is ~200 MB,
`sentence-transformers` ~50 MB.

---

## 4. Verify

```bash
python eval/run_baselines.py
```

You should see the gold-set summary, `scoring self-test: PASS`, and the three
baseline retrievers with `page_recall@5` roughly: random `0.055`,
first-k `0.024`, keyword overlap `0.157`.

If that runs, Days 1–2 are working on your machine.

---

## 5. The embedding model (Day 3)

The first time you run the dense retriever, this happens automatically:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```

That line downloads ~90 MB from Hugging Face and caches it in
`~/.cache/huggingface/` (Windows: `C:\Users\<you>\.cache\huggingface\`).
Every later run loads from cache — no network needed.

**Model options**, cheapest first:

| Model | Size | Notes |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB | fast, the standard baseline — start here |
| `BAAI/bge-small-en-v1.5` | ~130 MB | usually stronger on retrieval; needs a query prefix |
| `BAAI/bge-base-en-v1.5` | ~440 MB | stronger again, noticeably slower on CPU |

Start with MiniLM. Swapping models later is a one-line change, and the eval
harness will tell you whether the bigger model actually earned its runtime —
which is the entire point of building the harness before the retrievers.

**Heads-up on scale:** the corpus is ~48,600 chunks. Embedding all of it with
MiniLM on CPU takes roughly 10–25 minutes the first time. Embeddings are cached
to disk after that, so it's a one-off cost.

---

## 6. The frontend (Streamlit app)

`app.py` is a browser UI over the exact same pipeline `eval/run_pipeline.py`
runs from the CLI: retrieve → Verifier #1 (sufficiency gate) → answer →
Verifier #2 (citation check).

```powershell
pip install -r requirements.txt        # now includes streamlit + watchdog
$env:GROQ_API_KEY = 'gsk_...'           # same key the CLI uses
streamlit run app.py
```

It opens at http://localhost:8501.

Notes:

- **It needs the data pack** (`data/filings/*.htm`, `data/practice-questions.jsonl`)
  and an LLM key (`GROQ_API_KEY`, or `GEMINI_API_KEY`) in the shell you launch
  `streamlit` from — set them *before* `streamlit run`, not in the browser.
- **First query is slow.** Loading a filing triggers the MiniLM embedding
  download + that filing's embeddings, plus the ~90 MB cross-encoder reranker.
  Everything caches to `.cache/` afterwards; later questions are fast.
- **Launch from the activated venv.** If `where streamlit` (PowerShell) points
  outside `.venv\Scripts\`, it's the wrong interpreter and imports like
  `sentence-transformers` will fail — `.\.venv\Scripts\Activate.ps1` first.
- **Adding filings from the UI:** the sidebar has an uploader that accepts SEC
  filings as inline-XBRL `.htm` (as downloaded from EDGAR). It writes to
  `data/filings/` and parses on the spot; PDFs and plain HTML won't parse.

---

## Troubleshooting

**`GROQ_API_KEY is not set`** — set it in the shell you are running from, not a
different one. `[Environment]::SetEnvironmentVariable(...,'User')` only affects
terminals opened afterwards.

**`HTTP 403: error code 1010`** — Cloudflare rejecting the client, not a bad key.
The HTTP client sends a proper User-Agent to avoid this; if you see it, check you
are on the current `pipeline/llm.py`.

**`Groq quota exhausted -- wait NNNNs`** — the daily token cap, not per-minute
throttling. Everything is cached, so re-running after the reset resumes where it
stopped rather than starting over.

**Rate-limited mid-run** — raise `--rate` (seconds between calls). Counter-
intuitively this is *faster* than being throttled, because a 429 costs a full
retry ladder and still fails.

**`sentence-transformers` import errors** — you are on the wrong interpreter.
Activate the venv first; check with `where streamlit` (PowerShell) or
`which streamlit`.
