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

## Note on my sandbox vs your machine

I can read and write files in this folder, but I run Python in a separate Linux
sandbox with no PyPI access — so I can't install these packages or download
models myself. BM25 and dense retrieval get **written** here by me and **run**
by you. Paste the output back and we'll compare the ladder honestly.
