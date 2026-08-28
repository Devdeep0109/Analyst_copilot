"""
The Analyst Copilot — Streamlit frontend.
Run with: streamlit run app.py

Pipeline reproduced here (same as eval/run_pipeline.py):
    retrieve -> Verifier #1 (sufficiency gate) -> answer -> Verifier #2 (citation)
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# Pipeline imports
from eval.corpus import load_chunks, load_corpus
from eval.gold import load_gold
from pipeline.answer import Answerer
from pipeline.classify import classify, evidence_k
from pipeline.verify import Verifier
from pipeline.verify2 import check_citation
from pipeline.llm import get_client
from retrieval.hybrid import HybridRetriever
from retrieval.rerank import MS_MARCO, RerankRetriever

PROJECT_ROOT = Path(__file__).resolve().parent
FILINGS_DIR = PROJECT_ROOT / "data" / "filings"

# ============================================================================
# Page config & custom styling
# ============================================================================
st.set_page_config(
    page_title="The Analyst Copilot",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Professional styling - clean, modern, financial-tooling aesthetic
st.markdown("""
<style>
    /* Base styling */
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    
    /* Caption styling */
    .caption-text {
        font-size: 1.1rem;
        color: #475569;
        font-style: italic;
        margin-bottom: 2rem;
    }
    
    /* Cards and containers */
    .stContainer {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button[kind="primary"] {
        background: #0f766e;
        border: none;
        color: white;
        padding: 0.5rem 2rem;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #115e59;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Text area */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        font-size: 1rem;
        padding: 0.75rem;
        background: white;
    }
    
    .stTextArea textarea:focus {
        border-color: #0f766e;
        box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.1);
    }
    
    /* Select box */
    .stSelectbox select {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
    }
    
    /* Success/Warning/Info boxes */
    .stSuccess, .stWarning, .stInfo, .stError {
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid;
    }
    
    .stSuccess {
        background: #f0fdf4;
        border-color: #86efac;
        color: #166534;
    }
    
    .stWarning {
        background: #fef3c7;
        border-color: #fcd34d;
        color: #92400e;
    }
    
    .stInfo {
        background: #eff6ff;
        border-color: #93c5fd;
        color: #1e40af;
    }
    
    .stError {
        background: #fef2f2;
        border-color: #fca5a5;
        color: #991b1b;
    }
    
    /* Code blocks */
    .stCode {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        border-radius: 8px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #f8fafc;
    }
    
    /* Blockquote styling */
    blockquote {
        border-left: 4px solid #0f766e;
        margin: 1rem 0;
        padding: 0.75rem 1rem;
        background: #f0fdf4;
        border-radius: 0 8px 8px 0;
        color: #334155;
        font-style: italic;
    }
    
    /* Example buttons */
    .stButton > button[kind="secondary"] {
        background: white;
        border: 1px solid #cbd5e1;
        color: #475569;
        font-size: 0.9rem;
    }
    
    .stButton > button[kind="secondary"]:hover {
        border-color: #0f766e;
        color: #0f766e;
        background: #f0fdf4;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border-color: #e2e8f0;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #cbd5e1;
        border-radius: 8px;
        padding: 1rem;
        transition: border-color 0.2s;
    }
    
    .stFileUploader:hover {
        border-color: #0f766e;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Cached resources (models / clients are expensive to build)
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_llm_client():
    """One LLM client, shared by the answerer and Verifier #1 (shared rate limit)."""
    return get_client()


@st.cache_resource(show_spinner=False)
def load_retriever():
    """Hybrid + cross-encoder rerank retriever, built once."""
    return RerankRetriever(
        model_name=MS_MARCO,
        depth=20,
        base=HybridRetriever(method="weighted", candidate_pool=50),
    )


@st.cache_resource(show_spinner=False)
def load_answerer(_client):
    return Answerer(client=_client)


@st.cache_resource(show_spinner=False)
def load_verifier(_client):
    """Verifier #1 — the sufficiency gate. Abstains before answering when the
    retrieved evidence does not contain the facts needed."""
    return Verifier(client=_client)


@st.cache_resource(show_spinner=False)
def load_corpus_cached():
    """doc_name -> list[Chunk], for every filing on disk."""
    doc_names = sorted(p.stem for p in FILINGS_DIR.glob("*.htm"))
    return load_corpus(doc_names)


@st.cache_resource(show_spinner=False)
def load_gold_cached():
    try:
        return load_gold()
    except Exception:
        return []


# ============================================================================
# Session state
# ============================================================================
st.session_state.setdefault("history", [])
st.session_state.setdefault("indexed_docs", set())
st.session_state.setdefault("current_result", None)
st.session_state.setdefault("current_question", "")

# ============================================================================
# Header
# ============================================================================
st.markdown('<p class="main-title">📊 The Analyst Copilot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="caption-text">Answers from SEC filings, with a page citation you '
    'can check — or an honest "not found".</p>',
    unsafe_allow_html=True
)

# ---- API key check ----------------------------------------------------------
groq_key = os.environ.get("GROQ_API_KEY", "").strip()
gemini_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
if not groq_key and not gemini_key:
    st.error(
        "**No LLM API key found.**\n\n"
        "Set one in the shell you launched Streamlit from, then restart:\n\n"
        "```powershell\n"
        "$env:GROQ_API_KEY = 'gsk_...'      # free key: https://console.groq.com/keys\n"
        "```"
    )
    st.stop()

# ---- load everything -------------------------------------------------------
try:
    with st.spinner("Loading filings corpus (first run parses every filing — minutes)..."):
        corpus = load_corpus_cached()
    if not corpus:
        st.error(
            "**Corpus is empty.** Restore `data/filings/*.htm` per SETUP.md, then restart."
        )
        st.stop()
except Exception as e:
    st.error(
        "**Failed to load corpus.**\n\n"
        "Restore `data/filings/*.htm` and `data/practice-questions.jsonl` per "
        f"SETUP.md, then restart.\n\nError: {e}"
    )
    st.stop()

with st.spinner("Loading retriever + reranker model (~90 MB on first run)..."):
    client = get_llm_client()
    retriever = load_retriever()
    answerer = load_answerer(client)
    verifier = load_verifier(client)

# Filing list: the ones that have gold questions, plus anything added at runtime.
gold_questions = load_gold_cached()
gold_docs = {q.doc_name for q in gold_questions}
available_docs = sorted(d for d in corpus if d in gold_docs) or sorted(corpus)

# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.header("📁 Filing")
    selected_doc = st.selectbox(
        "Choose a filing to query",
        options=available_docs,
        index=0,
        help="The pipeline answers against exactly one filing at a time.",
    )

    st.divider()
    st.subheader("➕ Add a Filing")
    st.caption(
        "Upload a SEC filing as inline-XBRL **.htm** (as downloaded from EDGAR). "
        "It is saved to `data/filings/` and parsed on the spot."
    )
    upload = st.file_uploader("Filing (.htm)", type=["htm", "html"], label_visibility="collapsed")
    if upload is not None:
        raw_name = re.sub(r"\.html?$", "", upload.name, flags=re.I)
        doc_name = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_name).strip("_") or "uploaded_filing"
        dest = FILINGS_DIR / f"{doc_name}.htm"
        if st.button(f"Add \"{doc_name}\"", use_container_width=True):
            try:
                FILINGS_DIR.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(upload.getvalue())
                with st.spinner(f"Parsing {doc_name}..."):
                    chunks = load_chunks(doc_name)
                if not chunks:
                    dest.unlink(missing_ok=True)
                    st.error(
                        f"{doc_name} parsed to zero chunks — it is probably not an "
                        "inline-XBRL filing, or the download is truncated."
                    )
                else:
                    corpus[doc_name] = chunks          # cached dict is the same object
                    st.session_state.indexed_docs.discard(doc_name)
                    st.success(f"Added {doc_name} ({len(chunks)} chunks). Pick it above.")
                    st.rerun()
            except Exception as e:
                dest.unlink(missing_ok=True)
                st.error(f"Could not add {doc_name}: {e}")

    st.divider()
    st.markdown("**📊 How Answers Are Scored**")
    st.markdown(
        "**+1** correct answer with a mechanically verified page citation\n\n"
        "**0** an honest \"not found\" when the evidence isn't there\n\n"
        "**−1** a confident answer that turns out wrong\n\n"
        "The pipeline runs a sufficiency gate *before* answering and a substring "
        "citation check *after*, and abstains at either step."
    )
    
    st.divider()
    st.markdown("**🔑 API Status**")
    st.markdown(f"LLM: `{getattr(client, 'name', '?')}`")
    if groq_key:
        st.success("GROQ_API_KEY: ✅")
    else:
        st.info("GROQ_API_KEY: —")
    if gemini_key:
        st.success("GEMINI_API_KEY: ✅")
    else:
        st.info("GEMINI_API_KEY: —")

# ============================================================================
# Pipeline
# ============================================================================
def ensure_indexed(doc_name: str) -> None:
    if doc_name in st.session_state.indexed_docs:
        return
    if doc_name not in corpus:
        raise RuntimeError(f"Filing '{doc_name}' not in corpus.")
    with st.spinner(f"Indexing {doc_name} (BM25 + embeddings; cached after first time)..."):
        retriever.index(doc_name, corpus[doc_name])
    st.session_state.indexed_docs.add(doc_name)


def run_pipeline(question: str, doc_name: str) -> Dict[str, Any]:
    start = time.time()
    ensure_indexed(doc_name)

    cls = classify(question)
    k = evidence_k(cls)

    with st.spinner("Retrieving evidence..."):
        hits = retriever.search(doc_name, question, k)
    retriever.save_cache()

    if not hits:
        return {"question": question, "doc_name": doc_name, "hits": [], "cls": cls,
                "k": k, "verdict": None, "answer": None, "citation": None,
                "elapsed": time.time() - start}

    with st.spinner("Verifier #1 — is the evidence sufficient?"):
        verdict = verifier.judge(question, hits)
    verifier.save()

    with st.spinner("Answering..."):
        ans = answerer.answer(question, hits)
    answerer.save()

    cite = check_citation(ans.quote, ans.page, hits)

    return {"question": question, "doc_name": doc_name, "hits": hits, "cls": cls,
            "k": k, "verdict": verdict, "answer": ans, "citation": cite,
            "elapsed": time.time() - start}


# ============================================================================
# Result rendering
# ============================================================================
def render_result(r: Dict[str, Any], *, compact: bool = False) -> None:
    ans = r["answer"]
    cite = r["citation"]
    verdict = r["verdict"]

    if ans is None:
        st.info("**No evidence retrieved** for this question on this filing. Scores **0**.")
        return

    if ans.failed:
        st.error(f"**Answering failed:** {ans.quote}")
        return

    gated = verdict is not None and not verdict.sufficient

    if gated and ans.abstained:
        st.info(
            "**No supported answer — evidence is insufficient**\n\n"
            f"Verifier #1: *{verdict.reason}*\n\n"
            "Scores **0**, not −1. The needed figure isn't in the retrieved pages."
        )
        return
    if ans.abstained:
        st.info(
            "**No supported answer found**\n\n"
            "The model declined. Scores **0**, not −1 — the right call when the "
            "evidence is thin.\n\n"
            f"*{ans.quote or ''}*"
        )
        return

    # The model produced an answer.
    if compact:
        st.markdown(f"**Answer:** {ans.answer}")
    else:
        st.markdown(f"### 💡 Answer\n\n**{ans.answer}**")

    if ans.working and ans.working.strip():
        st.markdown("**🧮 Calculation:**")
        st.code(ans.working, language="text")

    if ans.quote and ans.quote.strip():
        st.markdown("**📄 Verbatim Quote:**")
        st.markdown(f"> {ans.quote.replace(chr(10), chr(10) + '> ')}")
    if ans.page is not None:
        st.markdown(f"**📍 Page {ans.page}**")

    # Verifier #2 — citation check
    if cite is not None and cite.label == "VERIFIED":
        st.success(f"✅ **Citation verified** — the quote is really on page {ans.page}.")
    else:
        reason = cite.reason if cite is not None else "not checked"
        st.warning(
            f"⚠️ **Citation NOT verified** — {reason}. The answer is shown, but its "
            "source could not be mechanically confirmed. Under the rubric this "
            "would be withheld (scores 0 instead of risking −1)."
        )

    # Verifier #1 — sufficiency gate
    if gated:
        st.warning(
            f"⚠️ **Verifier #1 flagged the evidence as insufficient:** *{verdict.reason}*. "
            "The pipeline would abstain here even though the model answered."
        )
    elif verdict is not None:
        st.caption(f"✓ Verifier #1: evidence sufficient — {verdict.reason}")

    if not compact:
        st.markdown("---")
        st.caption(
            f"**Filing:** {r['doc_name']} &nbsp;·&nbsp; "
            f"**Elapsed:** {r['elapsed']:.1f}s &nbsp;·&nbsp; "
            f"**Question type:** {r['cls'].label} &nbsp;·&nbsp; "
            f"**Pages retrieved:** {r['k']}"
        )
        
        summary = f"Answer: {ans.answer}\n"
        if ans.quote:
            summary += f'Quote: "{ans.quote}"\n'
        if ans.page is not None:
            summary += f"Page: {ans.page}\n"
        summary += f"Filing: {r['doc_name']}\n"
        st.markdown("**📋 Copy Summary:**")
        st.code(summary, language="text")


# ============================================================================
# Main area
# ============================================================================
st.markdown("### 💬 Ask a Question")
question = st.text_area(
    "Question",
    value=st.session_state.current_question,
    placeholder="e.g. What was the company's total revenue in the latest fiscal year?",
    height=100,
    label_visibility="collapsed",
)

st.markdown("**💡 Try an example:**")
examples = [
    "What was the total revenue (net sales) in the most recent fiscal year?",
    "What is total long-term debt on the balance sheet?",
    "Compute the operating margin for the latest fiscal year.",
]
example_cols = st.columns(3)
for col, ex in zip(example_cols, examples):
    if col.button(ex, use_container_width=True, key=f"ex_{ex}"):
        st.session_state.current_question = ex
        st.rerun()

ask = st.button("🔍 Ask", type="primary", use_container_width=True)

if ask and question.strip():
    st.session_state.current_question = question
    with st.container(border=True):
        try:
            result = run_pipeline(question, selected_doc)
            st.session_state.current_result = result
            st.session_state.history.insert(0, result)
            st.session_state.history = st.session_state.history[:10]
            render_result(result)
        except Exception as e:
            st.exception(e)
elif ask:
    st.warning("Please enter a question.")

# ============================================================================
# History
# ============================================================================
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📚 Session History")
    for i, r in enumerate(st.session_state.history):
        with st.expander(
            f"**{r['question'][:80]}** — {r['doc_name']} ({r['elapsed']:.1f}s)",
            expanded=(i == 0)
        ):
            render_result(r, compact=True)

st.markdown("---")
st.caption(
    "**The Analyst Copilot** · answers backed by verbatim quotes and verified "
    "page numbers — or an honest 'not found'."
)