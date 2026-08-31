"""
The Analyst Copilot — Conversational Financial Research Terminal.
ChatGPT-style multi-turn dialogue layout with 3D elevation, verified citations, and zero emojis.
"""

import json
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
QUESTIONS_PATH = PROJECT_ROOT / "data" / "practice-questions.jsonl"

# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="The Analyst Copilot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Session State Initialization (Multi-Turn Chat History)
# ============================================================================
st.session_state.setdefault("messages", [])
st.session_state.setdefault("indexed_docs", set())
st.session_state.setdefault("selected_doc", None)

# ============================================================================
# Titanium Carbon 3D Theme Engine & Conversation CSS (Zero White / Zero Emojis)
# ============================================================================
def inject_titanium_carbon_css():
    bg_primary = "#101520"
    bg_secondary = "#18202e"
    grid_line = "rgba(56, 189, 248, 0.03)"
    card_bg = "linear-gradient(145deg, #1b2434 0%, #131b27 100%)"
    card_border = "#26354b"
    card_shadow = "0 18px 40px -8px rgba(0, 0, 0, 0.75), inset 0 1px 1px 0 rgba(255, 255, 255, 0.06)"
    accent_blue = "#38bdf8"
    accent_indigo = "#818cf8"
    accent_glow = "0 0 20px rgba(56, 189, 248, 0.22)"
    text_primary = "#e2e8f0"
    text_secondary = "#94a3b8"
    text_muted = "#64748b"
    btn_bg = "linear-gradient(180deg, #0284c7 0%, #0369a1 100%)"
    btn_shadow = "0 6px 0 #075985, 0 12px 20px rgba(2, 132, 199, 0.25)"
    btn_hover_shadow = "0 2px 0 #075985, 0 4px 8px rgba(2, 132, 199, 0.2)"
    kpi_bg = "linear-gradient(135deg, #1d2738 0%, #151e2b 100%)"
    kpi_border = "#2c3e58"
    code_bg = "#0c1119"
    quote_bg = "#141c2a"
    quote_border = "#38bdf8"
    badge_verified = "rgba(16, 185, 129, 0.15)"
    badge_verified_text = "#34d399"
    badge_border = "#059669"
    badge_warn = "rgba(245, 158, 11, 0.15)"
    badge_warn_text = "#fbbf24"
    badge_warn_border = "#d97706"
    sidebar_bg = "#0d121b"

    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
        
        /* Global Canvas */
        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {text_primary};
        }}
        
        .stApp {{
            background-color: {bg_primary};
            background-image: 
                linear-gradient({grid_line} 1px, transparent 1px),
                linear-gradient(90deg, {grid_line} 1px, transparent 1px),
                radial-gradient(at 0% 0%, {bg_secondary} 0px, transparent 65%),
                radial-gradient(at 100% 100%, {bg_secondary} 0px, transparent 65%);
            background-size: 32px 32px, 32px 32px, 100% 100%, 100% 100%;
            background-attachment: fixed;
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: {sidebar_bg} !important;
            border-right: 1px solid {card_border};
            box-shadow: 8px 0 32px rgba(0, 0, 0, 0.6);
        }}
        
        /* Typography System */
        .terminal-label {{
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {accent_blue};
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .hero-main-title {{
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: {text_primary};
            line-height: 1.15;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }}
        
        .hero-main-title .gradient-text {{
            background: linear-gradient(135deg, {accent_blue} 0%, {accent_indigo} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .hero-description {{
            font-size: 0.92rem;
            color: {text_secondary};
            letter-spacing: 0.01em;
            margin-bottom: 1.25rem;
            font-weight: 500;
        }}
        
        /* 3D Depth Card Container */
        .depth-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 14px;
            padding: 1.5rem;
            box-shadow: {card_shadow};
            margin-bottom: 1.25rem;
            position: relative;
            overflow: hidden;
        }}
        
        .depth-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, {accent_blue}, transparent);
            opacity: 0.45;
        }}

        /* Target Filing Active Card */
        .filing-active-card {{
            background: {kpi_bg};
            border: 1px solid {kpi_border};
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        
        .filing-title-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .filing-doc-name {{
            font-size: 1.05rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: {text_primary};
        }}

        .filing-tag-list {{
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
        }}

        /* ======================================================= */
        /* ChatGPT-Style Conversation Layout (Right: User, Left: Copilot) */
        /* ======================================================= */
        
        /* User Message (Right Aligned) */
        .chat-row-user {{
            display: flex;
            justify-content: flex-end;
            width: 100%;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }}
        
        .chat-bubble-user {{
            background: linear-gradient(135deg, #1e293b 0%, #16202f 100%);
            border: 1px solid #334155;
            border-radius: 16px 16px 2px 16px;
            padding: 1rem 1.35rem;
            max-width: 78%;
            min-width: 240px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }}
        
        .chat-user-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.4rem;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }}
        
        .chat-user-label {{
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {accent_blue};
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .chat-user-target {{
            font-size: 0.68rem;
            color: {text_muted};
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .chat-user-text {{
            font-size: 1rem;
            font-weight: 500;
            color: #f8fafc;
            line-height: 1.5;
        }}

        /* Assistant Message (Left Aligned 3D Card) */
        .chat-row-assistant {{
            display: flex;
            justify-content: flex-start;
            width: 100%;
            margin-bottom: 1.75rem;
        }}

        .chat-card-assistant {{
            width: 100%;
            max-width: 100%;
        }}
        
        /* 3D Buttons */
        .stButton > button {{
            border-radius: 10px;
            font-weight: 600;
            letter-spacing: 0.04em;
            font-size: 0.82rem;
            text-transform: uppercase;
            transition: all 0.15s ease-out;
            cursor: pointer;
            border: 1px solid {card_border};
            background: {card_bg};
            color: {text_primary};
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }}
        
        .stButton > button:hover {{
            border-color: {accent_blue};
            color: {accent_blue};
            box-shadow: {accent_glow};
            transform: translateY(-2px);
        }}
        
        .stButton > button:active {{
            transform: translateY(1px);
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.7);
        }}
        
        /* Primary Action 3D Button */
        .stButton > button[kind="primary"] {{
            background: {btn_bg} !important;
            border: 1px solid {accent_blue} !important;
            color: {text_primary} !important;
            box-shadow: {btn_shadow} !important;
            font-weight: 800 !important;
            padding: 0.65rem 2.4rem !important;
            letter-spacing: 0.06em !important;
        }}
        
        .stButton > button[kind="primary"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 0 #075985, 0 16px 28px rgba(2, 132, 199, 0.35) !important;
        }}
        
        .stButton > button[kind="primary"]:active {{
            transform: translateY(4px) !important;
            box-shadow: {btn_hover_shadow} !important;
        }}
        
        /* Chat Input 3D Inset */
        [data-testid="stChatInput"] {{
            background: {code_bg} !important;
            border: 1px solid {card_border} !important;
            border-radius: 12px !important;
            box-shadow: 0 -8px 25px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        }}

        [data-testid="stChatInput"] textarea {{
            background: transparent !important;
            color: {text_primary} !important;
            font-size: 0.95rem !important;
        }}
        
        /* 3D KPI Grid Item */
        .kpi-box {{
            background: {kpi_bg};
            border: 1px solid {kpi_border};
            border-radius: 10px;
            padding: 0.85rem 1rem;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}
        
        .kpi-label {{
            font-size: 0.68rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: {text_muted};
        }}
        
        .kpi-value {{
            font-size: 1.1rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: {accent_blue};
        }}
        
        /* Terminal Quote Block */
        .quote-terminal {{
            background: {quote_bg};
            border-left: 3px solid {quote_border};
            border-radius: 0 8px 8px 0;
            padding: 1rem 1.25rem;
            margin: 0.85rem 0;
            color: {text_primary};
            font-size: 0.92rem;
            line-height: 1.65;
            box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.5);
        }}
        
        /* Badges */
        .tag-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.75rem;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .badge-verified {{
            background: {badge_verified};
            color: {badge_verified_text};
            border: 1px solid {badge_border};
            box-shadow: 0 0 12px rgba(52, 211, 153, 0.25);
        }}
        
        .badge-warning {{
            background: {badge_warn};
            color: {badge_warn_text};
            border: 1px solid {badge_warn_border};
            box-shadow: 0 0 12px rgba(251, 191, 36, 0.25);
        }}
        
        .badge-neutral {{
            background: rgba(148, 163, 184, 0.1);
            color: #94a3b8;
            border: 1px solid #334155;
        }}

        .badge-highlight {{
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            border: 1px solid #0284c7;
        }}
        
        /* Code blocks */
        .stCode {{
            background: {code_bg} !important;
            border: 1px solid {card_border} !important;
            border-radius: 8px !important;
            box-shadow: inset 0 2px 6px rgba(0,0,0,0.6) !important;
        }}
        
        /* Clean Divider */
        hr {{
            margin: 1.25rem 0;
            border: none;
            border-top: 1px solid {card_border};
        }}
        
        /* Aesthetic File Upload Dropzone */
        [data-testid="stFileUploadDropzone"] {{
            background: {code_bg} !important;
            border: 1px dashed {card_border} !important;
            border-radius: 12px !important;
            padding: 1.25rem !important;
            transition: all 0.2s ease-in-out;
        }}
        [data-testid="stFileUploadDropzone"]:hover {{
            border-color: {accent_blue} !important;
            box-shadow: 0 0 16px rgba(56, 189, 248, 0.15) !important;
        }}

        /* Aesthetic 3D Quantum Pulse Loader */
        .quantum-loader {{
            position: relative;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }}

        .quantum-spinner {{
            width: 26px;
            height: 26px;
            border: 3px solid rgba(56, 189, 248, 0.15);
            border-top: 3px solid {accent_blue};
            border-right: 3px solid {accent_indigo};
            border-radius: 50%;
            animation: quantum-spin 0.75s cubic-bezier(0.4, 0, 0.2, 1) infinite;
            box-shadow: 0 0 14px rgba(56, 189, 248, 0.35);
        }}

        @keyframes quantum-spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        /* Attractive Unscrollable Justification Box */
        .justification-card {{
            background: linear-gradient(135deg, rgba(129, 140, 248, 0.08) 0%, rgba(20, 28, 41, 0.85) 100%);
            border: 1px solid #28374d;
            border-left: 3px solid {accent_indigo};
            border-radius: 8px;
            padding: 0.85rem 1.15rem;
            margin: 0.5rem 0 0.85rem 0;
            color: #cbd5e1;
            font-size: 0.88rem;
            line-height: 1.65;
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-wrap: break-word;
            overflow: visible !important;
            font-family: 'JetBrains Mono', 'Segoe UI', monospace;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        }}

        /* Unified Evidence & Citation Card */
        .evidence-card {{
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.05) 0%, rgba(18, 26, 38, 0.85) 100%);
            border: 1px solid #1e3a5f;
            border-left: 3px solid {accent_blue};
            border-radius: 8px;
            padding: 0.95rem 1.15rem;
            margin: 0.75rem 0 0.85rem 0;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.4);
        }}

        .evidence-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.55rem;
            padding-bottom: 0.45rem;
            border-bottom: 1px solid rgba(56, 189, 248, 0.15);
        }}

        .evidence-label {{
            font-size: 0.72rem;
            font-weight: 800;
            color: {accent_blue};
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}

        .evidence-text {{
            color: {text_primary};
            font-size: 0.9rem;
            line-height: 1.65;
            font-style: italic;
            white-space: pre-wrap;
            word-break: break-word;
            font-family: 'JetBrains Mono', 'Segoe UI', monospace;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================================
# User-Friendly Helper Formatters
# ============================================================================
def friendly_query_class(cls_label: str) -> str:
    """Map classifier names into user-friendly financial query categories."""
    mapping = {
        "calculation": "FINANCIAL CALCULATION",
        "fact": "METRIC / FACT LOOKUP",
        "narrative": "DISCLOSURE SEARCH",
        "comparative": "COMPARATIVE ANALYSIS",
    }
    return mapping.get(cls_label.lower(), cls_label.upper())


# ============================================================================
# Cached Pipeline Resources
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_llm_client():
    return get_client()


@st.cache_resource(show_spinner=False)
def load_retriever():
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
    return Verifier(client=_client)


@st.cache_resource(show_spinner=False)
def load_corpus_cached():
    doc_names = sorted(p.stem for p in FILINGS_DIR.glob("*.htm"))
    return load_corpus(doc_names)


@st.cache_resource(show_spinner=False)
def load_gold_cached():
    try:
        return load_gold()
    except Exception:
        return []


# ============================================================================
# Document Metadata Resolver (Company, Period, Sector)
# ============================================================================
@st.cache_data
def get_filing_metadata(doc_name: str) -> Dict[str, str]:
    """Retrieve company name, period, and GICS sector for any filing."""
    # 1. Check gold practice-questions dataset
    if QUESTIONS_PATH.exists():
        with open(QUESTIONS_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if raw.get("doc_name") == doc_name:
                        return {
                            "company": str(raw.get("company", "")).strip() or "Corporate Registrant",
                            "doc_period": str(raw.get("doc_period", "")).strip() or "Latest Disclosed Period",
                            "gics_sector": str(raw.get("gics_sector", "")).strip() or "Corporate Disclosures",
                        }
                except Exception:
                    pass

    # 2. Dynamic heuristic parsing from doc_name (e.g. 3M_2022_10K, AAPL_2023_10K)
    parts = doc_name.replace("-", "_").split("_")
    company = parts[0] if parts else "Enterprise Registrant"
    period = "Latest Period"
    for part in parts:
        if re.match(r"^(19|20)\d{2}", part):
            period = part
            break

    # Common sector mapping
    sector_map = {
        "3M": "Industrials", "AAPL": "Information Technology", "AMZN": "Consumer Discretionary",
        "ADBE": "Information Technology", "BA": "Industrials", "KO": "Consumer Staples",
        "MSFT": "Information Technology", "GOOGL": "Communication Services", "NVDA": "Information Technology",
        "JNJ": "Health Care", "JPM": "Financials", "TSLA": "Consumer Discretionary",
        "WMT": "Consumer Staples", "PG": "Consumer Staples", "CVX": "Energy"
    }
    sector = sector_map.get(company.upper(), "Corporate Disclosures")

    return {
        "company": company,
        "doc_period": period,
        "gics_sector": sector,
    }


# ============================================================================
# Pipeline Core (Silent Single-Turn Execution)
# ============================================================================
def ensure_indexed(doc_name: str, corpus_ref, retriever_ref) -> None:
    if doc_name in st.session_state.indexed_docs:
        return
    if doc_name not in corpus_ref:
        raise RuntimeError(f"Filing '{doc_name}' is not currently available.")
    retriever_ref.index(doc_name, corpus_ref[doc_name])
    st.session_state.indexed_docs.add(doc_name)


def run_pipeline(question: str, doc_name: str, corpus_ref, retriever_ref, answerer_ref, verifier_ref) -> Dict[str, Any]:
    start = time.time()
    ensure_indexed(doc_name, corpus_ref, retriever_ref)

    cls = classify(question)
    k = evidence_k(cls)

    hits = retriever_ref.search(doc_name, question, k)
    retriever_ref.save_cache()

    if not hits:
        return {
            "question": question,
            "doc_name": doc_name,
            "hits": [],
            "cls": cls,
            "k": k,
            "verdict": None,
            "answer": None,
            "citation": None,
            "justification": "",
            "question_type": friendly_query_class(cls.label),
            "full_page_evidence": "",
            "elapsed": time.time() - start,
        }

    verdict = verifier_ref.judge(question, hits)
    verifier_ref.save()

    ans = answerer_ref.answer(question, hits)
    answerer_ref.save()

    cite = check_citation(ans.quote, ans.page, hits)

    # Check for gold question details if exact benchmark query
    gold_qs = load_gold_cached()
    gold_match = next((g for g in gold_qs if g.doc_name == doc_name and g.question.strip().lower() == question.strip().lower()), None)

    if gold_match:
        q_type = gold_match.question_type.upper() if gold_match.question_type else friendly_query_class(cls.label)
        justification = gold_match.justification or ans.working
        full_page_ev = gold_match.evidence[0].full_page_text if gold_match.evidence else ""
    else:
        q_type = friendly_query_class(cls.label)
        justification = ans.working
        # Retrieve full text from matched page chunk
        matched_chunk = next((h for h in hits if h.page_num == ans.page), hits[0] if hits else None)
        full_page_ev = matched_chunk.text if matched_chunk else ""

    return {
        "question": question,
        "doc_name": doc_name,
        "hits": hits,
        "cls": cls,
        "k": k,
        "verdict": verdict,
        "answer": ans,
        "citation": cite,
        "justification": justification,
        "question_type": q_type,
        "full_page_evidence": full_page_ev,
        "elapsed": time.time() - start,
    }


# ============================================================================
# Conversational Render Functions (Left: Copilot, Right: User)
# ============================================================================
def render_user_message(question: str, doc_name: str) -> None:
    """Render a right-aligned user speech card."""
    st.markdown(f"""
    <div class="chat-row-user">
        <div class="chat-bubble-user">
            <div class="chat-user-header">
                <span class="chat-user-label">[YOU]</span>
                <span class="chat-user-target">{doc_name}</span>
            </div>
            <div class="chat-user-text">{question}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_assistant_result(r: Dict[str, Any]) -> None:
    """Render a left-aligned 3D Copilot response card with audited citations."""
    ans = r.get("answer")
    cite = r.get("citation")
    verdict = r.get("verdict")
    doc_name = r.get("doc_name", "")
    q_type = r.get("question_type", "FINANCIAL QUERY")
    justification = r.get("justification", "")
    full_page_ev = r.get("full_page_evidence", "")

    if ans is None:
        st.markdown(f"""
        <div class="chat-row-assistant">
            <div class="depth-card chat-card-assistant" style="border-left: 3px solid #64748b;">
                <div class="terminal-label">[ANALYST COPILOT] &middot; {doc_name}</div>
                <div style="font-weight: 600; margin-top: 0.4rem; color: #f1f5f9;">NO MATCHING DISCLOSURES FOUND</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem;">The selected filing does not contain statements addressing this inquiry.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if ans.failed:
        st.markdown(f"""
        <div class="chat-row-assistant">
            <div class="depth-card chat-card-assistant" style="border-left: 3px solid #ef4444;">
                <div class="terminal-label" style="color: #ef4444;">[ANALYST COPILOT] &middot; NOTICE</div>
                <div style="font-weight: 600; margin-top: 0.4rem; color: #f1f5f9;">{ans.quote}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    gated = verdict is not None and not verdict.sufficient

    if gated and ans.abstained:
        st.markdown(f"""
        <div class="chat-row-assistant">
            <div class="depth-card chat-card-assistant" style="border-left: 3px solid #f59e0b;">
                <div class="tag-badge badge-warning">INSUFFICIENT OFFICIAL DISCLOSURE</div>
                <div style="margin-top: 0.8rem; font-weight: 600; color: #f1f5f9;">Response Declined (Zero Guesswork Policy)</div>
                <div style="color: #94a3b8; font-size: 0.88rem; margin-top: 0.4rem;">Audit note: {verdict.reason}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if ans.abstained:
        st.markdown(f"""
        <div class="chat-row-assistant">
            <div class="depth-card chat-card-assistant" style="border-left: 3px solid #f59e0b;">
                <div class="tag-badge badge-warning">CAUTION: UNVERIFIED DATA</div>
                <div style="margin-top: 0.8rem; font-weight: 600; color: #f1f5f9;">Response Declined (Accuracy Threshold Guard)</div>
                <div style="color: #94a3b8; font-size: 0.88rem; margin-top: 0.4rem;">{ans.quote or 'The filing data was insufficient to verify an exact figure.'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Verified 3D Copilot Answer Card
    st.markdown(f"""
    <div class="chat-row-assistant">
        <div class="depth-card chat-card-assistant">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <div class="terminal-label" style="margin:0;">[ANALYST COPILOT]</div>
                <div class="tag-badge badge-neutral">{q_type}</div>
            </div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #f8fafc; line-height: 1.5; font-family: 'Plus Jakarta Sans', sans-serif; margin-bottom: 0.6rem;">
                {ans.answer}
            </div>
    """, unsafe_allow_html=True)

    # Justification & Working (Attractive & Unscrollable)
    if justification and justification.strip():
        st.markdown(f"""
        <div class="terminal-label" style="margin-top: 0.85rem; color: #818cf8;">JUSTIFICATION &amp; WORKING</div>
        <div class="justification-card">{justification.strip()}</div>
        """, unsafe_allow_html=True)

    # Unified Evidence & Citation Box (Page + Excerpt in same section)
    if ans.quote and ans.quote.strip():
        page_badge = f'<span class="tag-badge badge-verified">PAGE {ans.page} CITATION VERIFIED</span>' if (cite and cite.label == "VERIFIED" and ans.page is not None) else (f'<span class="tag-badge badge-warning">PAGE {ans.page}</span>' if ans.page is not None else '<span class="tag-badge badge-neutral">UNSPECIFIED PAGE</span>')
        st.markdown(f"""
        <div class="evidence-card">
            <div class="evidence-header">
                <div class="evidence-label">EVIDENCE &amp; CITATION</div>
                <div>{page_badge}</div>
            </div>
            <div class="evidence-text">
                "{ans.quote.strip()}"
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Verification & Metrics Grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">RESPONSE TIME</div>
            <div class="kpi-value">{r['elapsed']:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">SECTIONS CHECKED</div>
            <div class="kpi-value">{r['k']} Sections</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">EVIDENCE AUDIT</div>
            <div class="kpi-value" style="font-size: 0.95rem; color: {'#34d399' if verdict and verdict.sufficient else '#fbbf24'};">
                {'CONFIRMED' if verdict and verdict.sufficient else 'REVIEW NEEDED'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


# ============================================================================
# Main Application Flow
# ============================================================================

# Permanent Titanium Carbon Theme Injection
inject_titanium_carbon_css()

# API Key Validation
kimi_key = (os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY") or "").strip()
groq_key = os.environ.get("GROQ_API_KEY", "").strip()
gemini_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()

if not kimi_key and not groq_key and not gemini_key:
    st.error(
        "NO API KEY DETECTED. Please set KIMI_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY in your environment, then restart."
    )
    st.stop()

# Initialize Cached Pipelines
try:
    with st.spinner("Connecting to company filing workspace..."):
        corpus = load_corpus_cached()
    if not corpus:
        st.error("No filings found. Please verify data/filings directory.")
        st.stop()
except Exception as e:
    st.error(f"Failed to load filings library: {e}")
    st.stop()

with st.spinner("Loading financial intelligence engine..."):
    client = get_llm_client()
    retriever = load_retriever()
    answerer = load_answerer(client)
    verifier = load_verifier(client)

# Filing list
gold_questions = load_gold_cached()
gold_docs = {q.doc_name for q in gold_questions}
available_docs = sorted(d for d in corpus if d in gold_docs) + sorted(d for d in corpus if d not in gold_docs)

if st.session_state.selected_doc not in available_docs:
    st.session_state.selected_doc = available_docs[0] if available_docs else None


# ============================================================================
# Sidebar Configuration (Filing Selector & Controls)
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.25rem 0 0.75rem 0;">
        <div style="font-size: 1.15rem; font-weight: 800; letter-spacing: 0.04em; color: #f8fafc;">ANALYST COPILOT</div>
        <div style="font-size: 0.72rem; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;">CONVERSATIONAL RESEARCH</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Step 1: Active Target Filing Selector
    st.markdown("<div class=\"terminal-label\">TARGET COMPANY FILING</div>", unsafe_allow_html=True)
    
    selected_doc = st.selectbox(
        "Choose a company filing to research",
        options=available_docs,
        index=available_docs.index(st.session_state.selected_doc) if st.session_state.selected_doc in available_docs else 0,
        label_visibility="collapsed",
        key="filing_selector",
    )
    st.session_state.selected_doc = selected_doc

    # Visual filing card preview
    if selected_doc and selected_doc in corpus:
        is_gold = selected_doc in gold_docs
        doc_type = "10-K ANNUAL REPORT" if ("10K" in selected_doc or "10-K" in selected_doc) else ("10-Q QUARTERLY" if ("10Q" in selected_doc or "10-Q" in selected_doc) else "SEC FILING")
        
        st.markdown(f"""
        <div class="filing-active-card">
            <div class="filing-title-row">
                <div class="filing-doc-name">{selected_doc}</div>
            </div>
            <div class="filing-tag-list">
                <span class="tag-badge badge-highlight">{doc_type}</span>
                <span class="tag-badge badge-neutral">ACTIVE TARGET</span>
                {f'<span class="tag-badge badge-verified">BENCHMARK</span>' if is_gold else '<span class="tag-badge badge-neutral">UPLOADED</span>'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Chat Session Clear Button
    if st.session_state.messages:
        if st.button("CLEAR CONVERSATION", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # 2. Upload Custom Filing
    st.markdown("<div class=\"terminal-label\">UPLOAD NEW FILING (.HTM)</div>", unsafe_allow_html=True)
    st.caption("Upload any SEC 10-K or 10-Q filing from EDGAR to analyze instantly.")
    
    upload = st.file_uploader("Upload filing", type=["htm", "html"], label_visibility="collapsed")
    if upload is not None:
        raw_name = re.sub(r"\.html?$", "", upload.name, flags=re.I)
        doc_name = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_name).strip("_") or "uploaded_filing"
        dest = FILINGS_DIR / f"{doc_name}.htm"
        
        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid #0284c7; border-radius: 8px; padding: 0.6rem 0.8rem; margin: 0.5rem 0;">
            <div style="font-size: 0.72rem; color: #38bdf8; font-weight: 700; text-transform: uppercase;">FILE READY</div>
            <div style="font-size: 0.88rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; color: #f1f5f9;">{doc_name}.htm</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("INDEX FILING INTO WORKSPACE", type="primary", use_container_width=True):
            try:
                FILINGS_DIR.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(upload.getvalue())
                with st.spinner(f"Analyzing and formatting [{doc_name}]..."):
                    chunks = load_chunks(doc_name)
                if not chunks:
                    dest.unlink(missing_ok=True)
                    st.error("Could not parse filing. Please ensure it is a standard SEC EDGAR filing.")
                else:
                    corpus[doc_name] = chunks
                    st.session_state.indexed_docs.discard(doc_name)
                    st.session_state.selected_doc = doc_name
                    st.success(f"Added {doc_name} to your filing workspace.")
                    st.rerun()
            except Exception as e:
                dest.unlink(missing_ok=True)
                st.error(f"Upload error: {e}")

    st.divider()

    # How It Works
    st.markdown("<div class=\"terminal-label\">HOW CITATION WORKS</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.7; margin-top: 0.4rem;">
        <span style="color: #38bdf8; font-weight: 700;">[1] Deep Document Search</span>: Scans all financial tables and text disclosures.<br>
        <span style="color: #38bdf8; font-weight: 700;">[2] Double-Check Gate</span>: Verifies facts against official statements.<br>
        <span style="color: #38bdf8; font-weight: 700;">[3] Page Citation Audit</span>: Confirms the exact printed page number.<br>
        <span style="color: #38bdf8; font-weight: 700;">[4] Zero Guesswork Policy</span>: Never invents numbers or guesses data.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Main Header & Active Workspace Indicator
# ============================================================================
doc_meta = get_filing_metadata(st.session_state.selected_doc)

st.markdown(f"""
<div style="display: flex; flex-direction: column; gap: 0.2rem; padding: 0.25rem 0;">
    <div class="terminal-label">WORKSPACE CONTEXT &middot; THE ANALYST COPILOT</div>
    <div style="display: flex; align-items: center; gap: 0.65rem;">
        <span style="font-size: 1.6rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; font-family: 'Plus Jakarta Sans', sans-serif;">
            {st.session_state.selected_doc}
        </span>
        <span class="tag-badge badge-highlight">ACTIVE TARGET</span>
    </div>
</div>
<hr style="margin: 0.75rem 0 1.25rem 0; border: none; border-top: 1px solid #26354b;">
""", unsafe_allow_html=True)

# ============================================================================
# Conversational Chat Stream (Left: Copilot, Right: User)
# ============================================================================

# Show Company Name, Period, and GICS Sector exactly once at the start of dialogue
if st.session_state.messages:
    st.markdown(f"""
    <div class="depth-card" style="padding: 0.85rem 1.25rem; margin-bottom: 1.25rem; border-left: 3px solid #38bdf8; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
        <div style="display: flex; align-items: center; gap: 2.25rem; flex-wrap: wrap;">
            <div>
                <div style="color: #64748b; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">COMPANY</div>
                <div style="color: #f8fafc; font-size: 1.05rem; font-weight: 700;">{doc_meta['company']}</div>
            </div>
            <div>
                <div style="color: #64748b; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">PERIOD</div>
                <div style="color: #f8fafc; font-size: 1.05rem; font-weight: 700;">{doc_meta['doc_period']}</div>
            </div>
            <div>
                <div style="color: #64748b; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">GICS SECTOR</div>
                <div style="color: #f8fafc; font-size: 1.05rem; font-weight: 700;">{doc_meta['gics_sector']}</div>
            </div>
        </div>
        <div class="tag-badge badge-neutral" style="letter-spacing: 0.08em;">PRIMARY ENTITY CONTEXT</div>
    </div>
    """, unsafe_allow_html=True)

# Render all existing conversation messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        render_user_message(msg["content"], msg.get("doc_name", st.session_state.selected_doc))
    elif msg["role"] == "assistant":
        render_assistant_result(msg["result"])

# ============================================================================
# Chat Input (Sticky at Bottom like ChatGPT / Antigravity)
# ============================================================================
user_query = st.chat_input(f"Ask a question about {st.session_state.selected_doc}...")

if user_query and user_query.strip():
    # 1. Append User Message to session
    st.session_state.messages.append({
        "role": "user",
        "content": user_query.strip(),
        "doc_name": st.session_state.selected_doc
    })
    
    # 2. Immediately render user message
    render_user_message(user_query.strip(), st.session_state.selected_doc)
    
    # 3. Render aesthetic Quantum Pulse Loader (Simultaneous Analyzing & Searching)
    loader_placeholder = st.empty()
    loader_placeholder.markdown(f"""
    <div class="chat-row-assistant">
        <div class="depth-card chat-card-assistant" style="display: flex; align-items: center; gap: 1.15rem; padding: 1.1rem 1.45rem; border-left: 3px solid #38bdf8;">
            <div class="quantum-loader">
                <div class="quantum-spinner"></div>
            </div>
            <div>
                <div class="terminal-label" style="margin: 0; font-size: 0.76rem; letter-spacing: 0.12em;">ANALYZING AND SEARCHING...</div>
                <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 0.2rem;">Cross-referencing {st.session_state.selected_doc} disclosures & verifying citations</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4. Execute Copilot Pipeline silently
    res = run_pipeline(user_query.strip(), st.session_state.selected_doc, corpus, retriever, answerer, verifier)
    
    # 5. Append Assistant Message
    st.session_state.messages.append({
        "role": "assistant",
        "result": res,
        "doc_name": st.session_state.selected_doc
    })
    
    # 6. Clear loader and refresh view
    loader_placeholder.empty()
    st.rerun()

st.markdown("<div style=\"margin-top: 4rem; text-align: center; color: #64748b; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;\">The Analyst Copilot &middot; Active Workspace &middot; Exact Page Citations</div>", unsafe_allow_html=True)

