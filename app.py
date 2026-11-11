import os
import glob
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

from database import (
    init_db,
    get_db_stats,
    get_all_books,
    delete_book,
    clear_all_chunks,
)
from ingest import (
    validate_pdf_pages,
    ingestion_task_manager,
    get_task_timing,
    MAX_PDF_PAGES,
)
from retrieval import (
    smart_retrieve,
    get_query_suggestions,
)

load_dotenv()

# ===================== Configuration =====================
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "nvidia/nemotron-3.5-bolt-e4b")

BOOKS_DIR = os.path.join(os.path.dirname(__file__), "books")
os.makedirs(BOOKS_DIR, exist_ok=True)

# ===================== Streamlit Page Setup =====================
st.set_page_config(
    page_title="Multi-Book RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Multi-Book RAG Assistant - Query across multiple story books with intelligent retrieval"
    }
)

# ===================== Enhanced CSS =====================
st.markdown("""
<style>
/* ===== LAYOUT & CORE THEME ===== */
.block-container { max-width: 1200px; padding-top: 1.2rem !important; padding-bottom: 3rem !important; }
body, p, h1, h2, h3, h4, h5, h6, input, textarea {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* ===== PROTECT STREAMLIT ICON FONTS (CRITICAL FOR LIGATURES) ===== */
[data-testid*="Icon"],
[data-testid*="stIcon"],
.material-symbols-rounded,
.material-symbols-outlined,
[class*="material-symbols"],
[class*="stIcon"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
    font-style: normal !important;
    font-weight: normal !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    word-wrap: normal !important;
    white-space: nowrap !important;
    direction: ltr !important;
    -webkit-font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}

/* ===== SIDEBAR ===== */
@media (min-width: 768px) {
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] { display: none !important; }

    section[data-testid="stSidebar"] {
        width: 380px !important;
        min-width: 380px !important;
        max-width: 380px !important;
        transform: none !important;
        margin-left: 0 !important;
        visibility: visible !important;
    }
}
section[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
section[data-testid="stSidebar"] h2 {
    color: #f59e0b; font-size: 1.15rem; font-weight: 700;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: .5rem;
    display: flex; align-items: center; gap: 8px;
}
section[data-testid="stSidebar"] h3 { font-size: .92rem; margin-top: .8rem; font-weight: 600; color: #e2e8f0; }

/* ===== STATS CARD ===== */
.sidebar-stats-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 1rem;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
}
.stats-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 10px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.8px; color: #94a3b8;
}
.stats-pulse-dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background-color: #10b981; box-shadow: 0 0 8px #10b981; margin-right: 6px;
}
.stats-tag {
    background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(99, 102, 241, 0.4);
    color: #c7d2fe; padding: 1px 6px; border-radius: 4px; font-size: 0.65rem;
}
.stats-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; text-align: center;
}
.stats-item {
    background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px; padding: 6px 4px;
}
.stats-num { font-size: 1.15rem; font-weight: 800; color: #f59e0b; line-height: 1.2; }
.stats-lbl { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; margin-top: 2px; }

/* ===== BUTTONS & EXPANDERS ===== */
.stButton > button {
    border-radius: 8px; font-weight: 500; font-size: 0.85rem;
    transition: all .2s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(255, 255, 255, 0.12);
}
.stButton > button:hover {
    border-color: #6366f1; transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
}
[data-testid="stExpander"] {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    background: rgba(17, 24, 39, 0.4) !important;
    margin: 8px 0 !important;
}

/* ===== APP HEADER ===== */
.app-header-container { text-align: center; margin-bottom: 1.6rem; }
.app-badge-header {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.35);
    color: #c7d2fe; font-size: 0.75rem; font-weight: 600; padding: 4px 14px;
    border-radius: 20px; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px;
}
.app-title {
    font-size: 2.4rem; font-weight: 800; text-align: center; margin: 0;
    color: #f8fafc;
    background: linear-gradient(135deg, #ffffff 0%, #fde047 50%, #f59e0b 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    letter-spacing: -0.5px;
}
.app-subtitle {
    font-size: 0.96rem; color: #94a3b8; text-align: center; max-width: 720px; margin: 0.6rem auto 1.2rem auto;
    line-height: 1.55;
}

/* ===== SUGGESTION BUTTONS & HEADER ===== */
.suggestions-header {
    display: flex; align-items: center; gap: 8px; margin: 1rem 0 0.6rem 0;
}
.suggestions-icon { font-size: 1.1rem; }
.suggestions-title { font-size: 0.92rem; font-weight: 700; color: #e2e8f0; letter-spacing: 0.3px; }
.suggestions-subtitle { font-size: 0.78rem; color: #64748b; margin-left: 4px; }

div[data-testid="column"] .stButton > button {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
    padding: 0.65rem 0.8rem !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    text-align: center !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.25s ease !important;
    height: 100% !important;
}
div[data-testid="column"] .stButton > button:hover {
    border-color: #f59e0b !important;
    color: #fbbf24 !important;
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.95) 0%, rgba(20, 28, 48, 0.95) 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(245, 158, 11, 0.25) !important;
}

/* ===== VISUAL CONFIDENCE METER ===== */
.confidence-meter-container {
    background: linear-gradient(135deg, rgba(20, 27, 45, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 12px 0 16px 0;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}
.confidence-meter-top {
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;
    font-size: 0.84rem;
}
.confidence-meter-title { color: #f1f5f9; font-weight: 600; display: flex; align-items: center; gap: 6px; }
.confidence-meter-pct {
    font-size: 0.76rem; font-weight: 700; padding: 3px 10px; border-radius: 20px;
    letter-spacing: 0.3px;
}
.confidence-meter-track {
    width: 100%; height: 7px; background: rgba(255, 255, 255, 0.08);
    border-radius: 6px; overflow: hidden; margin-bottom: 8px;
}
.confidence-meter-fill {
    height: 100%; border-radius: 6px; transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
}
.confidence-meter-bottom {
    display: flex; justify-content: space-between; font-size: 0.74rem; color: #94a3b8;
}

/* ===== RETRIEVAL DETAILS GRID ===== */
.retrieval-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 8px 0;
}
@media (max-width: 768px) {
    .retrieval-grid { grid-template-columns: repeat(2, 1fr); }
}
.retrieval-cell {
    background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px; padding: 8px 10px;
}
.rc-label { font-size: 0.64rem; font-weight: 700; color: #64748b; letter-spacing: 0.5px; }
.rc-val { font-size: 0.8rem; font-weight: 600; margin-top: 2px; }
.entity-chips-container {
    display: flex; align-items: center; flex-wrap: wrap; gap: 5px;
    margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.06);
}
.entity-chips-label { font-size: 0.72rem; color: #94a3b8; font-weight: 600; }
.entity-badge {
    display: inline-block; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.35);
    color: #c7d2fe; padding: 2px 8px; border-radius: 12px; font-size: 0.73rem; font-weight: 500;
}

/* ===== QUERY TYPE BADGES ===== */
.query-type-badge {
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; margin-bottom: 10px; letter-spacing: 0.3px;
}
.query-type-listing { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.35); }
.query-type-comparison { background: rgba(245, 158, 11, 0.15); color: #fde68a; border: 1px solid rgba(245, 158, 11, 0.35); }
.query-type-timeline { background: rgba(99, 102, 241, 0.15); color: #c7d2fe; border: 1px solid rgba(99, 102, 241, 0.35); }
.query-type-thematic { background: rgba(168, 85, 247, 0.15); color: #f0abfc; border: 1px solid rgba(168, 85, 247, 0.35); }
.query-type-factual { background: rgba(14, 165, 233, 0.15); color: #7dd3fc; border: 1px solid rgba(14, 165, 233, 0.35); }

/* ===== SOURCE CARDS ===== */
.source-card {
    background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px; padding: 12px 16px; margin: 10px 0; transition: all .2s;
}
.source-card:hover { border-color: rgba(99, 102, 241, 0.5); transform: translateY(-1px); }
.source-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.source-book { font-weight: 600; font-size: 0.84rem; color: #e2e8f0; }
.source-distance { font-size: 0.7rem; padding: 2px 8px; border-radius: 12px; font-weight: 600; }
.distance-good { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }
.distance-med { background: rgba(245, 158, 11, 0.2); color: #fde68a; border: 1px solid rgba(245, 158, 11, 0.4); }
.distance-bad { background: rgba(99, 102, 241, 0.2); color: #c7d2fe; border: 1px solid rgba(99, 102, 241, 0.4); }
.source-text { color: #cbd5e1; font-size: 0.85rem; line-height: 1.65; margin-top: 6px; }

/* ===== BOOK DISTRIBUTION ROW ===== */
.dist-row { margin-bottom: 8px; }
.dist-meta { display: flex; justify-content: space-between; font-size: 0.76rem; margin-bottom: 3px; }
.dist-book-name { color: #e2e8f0; font-weight: 500; }
.dist-book-stats { color: #94a3b8; }
.dist-track { height: 6px; border-radius: 4px; background: rgba(255, 255, 255, 0.08); overflow: hidden; }
.dist-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #6366f1, #38bdf8); }

/* ===== CARDS & FILE LIST ===== */
.pdf-file-card {
    border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px; margin: 6px 0;
    background: linear-gradient(145deg, #131b2e, #0f172a); transition: all .2s;
}
.pdf-file-card:hover { border-color: #f59e0b; }
.pdf-file-card .title { font-weight: 600; color: #f59e0b; font-size: .88rem; margin-bottom: 4px; }
.pdf-file-card .stats { color: #888; font-size: .78rem; }
.pdf-file-card .chapters { color: #a5b4fc; font-size: .78rem; margin-top: 4px; }
.status-indexed { color: #10b981; font-weight: 500; }
.status-pending { color: #f59e0b; font-weight: 500; }

/* ===== BACKGROUND INGESTION LOADER ===== */
.upload-limit-badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.35);
    color: #c7d2fe; font-size: 0.74rem; font-weight: 500; padding: 3px 8px;
    border-radius: 6px; margin-right: 6px; margin-bottom: 8px;
}
.loader-card {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(99, 102, 241, 0.45); border-radius: 12px; padding: 14px 16px;
    margin: 10px 0; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3); position: relative; overflow: hidden;
}
.loader-card::before {
    content: ''; position: absolute; top: 0; left: -100%; width: 200%; height: 2px;
    background: linear-gradient(90deg, transparent, #6366f1, #a855f7, transparent);
    animation: shimmer 2s infinite linear;
}
@keyframes shimmer { 0% { transform: translateX(0); } 100% { transform: translateX(50%); } }
.loader-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.loader-title { font-weight: 600; font-size: 0.86rem; color: #f5c542; display: flex; align-items: center; gap: 8px; }
.loader-pulse {
    display: inline-block; width: 9px; height: 9px; border-radius: 50%; background-color: #38bdf8;
    box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); animation: pulse-ring 1.4s infinite;
}
@keyframes pulse-ring {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(56, 189, 248, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
}
.loader-step { font-size: 0.78rem; color: #cbd5e1; margin-top: 4px; line-height: 1.4; }
.loader-progress-track {
    width: 100%; height: 7px; background: rgba(255, 255, 255, 0.08);
    border-radius: 6px; overflow: hidden; margin: 9px 0 7px 0;
}
.loader-progress-fill {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, #38bdf8, #6366f1, #a855f7);
    background-size: 200% 100%;
    animation: gradient-move 2.5s ease infinite;
    transition: width 0.4s ease;
}
@keyframes gradient-move {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.loader-completed {
    border-color: rgba(34, 197, 94, 0.4);
    background: linear-gradient(135deg, rgba(20, 83, 45, 0.25) 0%, rgba(15, 23, 42, 0.9) 100%);
}
.loader-completed::before { display: none; }
.loader-rejected {
    border-color: rgba(239, 68, 68, 0.4);
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.25) 0%, rgba(15, 23, 42, 0.9) 100%);
}
.loader-rejected::before { display: none; }
.loader-timer-row {
    display: flex; justify-content: space-between; align-items: center; font-size: 0.77rem; color: #94a3b8;
    margin-top: 6px; margin-bottom: 2px; padding-top: 5px; border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.loader-timer-row b { color: #f5c542; font-weight: 600; }
.loader-pct-badge {
    background: rgba(99, 102, 241, 0.25); border: 1px solid rgba(99, 102, 241, 0.5);
    color: #e0e7ff; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 700;
}
.loader-stage-pill {
    background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4);
    color: #38bdf8; padding: 2px 7px; border-radius: 6px; font-size: 0.72rem; font-weight: 600; margin-right: 6px;
}

/* ===== LOADING SPINNER & CHAT INPUT ===== */
.stSpinner > div { border-color: #6366f1 !important; }
.stChatInputContainer { border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 0.8rem; }
.stChatInputContainer textarea {
    border-radius: 12px !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    background: rgba(15, 23, 42, 0.8) !important;
    color: #f8fafc !important;
}
.stChatInputContainer textarea:focus {
    border-color: #f59e0b !important;
    box-shadow: 0 0 12px rgba(245, 158, 11, 0.25) !important;
}
#MainMenu, footer, [data-testid="stDeployButton"], .stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ===================== Sanity Checks =====================
if not NVIDIA_API_KEY:
    st.error("❌ Missing NVIDIA_API_KEY in .env file. Please add it and restart the app.")
    st.stop()

client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

try:
    init_db()
except Exception as db_err:
    st.error(f"❌ Failed to connect to the vector database: {db_err}")
    st.info("💡 Make sure PostgreSQL + pgvector is running (`docker start pgvector`).")
    st.stop()


# ===================== Helpers =====================
@st.cache_data(ttl=20)
def get_system_state():
    """Return database stats, PDF files on disk, and indexed chapters mapping."""
    try:
        db_stats = get_db_stats()
    except Exception:
        db_stats = {"total_chunks": 0, "total_books": 0, "books": []}

    pdf_files = glob.glob(os.path.join(BOOKS_DIR, "*.pdf"))
    pdf_info = []
    for p in pdf_files:
        name = os.path.basename(p)
        try:
            with open(p, "rb") as fh:
                reader = PdfReader(fh)
                pages_count = len(reader.pages)
        except Exception:
            pages_count = 0
        pdf_info.append({"path": p, "filename": name, "pages": pages_count})

    pdf_chapters = defaultdict(list)
    for b in db_stats.get("books", []):
        pdf_chapters[b["filename"]].append(b)

    indexed_map = {b["book_title"].lower(): b["chunk_count"] for b in db_stats["books"]}
    return db_stats, pdf_info, indexed_map, pdf_chapters


def get_query_type_class(query_type: str) -> str:
    """Get CSS class for query type badge."""
    classes = {
        "LISTING": "query-type-listing",
        "COMPARISON": "query-type-comparison",
        "TIMELINE": "query-type-timeline",
        "THEMATIC": "query-type-thematic",
        "FACTUAL": "query-type-factual",
    }
    return classes.get(query_type, "query-type-factual")


def get_query_type_label(query_type: str) -> str:
    """Get human-readable label for query type."""
    labels = {
        "LISTING": "📋 Cross-Book Listing",
        "COMPARISON": "⚖️ Multi-Book Comparison",
        "TIMELINE": "📅 Chronological Timeline",
        "THEMATIC": "🎭 Thematic Analysis",
        "FACTUAL": "❓ Direct Fact Lookup",
    }
    return labels.get(query_type, query_type)


STRATEGY_HUMAN_LABELS = {
    "stratified_multi_book": "Cross-Book Stratified Search",
    "entity_comparison": "Balanced Multi-Entity Comparison",
    "entity_timeline": "Chronological Arc & Timeline",
    "thematic_exploration": "Deep Thematic Exploration",
    "hybrid_precision": "Hybrid Precision & Vector Search",
}


def generate_answer_streaming(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    query_meta: Optional[Dict[str, Any]] = None,
):
    """Generate streaming answer from retrieved chunks with structured multi-book framing."""
    if not retrieved_chunks:
        yield "I couldn't find relevant passages in the indexed books to answer this question."
        return

    # Group passages by book title so the LLM has explicit narrative boundaries
    grouped_chunks = defaultdict(list)
    for idx, item in enumerate(retrieved_chunks, start=1):
        grouped_chunks[item.get("book_title", "Unknown Book")].append((idx, item))

    context_sections = []
    for book_title, items in grouped_chunks.items():
        passages_text = []
        for idx, item in items:
            page_num = item.get("page") or item.get("page_number", "?")
            snippet = item.get("text") or item.get("content", "")
            passages_text.append(f"[Excerpt {idx} | Page {page_num}]:\n{snippet}")
        context_sections.append(f"### 📖 BOOK: {book_title}\n" + "\n\n".join(passages_text))

    context = "\n\n---\n\n".join(context_sections)
    query_type = query_meta.get("query_type", "FACTUAL") if query_meta else "FACTUAL"

    if query_type == "LISTING":
        type_instructions = """
SPECIAL INSTRUCTIONS FOR LISTINGS:
- Identify ALL items, people, associates, or concepts mentioned across the provided excerpts.
- Group the response logically (e.g., by book, by allegiance/relationship, or by role).
- For each entry, provide their role or significance and cite the source [Book Title, Page X].
- Be thorough and synthesize data across all represented books.
"""
        max_tokens = 1800
    elif query_type == "COMPARISON":
        type_instructions = """
SPECIAL INSTRUCTIONS FOR COMPARISONS:
- Present a structured cross-book comparison between the subjects.
- Use clear sections:
  1. Key Parallels & Similarities
  2. Core Differences & Divergences
  3. Character Arcs & Motivations per Book/Subject
- Back each point with specific textual citations [Book Title, Page X].
- Maintain clear distinctions between different books and characters.
"""
        max_tokens = 1800
    elif query_type == "TIMELINE":
        type_instructions = """
SPECIAL INSTRUCTIONS FOR TIMELINES & ARCS:
- Chronologically trace major events, decisions, turning points, and consequences.
- Detail the progression from early events to climactic moments across the stories.
- Cite each event with [Book Title, Page X].
"""
        max_tokens = 1700
    elif query_type == "THEMATIC":
        type_instructions = """
SPECIAL INSTRUCTIONS FOR THEMATIC ANALYSIS:
- Explore the deeper meanings, symbolism, motifs, and underlying messages.
- Synthesize thematic evidence across all represented stories.
- Cite specific textual evidence [Book Title, Page X].
"""
        max_tokens = 1600
    else:
        type_instructions = """
- Answer the question directly and factually based on the excerpts.
- Provide supporting context and cite specific book titles and page numbers [Book Title, Page X].
- If a specific detail is not in the excerpts, explicitly state that it is not mentioned.
"""
        max_tokens = 1200

    system_prompt = f"""You are an expert literary research assistant with deep analytical capabilities across multi-book libraries.
Answer questions accurately and comprehensively based strictly on the provided book excerpts.

Core Principles:
1. Strict Grounding: Rely solely on facts and events from the excerpts. Never hallucinate outside assumptions.
2. Cross-Book Synthesis: When questions involve multiple books or characters, maintain clear distinctions between worlds and synthesize them thoroughly.
3. Citations: Support every claim with citations in the format [Book Title, Page X].
4. Transparency: If a requested detail is not found in the excerpts, state so explicitly.

{type_instructions}"""

    user_prompt = f"Book Context (organized by book):\n\n{context}\n\nQuestion: {question}"

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        try:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        except (IndexError, AttributeError):
            continue


def render_book_distribution(book_dist: Dict[str, int], max_items: int = 10):
    """Render visual book distribution bars."""
    if not book_dist:
        return

    total = sum(book_dist.values())
    sorted_items = sorted(book_dist.items(), key=lambda x: x[1], reverse=True)[:max_items]

    for book_title, count in sorted_items:
        pct = (count / total) * 100
        bar_width = min(pct, 100)
        short_title = book_title[:50] + "..." if len(book_title) > 50 else book_title
        st.markdown(f"""
        <div class="dist-row">
            <div class="dist-meta">
                <span class="dist-book-name">📖 {short_title}</span>
                <span class="dist-book-stats"><b>{count}</b> chunks ({pct:.0f}%)</span>
            </div>
            <div class="dist-track">
                <div class="dist-fill" style="width: {bar_width}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_cross_book_view(chunks: List[Dict[str, Any]], query_type: str):
    """Render cross-book comparison view."""
    if query_type not in ("COMPARISON", "TIMELINE") or not chunks:
        return

    by_book = defaultdict(list)
    for c in chunks:
        by_book[c.get("book_title", "Unknown Book")].append(c)

    if len(by_book) < 2:
        return

    st.markdown(f"#### 📊 Cross-Book Excerpts ({len(by_book)} Books Represented)")
    tab_titles = [f"📖 {title[:20]}... ({len(b_chunks)})" if len(title) > 20 else f"📖 {title} ({len(b_chunks)})" for title, b_chunks in by_book.items()]
    tabs = st.tabs(tab_titles)

    for tab, (book_title, book_chunks) in zip(tabs, by_book.items()):
        with tab:
            st.markdown(f"**{book_title}** — *{len(book_chunks)} passages retrieved*")
            for c in book_chunks:
                distance = c.get('distance', 0)
                if distance < 0.38:
                    dist_badge = f'<span class="source-distance distance-good">🟢 High Relevance (Dist: {distance:.3f})</span>'
                    border_accent = "#10b981"
                elif distance < 0.55:
                    dist_badge = f'<span class="source-distance distance-med">🟡 Moderate (Dist: {distance:.3f})</span>'
                    border_accent = "#f59e0b"
                else:
                    dist_badge = f'<span class="source-distance distance-bad">🔵 Contextual (Dist: {distance:.3f})</span>'
                    border_accent = "#6366f1"

                page_num = c.get("page") or c.get("page_number", "?")
                text_snippet = c.get("text") or c.get("content", "")
                st.markdown(
                    f"""<div class="source-card" style="border-left: 3px solid {border_accent};">
                    <div class="source-header">
                        <span class="source-book">📄 Page {page_num}</span>
                        {dist_badge}
                    </div>
                    <div class="source-text">{text_snippet[:420]}{"..." if len(text_snippet) > 420 else ""}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )


def render_quality_indicator(meta: Dict[str, Any]):
    """Render answer quality indicator with a visual confidence meter."""
    confidence = meta.get("confidence", 0)
    unique_books = meta.get("unique_books_retrieved", 0)
    book_count = meta.get("book_count", 1)
    coverage = meta.get("coverage_pct", 0)
    pct_val = max(5, min(100, int(confidence * 100)))

    if confidence > 0.65 and coverage > 40:
        bar_color = "linear-gradient(90deg, #10b981, #34d399)"
        badge_style = "background:rgba(16, 185, 129, 0.2); color:#6ee7b7; border:1px solid rgba(16, 185, 129, 0.4);"
        quality_label = "Strong Match"
        quality_icon = "🟢"
        note = f"High relevance across {unique_books} of {book_count} books"
    elif confidence > 0.35:
        bar_color = "linear-gradient(90deg, #f59e0b, #fbbf24)"
        badge_style = "background:rgba(245, 158, 11, 0.2); color:#fde68a; border:1px solid rgba(245, 158, 11, 0.4);"
        quality_label = "Moderate Match"
        quality_icon = "🟡"
        note = f"Relevant context found in {unique_books} of {book_count} books"
    else:
        bar_color = "linear-gradient(90deg, #6366f1, #818cf8)"
        badge_style = "background:rgba(99, 102, 241, 0.2); color:#c7d2fe; border:1px solid rgba(99, 102, 241, 0.4);"
        quality_label = "Partial Match"
        quality_icon = "🔵"
        note = f"Limited matches found ({unique_books} book(s) represented)"

    st.markdown(f"""
    <div class="confidence-meter-container">
        <div class="confidence-meter-top">
            <span class="confidence-meter-title">
                {quality_icon} <b>{quality_label}</b> &nbsp;<span style="color:#94a3b8; font-weight:normal;">({note})</span>
            </span>
            <span class="confidence-meter-pct" style="{badge_style}">
                {confidence:.0%} Match
            </span>
        </div>
        <div class="confidence-meter-track">
            <div class="confidence-meter-fill" style="width: {pct_val}%; background: {bar_color};"></div>
        </div>
        <div class="confidence-meter-bottom">
            <span>📚 <b>Library Coverage:</b> {coverage:.0f}% ({unique_books}/{book_count} books consulted)</span>
            <span>🎯 <b>Relevance Score:</b> {confidence:.2f} / 1.00</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_retrieval_metadata(meta: Dict[str, Any]):
    """Render structured retrieval details pill with modern metrics and entity chips."""
    raw_strategy = meta.get("strategy", "unknown")
    strategy_label = STRATEGY_HUMAN_LABELS.get(raw_strategy, raw_strategy.replace("_", " ").title())
    dynamic_k = meta.get("dynamic_k", 0)
    target_k = meta.get("target_k", 0)
    entities = meta.get("entities", [])
    confidence = meta.get("confidence", 0)
    unique_books = meta.get("unique_books_retrieved", 0)
    retrieval_time = meta.get("retrieval_time_ms", 0)

    st.markdown(f"""
    <div class="retrieval-grid">
        <div class="retrieval-cell">
            <div class="rc-label">STRATEGY</div>
            <div class="rc-val" style="color:#a5b4fc;">⚡ {strategy_label}</div>
        </div>
        <div class="retrieval-cell">
            <div class="rc-label">PASSAGES</div>
            <div class="rc-val" style="color:#fde047;">🎯 {dynamic_k} / {target_k} chunks</div>
        </div>
        <div class="retrieval-cell">
            <div class="rc-label">SEARCH SPEED</div>
            <div class="rc-val" style="color:#6ee7b7;">⏱️ {retrieval_time} ms</div>
        </div>
        <div class="retrieval-cell">
            <div class="rc-label">BOOKS CONSULTED</div>
            <div class="rc-val" style="color:#93c5fd;">📚 {unique_books} books</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if entities:
        chips_html = '<div class="entity-chips-container"><span class="entity-chips-label">🎯 Key Subjects Detected:</span> '
        for e in entities[:8]:
            chips_html += f'<span class="entity-badge">{e}</span>'
        if len(entities) > 8:
            chips_html += f'<span class="entity-badge">+{len(entities)-8} more</span>'
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)


def render_retrieved_passages(chunks: List[Dict[str, Any]], book_dist: Optional[Dict[str, int]] = None):
    """Render expandable retrieved passages grouped by book with distance badges and distribution."""
    if not chunks:
        return

    with st.expander(f"📜 View Retrieved Passages ({len(chunks)} chunks)", expanded=False):
        if book_dist:
            st.markdown("##### 📊 Library Passage Distribution")
            render_book_distribution(book_dist)
            st.divider()

        book_groups = defaultdict(list)
        for c in chunks:
            book_groups[c.get("book_title", "Unknown Book")].append(c)

        for book_title, group in book_groups.items():
            st.markdown(f"**📖 {book_title}** &nbsp;•&nbsp; `{len(group)} passages`")
            for item in group:
                distance = item.get('distance', 0)
                if distance < 0.38:
                    dist_badge = f'<span class="source-distance distance-good">🟢 High Relevance (Dist: {distance:.3f})</span>'
                    border_accent = "#10b981"
                elif distance < 0.55:
                    dist_badge = f'<span class="source-distance distance-med">🟡 Moderate (Dist: {distance:.3f})</span>'
                    border_accent = "#f59e0b"
                else:
                    dist_badge = f'<span class="source-distance distance-bad">🔵 Contextual (Dist: {distance:.3f})</span>'
                    border_accent = "#6366f1"

                page_num = item.get("page") or item.get("page_number", "?")
                text_snippet = item.get("text") or item.get("content", "")

                st.markdown(
                    f"""<div class="source-card" style="border-left: 3px solid {border_accent};">
                    <div class="source-header">
                        <span class="source-book">📄 Page {page_num}</span>
                        {dist_badge}
                    </div>
                    <div class="source-text">{text_snippet}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.divider()


# ===================== Background Ingestion Banner Fragment =====================
@st.fragment(run_every="2s")
def render_live_background_banner():
    """
    Isolated Streamlit fragment that auto-updates progress every 2 seconds
    WITHOUT rerunning the main page or freezing user interactions.
    """
    active = ingestion_task_manager.get_active_tasks()
    was_busy = st.session_state.get("_task_was_busy", False)

    if active:
        st.session_state["_task_was_busy"] = True
        st.markdown("#### ⏳ Background Ingestion Active")
        for t in active:
            status_text = "Processing..." if t["status"] == "processing" else "Queued"
            elapsed_str, eta_str, pct_str = get_task_timing(t)
            pages_label = f"{t.get('total_pages', 0):,} pages" if t.get("total_pages") else ""
            stage_pill = f'<span class="loader-stage-pill">{t.get("stage", "Processing")}</span>'
            pct_num = max(2.0, min(float(t.get("progress", 0.05)) * 100, 100.0))

            st.markdown(f"""
            <div class="loader-card">
                <div class="loader-header">
                    <span class="loader-title">
                        <span class="loader-pulse"></span> 📄 {t['filename']}
                    </span>
                    <div>
                        {stage_pill}
                        <span class="loader-pct-badge">{pct_str}</span>
                    </div>
                </div>
                <div class="loader-step">{t.get('step', 'Processing...')}</div>
                <div class="loader-progress-track">
                    <div class="loader-progress-fill" style="width: {pct_num:.1f}%;"></div>
                </div>
                <div class="loader-timer-row">
                    <span>⏱️ Elapsed: <b>{elapsed_str}</b></span>
                    <span>⏳ Est. Remaining: <b>{eta_str}</b></span>
                    <span>📚 Size: <b>{pages_label}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        if was_busy:
            st.session_state["_task_was_busy"] = False
            get_system_state.clear()
            st.rerun()

        finished = [t for t in ingestion_task_manager.get_all_tasks() if t["status"] in ("completed", "failed", "rejected")]
        if finished:
            for t in finished[-2:]:
                dur_str = t.get("duration_str") or "completed"
                if t["status"] == "completed":
                    chunks_ct = t.get('total_chunks', 0)
                    st.markdown(f"""
                    <div class="loader-card loader-completed">
                        <div class="loader-header">
                            <span class="loader-title" style="color:#4ade80;">
                                ✅ {t['filename']}
                            </span>
                            <span style="font-size:0.75rem; color:#86efac; font-weight:600;">Indexed in {dur_str}</span>
                        </div>
                        <div class="loader-step" style="color:#bbf7d0;">{t.get('step', f'Successfully indexed {chunks_ct:,} chunks!')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="loader-card loader-rejected">
                        <div class="loader-header">
                            <span class="loader-title" style="color:#f87171;">
                                ❌ {t['filename']}
                            </span>
                            <span style="font-size:0.75rem; color:#fca5a5; font-weight:600;">Failed</span>
                        </div>
                        <div class="loader-step" style="color:#fecaca;">{t.get('step', 'Processing failed')}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ===================== Session State =====================
for key, default in [
    ("messages", []),
    ("pending_question", None),
    ("show_book_manager", False),
    ("selected_query_type", None),
    ("queued_file_signatures", set()),
    ("_task_was_busy", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ===================== Sidebar =====================
with st.sidebar:
    st.markdown("## 📚 Library Knowledge Base")
    db_stats, pdf_info, indexed_map, pdf_chapters = get_system_state()

    total_pdfs = len([p for p in pdf_info if p["filename"] in pdf_chapters])
    total_chapters = sum(len(c) for c in pdf_chapters.values())

    st.markdown(f"""
    <div class="sidebar-stats-card">
        <div class="stats-header">
            <span><span class="stats-pulse-dot"></span>DB ONLINE</span>
            <span class="stats-tag">pgvector</span>
        </div>
        <div class="stats-grid">
            <div class="stats-item">
                <div class="stats-num">{total_pdfs}</div>
                <div class="stats-lbl">Books</div>
            </div>
            <div class="stats-item">
                <div class="stats-num">{total_chapters}</div>
                <div class="stats-lbl">Sections</div>
            </div>
            <div class="stats-item">
                <div class="stats-num">{db_stats['total_chunks']:,}</div>
                <div class="stats-lbl">Chunks</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Book Upload (Non-blocking background queue) ---
    st.markdown("### 📤 Upload Books")
    st.markdown(
        f'<span class="upload-limit-badge">🛡️ Limit: Max {MAX_PDF_PAGES:,} pages</span>'
        '<span class="upload-limit-badge">⚡ Non-blocking Background</span>',
        unsafe_allow_html=True
    )
    st.caption("Upload PDF story books. Processing runs in the background without freezing the UI.")

    uploaded = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
        help=f"Upload PDF story books (maximum {MAX_PDF_PAGES:,} pages). Processing runs in the background."
    )
    if uploaded:
        new_queued = 0
        for f in uploaded:
            file_sig = f"{f.name}_{f.size}"
            if file_sig not in st.session_state.queued_file_signatures:
                save_path = os.path.join(BOOKS_DIR, f.name)
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())

                is_valid, page_count, err_msg = validate_pdf_pages(save_path, max_pages=MAX_PDF_PAGES)
                if not is_valid:
                    try:
                        if os.path.exists(save_path):
                            os.remove(save_path)
                    except Exception:
                        pass
                    st.error(f"❌ **{f.name}** was rejected: {err_msg}")
                    st.session_state.queued_file_signatures.add(file_sig)
                else:
                    ingestion_task_manager.queue_file(save_path, f.name)
                    st.session_state.queued_file_signatures.add(file_sig)
                    st.toast(f"🚀 Queued '{f.name}' ({page_count:,} pages) for background indexing!", icon="📚")
                    new_queued += 1

        if new_queued > 0:
            get_system_state.clear()
            st.rerun()

    active_tasks_sidebar = ingestion_task_manager.get_active_tasks()
    if active_tasks_sidebar:
        cur_t = active_tasks_sidebar[0]
        el_str, eta_str, pct_str = get_task_timing(cur_t)
        pct_num = max(2.0, min(float(cur_t.get("progress", 0.05)) * 100, 100.0))
        st.markdown(f"""
        <div class="loader-card" style="margin-top:10px; padding:10px 12px;">
            <div class="loader-header">
                <span class="loader-title" style="font-size:0.82rem;">
                    <span class="loader-pulse"></span> 📄 {cur_t['filename'][:24]}...
                </span>
                <span class="loader-pct-badge">{pct_str}</span>
            </div>
            <div class="loader-step" style="font-size:0.75rem;">{cur_t.get('step', 'Processing...')}</div>
            <div class="loader-progress-track" style="height:5px; margin:6px 0;">
                <div class="loader-progress-fill" style="width:{pct_num:.1f}%;"></div>
            </div>
            <div class="loader-timer-row" style="margin-top:4px; font-size:0.7rem;">
                <span>⏱️ {el_str}</span>
                <span>⏳ ETA: {eta_str}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    finished_tasks_sidebar = [t for t in ingestion_task_manager.get_all_tasks() if t["status"] in ("completed", "failed", "rejected")]
    if finished_tasks_sidebar and not active_tasks_sidebar:
        if st.button("🧹 Clear Status Notifications", key="dismiss_tasks_btn", use_container_width=True):
            ingestion_task_manager.clear_all_finished()
            st.rerun()

    st.divider()

    # --- Search Scope ---
    st.markdown("### 🎯 Search Scope")
    scope_display_names = ["🌍 All Books (Entire Library)"]
    scope_value_map = {"🌍 All Books (Entire Library)": None}

    for p in pdf_info:
        if p["filename"] in pdf_chapters:
            chapter_count = len(pdf_chapters[p["filename"]])
            clean_name = os.path.splitext(p["filename"])[0].replace("_", " ").title()
            display_str = f"📖 {clean_name} ({chapter_count} sections)"
            scope_display_names.append(display_str)
            scope_value_map[display_str] = p["filename"]

    selected_scope = st.selectbox(
        "Search in:",
        options=scope_display_names,
        index=0,
        help="Select 'All Books' for multi-book comparisons, timelines, and listings, or select a single book.",
        label_visibility="collapsed",
    )
    filter_param = scope_value_map[selected_scope]

    if filter_param is None:
        st.caption(f"🌍 **All Books** ({total_pdfs} books, {total_chapters} sections indexed)")
    else:
        st.caption(f"📖 **{selected_scope}**")

    # --- Sync Button ---
    if st.button("⚡ Sync / Index Unprocessed Books", use_container_width=True, type="primary"):
        if not pdf_info:
            st.warning("Please upload or place PDF files in the `books/` folder first.")
        else:
            queued_count = 0
            rejected_count = 0
            for p in pdf_info:
                fname = p["filename"]
                chapters = pdf_chapters.get(fname, [])
                if not chapters:  # not yet indexed
                    fpath = p["path"]
                    is_valid, pages, err = validate_pdf_pages(fpath, max_pages=MAX_PDF_PAGES)
                    if not is_valid:
                        st.error(f"❌ Skipping '{fname}': {err}")
                        rejected_count += 1
                        continue
                    ingestion_task_manager.queue_file(fpath, fname)
                    queued_count += 1

            if queued_count > 0:
                st.toast(f"🚀 Queued {queued_count} unindexed book(s) for background processing!", icon="⚡")
                get_system_state.clear()
                st.rerun()
            elif rejected_count == 0:
                st.info("All PDF books in `./books` are already indexed!")

    st.divider()

    # --- Advanced Settings (Progressive Disclosure) ---
    with st.expander("⚙️ Advanced Retrieval Settings", expanded=False):
        k_mode = st.radio(
            "Passage Count (K Mode):",
            options=["🎯 Dynamic K (Auto-Adaptive)", "🎛️ Manual Fixed K"],
            index=0,
            help="Dynamic K automatically balances chunk count across books based on question type.",
        )

        if k_mode == "🎛️ Manual Fixed K":
            manual_k_val = st.slider(
                "Manual Chunks (K)",
                min_value=3,
                max_value=60,
                value=12,
                step=1,
                help="Fixed number of passages to retrieve.",
            )
        else:
            manual_k_val = None
            st.caption("🎯 **Dynamic K**: Automatically balances passages across books according to query intent.")

    # --- Library Management Expander ---
    with st.expander("📁 Library & Indexed Books", expanded=False):
        if pdf_info:
            book_filter_q = st.text_input("🔍 Filter library books...", placeholder="Type to filter...", key="lib_filter_input")
            filtered_pdfs = [p for p in pdf_info if not book_filter_q or book_filter_q.lower() in p["filename"].lower()]
            display_pdfs = filtered_pdfs[:20]

            for p in display_pdfs:
                chapters = pdf_chapters.get(p["filename"], [])
                is_indexed = len(chapters) > 0
                status_class = "status-indexed" if is_indexed else "status-pending"
                status_text = "Indexed" if is_indexed else "Not indexed"

                st.markdown(f"""
                <div class="pdf-file-card">
                    <div class="title">📄 {p['filename']}</div>
                    <div class="stats">{p['pages']} pages • <span class="{status_class}">● {status_text}</span></div>
                    <div class="chapters">{len(chapters)} indexed section(s)</div>
                </div>
                """, unsafe_allow_html=True)

            if len(filtered_pdfs) > 20:
                st.caption(f"Showing 20 of {len(filtered_pdfs)} books. Use the filter above to search.")
        else:
            st.caption("No PDF files in `books/` folder.")

        st.divider()
        if st.button("🗑️ Clear All Indexed Data", type="secondary", use_container_width=True):
            clear_all_chunks()
            st.success("All indexed data cleared!")
            get_system_state.clear()
            st.rerun()

    st.divider()

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ===================== Main UI =====================
st.markdown("""
<div class="app-header-container">
    <div class="app-badge-header">✨ AI-Powered Literary Research Engine</div>
    <div class="app-title">Multi-Book Story Assistant</div>
    <div class="app-subtitle">
        Explore characters, trace chronological timelines, and synthesize insights across multiple books with exact page citations.
    </div>
</div>
""", unsafe_allow_html=True)

# Live non-blocking background progress banner (Streamlit Fragment)
render_live_background_banner()

# Friendly onboarding when nothing indexed
if db_stats["total_chunks"] == 0 and not ingestion_task_manager.is_busy():
    st.markdown("""
    <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.25); border-radius:12px; padding:24px; text-align:center; margin:15px 0;">
        <h3 style="color:#f5c542; margin-bottom:8px;">📚 Welcome to Multi-Book Story Assistant</h3>
        <p style="color:#cbd5e1; font-size:0.95rem; margin-bottom:18px;">
            Query, compare, and explore your books with accurate passage citations and cross-book intelligence.
        </p>
        <div style="display:flex; justify-content:center; gap:24px; text-align:left; max-width:650px; margin:0 auto; font-size:0.88rem; color:#94a3b8;">
            <div><b>1. 📤 Upload</b><br>Drop PDF books in the sidebar</div>
            <div><b>2. ⚡ Background Index</b><br>Auto-extracts & embeds safely</div>
            <div><b>3. 💬 Ask Anything</b><br>Get answers with book & page citations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Helpful suggested questions
st.markdown("""
<div class="suggestions-header">
    <span class="suggestions-icon">💡</span>
    <span class="suggestions-title">Explore Your Library</span>
    <span class="suggestions-subtitle">• Click any prompt below to ask instantly</span>
</div>
""", unsafe_allow_html=True)

sample_questions = [
    ("👥 Character Cast", "List all main characters and their roles across the books"),
    ("⚖️ Cross-Book Compare", "Compare the protagonists, their motivations, and key differences"),
    ("📅 Story Timeline", "What are the major chronological events across the story?"),
    ("🎭 Central Themes", "What central themes and deeper meanings are explored in these books?"),
]
s_cols = st.columns(len(sample_questions))
for i, (label, q) in enumerate(sample_questions):
    if s_cols[i].button(label, key=f"sugg_btn_{i}", help=q, use_container_width=True):
        st.session_state.pending_question = q

st.divider()


# ===================== Chat History =====================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        meta = message.get("meta")
        if message["role"] == "assistant" and meta:
            query_type = meta.get("query_type", "UNKNOWN")
            qt_class = get_query_type_class(query_type)
            qt_label = get_query_type_label(query_type)
            st.markdown(f'<span class="query-type-badge {qt_class}">{qt_label}</span>', unsafe_allow_html=True)

        st.markdown(message["content"])

        if message["role"] == "assistant" and meta:
            # Quality indicator (Visual Confidence Meter)
            render_quality_indicator(meta)

            # Cross-book view
            if meta.get("query_type") in ("COMPARISON", "TIMELINE") and message.get("chunks"):
                render_cross_book_view(message["chunks"], meta.get("query_type"))

            # Expandable sources
            if message.get("chunks"):
                book_dist = meta.get("book_distribution", {})
                render_retrieved_passages(message["chunks"], book_dist)

            # Retrieval details inside progressive disclosure expander
            with st.expander("🔍 Retrieval Analytics & Entity Breakdown", expanded=False):
                render_retrieval_metadata(meta)


# ===================== Chat Input =====================
user_input = st.chat_input("Ask anything about your books...", key="chat_input")
active_question = user_input or st.session_state.pending_question

if active_question:
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": active_question})

    with st.chat_message("user"):
        st.markdown(active_question)

    with st.chat_message("assistant"):
        if db_stats["total_chunks"] == 0:
            msg = (
                "The Vector Database has 0 chunks indexed. "
                "Upload PDFs in the sidebar and wait for background indexing to complete."
            )
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            try:
                scope_desc = selected_scope if selected_scope != "All Books" else "all books"
                with st.spinner(f"🔍 Analyzing query and retrieving across {scope_desc}..."):
                    matches, meta = smart_retrieve(
                        question=active_question,
                        book_filter=filter_param,
                        manual_k=manual_k_val,
                    )

                if not matches:
                    no_results_msg = (
                        "I couldn't find any relevant passages in the indexed books for your question. "
                        "Try rephrasing your question, selecting a different search scope, or adding more books."
                    )
                    st.markdown(no_results_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": no_results_msg,
                        "meta": meta,
                        "chunks": []
                    })
                else:
                    # Query type badge shown upfront
                    query_type = meta.get("query_type", "UNKNOWN")
                    qt_class = get_query_type_class(query_type)
                    qt_label = get_query_type_label(query_type)
                    st.markdown(f'<span class="query-type-badge {qt_class}">{qt_label}</span>', unsafe_allow_html=True)

                    # Streaming answer
                    answer_placeholder = st.empty()
                    full_answer = ""
                    for chunk in generate_answer_streaming(active_question, matches, meta):
                        full_answer += chunk
                        answer_placeholder.markdown(full_answer + "▌")
                    answer_placeholder.markdown(full_answer)

                    # Quality indicator (Visual Confidence Meter)
                    render_quality_indicator(meta)

                    # Cross-book view
                    if meta.get("query_type") in ("COMPARISON", "TIMELINE"):
                        render_cross_book_view(matches, meta.get("query_type"))

                    # Expandable sources
                    book_dist = meta.get("book_distribution", {}) if meta else {}
                    render_retrieved_passages(matches, book_dist)

                    # Retrieval details inside progressive disclosure expander
                    with st.expander("🔍 Retrieval Analytics & Entity Breakdown", expanded=False):
                        render_retrieval_metadata(meta)

                    # Save message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "chunks": matches,
                        "meta": meta,
                    })

            except Exception as error:
                err_msg = f"❌ Failed to retrieve or generate answer: {error}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
