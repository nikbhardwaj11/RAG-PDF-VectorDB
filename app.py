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
/* ===== LAYOUT ===== */
.block-container { max-width: 1200px; padding-top: 1.5rem !important; }

/* ===== SIDEBAR: ALWAYS OPEN ON DESKTOP ===== */
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
section[data-testid="stSidebar"] { border-right: 1px solid #2d2d2d; }
section[data-testid="stSidebar"] h2 {
    color: #d3a625; font-size: 1.2rem; font-weight: 700;
    border-bottom: 1px solid #2d2d2d; padding-bottom: .5rem;
}
section[data-testid="stSidebar"] h3 { font-size: .95rem; margin-top: 1rem; }
section[data-testid="stSidebar"] .stButton > button { justify-content: flex-start; }

/* ===== CLEAN HEADER ===== */
#MainMenu, footer, [data-testid="stDeployButton"], .stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }

/* ===== WIDGETS ===== */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid #2d2d2d; border-radius: 10px; padding: 10px 12px;
}
[data-testid="stMetricLabel"] { font-size: .65rem; text-transform: uppercase; letter-spacing: .8px; color: #888; }
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; color: #f5c542; }
.stButton > button { border-radius: 8px; font-weight: 500; transition: all .2s; }
.stButton > button:hover { border-color: #6366f1; }
[data-testid="stExpander"] { border: 1px solid #2d2d2d; border-radius: 10px; }
[data-testid="stChatMessage"] { border-radius: 12px; }

/* ===== TITLE ===== */
.app-title {
    font-family: Georgia, serif; font-size: 2.5rem; font-weight: 700; text-align: center;
    background: linear-gradient(90deg, #d3a625, #f5c542, #d3a625);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.app-subtitle { font-size: 1.05rem; color: #a0a0a0; margin-bottom: 1.5rem; text-align: center; }

/* ===== RETRIEVAL BADGE ===== */
.retrieval-pill {
    padding: 12px 16px; border-radius: 12px; font-size: .85rem; line-height: 1.7;
    background: rgba(99, 102, 241, .1); border: 1px solid rgba(99, 102, 241, .3);
    color: #e0e7ff; margin-bottom: 1rem;
}
.confidence-high { color: #22c55e; font-weight: 600; }
.confidence-med  { color: #f59e0b; font-weight: 600; }
.confidence-low  { color: #ef4444; font-weight: 600; }
.entity-badge {
    display: inline-block; background: rgba(34, 197, 94, .15); border: 1px solid rgba(34, 197, 94, .3);
    color: #86efac; padding: 3px 10px; border-radius: 12px; margin: 2px; font-size: .78rem;
}
.cross-book-badge {
    background: rgba(251, 191, 36, .15); border: 1px solid rgba(251, 191, 36, .4);
    color: #fde047; padding: 3px 10px; border-radius: 12px; font-size: .72rem;
}

/* ===== CARDS ===== */
.book-card, .pdf-file-card {
    border: 1px solid #2d2d2d; border-radius: 10px; padding: 12px; margin: 6px 0;
    background: linear-gradient(145deg, #1a1a2e, #16213e); transition: all .2s;
}
.book-card:hover, .pdf-file-card:hover { border-color: #d3a625; }
.book-card h4 { margin: 0 0 6px 0; color: #d3a625; font-size: 1rem; }
.book-card .meta { color: #888; font-size: .82rem; }
.pdf-file-card .title { font-weight: 600; color: #d3a625; font-size: .9rem; margin-bottom: 4px; }
.pdf-file-card .stats { color: #888; font-size: .78rem; }
.pdf-file-card .chapters { color: #a5b4fc; font-size: .78rem; margin-top: 4px; }
.status-indexed { color: #22c55e; font-weight: 500; }
.status-pending { color: #f59e0b; font-weight: 500; }

/* ===== DISTRIBUTION BARS ===== */
.dist-label { font-size: .75rem; color: #a0a0a0; display: flex; justify-content: space-between; margin-bottom: 2px; }
.dist-label .book-name { color: #e0e7ff; font-weight: 500; }
.dist-label .book-count { color: #f5c542; font-weight: 600; }
.dist-bar { height: 10px; border-radius: 6px; margin-bottom: 10px; background: linear-gradient(90deg, #6366f1, #22c55e, #f59e0b); }

/* ===== BACKGROUND INGESTION LOADER & STATUS CARDS ===== */
.upload-limit-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #c7d2fe;
    font-size: 0.74rem;
    font-weight: 500;
    padding: 3px 8px;
    border-radius: 6px;
    margin-right: 6px;
    margin-bottom: 8px;
}
.loader-card {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(99, 102, 241, 0.45);
    border-radius: 10px;
    padding: 12px 14px;
    margin: 8px 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    position: relative;
    overflow: hidden;
}
.loader-card::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 200%; height: 2px;
    background: linear-gradient(90deg, transparent, #6366f1, #a855f7, transparent);
    animation: shimmer 2s infinite linear;
}
@keyframes shimmer {
    0% { transform: translateX(0); }
    100% { transform: translateX(50%); }
}
.loader-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}
.loader-title {
    font-weight: 600;
    font-size: 0.86rem;
    color: #f5c542;
    display: flex;
    align-items: center;
    gap: 8px;
}
.loader-pulse {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background-color: #38bdf8;
    box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7);
    animation: pulse-ring 1.4s infinite;
}
@keyframes pulse-ring {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(56, 189, 248, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
}
.loader-step {
    font-size: 0.77rem;
    color: #cbd5e1;
    margin-top: 4px;
    line-height: 1.4;
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
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.77rem;
    color: #94a3b8;
    margin-top: 6px;
    margin-bottom: 6px;
    padding-top: 5px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.loader-timer-row b {
    color: #f5c542;
    font-weight: 600;
}
.loader-pct-badge {
    background: rgba(99, 102, 241, 0.25);
    border: 1px solid rgba(99, 102, 241, 0.5);
    color: #e0e7ff;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 700;
}
.loader-stage-pill {
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(56, 189, 248, 0.4);
    color: #38bdf8;
    padding: 2px 7px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
}

/* ===== QUERY TYPE INDICATOR ===== */
.query-type-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: .75rem;
    font-weight: 600; margin-right: 8px; margin-bottom: 4px;
}
.query-type-listing { background: rgba(34, 197, 94, .2); color: #86efac; border: 1px solid rgba(34, 197, 94, .4); }
.query-type-comparison { background: rgba(251, 191, 36, .2); color: #fde047; border: 1px solid rgba(251, 191, 36, .4); }
.query-type-timeline { background: rgba(99, 102, 241, .2); color: #c7d2fe; border: 1px solid rgba(99, 102, 241, .4); }
.query-type-thematic { background: rgba(168, 85, 247, .2); color: #f0abfc; border: 1px solid rgba(168, 85, 247, .4); }
.query-type-factual { background: rgba(59, 130, 246, .2); color: #bfdbfe; border: 1px solid rgba(59, 130, 246, .4); }

/* ===== ANSWER QUALITY INDICATOR (CONFIDENCE METER) ===== */
.quality-indicator {
    display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px;
    border-radius: 8px; font-size: .8rem; font-weight: 500; margin-bottom: 12px;
}
.quality-high { background: rgba(34, 197, 94, .15); color: #86efac; border: 1px solid rgba(34, 197, 94, .3); }
.quality-medium { background: rgba(251, 191, 36, .15); color: #fde047; border: 1px solid rgba(251, 191, 36, .3); }
.quality-low { background: rgba(239, 68, 68, .15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, .3); }

/* ===== SOURCE CARD ===== */
.source-card {
    border: 1px solid #2d2d2d; border-radius: 8px; padding: 12px; margin: 8px 0;
    background: #16213e; transition: all .2s;
}
.source-card:hover { border-color: #d3a625; }
.source-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.source-book { color: #d3a625; font-weight: 600; font-size: .9rem; }
.source-page { color: #888; font-size: .75rem; }
.source-distance { font-size: .7rem; padding: 2px 8px; border-radius: 10px; }
.distance-good { background: rgba(34, 197, 94, .2); color: #86efac; }
.distance-med { background: rgba(251, 191, 36, .2); color: #fde047; }
.distance-bad { background: rgba(239, 68, 68, .2); color: #fca5a5; }
.source-text { color: #e0e0e0; font-size: .88rem; line-height: 1.6; }

/* ===== LOADING SPINNER & CHAT INPUT ===== */
.stSpinner > div { border-color: #6366f1 !important; }
.stChatInputContainer { border-top: 1px solid #2d2d2d; padding-top: 1rem; }
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
        bar_width = min(pct * 2.5, 100)
        short_title = book_title[:45] + "..." if len(book_title) > 45 else book_title
        st.markdown(f"""
        <div class="dist-label">
            <span class="book-name">{short_title}</span>
            <span class="book-count">{count} chunks ({pct:.0f}%)</span>
        </div>
        <div class="dist-bar" style="width: {bar_width}%;"></div>
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

    st.markdown("### 📊 Cross-Book View")
    tab_titles = [f"📘 {title[:22]}..." if len(title) > 22 else f"📘 {title}" for title in by_book.keys()]
    tabs = st.tabs(tab_titles)

    for tab, (book_title, book_chunks) in zip(tabs, by_book.items()):
        with tab:
            st.markdown(f"**{book_title}** — {len(book_chunks)} passages")
            for c in book_chunks:
                distance = c.get('distance', 0)
                dist_color = "#22c55e" if distance < 0.4 else "#f59e0b" if distance < 0.55 else "#ef4444"
                page_num = c.get("page") or c.get("page_number", "?")
                text_snippet = c.get("text") or c.get("content", "")
                st.markdown(
                    f"**Page {page_num}** &nbsp; "
                    f"<span style='color:{dist_color};'>● Distance: {distance:.3f}</span>",
                    unsafe_allow_html=True,
                )
                st.write(text_snippet[:400] + ("..." if len(text_snippet) > 400 else ""))
                st.divider()


def render_quality_indicator(meta: Dict[str, Any]):
    """Render answer quality indicator (confidence meter) with friendly user-facing labels."""
    confidence = meta.get("confidence", 0)
    unique_books = meta.get("unique_books_retrieved", 0)
    book_count = meta.get("book_count", 1)
    coverage = meta.get("coverage_pct", 0)

    if confidence > 0.65 and coverage > 40:
        quality_class = "quality-high"
        quality_label = "Strong Match"
        quality_icon = "✅"
        note = f"Found relevant passages across {unique_books} of {book_count} books"
    elif confidence > 0.35:
        quality_class = "quality-medium"
        quality_label = "Moderate Match"
        quality_icon = "⚠️"
        note = f"Found relevant passages in {unique_books} of {book_count} books"
    else:
        quality_class = "quality-low"
        quality_label = "Partial Match"
        quality_icon = "🔍"
        note = f"Limited matches found ({unique_books} book(s) represented)"

    st.markdown(f"""
    <div class="quality-indicator {quality_class}">
        {quality_icon} <b>{quality_label}</b> &nbsp;•&nbsp; {note} &nbsp;•&nbsp; Match Score: {confidence:.0%}
    </div>
    """, unsafe_allow_html=True)


def render_retrieval_metadata(meta: Dict[str, Any]):
    """Render structured retrieval details pill with human-readable labels and entity badges."""
    raw_strategy = meta.get("strategy", "unknown")
    strategy_label = STRATEGY_HUMAN_LABELS.get(raw_strategy, raw_strategy.replace("_", " ").title())
    dynamic_k = meta.get("dynamic_k", 0)
    target_k = meta.get("target_k", 0)
    entities = meta.get("entities", [])
    confidence = meta.get("confidence", 0)
    book_dist = meta.get("book_distribution", {})
    unique_books = meta.get("unique_books_retrieved", 0)
    retrieval_time = meta.get("retrieval_time_ms", 0)

    if confidence > 0.65:
        conf_class = "confidence-high"
        conf_label = "High"
    elif confidence > 0.35:
        conf_class = "confidence-med"
        conf_label = "Medium"
    else:
        conf_class = "confidence-low"
        conf_label = "Low"

    badge_html = '<div class="retrieval-pill">'
    badge_html += f'🔍 <b>Strategy:</b> {strategy_label} &nbsp;|&nbsp; '
    badge_html += f'🎯 <b>Passages Retrieved:</b> {dynamic_k}/{target_k} chunks &nbsp;|&nbsp; '
    badge_html += f'📈 <b>Match Score:</b> <span class="{conf_class}">{conf_label} ({confidence:.0%})</span> &nbsp;|&nbsp; '
    badge_html += f'⏱️ <b>Search Time:</b> {retrieval_time}ms'

    if entities:
        badge_html += '<br><div style="margin-top:6px; display:flex; align-items:center; flex-wrap:wrap; gap:4px;">'
        badge_html += '<span style="color:#94a3b8; font-size:0.75rem;">Key Subjects:</span> '
        for e in entities[:6]:
            badge_html += f'<span class="entity-badge">{e}</span>'
        if len(entities) > 6:
            badge_html += f'<span class="entity-badge">+{len(entities)-6} more</span>'
        badge_html += '</div>'

    if book_dist and unique_books > 1:
        badge_html += f' <div style="margin-top:4px;"><span class="cross-book-badge">📚 Cross-Book Synthesis ({unique_books} books)</span></div>'

    badge_html += '</div>'
    st.markdown(badge_html, unsafe_allow_html=True)


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
                <div class="loader-timer-row">
                    <span>⏱️ Elapsed: <b>{elapsed_str}</b></span>
                    <span>⏳ Est. Remaining: <b>{eta_str}</b></span>
                    <span>📚 Size: <b>{pages_label}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            prog = max(0.02, min(float(t.get("progress", 0.05)), 1.0))
            st.progress(prog)
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
    st.markdown("## 📚 Knowledge Base")
    st.caption("PostgreSQL + pgvector Vector Database")

    db_stats, pdf_info, indexed_map, pdf_chapters = get_system_state()

    total_pdfs = len([p for p in pdf_info if p["filename"] in pdf_chapters])
    total_chapters = sum(len(c) for c in pdf_chapters.values())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunks", f"{db_stats['total_chunks']:,}")
    with col2:
        st.metric("PDFs", total_pdfs)
    with col3:
        st.metric("Chapters", total_chapters)

    st.divider()

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
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.35); border-radius:8px; padding:10px; margin-top:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.82rem; font-weight:600; color:#f5c542;">⏳ Indexing in Background</span>
                <span style="font-size:0.75rem; font-weight:700; color:#38bdf8;">{pct_str}</span>
            </div>
            <div style="font-size:0.77rem; color:#e2e8f0; margin-top:3px; word-break:break-all;">📄 {cur_t['filename'][:28]}...</div>
            <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#94a3b8; margin-top:6px; border-top:1px solid rgba(255,255,255,0.06); padding-top:4px;">
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
            for p in pdf_info:
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
st.markdown('<div class="app-title">📚 Multi-Book Story Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'Ask questions across your story books. Answers are synthesized and strictly cited with exact book and page numbers.'
    '</div>',
    unsafe_allow_html=True,
)

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
st.markdown("##### 💡 Try asking:")
sample_questions = [
    ("👥 Characters", "List all main characters and their roles across the books"),
    ("⚖️ Comparison", "Compare the protagonists, their motivations, and key differences"),
    ("📅 Timeline", "What are the major chronological events across the story?"),
    ("🎭 Themes", "What central themes and deeper meanings are explored in these books?"),
]
s_cols = st.columns(len(sample_questions))
for i, (label, q) in enumerate(sample_questions):
    if s_cols[i].button(label, key=f"sugg_btn_{i}", help=q, use_container_width=True):
        st.session_state.pending_question = q

st.divider()


# ===================== Chat History =====================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "meta" in message and message["meta"]:
            m = message["meta"]
            query_type = m.get("query_type", "UNKNOWN")

            # Query type badge
            qt_class = get_query_type_class(query_type)
            qt_label = get_query_type_label(query_type)
            st.markdown(f'<span class="query-type-badge {qt_class}">{qt_label}</span>', unsafe_allow_html=True)

            # Quality indicator (Confidence Meter)
            render_quality_indicator(m)

            # Retrieval details inside progressive disclosure expander
            with st.expander("🔍 Retrieval Analytics & Entity Breakdown", expanded=False):
                render_retrieval_metadata(m)

        st.markdown(message["content"])

        # Cross-book view
        if "chunks" in message and message["chunks"] and "meta" in message:
            meta = message["meta"]
            if meta.get("query_type") in ("COMPARISON", "TIMELINE"):
                render_cross_book_view(message["chunks"], meta.get("query_type"))

        # Expandable sources
        if "chunks" in message and message["chunks"]:
            chunks = message["chunks"]
            with st.expander(f"📜 View Retrieved Passages ({len(chunks)} chunks)"):
                if "meta" in message and message["meta"]:
                    book_dist = message["meta"].get("book_distribution", {})
                    if book_dist:
                        st.markdown("**📊 Book Distribution:**")
                        render_book_distribution(book_dist)

                st.divider()
                book_groups = {}
                for c in chunks:
                    bt = c.get("book_title", "Unknown")
                    book_groups.setdefault(bt, []).append(c)

                for book_title, group in book_groups.items():
                    st.markdown(f'**📘 {book_title}** ({len(group)} chunks)')
                    for item in group:
                        distance = item.get('distance', 0)
                        dist_class = "distance-good" if distance < 0.4 else "distance-med" if distance < 0.55 else "distance-bad"
                        page_num = item.get("page") or item.get("page_number", "?")
                        text_snippet = item.get("text") or item.get("content", "")
                        st.markdown(
                            f"""<div class="source-card">
                            <div class="source-header">
                                <span class="source-book">Page {page_num}</span>
                                <span class="source-distance {dist_class}">Distance: {distance:.4f}</span>
                            </div>
                            <div class="source-text">{text_snippet}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    st.divider()


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
                    # Streaming answer
                    answer_placeholder = st.empty()
                    full_answer = ""
                    for chunk in generate_answer_streaming(active_question, matches, meta):
                        full_answer += chunk
                        answer_placeholder.markdown(full_answer + "▌")
                    answer_placeholder.markdown(full_answer)

                    # Query type badge
                    query_type = meta.get("query_type", "UNKNOWN")
                    qt_class = get_query_type_class(query_type)
                    qt_label = get_query_type_label(query_type)
                    st.markdown(f'<span class="query-type-badge {qt_class}">{qt_label}</span>', unsafe_allow_html=True)

                    # Quality indicator (Confidence Meter)
                    render_quality_indicator(meta)

                    # Retrieval details inside progressive disclosure expander
                    with st.expander("🔍 Retrieval Analytics & Entity Breakdown", expanded=False):
                        render_retrieval_metadata(meta)

                    # Cross-book view
                    if meta.get("query_type") in ("COMPARISON", "TIMELINE"):
                        render_cross_book_view(matches, meta.get("query_type"))

                    # Expandable sources
                    with st.expander(f"📜 View Retrieved Passages ({len(matches)} chunks)"):
                        if book_dist:
                            st.markdown("**📊 Book Distribution:**")
                            render_book_distribution(book_dist)
                        st.divider()
                        book_groups = {}
                        for c in matches:
                            bt = c.get("book_title", "Unknown")
                            book_groups.setdefault(bt, []).append(c)

                        for book_title, group in book_groups.items():
                            st.markdown(f'**📘 {book_title}** ({len(group)} chunks)')
                            for item in group:
                                distance = item.get('distance', 0)
                                dist_class = "distance-good" if distance < 0.4 else "distance-med" if distance < 0.55 else "distance-bad"
                                page_num = item.get("page") or item.get("page_number", "?")
                                text_snippet = item.get("text") or item.get("content", "")
                                st.markdown(
                                    f"""<div class="source-card">
                                    <div class="source-header">
                                        <span class="source-book">Page {page_num}</span>
                                        <span class="source-distance {dist_class}">Distance: {distance:.4f}</span>
                                    </div>
                                    <div class="source-text">{text_snippet}</div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )
                            st.divider()

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
