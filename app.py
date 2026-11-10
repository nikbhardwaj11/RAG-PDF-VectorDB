import os
import glob
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from database import (
    init_db,
    get_connection,
    search_chunks,
    search_chunks_hybrid,
    search_chunks_stratified,
    get_db_stats,
    get_all_books,
    delete_book,
    clear_all_chunks,
)
from ingest import create_embeddings, scan_and_ingest_books
from retrieval import (
    smart_retrieve,
    get_query_suggestions,
    extract_entities,
    classify_query,
)

load_dotenv()

# ===================== Configuration =====================
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b")

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
    /* hide the collapse/expand button (names differ between Streamlit versions) */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] { display: none !important; }

    /* fixed width, and stay open even if the browser remembered "collapsed" */
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

/* ===== CLEAN HEADER (keep it, so the running indicator still shows) ===== */
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

/* ===== ANSWER QUALITY INDICATOR ===== */
.quality-indicator {
    display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px;
    border-radius: 8px; font-size: .8rem; font-weight: 500;
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

/* ===== LOADING SPINNER ===== */
.stSpinner > div { border-color: #6366f1 !important; }

/* ===== CHAT INPUT ===== */
.stChatInputContainer { border-top: 1px solid #2d2d2d; padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

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
@st.cache_data(ttl=30)
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
            from pypdf import PdfReader
            reader = PdfReader(p)
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
        "LISTING": "📋 Listing",
        "COMPARISON": "⚖️ Comparison",
        "TIMELINE": "📅 Timeline",
        "THEMATIC": "🎭 Thematic",
        "FACTUAL": "❓ Factual",
    }
    return labels.get(query_type, query_type)


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
            passages_text.append(f"[Excerpt {idx} | Page {item['page']}]:\n{item['text']}")
        context_sections.append(f"### 📖 BOOK: {book_title}\n" + "\n\n".join(passages_text))

    context = "\n\n---\n\n".join(context_sections)

    query_type = query_meta.get("query_type", "FACTUAL") if query_meta else "FACTUAL"

    if query_type == "LISTING":
        type_instructions = """
SPECIAL INSTRUCTIONS FOR LISTINGS:
- Identify ALL items, people, associates, or concepts mentioned across the provided excerpts.
- Group the response logically (e.g., by book, by allegiance/relationship, or by role).
- For each entry, provide their role or significance and cite the source [Book: ..., Page ...].
- Be thorough and include every figure or item supported by the excerpts.
- If the excerpts don't contain a complete list, state what you found and note the limitation.
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
- Back each point with specific textual citations [Book: ..., Page ...].
- Maintain clear distinctions between different books/characters.
- If comparing characters from different books, explicitly note which book each character is from.
"""
        max_tokens = 1700
    elif query_type == "TIMELINE":
        type_instructions = """
SPECIAL INSTRUCTIONS FOR TIMELINES & ARCS:
- Chronologically trace major events, decisions, turning points, and consequences.
- Detail the progression from early events to climactic moments.
- Cite each event with [Book: ..., Page ...].
- Use clear temporal markers (Beginning, Middle, Climax, Resolution).
- If the character spans multiple books, trace their arc across all books.
"""
        max_tokens = 1600
    elif query_type == "THEMATIC":
        type_instructions = """
SPECIAL INSTRUCTIONS FOR THEMATIC ANALYSIS:
- Explore the deeper meanings, symbolism, motifs, and underlying messages.
- Synthesize thematic evidence across all represented stories.
- Explain WHY and HOW events symbolize these themes, citing [Book: ..., Page ...].
- Connect specific textual evidence to broader thematic claims.
"""
        max_tokens = 1500
    else:
        type_instructions = """
- Answer the question directly and factually in the opening paragraph.
- Provide supporting context and detail directly from the excerpts.
- Cite specific book titles and page numbers [Book: ..., Page ...].
- If a specific detail is not in the excerpts, explicitly state that the provided text does not contain that information.
"""
        max_tokens = 1100

    system_prompt = f"""
You are an expert literary research assistant with deep analytical capabilities across multi-book libraries.
Answer questions accurately and comprehensively based strictly on the provided book excerpts.

Core Principles:
1. Strict Grounding: Answer strictly using facts and events from the excerpts. Never hallucinate or import outside assumptions.
2. Cross-Book Synthesis: When questions involve multiple books or characters, maintain clear distinctions between the worlds and synthesize them thoughtfully.
3. Source Citations: Support every claim with citations in the format [Book Title, Page X].
4. Transparency: If a requested detail is not found in the excerpts, explicitly state that the provided text does not contain that information.
5. Accuracy Over Completeness: It's better to say "The excerpts don't mention X" than to guess.

{type_instructions}
"""

    user_prompt = f"Book Context (organized by book):\n\n{context}\n\nQuestion: {question}"

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
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
        by_book[c["book_title"]].append(c)

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
                st.markdown(
                    f"**Page {c['page']}** &nbsp; "
                    f"<span style='color:{dist_color};'>● Distance: {distance:.3f}</span>",
                    unsafe_allow_html=True,
                )
                st.write(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))
                st.divider()


def render_quality_indicator(meta: Dict[str, Any]):
    """Render answer quality indicator based on retrieval metadata."""
    confidence = meta.get("confidence", 0)
    unique_books = meta.get("unique_books_retrieved", 0)
    book_count = meta.get("book_count", 1)
    coverage = meta.get("coverage_pct", 0)

    # Determine quality based on multiple factors
    if confidence > 0.7 and coverage > 50:
        quality_class = "quality-high"
        quality_label = "High Confidence"
        quality_icon = "✅"
    elif confidence > 0.4 and coverage > 25:
        quality_class = "quality-medium"
        quality_label = "Medium Confidence"
        quality_icon = "⚠️"
    else:
        quality_class = "quality-low"
        quality_label = "Low Confidence"
        quality_icon = "❌"

    st.markdown(f"""
    <div class="quality-indicator {quality_class}">
        {quality_icon} {quality_label} — Coverage: {coverage:.0f}% ({unique_books}/{book_count} books) • Retrieval Confidence: {confidence:.0%}
    </div>
    """, unsafe_allow_html=True)


# ===================== Session State =====================
for key, default in [
    ("messages", []),
    ("pending_question", None),
    ("show_book_manager", False),
    ("selected_query_type", None),
    ("last_retrieval_meta", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ===================== Sidebar =====================
with st.sidebar:
    # --- Knowledge Base Stats ---
    st.markdown("## 📚 Knowledge Base")
    st.caption("PostgreSQL + pgvector Vector Database")

    db_stats, pdf_info, indexed_map, pdf_chapters = get_system_state()

    total_pdfs = len([p for p in pdf_info if p["filename"] in pdf_chapters])
    total_chapters = sum(len(c) for c in pdf_chapters.values())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunks", f"{db_stats['total_chunks']:,}")
    with col2:
        st.metric("PDFs Indexed", total_pdfs)
    with col3:
        st.metric("Chapters", total_chapters)

    st.divider()

    # --- Book Upload ---
    st.markdown("### 📤 Upload Books")
    st.caption("Drag & drop PDF story books. Saved to `books/` folder.")
    uploaded = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Supports single books and compilations (PDFs with bookmarks/TOC)"
    )
    if uploaded:
        for f in uploaded:
            save_path = os.path.join(BOOKS_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.read())
        st.success(f"✅ Uploaded {len(uploaded)} file(s). Click 'Sync / Index Books' to index them.")
        st.rerun()

    st.divider()

    # --- Search Scope ---
    st.markdown("### 🎯 Search Scope")
    scope_options = ["All Books"]
    for p in pdf_info:
        if p["filename"] in pdf_chapters:
            chapter_count = len(pdf_chapters[p["filename"]])
            scope_options.append(f"📄 {p['filename']} ({chapter_count} chapters)")

    selected_scope = st.selectbox(
        "Search in:",
        options=scope_options,
        index=0,
        help="🔍 'All Books' searches across all PDFs. Select a specific PDF to limit scope.",
        label_visibility="collapsed",
    )

    if selected_scope == "All Books":
        filter_param = None
        scope_info = f"🌍 **All Books** ({total_pdfs} PDFs, {total_chapters} chapters)"
    else:
        filter_param = selected_scope.split("📄 ")[1].split(" (")[0]
        chapter_count = len(pdf_chapters.get(filter_param, []))
        scope_info = f"📄 **{filter_param}** ({chapter_count} chapters)"

    st.caption(scope_info)
    st.caption("💡 Use 'All Books' for cross-book comparisons, timelines, and listings.")

    st.divider()

    # --- Dynamic K ---
    st.markdown("### 🧠 Retrieval Engine (Dynamic K)")
    k_mode = st.radio(
        "K Mode:",
        options=["🎯 Dynamic K (Auto)", "🎛️ Manual Fixed K"],
        index=0,
        help="Dynamic K automatically picks the best chunk count based on question type.",
        label_visibility="collapsed",
    )

    if k_mode == "🎛️ Manual Fixed K":
        manual_k_val = st.slider(
            "Manual Chunks (K)",
            min_value=3,
            max_value=60,
            value=10,
            step=1,
            help="Fixed number of passages to retrieve.",
        )
    else:
        manual_k_val = None
        st.info("🎯 **Dynamic K**: Listings/comparisons/timelines use more chunks; factual questions use fewer for precision.")

    st.divider()

    # --- PDF Files on Disk ---
    st.markdown("### 📁 PDF Files in `./books`")
    if pdf_info:
        for p in pdf_info:
            chapters = pdf_chapters.get(p["filename"], [])
            is_indexed = len(chapters) > 0
            status_class = "status-indexed" if is_indexed else "status-pending"
            status_text = "✅ Indexed" if is_indexed else "⏳ Not indexed"

            with st.container():
                st.markdown(f"""
                <div class="pdf-file-card">
                    <div class="title">📄 {p['filename']}</div>
                    <div class="stats">{p['pages']} pages • <span class="{status_class}">{status_text}</span></div>
                    <div class="chapters">{len(chapters)} chapters: {', '.join([c['book_title'][:28] for c in chapters[:4]])}{'...' if len(chapters) > 4 else ''}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No PDF files in `books/` folder. Upload some to get started!")

    # --- Index Button ---
    if st.button("⚡ Sync / Index Books into Vector DB", use_container_width=True, type="primary"):
        if not pdf_info:
            st.warning("Please upload or place PDF files in the `books/` folder first.")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(msg, pct):
                status_text.text(msg)
                progress_bar.progress(min(max(pct, 0.0), 1.0))

            try:
                with st.spinner("Indexing books into Vector DB..."):
                    results = scan_and_ingest_books(
                        books_dir=BOOKS_DIR,
                        progress_callback=update_progress,
                    )
                status_text.success("✅ Indexing complete!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                status_text.error(f"❌ Indexing failed: {e}")

    st.divider()

    # --- Book Manager ---
    if st.button("🗂️ Manage Indexed Books", use_container_width=True):
        st.session_state.show_book_manager = not st.session_state.show_book_manager
        st.rerun()

    if st.session_state.show_book_manager:
        st.markdown("### 🗂️ Book Manager")
        books = get_all_books()
        if books:
            by_pdf = defaultdict(list)
            for b in books:
                by_pdf[b["filename"]].append(b)

            for pdf_name, chapters in by_pdf.items():
                with st.expander(f"📄 {pdf_name} ({len(chapters)} chapters)", expanded=False):
                    for book in chapters:
                        st.markdown(f"""
                        <div class="book-card">
                            <h4>{book['book_title']}</h4>
                            <div class="meta">Chunks: {book['chunk_count']} • Pages: {book['min_page']}–{book['max_page']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"🗑️ Delete", key=f"del_{book['book_title']}_{pdf_name}", use_container_width=True):
                            deleted = delete_book(book['book_title'])
                            st.success(f"Deleted {deleted} chunks for '{book['book_title']}'")
                            st.rerun()
        else:
            st.info("No books indexed yet.")

        st.divider()
        if st.button("🗑️ Clear ALL Data", type="secondary", use_container_width=True):
            if st.checkbox("I understand this deletes everything permanently", key="confirm_clear_all"):
                clear_all_chunks()
                st.success("All data cleared!")
                st.rerun()

    st.divider()

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ===================== Main UI =====================
st.markdown('<div class="app-title">📚 Multi-Book RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'Conversational RAG with <b>Dynamic K</b> retrieval across multiple story books. '
    'Upload PDFs, index them, and ask questions with full source citations.'
    '</div>',
    unsafe_allow_html=True,
)

# Guidance when nothing indexed
if db_stats["total_chunks"] == 0:
    st.info(
        "**No books indexed yet.**\n\n"
        "👈 Upload PDF story books in the sidebar, then click **'⚡ Sync / Index Books into Vector DB'** "
        "to populate the vector database."
    )

# Dynamic query suggestions
st.markdown("### 💡 Suggested Questions")
suggestions = get_query_suggestions()
cols = st.columns(len(suggestions))
for i, (label, q) in enumerate(suggestions):
    if cols[i].button(label, key=f"sample_{i}", help=q, use_container_width=True):
        st.session_state.pending_question = q

# Query type quick-selector
st.markdown("### 🎯 Query Type Quick-Select")
qt_cols = st.columns(5)
query_types = ["LISTING", "COMPARISON", "TIMELINE", "THEMATIC", "FACTUAL"]
qt_labels = ["📋 Listing", "⚖️ Comparison", "📅 Timeline", "🎭 Thematic", "❓ Factual"]
for i, (qt, label) in enumerate(zip(query_types, qt_labels)):
    with qt_cols[i]:
        if st.button(label, key=f"qt_{qt}", use_container_width=True, help=f"Show {qt} example questions"):
            st.session_state.selected_query_type = qt
            type_suggestions = get_query_suggestions(qt)
            for j, (lbl, q) in enumerate(type_suggestions):
                st.caption(f"• {q}")

st.divider()


# ===================== Chat History =====================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "meta" in message and message["meta"]:
            m = message["meta"]
            query_type = m.get("query_type", "UNKNOWN")
            strategy = m.get("strategy", "unknown")
            dynamic_k = m.get("dynamic_k", 0)
            target_k = m.get("target_k", 0)
            entities = m.get("entities", [])
            confidence = m.get("confidence", 0)
            book_dist = m.get("book_distribution", {})
            unique_books = m.get("unique_books_retrieved", 0)
            retrieval_time = m.get("retrieval_time_ms", 0)

            # Query type badge
            qt_class = get_query_type_class(query_type)
            qt_label = get_query_type_label(query_type)
            st.markdown(f'<span class="query-type-badge {qt_class}">{qt_label}</span>', unsafe_allow_html=True)

            # Confidence color
            if confidence > 0.7:
                conf_class = "confidence-high"
                conf_label = "High"
            elif confidence > 0.4:
                conf_class = "confidence-med"
                conf_label = "Medium"
            else:
                conf_class = "confidence-low"
                conf_label = "Low"

            # Build retrieval badge
            badge_html = f'<div class="retrieval-pill">'
            badge_html += f'🎯 <b>Dynamic K:</b> {dynamic_k}/{target_k} chunks &nbsp;|&nbsp; '
            badge_html += f'⚡ <b>Strategy:</b> <code>{strategy}</code> &nbsp;|&nbsp; '
            badge_html += f'🎯 <b>Confidence:</b> <span class="{conf_class}">{conf_label} ({confidence:.0%})</span>'
            badge_html += f' &nbsp;|&nbsp; ⏱️ <b>Retrieval:</b> {retrieval_time}ms'

            if entities:
                badge_html += f'<br><div style="margin-top:6px;">'
                for e in entities[:6]:
                    badge_html += f'<span class="entity-badge">{e}</span>'
                if len(entities) > 6:
                    badge_html += f'<span class="entity-badge">+{len(entities)-6} more</span>'
                badge_html += '</div>'

            if book_dist and unique_books > 2:
                badge_html += f' <span class="cross-book-badge">📚 Cross-book ({unique_books} books)</span>'

            badge_html += '</div>'
            st.markdown(badge_html, unsafe_allow_html=True)

            # Quality indicator
            render_quality_indicator(m)

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
                    bt = c["book_title"]
                    book_groups.setdefault(bt, []).append(c)

                for book_title, group in book_groups.items():
                    st.markdown(f'**📘 {book_title}** ({len(group)} chunks)')
                    for item in group:
                        distance = item.get('distance', 0)
                        dist_color = "#22c55e" if distance < 0.4 else "#f59e0b" if distance < 0.55 else "#ef4444"
                        dist_class = "distance-good" if distance < 0.4 else "distance-med" if distance < 0.55 else "distance-bad"
                        st.markdown(
                            f"""<div class="source-card">
                            <div class="source-header">
                                <span class="source-book">Page {item['page']}</span>
                                <span class="source-distance {dist_class}">Distance: {distance:.4f}</span>
                            </div>
                            <div class="source-text">{item['text']}</div>
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
                "Upload PDFs and click **'⚡ Sync / Index Books into Vector DB'** first."
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

                # Handle case where no chunks retrieved
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

                    # Retrieval metadata badge
                    strategy = meta.get("strategy", "unknown")
                    dynamic_k = meta.get("dynamic_k", 0)
                    target_k = meta.get("target_k", 0)
                    entities = meta.get("entities", [])
                    confidence = meta.get("confidence", 0)
                    book_dist = meta.get("book_distribution", {})
                    unique_books = meta.get("unique_books_retrieved", 0)
                    retrieval_time = meta.get("retrieval_time_ms", 0)

                    if confidence > 0.7:
                        conf_class = "confidence-high"
                        conf_label = "High"
                    elif confidence > 0.4:
                        conf_class = "confidence-med"
                        conf_label = "Medium"
                    else:
                        conf_class = "confidence-low"
                        conf_label = "Low"

                    badge_html = f'<div class="retrieval-pill">'
                    badge_html += f'🎯 <b>Dynamic K:</b> {dynamic_k}/{target_k} chunks &nbsp;|&nbsp; '
                    badge_html += f'⚡ <b>Strategy:</b> <code>{strategy}</code> &nbsp;|&nbsp; '
                    badge_html += f'🎯 <b>Confidence:</b> <span class="{conf_class}">{conf_label} ({confidence:.0%})</span>'
                    badge_html += f' &nbsp;|&nbsp; ⏱️ <b>Retrieval:</b> {retrieval_time}ms'

                    if entities:
                        badge_html += f'<br><div style="margin-top:6px;">'
                        for e in entities[:6]:
                            badge_html += f'<span class="entity-badge">{e}</span>'
                        if len(entities) > 6:
                            badge_html += f'<span class="entity-badge">+{len(entities)-6} more</span>'
                        badge_html += '</div>'

                    if book_dist and unique_books > 2:
                        badge_html += f' <span class="cross-book-badge">📚 Cross-book ({unique_books} books)</span>'

                    badge_html += '</div>'
                    st.markdown(badge_html, unsafe_allow_html=True)

                    # Quality indicator
                    render_quality_indicator(meta)

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
                            bt = c["book_title"]
                            book_groups.setdefault(bt, []).append(c)

                        for book_title, group in book_groups.items():
                            st.markdown(f'**📘 {book_title}** ({len(group)} chunks)')
                            for item in group:
                                distance = item.get('distance', 0)
                                dist_color = "#22c55e" if distance < 0.4 else "#f59e0b" if distance < 0.55 else "#ef4444"
                                dist_class = "distance-good" if distance < 0.4 else "distance-med" if distance < 0.55 else "distance-bad"
                                st.markdown(
                                    f"""<div class="source-card">
                                    <div class="source-header">
                                        <span class="source-book">Page {item['page']}</span>
                                        <span class="source-distance {dist_class}">Distance: {distance:.4f}</span>
                                    </div>
                                    <div class="source-text">{item['text']}</div>
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