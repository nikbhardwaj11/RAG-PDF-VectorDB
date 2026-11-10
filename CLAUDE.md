# RAG-PDF-VectorDB - Complete Codebase Analysis & Documentation

> **Generated**: 2026-11-10
> **Purpose**: Comprehensive understanding of the Multi-Book RAG system for AI/RAG learning and management presentation

---

## 📋 Project Overview

**Multi-Book RAG Assistant** is a production-grade Retrieval-Augmented Generation system that enables users to:
- Upload and index multiple PDF books (fiction, non-fiction, compilations)
- Query across all books with intelligent retrieval strategies
- Get answers with source citations (book title, page number)
- Visualize retrieval metadata (strategy, confidence, cross-book distribution)

### Tech Stack
| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit (chat interface, sidebar management) |
| **Vector DB** | PostgreSQL + pgvector (HNSW index, FTS GIN index) |
| **Embeddings** | NVIDIA Nemotron 3.5 Embed 1B (2048-dim) |
| **LLM/Chat** | NVIDIA Nemotron 3.5 Bolt E4B |
| **PDF Processing** | pypdf (text extraction, outline detection) |
| **Parallelization** | ThreadPoolExecutor (8 workers for embeddings) |

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MULTI-BOOK RAG SYSTEM                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   USER       │     │   APP.PY     │     │  RETRIEVAL   │     │  DATABASE    │
│  INTERFACE   │────▶│  (STREAMLIT) │────▶│   .PY        │────▶│  (POSTGRESQL │
│  (CHAT UI)   │     │              │     │              │     │  + PGVECTOR) │
└──────────────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
                            │                    │                    │
                            ▼                    ▼                    ▼
                     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
                     │  INGEST.PY   │     │  EMBEDDINGS  │     │  VECTOR      │
                     │  (PDF PROC)  │────▶│  (NVIDIA     │────▶│  SEARCH      │
                     │              │     │   NEMOTRON   │     │  (HNSW +     │
                     └──────────────┘     │   3.5 EMBED) │     │   FTS + RRF) │
                                          └──────────────┘     └──────────────┘
```

---

## 📁 File Structure

```
RAG-PDF-VectorDB/
├── app.py                    # Main Streamlit application (39 KB)
├── database.py               # PostgreSQL + pgvector operations (20 KB)
├── ingest.py                 # PDF ingestion pipeline (15 KB)
├── retrieval.py              # Smart retrieval with Dynamic K (36 KB)
├── requirements.txt          # Dependencies
├── .env                      # Configuration (create from example)
├── books/                    # PDF storage (gitignored)
├── chroma_db/                # Legacy ChromaDB (unused)
├── test_db.py                # Database connection test
├── test_vector.py            # Vector operations test
├── test_retrieval.py         # Query analysis tests
├── RAG_SYSTEM_DOCUMENTATION.md  # Complete architecture docs (34 KB)
├── FLOW_DIAGRAMS.md          # Visual flow diagrams (24 KB)
└── README.md                 # Project documentation (4.5 KB)
```

---

## 🔧 Core Components Deep Dive

### 1. `database.py` - PostgreSQL + pgvector Layer

**Purpose**: Low-level database operations with pgvector support

**Key Functions**:
```python
init_db()                              # Create tables, indexes, extensions
get_connection()                       # Psycopg connection with vector support

# Search Functions (4 strategies)
search_chunks()                        # Pure vector cosine similarity
search_chunks_hybrid()                 # Vector + FTS with RRF
search_chunks_stratified()             # Window functions: per-book top-K
search_chunks_by_entity()              # Entity FTS + vector, stratified

# Management
get_db_stats()                         # Total chunks, books, per-book counts
get_all_books()                        # List all indexed books with stats
delete_book(book_title)                # Delete by book_title
clear_all_chunks()                     # Truncate table
```

**Schema**:
```sql
CREATE TABLE document_chunks (
    id              BIGSERIAL PRIMARY KEY,
    source_id       VARCHAR(64) NOT NULL,      -- filename_start_end
    filename        VARCHAR(255) NOT NULL,     -- original PDF filename
    book_title      VARCHAR(255) NOT NULL,     -- human-readable title
    page_number     INTEGER NOT NULL,          -- page in PDF
    chunk_index     INTEGER NOT NULL,          -- order within book
    content         TEXT NOT NULL,             -- chunk text
    embedding       vector(2048),              -- Nemotron 3.5 Embed
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Critical Indexes
CREATE INDEX idx_doc_chunks_fts ON document_chunks 
    USING gin(to_tsvector('english', content));
CREATE INDEX idx_doc_chunks_halfvec ON document_chunks 
    USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops);
```

---

### 2. `ingest.py` - PDF Processing Pipeline

**Purpose**: Extract, chunk, embed, and index PDF books

**Pipeline**:
```
PDF FILE → Detect Type (Compilation vs Single) → 
Extract Text (per page range) → 
Chunk (1200 chars, 300 overlap) → 
Parallel Embed (8 workers, Nemotron) → 
Batch Insert to PostgreSQL
```

**Key Functions**:
```python
# Text Extraction
extract_pdf_pages_range(file, start, end) → [{page, text}, ...]

# Chunking
chunk_pages(pages) → [{text, page}, ...]

# Embeddings (Parallel)
create_embeddings(texts, input_type="passage") → [[float], ...]

# Book Detection
get_compilation_book_ranges(reader) → [{book_title, start_page, end_page}, ...]

# Main Entry
scan_and_ingest_books(books_dir, progress_callback)
```

**Compilation PDF Support**: Automatically detects PDF outline/bookmarks to split one PDF into multiple logical books.

---

### 3. `retrieval.py` - Smart Retrieval Engine (Core Intelligence)

**Purpose**: Query analysis, Dynamic K calculation, 4 retrieval strategies

**Query Processing Flow**:
```
USER QUESTION
     │
     ▼
┌──────────────────────┐
│ ENTITY EXTRACTION    │  → ["Harry Potter", "Mowgli"]
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ QUERY CLASSIFICATION │  → ("COMPARISON", 0.67, entities)
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ DYNAMIC K CALCULATION│  → final_k = 20
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ STRATEGY ROUTING     │  → strategy_entity_comparison()
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ DATABASE SEARCH      │  → Vector + FTS + Window Functions
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ POST-PROCESS         │  → Sort, Deduplicate, Metadata
└──────────────────────┘
```

**4 Retrieval Strategies**:

| Strategy | Query Type | Technique | Best For |
|----------|------------|-----------|----------|
| **Stratified Multi-Book** | LISTING | `ROW_NUMBER() OVER (PARTITION BY book_title)` | "List all characters" |
| **Entity Comparison** | COMPARISON | Sub-query per entity + round-robin merge | "Compare Harry and Mowgli" |
| **Entity Timeline** | TIMELINE | Entity FTS + vector, chronological sort | "Timeline of Harry's life" |
| **Hybrid Precision** | THEMATIC/FACTUAL | RRF (Vector + FTS) + gap pruning | "Why did...", "What is..." |

**Dynamic K Configuration**:
```python
DYNAMIC_K_CONFIG = {
    "LISTING":      {"base": 21, "per_book": 3,  "min": 15, "max": 50},
    "COMPARISON":   {"base": 20, "per_entity": 10, "min": 15, "max": 40},
    "TIMELINE":     {"base": 18, "per_book": 2,  "min": 12, "max": 35},
    "THEMATIC":     {"base": 10, "per_book": 1,  "min": 8,  "max": 20},
    "FACTUAL":      {"base": 5,  "per_book": 0,  "min": 3,  "max": 10},
}
```

---

### 4. `app.py` - Streamlit Interface

**Purpose**: Chat UI, book management, visualization

**UI Sections**:
```
Sidebar:
├── Knowledge Base Stats (chunks, PDFs, chapters)
├── Upload PDFs (drag & drop)
├── Search Scope (All Books / Specific PDF)
├── Dynamic K / Manual K toggle
├── PDF Files list with index status
├── Sync/Index Button (with progress bar)
├── Book Manager (expandable - view/delete books)
└── Clear Chat/DB buttons

Main:
├── Title + Subtitle
├── Suggestion Pills (dynamic per query type)
├── Query Type Quick-Selector (5 buttons)
├── Chat History
│   ├── User messages
│   └── Assistant messages
│       ├── Retrieval metadata pill (type, strategy, K, confidence, entities)
│       ├── Cross-book view (tabs for COMPARISON/TIMELINE)
│       ├── Streaming answer
│       └── Expandable sources (grouped by book, with distances)
└── Chat Input
```

**Session State**:
```python
st.session_state.messages          # Chat history with metadata
st.session_state.pending_question  # From suggestion pills
st.session_state.show_book_manager # Book manager toggle
st.session_state.selected_query_type # Query type selector
```

---

## 🧠 Key AI/RAG Concepts Demonstrated

### 1. **Hybrid Search (Vector + FTS + RRF)**
- Vector similarity for semantic matching
- Full-text search for exact keyword matching
- Reciprocal Rank Fusion combines both rankings

### 2. **Stratified Retrieval (Window Functions)**
- PostgreSQL `ROW_NUMBER() OVER (PARTITION BY book_title)`
- Guarantees balanced representation across books
- Prevents one large book from dominating results

### 3. **Dynamic K (Adaptive Retrieval)**
- Adjusts chunk count based on query complexity
- LISTING: broad coverage (up to 50 chunks)
- FACTUAL: precision focus (3-10 chunks)

### 4. **Entity-Aware Retrieval**
- Extracts entities from questions without hardcoded lists
- Routes to entity-specific search strategies
- Enables implicit cross-book comparison

### 5. **Query Classification (Pattern-Based)**
- Regex patterns for 5 query types
- Confidence scoring for strategy selection
- Contextual boosts (e.g., 2+ entities → COMPARISON)

### 6. **Query Expansion**
- Type-specific expansion terms for better recall
- THEMATIC: "theme symbolism meaning significance..."
- COMPARISON: "similarities differences personality traits..."

### 7. **Reranking (Multi-Signal)**
- Vector similarity (70%) + Entity presence (20%) + Term overlap (10%)
- Timeline position bonus for chronological queries
- Gap pruning for FACTUAL queries

---

## 📊 Database Configuration

### Environment Variables (`.env`)
```env
# NVIDIA API
NVIDIA_API_KEY=your_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_CHAT_MODEL=nvidia/nemotron-3.5-bolt-e4b
NVIDIA_EMBED_MODEL=nvidia/nemotron-3.5-embed-1b

# PostgreSQL
PGHOST=localhost
PGPORT=5432
PGDATABASE=ragdb
PGUSER=raguser
PGPASSWORD=ragpassword
```

### Models
| Purpose | Model | Dimensions |
|---------|-------|------------|
| Chat/Generation | Nemotron 3.5 Bolt E4B | - |
| Embeddings | Nemotron 3.5 Embed 1B | 2048 |

---

## 🚀 Running the System

```bash
# 1. Start PostgreSQL with pgvector
docker run -d --name pgvector \
  -e POSTGRES_DB=ragdb -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=ragpassword -p 5432:5432 \
  pgvector/pgvector:pg16

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure .env with your NVIDIA API key

# 4. Add PDFs to books/ folder or upload via UI

# 5. Run app
streamlit run app.py

# 6. Click "Sync / Index Books" in sidebar
```

---

## 🧪 Testing

```bash
# Test database connection
python test_db.py

# Test vector operations
python test_vector.py

# Test query analysis (no DB needed)
python test_retrieval.py
```

---

## 🎯 Key Differentiators for Presentation

| Feature | Why It Matters |
|---------|----------------|
| **Stratified retrieval** | Prevents one book dominating results; guarantees each book represented |
| **Dynamic K** | Factual questions get 5 chunks (fast, precise); listings get 50 (comprehensive) |
| **Implicit cross-book comparison** | "Compare protagonists" works without naming characters |
| **Entity-aware** | Detects "Harry Potter" as single entity, not "Harry" + "Potter" |
| **Visual metadata** | Users see exactly which books, how many chunks, confidence, strategy |
| **Compilation PDF support** | One PDF with bookmarks → multiple indexed books automatically |
| **Parallel embeddings** | 8-worker ThreadPoolExecutor for fast ingestion |

---

## 📚 Learning Resources for AI/RAG

### Core Concepts Covered:
1. **RAG Architecture** - Retrieval → Augmentation → Generation
2. **Vector Databases** - pgvector, HNSW indexes, cosine similarity
3. **Hybrid Search** - Combining dense + sparse retrieval
4. **Query Understanding** - Classification, entity extraction, intent detection
5. **Adaptive Retrieval** - Dynamic K, strategy routing
6. **Cross-Document Reasoning** - Multi-book synthesis, comparison, timelines
7. **Production RAG** - Streaming, citations, metadata visualization, UI/UX

### Advanced Topics to Explore:
- Reranking with cross-encoders
- Query rewriting/decomposition
- Agentic RAG (multi-hop reasoning)
- Evaluation frameworks (RAGAS, retrieval metrics)
- Long-context handling
- Knowledge graphs integration

---

## 🔄 Ingestion vs Query Flow Summary

### INGESTION (One-time, batch)
```
PDF → Outline Detection → Page Ranges → Text Extraction → 
Chunking (1200/300) → Parallel Embedding → Batch Insert → pgvector + FTS
```

### QUERY (Real-time, per request)
```
Question → Entity Extraction → Classification → Dynamic K → 
Strategy Selection → Search (Vector/FTS/Window) → 
Reranking → Deduplication → Context Building → LLM Generation → 
Streaming Answer + Metadata + Source Citations
```

---

*This documentation covers the complete RAG-PDF-VectorDB codebase as of 2026-11-10. Use for learning, presentation, and further development.*