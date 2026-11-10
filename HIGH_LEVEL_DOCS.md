# High-Level Documentation - Multi-Book RAG System

> **Target Audience**: Manager, stakeholders, team members  
> **Purpose**: Understand each major component/topic without code details

---

## 📚 Topic 1: What is RAG (Retrieval-Augmented Generation)?

### The Problem
LLMs have knowledge cutoff dates and hallucinate facts. They can't access your private documents.

### The RAG Solution
```
User Question → Retrieve Relevant Passages → Inject into LLM Prompt → Generate Grounded Answer
```

### Our Implementation
- **Retrieval**: PostgreSQL + pgvector (semantic + keyword search)
- **Augmentation**: Chunks organized by book with page citations
- **Generation**: NVIDIA Nemotron 3.5 Bolt with streaming

### Why It Matters
- **Accuracy**: Answers cite exact book pages
- **Freshness**: New PDFs indexed instantly
- **Control**: You own the knowledge base
- **Auditability**: Every answer shows its sources

---

## 📚 Topic 2: Vector Databases & Embeddings

### What are Embeddings?
Text → Mathematical vectors (2048 numbers) → Similar meaning = Close in vector space

```
"Harry Potter wizard"  →  [0.12, -0.05, 0.87, ...]  (2048 dimensions)
"Hermione magic spell" →  [0.11, -0.04, 0.85, ...]  (close distance)
"Car engine repair"    →  [0.92, 0.31, -0.12, ...]  (far distance)
```

### Our Setup
| Component | Choice | Reason |
|-----------|--------|--------|
| **Model** | Nemotron 3.5 Embed 1B | Fast, 2048-dim, good quality |
| **Database** | PostgreSQL + pgvector | Production-grade, SQL + vector |
| **Index** | HNSW (halfvec) | Fast approximate search |
| **Distance** | Cosine similarity | Standard for embeddings |

### Search Types
1. **Vector Search**: Semantic similarity (embedding ↔ embedding)
2. **Full-Text Search**: Keyword matching (PostgreSQL FTS)
3. **Hybrid (RRF)**: Best of both combined

---

## 📚 Topic 3: Multi-Book Architecture

### The Challenge
- Multiple PDFs in one system
- Some PDFs = single book, others = compilations (anthologies)
- Questions span books: "Compare heroes across all books"

### Our Solution: Logical Book Model
```
PDF File: "classics_compilation.pdf"
    ├── Book: "The Jungle Book" (pages 1-50)
    ├── Book: "Sherlock Holmes" (pages 51-120)
    └── Book: "Dracula" (pages 121-200)

PDF File: "harry_potter.pdf"  
    └── Book: "Harry Potter" (all pages, no outline)
```

### Database Schema
```sql
document_chunks:
  - book_title: "The Jungle Book"     ← Human-readable
  - filename: "classics_compilation.pdf"  ← Source file
  - page_number: 15                   ← Page in PDF
  - chunk_index: 3                    ← Order within book
```

### Benefits
- **Granular citations**: "Page 15 of The Jungle Book"
- **Cross-book queries**: Natural "across all books" search
- **Flexible management**: Delete/update per logical book

---

## 📚 Topic 4: Dynamic K - Adaptive Retrieval

### The Insight
Different questions need different amounts of context:
- "What is the capital?" → Need 1-2 precise chunks
- "List all characters" → Need 50+ chunks across books

### Our Dynamic K System
| Query Type | Example | Chunks Retrieved | Strategy |
|------------|---------|------------------|----------|
| **FACTUAL** | "Who is the mentor?" | 3-10 | High precision |
| **THEMATIC** | "Explain the theme" | 8-20 | Semantic breadth |
| **TIMELINE** | "Hero's journey" | 12-35 | Chronological |
| **COMPARISON** | "Compare X and Y" | 15-40 | Per-entity balance |
| **LISTING** | "All characters" | 15-50 | Broad coverage |

### Formula Examples (2 books)
```
FACTUAL:      base=5                          → 5 chunks
THEMATIC:     base=10 + 1×2 books             → 12 chunks
TIMELINE:     base=18 + 2×(2-1) books         → 20 chunks
COMPARISON:   10 × 2 entities                 → 20 chunks
LISTING:      3 × 2 books = 6 → clamp to min 15 → 15 chunks
```

### Why It Matters
- **Speed**: Factual queries fast (fewer chunks → less LLM context)
- **Quality**: Complex queries comprehensive (more chunks → better coverage)
- **Cost**: Optimized token usage

---

## 📚 Topic 5: Four Retrieval Strategies

### Strategy 1: Stratified Multi-Book (LISTING)
**Query**: "List all characters across all books"

**How it Works**:
```sql
ROW_NUMBER() OVER (PARTITION BY book_title ORDER BY similarity)
```
→ Gets top-K chunks **per book**, not globally

**Result**: 10 chunks from Book A + 10 chunks from Book B (balanced!)

**Why**: Prevents one large book dominating "list everything" answers

---

### Strategy 2: Entity Comparison (COMPARISON)
**Query**: "Compare Mowgli and Harry Potter"

**How it Works**:
```
For EACH entity detected:
  1. Build entity-specific query: "Mowgli personality development relationships"
  2. Search all books for that entity
  3. Get per-book top chunks

MERGE: Round-robin [Mowgli-1, Harry-1, Mowgli-2, Harry-2...]
```

**Result**: Balanced representation of both characters across all books

**Why**: "Compare" needs deep, balanced coverage of each entity

---

### Strategy 3: Entity Timeline (TIMELINE)
**Query**: "Timeline of Harry Potter's life"

**How it Works**:
```
1. Detect primary entity: "Harry Potter"
2. Build timeline query: "Harry Potter major events actions story"
3. Search with entity focus + vector similarity
4. Sort results by (book_title, page_number) → Chronological!
```

**Result**: Chronological passages showing character arc

**Why**: Timelines need temporal ordering, not just relevance

---

### Strategy 4: Hybrid Precision (THEMATIC/FACTUAL)
**Query**: "Why did Harry choose to fight?" or "What is the capital?"

**How it Works**:
```
1. Vector search (semantic) → ranked by cosine distance
2. Full-text search (keywords) → ranked by tf-idf
3. RRF Fusion: score = 1/(60+vec_rank) + 1/(60+fts_rank)
4. Gap pruning: stop when distance jumps > 0.15
```

**Result**: High-precision, focused chunks

**Why**: Fact/thematic queries need precision, not breadth

---

## 📚 Topic 6: Query Understanding Pipeline

### Step 1: Entity Extraction
```
Input: "Summarize Harry Potter's journey across the books"
         │
         ├── Normalize: "Harry Potter's" → "Harry Potter"
         ├── Remove first stop word: "Summarize" → removed
         ├── Multi-word match: "Harry Potter" ✓
         ├── Filter: "journey", "books" (too generic)
         └── Result: ["Harry Potter"]
```

**Techniques**: Capitalized phrase detection, possessive normalization, stop-word filtering, deduplication (prefer longer)

---

### Step 2: Query Classification
```
Input: "Compare Mowgli and Harry Potter"

Patterns Matched:
  LISTING:     0 matches
  COMPARISON:  "compare" ✓ (weight 1.0) 
  TIMELINE:    0 matches
  THEMATIC:    0 matches
  FACTUAL:     0 matches

Contextual Boost: 2 entities detected → +2.0 to COMPARISON

Result: COMPARISON, confidence=0.67, entities=["Mowgli", "Harry Potter"]
```

**Patterns per Type**: 5-8 regex patterns + keyword sets per query type

---

### Step 3: Strategy Routing
```python
if query_type == "LISTING" and scope == "All Books":
    → strategy_stratified_listing()
elif query_type == "COMPARISON" and scope == "All Books":
    → strategy_entity_comparison()
elif query_type == "TIMELINE" and scope == "All Books":
    → strategy_entity_timeline()
else:
    → strategy_hybrid_precision()  # THEMATIC, FACTUAL, or filtered
```

---

## 📚 Topic 7: Ingestion Pipeline

### PDF Processing Flow
```
1. SCAN books/ folder for *.pdf
       │
       ├── HAS outline/bookmarks → COMPILATION
       │   → Extract: [{book_title, start_page, end_page}, ...]
       │
       └── NO outline → SINGLE BOOK
           → Title from filename, pages 1-N
       │
2. FOR EACH BOOK RANGE:
       │
       ├── DEDUP CHECK: is_book_indexed()? → SKIP
       │
       ├── EXTRACT TEXT: Per page, skip empty (scanned PDF safe)
       │
       ├── CHUNK: 1200 chars, 300 overlap
       │
       ├── EMBED: Parallel (8 workers), Nemotron, retry with backoff
       │
       └── INSERT: Batch executemany → PostgreSQL
```

### Key Features
- **Idempotent**: Re-run safely, skips already-indexed books
- **Compilation support**: One PDF → multiple logical books
- **Progress callbacks**: Real-time UI updates
- **Error handling**: Graceful skip for scanned/image PDFs

---

## 📚 Topic 8: Streamlit UI Architecture

### State Management
```python
st.session_state.messages          # Full chat history with metadata
st.session_state.pending_question  # From suggestion pills
st.session_state.show_book_manager # Sidebar toggle
st.session_state.selected_query_type # Quick-selector state
```

### Key UI Components

| Component | Purpose |
|-----------|---------|
| **Suggestion Pills** | One-click example questions per query type |
| **Query Type Selector** | Force specific strategy, see examples |
| **Search Scope** | Filter: All Books vs Single PDF |
| **Dynamic K Toggle** | Auto vs Manual chunk count |
| **Retrieval Metadata Pills** | Show strategy, K, confidence, entities |
| **Cross-Book Tabs** | Side-by-side view for COMPARISON/TIMELINE |
| **Expandable Sources** | Grouped by book, with distance scores |

### Visual Feedback
- **Color-coded confidence**: Green (>70%) / Yellow (>40%) / Red
- **Distance badges**: Green (<0.4) / Yellow (<0.55) / Red
- **Book distribution bars**: Visual % per book in results
- **Quality indicator**: Overall answer confidence assessment

---

## 📚 Topic 9: Database & Search Internals

### Indexes Created
```sql
-- B-tree for metadata filtering
CREATE INDEX idx_doc_chunks_book_title ON document_chunks(book_title);
CREATE INDEX idx_doc_chunks_filename ON document_chunks(filename);

-- GIN for Full-Text Search
CREATE INDEX idx_doc_chunks_fts 
    ON document_chunks USING gin(to_tsvector('english', content));

-- HNSW for Vector Search (approximate, fast)
CREATE INDEX idx_doc_chunks_halfvec 
    ON document_chunks USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops);
```

### Search Queries

**Pure Vector**:
```sql
SELECT * FROM document_chunks
WHERE book_title ILIKE '%filter%'
ORDER BY embedding <=> $query_vector
LIMIT $top_k;
```

**Hybrid (RRF)**:
```sql
WITH candidates AS (
  SELECT *,
    ROW_NUMBER() OVER (ORDER BY embedding <=> $qv) AS vec_rank,
    ROW_NUMBER() OVER (ORDER BY ts_rank(...) DESC) AS fts_rank
  FROM document_chunks
  WHERE embedding <=> $qv < 0.68 OR content @@ $fts_query
  LIMIT 60
)
SELECT *, (1.0/(60+vec_rank)) + (1.0/(60+fts_rank)) AS rrf_score
FROM candidates
ORDER BY rrf_score DESC LIMIT $top_k;
```

**Stratified (Window Function)**:
```sql
WITH ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY book_title ORDER BY embedding <=> $qv) AS rank_in_book
  FROM document_chunks
  WHERE embedding <=> $qv <= 0.68 OR content @@ $fts_query
)
SELECT * FROM ranked WHERE rank_in_book <= $per_book_k
ORDER BY book_title, page_number;
```

---

## 📚 Topic 10: Production Considerations

### Performance
| Aspect | Approach |
|--------|----------|
| **Embedding speed** | 8 parallel workers, async-friendly |
| **Search speed** | HNSW index (~ms), FTS GIN index |
| **Token efficiency** | Dynamic K prevents over-retrieval |
| **Streaming** | Token-by-token LLM response |

### Reliability
- **Idempotent ingestion**: Safe to re-run
- **Graceful degradation**: Hybrid → pure vector fallback
- **Retry logic**: Exponential backoff for embedding API
- **Connection pooling**: Single connection per request

### Scalability
- **pgvector**: Handles millions of vectors
- **Partitioning**: Can shard by book_title if needed
- **Async ready**: Architecture supports async DB drivers

### Monitoring (via UI)
- Retrieval time (ms)
- Chunks retrieved vs target
- Book distribution %
- Confidence scores
- Entity detection results

---

## 📚 Topic 11: Extending the System

### Easy Customizations
```python
# ingest.py - Chunking
CHUNK_SIZE = 1200      # Larger = more context, fewer chunks
CHUNK_OVERLAP = 300    # More overlap = better boundary continuity
MAX_EMBED_WORKERS = 8  # Adjust for API rate limits

# retrieval.py - Dynamic K
DYNAMIC_K_CONFIG["FACTUAL"]["max"] = 15  # More context for facts

# database.py - Search thresholds
max_distance=0.68      # Stricter = more precise, fewer results
```

### Advanced Extensions
1. **Cross-encoder reranking**: Add re-ranking model after retrieval
2. **Query rewriting**: LLM rewrites questions for better search
3. **Multi-hop retrieval**: Follow-up searches for complex queries
4. **Knowledge graph**: Extract entities/relations, build graph
5. **Evaluation**: Add RAGAS metrics, retrieval recall@k
6. **Multi-modal**: Add image/pdf figure extraction

---

## 📚 Topic 12: AI/RAG Learning Path

### This Project Teaches:
| Concept | Where It's Demonstrated |
|---------|------------------------|
| RAG Architecture | app.py → retrieval.py → database.py |
| Vector Search | database.py search functions |
| Hybrid Search | search_chunks_hybrid() with RRF |
| Window Functions | search_chunks_stratified() |
| Query Understanding | retrieval.py classify/extract |
| Adaptive Retrieval | Dynamic K + Strategy routing |
| Cross-Doc Reasoning | Entity comparison/timeline |
| Production RAG | Streaming, citations, metadata UI |

### Next Steps for Learning:
1. **Run the system** with sample PDFs
2. **Trace a query** through logs/UI metadata
3. **Modify Dynamic K** values, observe changes
4. **Add a new query type** (e.g., "CHARACTER_ANALYSIS")
5. **Implement cross-encoder reranking**
6. **Add evaluation harness** with ground truth

---

## 📋 Summary for Management

| Metric | Value |
|--------|-------|
| **Core Files** | 4 Python modules (~110 KB total) |
| **Strategies** | 4 retrieval strategies |
| **Query Types** | 5 (LISTING, COMPARISON, TIMELINE, THEMATIC, FACTUAL) |
| **Dynamic K Range** | 3-50 chunks (adaptive) |
| **Embedding Model** | Nemotron 3.5 Embed 1B (2048-dim) |
| **LLM** | Nemotron 3.5 Bolt E4B |
| **Vector DB** | PostgreSQL + pgvector (HNSW + FTS) |
| **UI** | Streamlit with real-time metadata |
| **Ingestion** | Parallel, idempotent, compilation-aware |

### Key Innovations
1. **Stratified retrieval** via PostgreSQL window functions
2. **Entity-aware strategy routing** without hardcoded entities
3. **Dynamic K** adaptive to query complexity
4. **Compilation PDF support** via outline detection
5. **Visual metadata** for trust/transparency

---

*High-level documentation for Multi-Book RAG Assistant - Understanding each topic without code diving*