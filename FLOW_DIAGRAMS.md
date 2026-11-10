# Visual Flow Diagrams - Quick Reference

---

## 1. Complete System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-BOOK RAG ARCHITECTURE                        │
└────────────────────────────────────────────────────────────────────────────┘

  ┌─────────┐      ┌──────────┐      ┌────────────┐      ┌────────────────┐
  │  PDF    │─────▶│ INGEST   │─────▶│ EMBEDDINGS │─────▶│ POSTGRESQL +   │
  │  FILES  │      │ PIPELINE │      │ (Nemotron) │      │ PGVECTOR DB    │
  └─────────┘      └──────────┘      └────────────┘      └───────┬────────┘
         │                                                       │
         │                ┌──────────┐                          │
         │                │ QUERY    │                          │
         └───────────────▶│ PROCESSOR│                          │
                          │(retrieval)                          │
                          └────┬─────┘                          │
                               │                                │
                               ▼                                │
                    ┌─────────────────────┐                     │
                    │  4 STRATEGIES       │                     │
                    ├─────────────────────┤                     │
                    │ • Stratified Listing│                     │
                    │ • Entity Comparison │                     │
                    │ • Entity Timeline   │                     │
                    │ • Hybrid Precision  │                     │
                    └──────────┬──────────┘                     │
                               │                                │
                               ▼                                │
                    ┌─────────────────────┐                     │
                    │  SEARCH (Vector +   │                     │
                    │  FTS + Window Fn)   │                     │
                    └──────────┬──────────┘                     │
                               │                                │
                               ▼                                │
                    ┌─────────────────────┐                     │
                    │  RANKED CHUNKS +    │                     │
                    │  METADATA           │                     │
                    └──────────┬──────────┘                     │
                               │                                │
                               ▼                                │
                    ┌─────────────────────┐                     │
                    │  CONTEXT BUILDING   │                     │
                    │  + NEMOTRON CHAT    │                     │
                    └──────────┬──────────┘                     │
                               │                                │
                               ▼                                ▼
                    ┌─────────────────────┐            ┌────────────────┐
                    │  STREAMLIT UI       │            │  SOURCE CITES  │
                    │  • Answer           │            │  • Book title  │
                    │  • Retrieval pills  │            │  • Page number │
                    │  • Cross-book tabs  │            │  • Distance    │
                    │  • Expandable srcs  │            └────────────────┘
                    └─────────────────────┘
```

---

## 2. Ingestion Pipeline Detail

```
PDF IN books/
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ SCAN & DETECT PDF TYPE                                       │
│                                                              │
│  PDFReader → .outline (bookmarks/TOC)                       │
│       │                                                       │
│       ├─ HAS OUTLINE ──▶ COMPILATION PDF                     │
│       │     Extract: [{book_title, start_page, end_page},..]│
│       │                                                       │
│       └─ NO OUTLINE ──▶ SINGLE BOOK                          │
│             Title = filename (cleaned)                       │
│             Range = pages 1 to N                             │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ FOR EACH BOOK RANGE                                          │
│                                                              │
│  1. DEDUP CHECK ──▶ is_book_indexed(book_title)             │
│     └─ YES: Skip, log "already indexed"                      │
│                                                              │
│  2. EXTRACT TEXT ──▶ extract_pdf_pages_range()              │
│     └─ Iterate pages, skip empty (scanned PDF safe)         │
│     └─ Return: [{page, text}, ...]                          │
│                                                              │
│  3. CHUNK ──▶ chunk_pages()                                  │
│     └─ 1200 chars, 300 overlap                               │
│     └─ Return: [{text, page}, ...]                          │
│                                                              │
│  4. EMBED ──▶ create_embeddings() [PARALLEL: 8 workers]     │
│     └─ ThreadPoolExecutor + retry with backoff              │
│     └─ Return: [[float × 2048], ...]                        │
│                                                              │
│  5. INSERT ──▶ insert_chunks() [BATCH]                      │
│     └─ (source_id, filename, book_title, page, idx,        │
│        content, embedding::vector)                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Query Processing Pipeline

```
USER QUESTION: "Compare Mowgli and Harry Potter"
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: ENTITY EXTRACTION                                    │
│                                                              │
│  Input: "Compare Mowgli and Harry Potter"                   │
│                                                              │
│  • Normalize possessive: none                               │
│  • First word "Compare" ∈ STOP_WORDS → remove               │
│  • Extraction text: "Mowgli and Harry Potter"               │
│  • Multi-word regex: "Harry Potter"                         │
│  • Single words: "Mowgli", "Harry", "Potter", "And"         │
│  • Filter: "And" ∈ STOP_WORDS, "Harry"/"Potter" in multi    │
│  • Result: ["Harry Potter", "Mowgli"]                       │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: QUERY CLASSIFICATION                                 │
│                                                              │
│  Patterns checked against: "compare mowgli and harry potter"│
│                                                              │
│  LISTING:     0 matches                                      │
│  COMPARISON:  "compare" ✓ (1.0)                              │
│  TIMELINE:    0 matches                                      │
│  THEMATIC:    0 matches                                      │
│                                                              │
│  Best: COMPARISON, score=1.0, confidence=0.33               │
│  Entities: ["Harry Potter", "Mowgli"]                       │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: DYNAMIC K CALCULATION                                │
│                                                              │
│  Config: base=20, per_entity=10, min=15, max=40             │
│  book_count = 2 (from DB: Jungle Book + Harry Potter)       │
│  entity_count = 2                                            │
│                                                              │
│  k = 10 × 2 = 20                                            │
│  clamp(15, 40, 20) = 20                                     │
│                                                              │
│  Final K = 20                                                │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: STRATEGY ROUTING                                     │
│                                                              │
│  query_type = COMPARISON                                     │
│  scope = "All Books" (no filter)                             │
│                                                              │
│  → strategy_entity_comparison()                              │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 5: ENTITY COMPARISON STRATEGY                           │
│                                                              │
│  FOR "Harry Potter":                                        │
│    query = "Harry Potter character personality development  │
│             role relationships story arc"                    │
│    embed → search_chunks_by_entity()                         │
│    → 10 chunks (5 per book × 2 books)                       │
│                                                              │
│  FOR "Mowgli":                                              │
│    query = "Mowgli character personality development ...    │
│    embed → search_chunks_by_entity()                         │
│    → 10 chunks (5 per book × 2 books)                       │
│                                                              │
│  MERGE: Round-robin [HP1, M1, HP2, M2, ...]                │
│  LIMIT: 20 chunks                                            │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 6: POST-PROCESS                                         │
│                                                              │
│  • Sort by book_title, page_number                          │
│  • Deduplicate by chunk ID                                   │
│  • book_distribution = {"Harry Potter": 10, "Jungle Book": 10}│
│  • Metadata = {                                              │
│      query_type: "COMPARISON",                               │
│      strategy: "entity_comparison",                          │
│      dynamic_k: 20, target_k: 20,                           │
│      entities: ["Harry Potter", "Mowgli"],                  │
│      confidence: 0.33,                                       │
│      book_distribution: {...},                               │
│      reasoning: "Comparison of Harry Potter, Mowgli: ..."   │
│    }                                                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Four Retrieval Strategies Visual

### A. STRATIFIED LISTING (LISTING queries)

```
QUERY: "List all characters across all books"
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ search_chunks_stratified()                          │
│                                                     │
│ SQL: ROW_NUMBER() OVER (PARTITION BY book_title    │
│       ORDER BY embedding <=> query_vector)         │
│                                                     │
│ per_book_k = final_k / book_count = 20 / 2 = 10   │
└─────────────────────────────────────────────────────┘
      │
      ▼
RESULT: 10 chunks from Book A + 10 chunks from Book B
        (Balanced! No single book dominates)
```

### B. ENTITY COMPARISON (COMPARISON queries)

```
QUERY: "Compare Harry and Mowgli" (entities detected)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ FOR EACH ENTITY:                                    │
│                                                     │
│ Entity "Harry Potter":                              │
│   search_chunks_by_entity("Harry Potter", ...)     │
│   → FTS match + vector similarity                   │
│   → per_book_k = 5 per book                        │
│                                                     │
│ Entity "Mowgli":                                    │
│   search_chunks_by_entity("Mowgli", ...)           │
│   → FTS match + vector similarity                   │
│   → per_book_k = 5 per book                        │
│                                                     │
│ MERGE: Round-robin [H1, M1, H2, M2, ...]          │
└─────────────────────────────────────────────────────┘
      │
      ▼
RESULT: Balanced entity representation across books
```

### C. ENTITY TIMELINE (TIMELINE queries)

```
QUERY: "Timeline of Harry Potter's life" (entity: "Harry Potter")
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ search_chunks_by_entity("Harry Potter", ...)       │
│                                                     │
│ query = "Harry Potter major events actions         │
│          story key moments timeline"               │
│                                                     │
│ per_book_k = final_k / book_count                  │
│                                                     │
│ ORDER BY book_title, page_number                   │
│ → Chronological within each book                   │
└─────────────────────────────────────────────────────┘
      │
      ▼
RESULT: Chronological passages from each book
```

### D. HYBRID PRECISION (THEMATIC/FACTUAL queries)

```
QUERY: "Why did Harry choose to fight?" (THEMATIC)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ search_chunks_hybrid()                              │
│                                                     │
│ RRF = 1/(60+vec_rank) + 1/(60+fts_rank)           │
│                                                     │
│ Candidates: 60 (vector ∪ FTS)                      │
│ Final: top_k (e.g., 10)                            │
│                                                     │
│ GAP PRUNING: Stop if distance jump > 0.15          │
│              or distance > 0.62                    │
└─────────────────────────────────────────────────────┘
      │
      ▼
RESULT: High-precision relevant chunks only
```

---

## 5. Dynamic K Decision Tree

```
QUERY TYPE
    │
    ├─▶ LISTING
    │     K = 3 × book_count
    │     Range: 15-50
    │
    ├─▶ COMPARISON
    │     K = 10 × entity_count (min 2)
    │     Range: 15-40
    │
    ├─▶ TIMELINE
    │     K = 18 + 2 × (book_count - 1)
    │     Range: 12-35
    │
    ├─▶ THEMATIC
    │     K = 10 + 1 × book_count
    │     Range: 8-20
    │
    └─▶ FACTUAL
          K = 5 (fixed)
          Range: 3-10

EXAMPLES:
─────────────────────────────────────────────
2 books, 0 entities:
  LISTING:     3×2=6  → clamp→15
  COMPARISON:  10×2=20
  TIMELINE:    18+2=20
  THEMATIC:    10+2=12
  FACTUAL:     5

3 books, 2 entities:
  LISTING:     3×3=9  → clamp→15
  COMPARISON:  10×2=20
  TIMELINE:    18+4=22
  THEMATIC:    10+3=13
  FACTUAL:     5
```

---

## 6. UI State Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STREAMLIT APP STATE                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  session_state.messages = [                                 │
│    {"role": "user", "content": "..."},                      │
│    {"role": "assistant", "content": "...",                  │
│     "chunks": [...], "meta": {...}}                         │
│  ]                                                          │
│                                                             │
│  session_state.pending_question = "question from pill"      │
│  session_state.show_book_manager = True/False               │
│  session_state.selected_query_type = "COMPARISON"           │
│                                                             │
└─────────────────────────────────────────────────────────────┘

SIDEBAR STATE:
─────────────────────────────────────────────────────────────
selected_scope = "All Books" | "📄 filename (N chapters)"
k_mode = "🎯 Dynamic K (Auto)" | "🎛️ Manual Fixed K"
manual_k_val = 8 (if manual)

MAIN UI FLOW:
─────────────────────────────────────────────────────────────
1. Render sidebar → get system state (stats, pdf_chapters)
2. Render suggestion pills (from get_query_suggestions())
3. Render query type buttons
4. Render chat history:
   For each assistant message:
     - Show retrieval pill (type, strategy, K, confidence, entities)
     - If COMPARISON/TIMELINE: show cross-book tabs
     - Show answer
     - Expandable: book distribution bars + grouped sources
5. Chat input → process → append to messages → rerun
```

---

## 7. Database Query Patterns

### Hybrid Search (RRF)
```sql
WITH candidates AS (
  SELECT *,
    ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS vec_rank,
    ROW_NUMBER() OVER (ORDER BY ts_rank(...) DESC) AS fts_rank
  FROM document_chunks
  WHERE embedding <=> $1 < 0.68
     OR content @@ $2
  LIMIT 60
)
SELECT *,
  (1.0/(60+vec_rank)) + (1.0/(60+fts_rank)) AS rrf_score
FROM candidates
ORDER BY rrf_score DESC
LIMIT $3;
```

### Stratified (Window Function)
```sql
WITH ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY book_title
      ORDER BY embedding <=> $1
    ) AS rank_in_book
  FROM document_chunks
  WHERE embedding <=> $1 <= $2
     OR content @@ $3
)
SELECT * FROM ranked
WHERE rank_in_book <= $4
ORDER BY book_title, page_number;
```

### Entity Search
```sql
WITH entity_matches AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY book_title
      ORDER BY embedding <=> $1
    ) AS rank_in_book
  FROM document_chunks
  WHERE content @@ $2
     OR embedding <=> $1 <= $3
)
SELECT * FROM entity_matches
WHERE rank_in_book <= $4
ORDER BY book_title, page_number;
```

---

## 8. Key Configuration Constants

```python
# ingest.py
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
MAX_EMBED_WORKERS = 8

# retrieval.py
DYNAMIC_K_CONFIG = {
    "LISTING":      {"base": 21, "per_book": 3, "min": 15, "max": 50},
    "COMPARISON":   {"base": 20, "per_entity": 10, "min": 15, "max": 40},
    "TIMELINE":     {"base": 18, "per_book": 2, "min": 12, "max": 35},
    "THEMATIC":     {"base": 10, "per_book": 1, "min": 8, "max": 20},
    "FACTUAL":      {"base": 5, "per_book": 0, "min": 3, "max": 10},
}

# Distance thresholds
MAX_DISTANCE_STRATIFIED = 0.68
MAX_DISTANCE_ENTITY = 0.65
MAX_DISTANCE_HYBRID_CANDIDATE = 0.68
GAP_PRUNE_THRESHOLD = 0.15
GAP_PRUNE_MAX_DIST = 0.62
```

---

*Flow diagrams for Multi-Book RAG Assistant - Quick Reference*