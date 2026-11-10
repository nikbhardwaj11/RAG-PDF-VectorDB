# Vector Databases & Embeddings - Technical Deep Dive

## What are Embeddings?

Embeddings convert text into high-dimensional vectors where **semantic similarity = geometric proximity**.

```
Text                    →  2048-dimensional vector
─────────────────────────────────────────────────
"Harry Potter wizard"   →  [0.12, -0.05, 0.87, 0.23, ...]
"Hermione casts spell"  →  [0.11, -0.04, 0.85, 0.25, ...]  ← CLOSE (similar meaning)
"Car engine repair"     →  [0.92, 0.31, -0.12, -0.45, ...]  ← FAR (different topic)
```

**Distance Metric**: Cosine similarity (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)

---

## Our Embedding Pipeline

### Model: NVIDIA Nemotron 3.5 Embed 1B
- **Dimensions**: 2048
- **Input types**: `passage` (documents) / `query` (questions)
- **Throughput**: ~8 parallel requests via ThreadPoolExecutor
- **Retry logic**: Exponential backoff (1s, 2s, 3s, 4s)

### Ingestion Flow
```python
# 1. Chunk text (1200 chars, 300 overlap)
chunks = chunk_pages(pages)  # → [{"text": "...", "page": 5}, ...]

# 2. Generate embeddings in parallel
vectors = create_embeddings(
    texts=[c["text"] for c in chunks],
    input_type="passage",  # Document mode
    progress_callback=update_ui
)

# 3. Batch insert to PostgreSQL
records = [(source_id, filename, book_title, page, idx, text, vec) 
           for idx, (text, vec) in enumerate(zip(chunks, vectors))]
insert_chunks(records)
```

---

## PostgreSQL + pgvector Setup

### Table Schema
```sql
CREATE TABLE document_chunks (
    id              BIGSERIAL PRIMARY KEY,
    source_id       VARCHAR(64),       -- filename_start_end
    filename        VARCHAR(255),      -- original PDF
    book_title      VARCHAR(255),      -- logical book name
    page_number     INTEGER,           -- page in PDF
    chunk_index     INTEGER,           -- order in book
    content         TEXT,              -- chunk text
    embedding       vector(2048),      -- Nemotron embedding
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

### Critical Indexes

| Index | Type | Purpose |
|-------|------|---------|
| `idx_doc_chunks_book_title` | B-tree | Filter by book |
| `idx_doc_chunks_filename` | B-tree | Filter by PDF |
| `idx_doc_chunks_fts` | GIN | Full-text search |
| `idx_doc_chunks_halfvec` | HNSW | Vector similarity (ANN) |

### HNSW Index (Approximate Nearest Neighbor)
```sql
CREATE INDEX idx_doc_chunks_halfvec 
ON document_chunks 
USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Parameters**:
- `m=16`: Graph connectivity (higher = better recall, more memory)
- `ef_construction=64`: Build-time search width
- `halfvec`: 16-bit floats (half memory, minimal quality loss)

---

## Search Strategies

### 1. Pure Vector Search
```sql
SELECT *, (embedding <=> $query_vec) AS distance
FROM document_chunks
WHERE book_title ILIKE '%filter%'
ORDER BY embedding <=> $query_vec
LIMIT $top_k;
```
- **Use case**: Semantic similarity only
- **Speed**: ~10-50ms with HNSW

### 2. Full-Text Search (PostgreSQL FTS)
```sql
SELECT *, ts_rank_cd(to_tsvector('english', content), 
       websearch_to_tsquery('english', $fts_query)) AS rank
FROM document_chunks
WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', $fts_query)
ORDER BY rank DESC
LIMIT $top_k;
```
- **Use case**: Exact keyword matching
- **Features**: Stemming, ranking, OR/AND/NOT operators

### 3. Hybrid Search (RRF - Reciprocal Rank Fusion)
```sql
WITH candidates AS (
  SELECT *,
    ROW_NUMBER() OVER (ORDER BY embedding <=> $qv) AS vec_rank,
    ROW_NUMBER() OVER (ORDER BY ts_rank(...) DESC) AS fts_rank
  FROM document_chunks
  WHERE embedding <=> $qv < 0.68 OR content @@ $fts_q
  LIMIT 60
)
SELECT *, (1.0/(60+vec_rank)) + (1.0/(60+fts_rank)) AS rrf_score
FROM candidates
ORDER BY rrf_score DESC LIMIT $top_k;
```

**Why RRF?**
- No score normalization needed
- Robust to different score distributions
- Combines semantic + keyword strengths

### 4. Stratified Search (Window Functions)
```sql
WITH ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY book_title 
      ORDER BY embedding <=> $qv
    ) AS rank_in_book
  FROM document_chunks
  WHERE embedding <=> $qv <= 0.68 OR content @@ $fts_q
)
SELECT * FROM ranked 
WHERE rank_in_book <= $per_book_k
ORDER BY book_title, page_number;
```

**Why Stratified?**
- Guarantees representation from each book
- Prevents large books from dominating
- Essential for "list all" queries

---

## Configuration Constants

```python
# database.py
DEFAULT_CANDIDATE_MULTIPLIER = 4    # Hybrid: retrieve 4× final K for RRF
MAX_CANDIDATES = 200                # Cap for candidate pool
RRF_CONSTANT = 60                   # RRF parameter (k in 1/(k+rank))

# Distance thresholds
MAX_DISTANCE_STRATIFIED = 0.68      # Stratified relevance cutoff
MAX_DISTANCE_ENTITY = 0.65          # Entity search cutoff
MAX_DISTANCE_HYBRID_CANDIDATE = 0.68 # Hybrid candidate cutoff
GAP_PRUNE_THRESHOLD = 0.15          # Gap pruning trigger
GAP_PRUNE_MAX_DIST = 0.62           # Absolute distance cap for facts
```

---

## Performance Characteristics

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| Single embedding | ~200ms | Parallel batch reduces per-chunk |
| Vector search (HNSW) | 10-50ms | Sub-linear scaling |
| FTS search | 5-20ms | GIN index |
| Hybrid RRF | 30-100ms | Two searches + fusion |
| Stratified | 50-150ms | Window function overhead |

---

## For Your Presentation

> "We use PostgreSQL with pgvector - it's production-grade SQL database with native vector support. Our embeddings are 2048-dimensional vectors from NVIDIA's Nemotron model. We combine three search methods: semantic vector search, keyword full-text search, and a hybrid fusion. The key innovation is stratified retrieval using PostgreSQL window functions - this guarantees balanced results across multiple books instead of letting the biggest book win."