# Extending the System & Best Practices

## Easy Customizations (No Code Architecture Changes)

### 1. Chunking Parameters (`ingest.py`)
```python
CHUNK_SIZE = 1200      # Characters per chunk
CHUNK_OVERLAP = 300    # Overlap between chunks
MAX_EMBED_WORKERS = 8  # Parallel embedding requests
```

| Use Case | CHUNK_SIZE | CHUNK_OVERLAP |
|----------|------------|---------------|
| **Dense technical docs** | 800 | 200 |
| **Narrative fiction (default)** | 1200 | 300 |
| **Long-form articles** | 1500 | 400 |
| **Short Q&A pairs** | 500 | 100 |

**Trade-offs**:
- Smaller chunks → more precise, more chunks, higher embedding cost
- Larger chunks → more context, fewer chunks, risk of dilution
- More overlap → better boundary continuity, more chunks

---

### 2. Dynamic K Values (`retrieval.py`)
```python
DYNAMIC_K_CONFIG = {
    "LISTING":      {"base": 21, "per_book": 3,  "min": 15, "max": 50},
    "COMPARISON":   {"base": 20, "per_entity": 10, "min": 15, "max": 40},
    "TIMELINE":     {"base": 18, "per_book": 2,  "min": 12, "max": 35},
    "THEMATIC":     {"base": 10, "per_book": 1,  "min": 8,  "max": 20},
    "FACTUAL":      {"base": 5,  "per_book": 0,  "min": 3,  "max": 10},
}
```

**Adjust for**:
- More books → increase `per_book` values
- Longer context window LLM → increase `max` values
- Cost-sensitive → decrease `max` values

---

### 3. Distance Thresholds (`database.py` / `retrieval.py`)
```python
# In search_chunks_stratified
max_distance=0.68        # Stricter = fewer, more relevant

# In search_chunks_hybrid candidate filter
WHERE (embedding <=> $qv) < 0.68

# In hybrid precision gap pruning
GAP_PRUNE_THRESHOLD = 0.15    # Stop when distance jumps this much
GAP_PRUNE_MAX_DIST = 0.62     # Absolute cutoff
```

**Nemotron 3.5 Embed distance guide**:
- < 0.35: Very high relevance
- 0.35-0.50: High relevance  
- 0.50-0.65: Moderate relevance
- 0.65-0.75: Low relevance
- > 0.75: Likely irrelevant

---

### 4. Query Patterns (`retrieval.py`)
Add new patterns to `QUERY_PATTERNS`:

```python
QUERY_PATTERNS = {
    "CHARACTER_ANALYSIS": {  # NEW TYPE
        "patterns": [
            r"\banalyze\s+\w+\s+character\b",
            r"\bcharacter\s+analysis\b",
            r"\bpsychology\s+of\b",
        ],
        "weight": 1.2,
        "keywords": {"analyze", "character", "psychology", "personality", "motivation"},
    },
    # ... existing types
}
```

Then add strategy routing in `smart_retrieve()`.

---

## Advanced Extensions

### 1. Cross-Encoder Reranking (Quality Boost)
```python
# After initial retrieval, rerank with cross-encoder
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_with_cross_encoder(query, chunks, top_k=10):
    pairs = [(query, c["text"]) for c in chunks]
    scores = cross_encoder.predict(pairs)
    
    for chunk, score in zip(chunks, scores):
        chunk["cross_encoder_score"] = float(score)
    
    return sorted(chunks, key=lambda c: c["cross_encoder_score"], reverse=True)[:top_k]
```

**Integration point**: After `rerank_chunks()` in each strategy

---

### 2. Query Rewriting (Recall Boost)
```python
def rewrite_query_for_search(question, query_type):
    """LLM rewrites question for better retrieval"""
    rewrite_prompts = {
        "COMPARISON": "Rewrite as a search query to find passages comparing {entities}",
        "TIMELINE": "Rewrite as a search query for chronological events about {entity}",
        "THEMATIC": "Rewrite as a search query for thematic analysis of {topic}",
    }
    # Call LLM, return optimized query
    # Use rewritten query for embedding + FTS
```

---

### 3. Multi-Hop Retrieval (Complex Reasoning)
```python
def multi_hop_retrieve(question, max_hops=3):
    """Iterative retrieval for complex questions"""
    all_chunks = []
    current_question = question
    
    for hop in range(max_hops):
        chunks, meta = smart_retrieve(current_question)
        all_chunks.extend(chunks)
        
        # LLM decides if more info needed
        next_question = llm_decide_next_hop(question, chunks)
        if not next_question:
            break
        current_question = next_question
    
    return deduplicate_chunks(all_chunks)
```

---

### 4. Knowledge Graph Integration
```python
# Extract entities + relations during ingestion
def extract_knowledge_graph(chunk_text):
    """Returns (entities, relations) from text"""
    # Use NER + relation extraction
    # Store in separate KG table
    pass

# Query-time: traverse graph for multi-hop
def kg_enhanced_retrieve(question):
    entities = extract_entities(question)
    related = traverse_knowledge_graph(entities, depth=2)
    # Boost chunks mentioning related entities
    pass
```

---

### 5. Evaluation Framework
```python
# Add test cases with expected behavior
EVAL_CASES = [
    {
        "question": "Compare Harry Potter and Mowgli",
        "expected_type": "COMPARISON",
        "expected_entities": ["Harry Potter", "Mowgli"],
        "min_chunks_per_entity": 5,
        "must_cite_both_books": True,
    },
    {
        "question": "List all characters",
        "expected_type": "LISTING",
        "min_books_represented": 2,
        "min_total_chunks": 15,
    },
]

def run_evaluation():
    results = []
    for case in EVAL_CASES:
        chunks, meta = smart_retrieve(case["question"])
        # Check type, entities, coverage, citations
        passed = evaluate_case(case, chunks, meta)
        results.append({"case": case, "passed": passed})
    return results
```

---

## Best Practices

### 1. Ingestion Best Practices

| Practice | Why |
|----------|-----|
| **Use compilation PDFs with bookmarks** | Automatic multi-book indexing |
| **Clean PDF text before ingestion** | Better embeddings, less noise |
| **Monitor chunk quality** | Check `chunk_pages()` output |
| **Batch embeddings** | 8x faster than sequential |
| **Idempotent runs** | Safe to re-run anytime |

### 2. Retrieval Best Practices

| Practice | Why |
|----------|-----|
| **Use Dynamic K (default)** | Optimal for each query type |
| **Filter scope when known** | Faster, more precise |
| **Monitor confidence scores** | Low confidence = hallucination risk |
| **Check book distribution** | Ensure cross-book coverage |

### 3. UI/UX Best Practices

| Practice | Why |
|----------|-----|
| **Show retrieval metadata** | Builds trust, enables debugging |
| **Group sources by book** | Natural reading order |
| **Stream answers** | Perceived speed, engagement |
| **Color-code confidence** | Instant quality signal |

### 4. Production Deployment

```bash
# Docker Compose for production
version: '3.8'
services:
  pgvector:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: ragdb
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ${PGPASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  app:
    build: .
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
      - PGHOST=pgvector
    ports:
      - "8501:8501"
    depends_on:
      - pgvector

volumes:
  pgdata:
```

---

## Monitoring & Observability

### Key Metrics to Track
```python
# In smart_retrieve() metadata
{
    "retrieval_time_ms": 145,        # Latency
    "dynamic_k": 20,                  # Chunks retrieved
    "target_k": 20,                   # Chunks requested
    "unique_books_retrieved": 2,      # Cross-book coverage
    "coverage_pct": 100.0,            # % of books represented
    "confidence": 0.67,               # Classification confidence
    "strategy": "entity_comparison",  # Strategy used
}
```

### Alerting Thresholds
| Metric | Warning | Critical |
|--------|---------|----------|
| Retrieval latency | >500ms | >2s |
| Confidence | <0.4 | <0.2 |
| Coverage % | <50% | <25% |
| Chunks retrieved | < target_k * 0.5 | < 3 |

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| **One book dominates results** | No stratified search | Use LISTING type or check strategy routing |
| **Low recall for comparisons** | Entities not detected | Add to CANONICAL_ENTITIES or improve extraction |
| **Slow ingestion** | Sequential embeddings | Increase MAX_EMBED_WORKERS, check API limits |
| **Hallucinated answers** | Low confidence retrieval | Increase Dynamic K, check distance thresholds |
| **Missing cross-book results** | Scope filter applied | Check sidebar "All Books" selected |

---

## For Your Presentation

> "The system is designed for extension. Easy knobs include chunk size, Dynamic K values, distance thresholds, and query patterns - all in constants at the top of files. Advanced extensions like cross-encoder reranking, query rewriting, multi-hop retrieval, and knowledge graphs integrate at well-defined points in the pipeline. We've also built in evaluation cases and observability metadata for production monitoring."