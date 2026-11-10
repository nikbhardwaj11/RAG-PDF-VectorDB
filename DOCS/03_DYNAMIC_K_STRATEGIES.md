# Dynamic K & Retrieval Strategies - The Intelligence Layer

## The Core Insight

**Not all questions are equal.** A factual question needs precision (few, highly relevant chunks). A listing question needs breadth (many chunks across books). Our system automatically detects question type and adapts.

---

## Dynamic K Configuration

```python
DYNAMIC_K_CONFIG = {
    "LISTING":      {"base": 21, "per_book": 3,  "min": 15, "max": 50},
    "COMPARISON":   {"base": 20, "per_entity": 10, "min": 15, "max": 40},
    "TIMELINE":     {"base": 18, "per_book": 2,  "min": 12, "max": 35},
    "THEMATIC":     {"base": 10, "per_book": 1,  "min": 8,  "max": 20},
    "FACTUAL":      {"base": 5,  "per_book": 0,  "min": 3,  "max": 10},
}
```

### Formula Examples (2 books indexed)

| Query Type | Formula | Calculation | Final K |
|------------|---------|-------------|---------|
| **FACTUAL** | `base` | 5 | **5** |
| **THEMATIC** | `base + per_book × books` | 10 + 1×2 | **12** |
| **TIMELINE** | `base + per_book × (books-1)` | 18 + 2×1 | **20** |
| **COMPARISON** | `per_entity × entities` | 10 × 2 | **20** |
| **LISTING** | `per_book × books` (clamped to min) | 3×2=6 → clamp 15 | **15** |

### With 3 Books, 2 Entities

| Query Type | Final K |
|------------|---------|
| FACTUAL | 5 |
| THEMATIC | 13 |
| TIMELINE | 22 |
| COMPARISON | 20 |
| LISTING | 15 (min clamp) |

---

## Four Retrieval Strategies

### Strategy 1: Stratified Multi-Book (LISTING)
**Trigger**: `query_type == "LISTING"` AND `scope == "All Books"`

**Examples**: "List all characters", "All themes across books", "Enumerate magical creatures"

**Algorithm**:
```sql
ROW_NUMBER() OVER (PARTITION BY book_title ORDER BY similarity)
```
→ Gets top-K chunks **per book**, then combines

**Result**: 10 chunks from Book A + 10 chunks from Book B = balanced!

**Why**: Prevents one large book from dominating "list everything" answers

---

### Strategy 2: Entity Comparison (COMPARISON)
**Trigger**: `query_type == "COMPARISON"` AND `scope == "All Books"`

**Examples**: "Compare Harry and Mowgli", "Differences between protagonists", "Villains across stories"

**Algorithm**:
```python
# For EACH detected entity:
for entity in entities:  # e.g., ["Harry Potter", "Mowgli"]
    entity_query = f"{entity} character personality development relationships"
    q_vec = embed(entity_query)
    entity_chunks = search_chunks_by_entity(entity, q_vec, per_book_k)
    entity_chunks = rerank_for_entity(entity_chunks, entity)

# Round-robin merge for balanced representation
balanced = []
for i in range(max_chunks):
    for entity in entities:
        if i < len(entity_chunks[entity]):
            balanced.append(entity_chunks[entity][i])

return balanced[:final_k]
```

**Result**: Alternating chunks [Harry-1, Mowgli-1, Harry-2, Mowgli-2...]

**Why**: "Compare" needs deep, equal coverage of each entity

---

### Strategy 3: Entity Timeline (TIMELINE)
**Trigger**: `query_type == "TIMELINE"` AND `scope == "All Books"`

**Examples**: "Timeline of Harry's life", "Major events in order", "Character arc of Mowgli"

**Algorithm**:
```python
target_entity = entities[0]  # Primary entity

# Build timeline-focused query
q_text = f"{target_entity} major events actions story key moments timeline journey"
q_vec = embed(q_text)

chunks = search_chunks_by_entity(target_entity, q_vec, per_book_k)

# Sort chronologically
chunks = sort_chunks_by_book_and_page(chunks)  # book_title, page_number
```

**Result**: Chronological passages showing character arc across books

**Why**: Timelines need temporal ordering, not just relevance

---

### Strategy 4: Hybrid Precision (THEMATIC/FACTUAL)
**Trigger**: Everything else (THEMATIC, FACTUAL, or filtered scope)

**Examples**: "Why did Harry fight?", "What is the capital?", "Explain the magic system"

**Algorithm**:
```python
# 1. Hybrid RRF search (vector + FTS)
chunks = search_chunks_hybrid(q_vec, fts_query, top_k=final_k*2)

# 2. Multi-signal reranking
for chunk in chunks:
    dist_score = (1 - distance) * 0.70          # Semantic similarity
    entity_score = 0.20 if entity_in_chunk else 0  # Entity presence
    term_score = min(overlap * 0.025, 0.10)      # Keyword overlap
    position_score = timeline_bonus if TIMELINE  # Page position
    chunk.rerank_score = dist_score + entity_score + term_score + position_score

# 3. Gap pruning (FACTUAL only)
# Stop when distance jumps significantly
pruned = [chunks[0]]
for i in range(1, len(chunks)):
    gap = chunks[i].distance - chunks[i-1].distance
    if gap > 0.15 or chunks[i].distance > 0.62:
        break
    pruned.append(chunks[i])
```

**Result**: High-precision, focused chunks with quality filtering

**Why**: Fact/thematic queries need precision, not breadth

---

## Strategy Routing Logic

```python
def smart_retrieve(question, book_filter, manual_k):
    # 1. Analyze
    query_type, confidence, entities = classify_query(question)
    book_count = get_book_count()
    
    # 2. Calculate K
    dynamic_k = calculate_dynamic_k(query_type, entities, book_count)
    final_k = manual_k or dynamic_k
    
    # 3. Route to strategy
    if query_type == "LISTING" and not book_filter:
        chunks = strategy_stratified_listing(...)
        strategy = "stratified_multi_book"
    elif query_type == "COMPARISON" and not book_filter:
        chunks = strategy_entity_comparison(...)
        strategy = "entity_comparison"
    elif query_type == "TIMELINE" and not book_filter:
        chunks = strategy_entity_timeline(...)
        strategy = "entity_timeline"
    else:
        chunks = strategy_hybrid_precision(...)
        strategy = "hybrid_precision"
    
    # 4. Post-process
    chunks = deduplicate_chunks(chunks)
    chunks = sort_chunks_by_book_and_page(chunks)
    
    # 5. Metadata
    metadata = {
        "query_type": query_type,
        "strategy": strategy,
        "dynamic_k": len(chunks),
        "target_k": final_k,
        "entities": entities,
        "confidence": confidence,
        "book_distribution": Counter(c.book_title for c in chunks),
        ...
    }
    
    return chunks, metadata
```

---

## Query Type Detection

### Pattern-Based Classification

```python
QUERY_PATTERNS = {
    "LISTING": {
        "patterns": [
            r"\blist\s+(all|every|the)\b",
            r"\ball\s+(characters?|themes?|books?)\b",
            r"\bacross\s+all\s+books?\b",
            r"\benumerate\b", r"\bcatalog\b",
        ],
        "keywords": {"list", "all", "every", "enumerate", "catalog"},
    },
    "COMPARISON": {
        "patterns": [
            r"\bcompare\b", r"\bcontrast\b", r"\bversus\b", r"\bvs\.?\b",
            r"\bdifference\s+between\b", r"\bsimilarities\b", r"\bdifferences\b",
        ],
        "keywords": {"compare", "contrast", "versus", "difference", "similarities"},
    },
    "TIMELINE": {
        "patterns": [
            r"\btimeline\b", r"\bmajor\s+events\b", r"\blife\s+of\b",
            r"\bjourney\s+of\b", r"\bcharacter\s+(development|arc|growth)\b",
        ],
        "keywords": {"timeline", "events", "life", "journey", "arc", "chronological"},
    },
    "THEMATIC": {
        "patterns": [
            r"\bwhy\s+did\b", r"\bhow\s+did\b", r"\bexplain\s+(why|how)\b",
            r"\bwhat\s+led\s+to\b", r"\bsignificance\s+of\b", r"\btheme\s+of\b",
        ],
        "keywords": {"why", "how", "explain", "reason", "cause", "theme", "symbolism"},
    },
    "FACTUAL": {
        "patterns": [
            r"\bwhat\s+is\b", r"\bwho\s+is\b", r"\bwhere\s+is\b", r"\bwhen\s+is\b",
            r"\bname\s+the\b", r"\bidentify\b", r"\bdefine\b",
        ],
        "keywords": {"what", "who", "where", "when", "which", "name", "define"},
    },
}
```

### Confidence Calculation
```python
# Each pattern match adds weight (1.0)
# Max ~3-4 patterns per type
confidence = min(total_score / 3.0, 1.0)

# Contextual boosts:
# - 2+ entities → +2.0 to COMPARISON
# - "X vs Y" pattern → +2.5 to COMPARISON  
# - "timeline"/"journey" → +1.8 to TIMELINE
# - "list all" → +2.0 to LISTING
# - "significance"/"symbolism" → +1.8 to THEMATIC
# - "who/what/where/when is" → +1.5 to FACTUAL
```

---

## For Your Presentation

> "Dynamic K is our adaptive retrieval system. Instead of always retrieving 10 chunks, we analyze the question type and adjust: factual questions get 5 chunks for speed and precision, while 'list all characters' gets 50 chunks for comprehensive coverage. We have four specialized strategies - stratified for listings, entity-decomposed for comparisons, chronological for timelines, and hybrid precision for facts. The system automatically routes each question to the right strategy."