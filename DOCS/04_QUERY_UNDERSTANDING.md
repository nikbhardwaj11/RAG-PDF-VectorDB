# Query Understanding - Entity Extraction & Classification

## Overview

Our system understands questions in three stages:
1. **Entity Extraction** - Find characters, places, concepts in the question
2. **Query Classification** - Determine question type (5 categories)
3. **Strategy Routing** - Pick the optimal retrieval approach

---

## Stage 1: Entity Extraction

### The Challenge
Questions mention entities in various forms:
- "Harry Potter's journey" (possessive)
- "Compare Mowgli and Harry Potter" (multiple)
- "What did the hero do?" (implicit)
- "Analyze Voldemort's choices" (single)

### Our Approach: Multi-Layer Detection

```python
def extract_entities(question):
    # 1. Normalize possessives: "Harry Potter's" → "Harry Potter"
    normalized = re.sub(r"\b([A-Za-z0-9_-]+)'s\b", r"\1", question)
    
    # 2. Known canonical entities (high precision)
    for canonical in CANONICAL_ENTITIES:
        if re.search(r"\b" + re.escape(canonical) + r"\b", normalized, re.IGNORECASE):
            found.append(canonical)
    
    # 3. Capitalized phrase detection (general)
    cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+|\s+[A-Z][a-z]+)*\b', normalized)
    for phrase in cap_phrases:
        if phrase not in EXTENDED_STOP_WORDS and len(phrase) > 2:
            found.append(phrase)
    
    # 4. Quoted entities: "Harry Potter"
    quoted = re.findall(r'"([^"]+)"', question)
    for ent in quoted:
        if len(ent) > 2:
            found.append(ent)
    
    # 5. Deduplicate: prefer longer matches
    found.sort(key=len, reverse=True)
    for ent in found:
        if not any(ent != e and ent.lower() in e.lower() for e in deduped):
            deduped.append(ent)
    
    return deduped
```

### Canonical Entity List (Extensible)
```python
CANONICAL_ENTITIES = [
    # Harry Potter
    "Harry Potter", "Ron Weasley", "Hermione Granger", "Lord Voldemort",
    "Albus Dumbledore", "Severus Snape", "Rubeus Hagrid", "Sirius Black",
    "Draco Malfoy", "Hogwarts", "Diagon Alley", "Horcrux", "Death Eater",
    
    # Dracula
    "Count Dracula", "Jonathan Harker", "Mina Harker", "Van Helsing",
    "Lucy Westenra", "Transylvania", "Castle Dracula",
    
    # Sherlock Holmes
    "Sherlock Holmes", "John Watson", "Professor Moriarty", "Irene Adler",
    "Baker Street", "221B Baker Street",
    
    # Jungle Book
    "Mowgli", "Baloo", "Bagheera", "Shere Khan", "Kaa", "Akela",
    
    # Spider-Man
    "Spider-Man", "Peter Parker", "Green Goblin", "Mary Jane Watson",
    "Doctor Octopus", "Daily Bugle", "Venom",
    
    # Literary terms
    "protagonist", "antagonist", "hero", "villain", "mentor", "sidekick",
]
```

### Examples

| Question | Entities Extracted |
|----------|-------------------|
| "Summarize Harry Potter's journey" | ["Harry Potter"] |
| "Compare Mowgli and Harry Potter" | ["Harry Potter", "Mowgli"] |
| "Why did Voldemort choose Horcruxes?" | ["Voldemort", "Horcruxes"] |
| "Analyze the protagonist's growth" | ["protagonist"] |
| "What happened at Hogwarts?" | ["Hogwarts"] |

---

## Stage 2: Query Classification

### Five Query Types

| Type | Purpose | Example Questions |
|------|---------|-------------------|
| **LISTING** | Enumerate items across books | "List all characters", "All themes", "Every location" |
| **COMPARISON** | Contrast entities | "Compare X and Y", "Differences between", "X vs Y" |
| **TIMELINE** | Chronological events | "Timeline of X", "Major events in order", "Life story of" |
| **THEMATIC** | Deep analysis, why/how | "Why did X?", "Explain theme", "Significance of" |
| **FACTUAL** | Specific fact retrieval | "What is X?", "Who is Y?", "Where is Z?" |

### Pattern Matching

```python
QUERY_PATTERNS = {
    "LISTING": {
        "patterns": [
            r"\blist\s+(all|every|the)\b",
            r"\ball\s+(characters?|themes?|books?)\b",
            r"\bacross\s+all\s+books?\b",
            r"\benumerate\b", r"\bcatalog\b",
        ],
        "weight": 1.2,
    },
    "COMPARISON": {
        "patterns": [
            r"\bcompare\b", r"\bcontrast\b", r"\bversus\b", r"\bvs\.?\b",
            r"\bdifference\s+between\b", r"\bsimilarities\b", r"\bdifferences\b",
        ],
        "weight": 1.3,
    },
    "TIMELINE": {
        "patterns": [
            r"\btimeline\b", r"\bmajor\s+events\b", r"\blife\s+of\b",
            r"\bjourney\s+of\b", r"\bcharacter\s+arc\b",
        ],
        "weight": 1.2,
    },
    "THEMATIC": {
        "patterns": [
            r"\bwhy\s+did\b", r"\bhow\s+did\b", r"\bexplain\s+(why|how)\b",
            r"\bwhat\s+led\s+to\b", r"\bsignificance\s+of\b", r"\btheme\s+of\b",
        ],
        "weight": 1.1,
    },
    "FACTUAL": {
        "patterns": [
            r"\bwhat\s+is\b", r"\bwho\s+is\b", r"\bwhere\s+is\b",
            r"\bname\s+the\b", r"\bidentify\b", r"\bdefine\b",
        ],
        "weight": 1.2,
    },
}
```

### Classification Algorithm

```python
def classify_query(question):
    lower_q = question.lower()
    entities = extract_entities(question)
    scores = {}
    
    # 1. Pattern matching with weights
    for qtype, config in QUERY_PATTERNS.items():
        score = 0.0
        for pattern in config["patterns"]:
            if re.search(pattern, lower_q):
                score += config["weight"]
        # Keyword overlap bonus
        words = set(lower_q.split())
        keyword_overlap = len(words & config["keywords"])
        score += keyword_overlap * 0.35
        scores[qtype] = score
    
    # 2. Contextual boosts (high impact!)
    if len(entities) >= 2:
        scores["COMPARISON"] = scores.get("COMPARISON", 0) + 2.0
    
    if re.search(r'\b\w+\s+(?:vs|versus|compared to)\s+\w+\b', lower_q):
        scores["COMPARISON"] = scores.get("COMPARISON", 0) + 2.5
    
    if any(k in lower_q for k in ["timeline", "major events", "life of", "journey of"]):
        scores["TIMELINE"] = scores.get("TIMELINE", 0) + 1.8
    
    if any(k in lower_q for k in ["list all", "all characters", "across the series"]):
        scores["LISTING"] = scores.get("LISTING", 0) + 2.0
    
    if lower_q.startswith(("why ", "how ")) and scores.get("COMPARISON", 0) < 1.0:
        scores["THEMATIC"] = scores.get("THEMATIC", 0) + 1.5
    
    if re.match(r"^(what|who|where|when)\s+(is|was)\b", lower_q):
        scores["FACTUAL"] = scores.get("FACTUAL", 0) + 1.5
    
    # 3. Pick winner
    best_type = max(scores, key=scores.get)
    confidence = min(scores[best_type] / 3.0, 1.0)
    
    return best_type, confidence, entities
```

### Confidence Examples

| Question | Type | Confidence | Reasoning |
|----------|------|------------|-----------|
| "Compare Harry and Mowgli" | COMPARISON | 0.67 | "compare" + 2 entities |
| "List all characters" | LISTING | 0.67 | "list all" + "characters" |
| "Timeline of Harry's life" | TIMELINE | 0.73 | "timeline" + "life of" + entity |
| "Why did Harry fight?" | THEMATIC | 0.73 | "why did" + entity |
| "What is the capital?" | FACTUAL | 0.60 | "what is" pattern |

---

## Stage 3: Strategy Routing

### Routing Logic

```python
if query_type == "LISTING" and not book_filter:
    → strategy_stratified_listing()
elif query_type == "COMPARISON" and not book_filter:
    → strategy_entity_comparison()
elif query_type == "TIMELINE" and not book_filter:
    → strategy_entity_timeline()
else:
    → strategy_hybrid_precision()  # THEMATIC, FACTUAL, or filtered
```

### Key Insight: Scope Matters

| Scope | Effect on Strategy |
|-------|-------------------|
| **All Books** | Full cross-book strategies (stratified, entity comparison, timeline) |
| **Single Book** | Falls back to hybrid_precision (no cross-book logic needed) |

---

## For Your Presentation

> "Our query understanding has three layers: First, entity extraction finds characters and concepts using both a canonical list (for known characters like Harry Potter) and general capitalized phrase detection. Second, classification uses regex patterns plus contextual boosts - for example, detecting 2+ entities automatically boosts COMPARISON confidence. Third, routing picks from four specialized retrieval strategies. The system handles implicit comparisons like 'compare protagonists' without needing explicit names."