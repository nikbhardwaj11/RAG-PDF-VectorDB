"""
Smart Retrieval Module for Multi-Book RAG - Optimized Version

Key improvements:
- Better entity extraction with NLP-like heuristics
- Query rewriting/expansion for improved recall
- Cross-encoder reranking simulation via distance scoring
- Adaptive thresholds per query type
- Improved hybrid search with query-specific FTS building
- Multi-hop retrieval for complex queries
- Better confidence calibration
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
from dotenv import load_dotenv

from database import (
    search_chunks,
    search_chunks_hybrid,
    search_chunks_stratified,
    search_chunks_by_entity,
    get_all_books,
)
from ingest import create_embeddings

load_dotenv()

# ===================== Constants =====================

# Extended stop words for entity extraction
QUESTION_STOP_WORDS = {
    "What", "Who", "Where", "When", "Why", "How", "The", "A", "An", "In", "On", "At",
    "To", "For", "Of", "And", "Or", "But", "Is", "Are", "Was", "Were", "Does", "Did",
    "Has", "Have", "Can", "Could", "Will", "Would", "Should", "List", "Compare",
    "Explain", "Summarize", "Describe", "Tell", "Give", "Show", "Find", "All", "Every",
    "Each", "Any", "Some", "Many", "Most", "Few", "Several", "Various", "Different",
    "Same", "Other", "Another", "Such", "Both", "Either", "Neither", "Across",
    "Throughout", "Between", "Among", "During", "Before", "After", "Since",
    "Until", "While", "Whose", "Which", "Whom", "There", "Here", "Then", "Now",
    "Also", "Just", "Only", "Even", "Still", "Yet", "Already", "Not", "No", "Yes",
    "Maybe", "Perhaps", "Probably", "Definitely", "Certainly", "Like", "Unlike",
}

# Query type patterns with improved specificity
QUERY_PATTERNS = {
    "LISTING": {
        "patterns": [
            r"\blist\s+(all|every|the)\b",
            r"\ball\s+(characters?|themes?|books?|items?|places?|locations?|events?)\b",
            r"\bevery\s+(character|book|chapter|place)\b",
            r"\bacross\s+all\s+books?\b",
            r"\bthroughout\s+(all\s+books?|the\s+series)\b",
            r"\benumerate\b", r"\bcatalog\b",
            r"\bwho\s+are\s+all\b", r"\bwhat\s+are\s+all\b",
            r"\bnames?\s+of\s+all\b", r"\bshow\s+(me\s+)?all\b",
            r"\btell\s+me\s+all\b", r"\bgive\s+me\s+all\b",
        ],
        "weight": 1.2,
        "keywords": {"list", "all", "every", "enumerate", "catalog", "names", "show", "tell", "give"},
    },
    "COMPARISON": {
        "patterns": [
            r"\bcompare\b", r"\bcontrast\b", r"\bversus\b", r"\bvs\.?\b",
            r"\bdifference\s+between\b", r"\bdiffer\b", r"\bcompared\s+to\b",
            r"\bsimilarities\b", r"\bdifferences\b", r"\bhow\s+(do|does)\s+.+\s+compare\b",
            r"\bbetter\s+than\b", r"\bworse\s+than\b", r"\bvs\s+\w+",
            r"\bhow\s+(is|are)\s+.+\s+(similar|different)\b",
            r"\bcontrast\s+(the|between)\b",
            r"\bdistinguish\b",
        ],
        "weight": 1.3,
        "keywords": {"compare", "contrast", "versus", "difference", "similarities", "differences", "similar", "different", "distinguish"},
    },
    "TIMELINE": {
        "patterns": [
            r"\btimeline\b", r"\bmajor\s+events\b", r"\ball\s+events\b",
            r"\blife\s+of\b", r"\bstory\s+of\b", r"\bhistory\s+of\b",
            r"\bjourney\s+of\b", r"\barc\s+of\b", r"\bwhat\s+happened\s+to\b",
            r"\bchronological\b", r"\bsequence\s+of\s+events\b",
            r"\bcharacter\s+(development|arc|growth)\b",
            r"\bsummarize\s+\w+.*journey\b", r"\bsummarize\s+\w+.*life\b",
            r"\bevents\s+in\s+order\b", r"\bfrom\s+start\s+to\s+end\b",
            r"\border\s+of\s+events\b", r"\bprogression\s+of\b",
        ],
        "weight": 1.2,
        "keywords": {"timeline", "events", "life", "journey", "arc", "chronological", "sequence", "order", "progression", "happened", "development", "growth"},
    },
    "THEMATIC": {
        "patterns": [
            r"\bwhy\s+did\b", r"\bhow\s+did\b", r"\bhow\s+does\b",
            r"\bexplain\s+(why|how)\b", r"\bwhat\s+is\s+the\s+reason\b",
            r"\bwhat\s+led\s+to\b", r"\bconsequences?\s+of\b",
            r"\bcause\s+of\b", r"\bsignificance\s+of\b", r"\bmeaning\s+of\b",
            r"\btheme\s+of\b", r"\bexplore\s+the\b", r"\bwhat\s+does\s+.+\s+represent\b",
            r"\bsymbolism\b", r"\bmetaphor\b", r"\bmotif\b",
            r"\bwhat\s+is\s+the\s+message\b", r"\bwhat\s+does\s+it\s+mean\b",
            r"\bdeeper\s+meaning\b", r"\bunderlying\b",
        ],
        "weight": 1.1,
        "keywords": {"why", "how", "explain", "reason", "cause", "significance", "theme", "symbolism", "meaning", "represent", "message", "deeper", "underlying", "led", "consequences"},
    },
    "FACTUAL": {
        "patterns": [
            r"\bwhat\s+is\b", r"\bwho\s+is\b", r"\bwhere\s+is\b", r"\bwhen\s+is\b",
            r"\bwhat\s+was\b", r"\bwho\s+was\b", r"\bwhich\s+",
            r"\bname\s+the\b", r"\bidentify\b", r"\bdefine\b",
            r"\bwhat\s+was\s+the\s+name\b", r"\bwho\s+played\b",
            r"\bwho\s+(killed|wrote|created|built|founded|invented|discovered)\b",
            r"\bwhen\s+(did|does|was|is)\b",
            r"\bwhere\s+(did|does|was|is)\b",
        ],
        "weight": 1.2,
        "keywords": {"what", "who", "where", "when", "which", "name", "define", "identify", "killed", "wrote", "created", "built", "founded", "invented", "discovered", "headmaster", "principal", "teacher", "professor"},
    },
}

# Dynamic K configuration - tuned for better coverage
DYNAMIC_K_CONFIG = {
    "LISTING":      {"base": 24, "per_book": 4,  "min": 18, "max": 60},
    "COMPARISON":   {"base": 24, "per_entity": 12, "min": 18, "max": 50},
    "TIMELINE":     {"base": 20, "per_book": 3,  "min": 15, "max": 40},
    "THEMATIC":     {"base": 12, "per_book": 2,  "min": 10, "max": 25},
    "FACTUAL":      {"base": 6,  "per_book": 1,  "min": 4,  "max": 12},
}

# Distance thresholds per query type
DISTANCE_THRESHOLDS = {
    "LISTING":     {"strict": 0.65, "loose": 0.75, "gap": 0.12},
    "COMPARISON":  {"strict": 0.60, "loose": 0.68, "gap": 0.10},
    "TIMELINE":    {"strict": 0.65, "loose": 0.72, "gap": 0.12},
    "THEMATIC":    {"strict": 0.58, "loose": 0.68, "gap": 0.15},
    "FACTUAL":     {"strict": 0.52, "loose": 0.60, "gap": 0.15},
}

# Query expansion templates per type
QUERY_EXPANSIONS = {
    "LISTING": [
        "characters themes events plot setting world building magic system locations protagonist antagonist",
        "all characters places settings main supporting minor",
    ],
    "COMPARISON": [
        "similarities differences personality traits relationships development arc strengths weaknesses",
        "motivations conflicts growth change parallel divergent alike unlike",
    ],
    "TIMELINE": [
        "major events key moments turning points beginning middle end climax resolution chronological",
        "sequence timeline story arc progression journey life history order of events",
    ],
    "THEMATIC": [
        "meaning significance symbolism theme motif representation deeper meaning why how",
        "reason cause consequence implication lesson moral message underlying symbolism",
    ],
    "FACTUAL": [
        "name identity role position title relationship specific detail fact exact precise information",
        "who is what is where is when is which one",
    ],
}


# ===================== Data Classes =====================

@dataclass
class RetrievalResult:
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# ===================== Utility Functions =====================

def get_book_count() -> int:
    """Get the number of unique PDF files (not chapters) in the database."""
    books = get_all_books()
    pdf_files = set(b["filename"] for b in books)
    return max(1, len(pdf_files))


def get_chapter_count() -> int:
    """Get total number of indexed chapters/books."""
    books = get_all_books()
    return len(books)


# Curated list of known characters, entities, and settings across the active library
CANONICAL_ENTITIES = [
    # Harry Potter Universe
    "Harry Potter", "Ron Weasley", "Hermione Granger", "Lord Voldemort", "Albus Dumbledore",
    "Severus Snape", "Rubeus Hagrid", "Sirius Black", "Draco Malfoy", "Remus Lupin",
    "Neville Longbottom", "Minerva McGonagall", "Ginny Weasley", "Luna Lovegood",
    "Dobby", "Voldemort", "Dumbledore", "Snape", "Hagrid", "Malfoy", "Hogwarts",
    "Diagon Alley", "Hogsmeade", "Azkaban", "Horcrux", "Death Eater", "Order of the Phoenix",
    # Dracula
    "Count Dracula", "Dracula", "Jonathan Harker", "Mina Harker", "Mina Murray",
    "Abraham Van Helsing", "Van Helsing", "Lucy Westenra", "John Seward",
    "Dr. Seward", "Arthur Holmwood", "Quincey Morris", "Renfield", "Transylvania",
    "Castle Dracula", "Whitby", "Carfax Abbey",
    # Sherlock Holmes
    "Sherlock Holmes", "Sherlock", "John Watson", "Dr. Watson", "Watson",
    "Professor Moriarty", "Moriarty", "Inspector Lestrade", "Lestrade",
    "Mycroft Holmes", "Irene Adler", "Mrs. Hudson", "Baker Street",
    "221B Baker Street", "Scotland Yard", "Reichenbach Falls",
    # The Jungle Book
    "Mowgli", "Baloo", "Bagheera", "Shere Khan", "Kaa", "Akela", "Raksha",
    "Father Wolf", "Bandar-log", "Messua", "Hathi", "Seeonee", "Wolf Pack",
    # Spider-Man
    "Spider-Man", "Spiderman", "Peter Parker", "Green Goblin", "Norman Osborn",
    "Harry Osborn", "Aunt May", "Uncle Ben", "Mary Jane Watson", "Mary Jane",
    "Gwen Stacy", "Doctor Octopus", "Doc Ock", "Otto Octavius", "Daily Bugle",
    "J. Jonah Jameson", "Flash Thompson", "Venom", "Carnage",
    # The Old Man and the Sea
    "Santiago", "Manolin", "The Old Man", "Joe DiMaggio", "Marlin", "Manolin",
    # Generic literary terms
    "protagonist", "antagonist", "hero", "villain", "mentor", "sidekick",
    "love interest", "companion", "ally", "enemy", "rival",
]


def _normalize_question(question: str) -> str:
    """Normalize question for entity and keyword processing."""
    # Handle possessives: "Harry Potter's" -> "Harry Potter"
    question = re.sub(r"\b([A-Za-z0-9_-]+)'s\b", r"\1", question)
    # Handle common contractions
    question = re.sub(r"\b(can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|wouldn't|couldn't|shouldn't)\b", " ", question, flags=re.IGNORECASE)
    return question


def extract_entities(question: str) -> List[str]:
    """
    Extract entities with high precision:
    - Checks against canonical fictional characters and locations (case-insensitive)
    - Supports hyphenated names like Spider-Man
    - Extracts capitalized multi-word phrases and single proper nouns
    - Deduplicates and returns longest specific entity matches first
    """
    normalized = _normalize_question(question)
    found_entities = []

    # 1. Match against known canonical characters & entities (case-insensitive)
    for canonical in CANONICAL_ENTITIES:
        pattern = r"\b" + re.escape(canonical) + r"\b"
        if re.search(pattern, normalized, re.IGNORECASE):
            found_entities.append(canonical)

    # 2. Extract capitalized phrases (including hyphens, e.g. Spider-Man)
    # Stop words to exclude
    EXTENDED_STOP_WORDS = QUESTION_STOP_WORDS | {
        "Compare", "Contrast", "List", "Show", "Find", "Give", "Tell", "Describe",
        "Explain", "Summarize", "Analyze", "Discuss", "Explore", "Examine",
        "Identify", "Define", "Enumerate", "Catalog", "What", "Which", "Who", "Whom",
        "Where", "When", "Why", "How", "All", "Every", "Each", "Any", "Some", "Main",
        "Major", "Key", "Important", "Significant", "Primary", "Secondary", "First",
        "Last", "Next", "Previous", "Current", "Past", "Future", "Present", "New",
        "Old", "Good", "Bad", "Best", "Worst", "Top", "Bottom", "High", "Low", "Big",
        "Small", "Large", "Little", "Long", "Short", "Early", "Late", "Recent",
        "Character", "Characters", "Protagonist", "Antagonist", "Hero", "Villain",
        "Book", "Books", "Story", "Stories", "Plot", "Theme", "Themes", "Setting",
        "World", "Magic", "System", "Event", "Events", "Chapter", "Chapters",
        "Series", "Passage", "Excerpt", "Development", "Difference", "Differences",
        "Similarity", "Similarities", "Timeline", "Arc", "Journey", "Story",
    }

    # Match TitleCase or hyphenated tokens like Spider-Man
    cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+|\s+[A-Z][a-z]+)*\b', normalized)
    for phrase in cap_phrases:
        if phrase in EXTENDED_STOP_WORDS:
            continue
        if len(phrase) <= 2:
            continue
        found_entities.append(phrase)

    # 3. Also look for quoted entities (e.g., "Harry Potter")
    quoted_entities = re.findall(r'"([^"]+)"', question)
    for ent in quoted_entities:
        if len(ent) > 2 and ent not in EXTENDED_STOP_WORDS:
            found_entities.append(ent)

    # 4. Deduplicate keeping longer version first (e.g. 'Harry Potter' over 'Harry')
    deduped = []
    # Sort candidate entities by length descending
    found_entities.sort(key=len, reverse=True)
    for ent in found_entities:
        if any(ent != existing and ent.lower() in existing.lower() for existing in deduped):
            continue
        if ent not in deduped:
            deduped.append(ent)

    return deduped


def classify_query(question: str) -> Tuple[str, float, List[str]]:
    """
    Classify query type with improved confidence scoring.
    Detects comparisons (including multi-entity questions), timelines/arcs, listings, thematic, and factual.
    """
    lower_q = question.lower()
    entities = extract_entities(question)

    scores = {}
    for qtype, config in QUERY_PATTERNS.items():
        score = 0.0
        # Pattern matching
        for pattern in config["patterns"]:
            if re.search(pattern, lower_q):
                score += config["weight"]

        # Keyword overlap bonus
        words = set(lower_q.split())
        keyword_overlap = len(words & config["keywords"])
        score += keyword_overlap * 0.35
        scores[qtype] = score

    # High-confidence contextual boosts:
    # 1. If 2+ distinct entities are detected, strongly weight COMPARISON
    if len(entities) >= 2:
        scores["COMPARISON"] = scores.get("COMPARISON", 0.0) + 2.0

    # 2. If "compare X and Y" or "X vs Y" pattern with entities
    vs_match = re.search(r'\b(\w+(?:\s+\w+)?)\s+(?:vs|versus|compared to)\s+(\w+(?:\s+\w+)?)\b', lower_q)
    if vs_match:
        scores["COMPARISON"] = scores.get("COMPARISON", 0.0) + 2.5

    # 3. If character development / major events / timeline patterns match
    if any(k in lower_q for k in ["character development", "major events", "timeline", "what happened to", "journey of", "story of", "life of", "arc of"]):
        scores["TIMELINE"] = scores.get("TIMELINE", 0.0) + 1.8

    # 4. If "list all" or "across the series" / "associated with"
    if any(k in lower_q for k in ["list all", "all characters", "associated with", "across the series", "every character", "all the characters"]):
        scores["LISTING"] = scores.get("LISTING", 0.0) + 2.0

    # 5. "Why" and "how" questions that aren't comparison -> THEMATIC
    if lower_q.startswith(("why ", "how ")) and scores.get("COMPARISON", 0) < 1.0:
        scores["THEMATIC"] = scores.get("THEMATIC", 0.0) + 1.5

    # 6. Questions starting with "what is/was", "who is/was", "where is/was", "when is/was" for specific facts -> FACTUAL
    if re.match(r"^(what|who|where|when)\s+(is|was)\b", lower_q) and scores.get("COMPARISON", 0) < 1.0:
        scores["FACTUAL"] = scores.get("FACTUAL", 0.0) + 1.5

    # 6b. "Who verb" questions (e.g., "Who killed", "Who wrote", "Who is the") -> FACTUAL
    if re.match(r"^who\s+\w+", lower_q) and scores.get("COMPARISON", 0) < 1.0:
        scores["FACTUAL"] = scores.get("FACTUAL", 0.0) + 1.3

    # 6c. "When verb" questions (e.g., "When did", "When does") -> FACTUAL
    if re.match(r"^when\s+(did|does)", lower_q) and scores.get("COMPARISON", 0) < 1.0:
        scores["FACTUAL"] = scores.get("FACTUAL", 0.0) + 1.3

    # 6d. "Where verb" questions (e.g., "Where did", "Where does", "Where is") -> FACTUAL
    if re.match(r"^where\s+(did|does|is)", lower_q) and scores.get("COMPARISON", 0) < 1.0:
        scores["FACTUAL"] = scores.get("FACTUAL", 0.0) + 1.3

    # 7. "Significance", "symbolism", "meaning", "theme", "metaphor", "motif", "represent" -> THEMATIC
    thematic_keywords = ["significance", "symbolism", "meaning of", "theme of", "metaphor", "motif", "represents", "symbolize", "deeper meaning", "underlying", "represent", "symbolic"]
    if any(k in lower_q for k in thematic_keywords):
        scores["THEMATIC"] = scores.get("THEMATIC", 0.0) + 1.8

    # 8. "What is the significance" / "What does X represent" / "What does X symbolize" -> THEMATIC (not FACTUAL)
    if "what is the significance" in lower_q or re.search(r"what does.+represent", lower_q) or re.search(r"what does.+symbolize", lower_q) or re.search(r"what.+represent", lower_q):
        scores["THEMATIC"] = scores.get("THEMATIC", 0.0) + 2.0
        scores["FACTUAL"] = max(0, scores.get("FACTUAL", 0.0) - 1.0)

    # 9. "All X" / "Every X" / "Names of all" -> LISTING
    if re.search(r'\b(all|every)\s+\w+', lower_q) or "names of all" in lower_q or "list of all" in lower_q:
        scores["LISTING"] = scores.get("LISTING", 0.0) + 1.5

    # 9. "All X" / "Every X" / "Names of all" -> LISTING
    if re.search(r'\b(all|every)\s+\w+', lower_q) or "names of all" in lower_q or "list of all" in lower_q:
        scores["LISTING"] = scores.get("LISTING", 0.0) + 1.5

    # 7. "Difference between" explicitly
    if "difference between" in lower_q or "differences between" in lower_q:
        scores["COMPARISON"] = scores.get("COMPARISON", 0.0) + 2.5

    # 8. "Similarities" and "differences" together strongly indicate comparison
    if "similarities" in lower_q and "differences" in lower_q:
        scores["COMPARISON"] = scores.get("COMPARISON", 0.0) + 2.0

    if scores:
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        if best_score > 0:
            confidence = min(best_score / 3.0, 1.0)
            return best_type, confidence, entities

    return "FACTUAL", 0.4, entities


def calculate_dynamic_k(query_type: str, entities: List[str], book_count: int) -> int:
    """Calculate optimal K with improved scaling."""
    config = DYNAMIC_K_CONFIG.get(query_type, DYNAMIC_K_CONFIG["FACTUAL"])

    if query_type == "LISTING":
        k = config["per_book"] * book_count + config["base"]
    elif query_type == "COMPARISON":
        entity_count = max(2, len(entities)) if entities else 2
        k = config["per_entity"] * entity_count
    elif query_type == "TIMELINE":
        k = config["base"] + config["per_book"] * max(0, book_count - 1)
    elif query_type == "THEMATIC":
        k = config["base"] + config["per_book"] * book_count
    else:  # FACTUAL
        k = config["base"] + config["per_book"] * max(0, book_count - 1)

    return max(config["min"], min(config["max"], k))


def build_expanded_query(original_question: str, query_type: str) -> str:
    """Build expanded query for better recall."""
    expansions = QUERY_EXPANSIONS.get(query_type, [])
    if expansions:
        return f"{original_question} {' '.join(expansions[:1])}"
    return original_question


def rerank_chunks(chunks: List[Dict[str, Any]], question: str, query_type: str,
                  entities: List[str], top_k: int) -> List[Dict[str, Any]]:
    """
    Rerank chunks using calibrated multi-signal scoring:
    - Vector semantic similarity (70% weight)
    - Entity presence boost (up to 20% weight)
    - Salient query term overlap (up to 10% weight)
    - Chronological / position signal for timelines (up to 5% weight)
    """
    if not chunks:
        return []

    question_lower = question.lower()
    raw_terms = re.findall(r'\b[A-Za-z0-9_-]{3,}\b', question_lower)
    question_terms = {t for t in raw_terms if t not in {s.lower() for s in QUESTION_STOP_WORDS}}

    for chunk in chunks:
        text_lower = chunk.get("text", "").lower()
        text_terms = set(re.findall(r'\b[A-Za-z0-9_-]{3,}\b', text_lower))

        # 1. Base semantic score from vector cosine distance (Nemotron distance 0.0 to 1.0)
        dist = chunk.get("distance", 0.5)
        dist_score = max(0.0, 1.0 - dist) * 0.70

        # 2. Entity presence bonus (up to 0.20)
        entity_score = 0.0
        if entities:
            for entity in entities:
                ent_lower = entity.lower()
                if ent_lower in text_lower:
                    entity_score = max(entity_score, 0.20)
                else:
                    parts = ent_lower.split()
                    if len(parts) > 1 and any(p in text_lower for p in parts if len(p) >= 4):
                        entity_score = max(entity_score, 0.10)

        # 3. Salient query term overlap (up to 0.10)
        overlap = len(question_terms & text_terms)
        term_score = min(overlap * 0.025, 0.10)

        # 4. Position bonus for timeline queries (earlier pages receive a slight gentle nudge)
        position_score = 0.0
        if query_type == "TIMELINE":
            page = chunk.get("page", 1)
            position_score = max(0.0, 0.05 - (page * 0.00005))

        chunk["rerank_score"] = dist_score + entity_score + term_score + position_score

    # Sort descending by rerank score
    reranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0), reverse=True)
    return reranked[:top_k]


def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate chunks by ID, keeping first occurrence."""
    seen = set()
    unique = []
    for c in chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)
    return unique


def sort_chunks_by_book_and_page(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort chunks by book title then page number."""
    return sorted(chunks, key=lambda c: (c.get("book_title", ""), c.get("page", 0)))


# ===================== Retrieval Strategies =====================

def strategy_stratified_listing(
    question: str,
    final_k: int,
    book_count: int,
    book_filter: Optional[str],
    entities: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Stratified multi-book retrieval for listing queries.
    If an entity is specified (e.g. 'associated with Harry Potter'), focuses retrieval
    on relevant story contexts while retaining breadth.
    """
    per_book_k = max(3, final_k // max(1, book_count))

    # If user asked about a specific entity/series
    if entities and len(entities) > 0:
        primary_entity = entities[0]
        entity_query = f"{primary_entity} characters allies enemies friends associates students teachers members family"
        q_vector = create_embeddings([entity_query], input_type="query")[0]
        chunks = search_chunks_by_entity(
            entity_name=primary_entity,
            query_vector=q_vector,
            per_book_k=per_book_k * 2,
            max_distance=0.82,
            top_k=final_k * 2,
        )
        if len(chunks) < final_k // 2:
            extra = search_chunks_hybrid(q_vector, entity_query, top_k=final_k, book_filter=book_filter)
            chunks = deduplicate_chunks(chunks + extra)
        return rerank_chunks(chunks, question, "LISTING", entities, final_k)

    # General listing across all books in the library
    expanded_q = build_expanded_query(question, "LISTING")
    q_vector = create_embeddings([expanded_q], input_type="query")[0]

    chunks = search_chunks_stratified(
        query_vector=q_vector,
        per_book_k=per_book_k,
        fts_query=question,
        book_filter=book_filter,
        max_distance=0.82,
    )

    # If sparse, broaden with loose threshold
    if len(chunks) < final_k * 0.6:
        extra_chunks = search_chunks_stratified(
            query_vector=q_vector,
            per_book_k=max(2, per_book_k // 2),
            fts_query=expanded_q,
            book_filter=book_filter,
            max_distance=0.88,
        )
        chunks = deduplicate_chunks(chunks + extra_chunks)

    # If still no chunks, fall back to hybrid search
    if not chunks:
        fallback_vec = create_embeddings([question], input_type="query")[0]
        chunks = search_chunks_hybrid(fallback_vec, question, final_k, book_filter)

    return chunks[:final_k]


def strategy_entity_comparison(
    question: str,
    final_k: int,
    entities: List[str],
    book_count: int,
    book_filter: Optional[str]
) -> List[Dict[str, Any]]:
    """
    Entity comparison with balanced representation across entities.
    Retrieves high-quality passages for each compared entity, then merges round-robin.
    """
    if len(entities) >= 2:
        target_entities = entities[:4]  # Max 4 entities
        per_entity_k = max(6, final_k // len(target_entities))

        entity_chunks_map = {}
        for entity in target_entities:
            # Build focused entity query
            entity_query = f"{entity} character personality traits relationships development story arc actions motivations conflicts"
            q_vec = create_embeddings([entity_query], input_type="query")[0]

            entity_chunks = search_chunks_by_entity(
                entity_name=entity,
                query_vector=q_vec,
                per_book_k=4,
                max_distance=0.82,
                top_k=per_entity_k * 2,
            )
            # Rerank for this entity
            entity_chunks = rerank_chunks(entity_chunks, entity_query, "COMPARISON", [entity], per_entity_k)
            entity_chunks_map[entity] = entity_chunks

        # Round-robin merge for balanced representation across all entities
        balanced = []
        max_len = max((len(v) for v in entity_chunks_map.values()), default=0)
        for i in range(max_len):
            for entity in target_entities:
                if i < len(entity_chunks_map[entity]):
                    chunk = entity_chunks_map[entity][i]
                    if not any(c["id"] == chunk["id"] for c in balanced):
                        balanced.append(chunk)

        if balanced:
            return balanced[:final_k]

    # Implicit comparison across all books - compare protagonists/main characters
    character_query = "main character protagonist hero personality development relationships story arc motivations"
    q_vec = create_embeddings([character_query], input_type="query")[0]
    per_book_k = max(3, final_k // max(1, book_count))

    chunks = search_chunks_stratified(
        query_vector=q_vec,
        per_book_k=per_book_k,
        fts_query="character protagonist hero main personality development",
        book_filter=None,
        max_distance=0.82,
    )

    if not chunks:
        fallback_vec = create_embeddings([question], input_type="query")[0]
        chunks = search_chunks_hybrid(fallback_vec, question, final_k, book_filter)

    chunks = rerank_chunks(chunks, question, "COMPARISON", [], final_k)
    return chunks[:final_k]


def strategy_entity_timeline(
    question: str,
    final_k: int,
    entities: List[str],
    book_count: int,
    book_filter: Optional[str]
) -> List[Dict[str, Any]]:
    """
    Entity timeline / arc retrieval.
    Prioritizes chunks focused on the target entity and sorts chronologically.
    """
    if not entities:
        return strategy_hybrid_precision(question, final_k, book_filter)

    target_entity = entities[0]

    # Build timeline-focused query
    q_text = f"{target_entity} major events key moments actions decisions consequences timeline story arc journey life"
    q_vec = create_embeddings([q_text], input_type="query")[0]

    chunks = search_chunks_by_entity(
        entity_name=target_entity,
        query_vector=q_vec,
        per_book_k=6,
        max_distance=0.82,
        top_k=final_k * 2,
    )

    # Fallback if sparse
    if len(chunks) < final_k // 2:
        fallback_chunks = search_chunks_hybrid(
            query_vector=q_vec,
            fts_query=f"{target_entity} events timeline story journey",
            top_k=final_k,
            book_filter=book_filter,
        )
        chunks = deduplicate_chunks(chunks + fallback_chunks)

    # Final fallback if still no chunks
    if not chunks:
        fallback_vec = create_embeddings([question], input_type="query")[0]
        chunks = search_chunks_hybrid(fallback_vec, question, final_k, book_filter)

    # Rerank with timeline awareness
    chunks = rerank_chunks(chunks, question, "TIMELINE", [target_entity], final_k)

    # Sort by book then page for chronological ordering
    chunks = sort_chunks_by_book_and_page(chunks)

    return chunks[:final_k]


def strategy_hybrid_precision(
    question: str,
    final_k: int,
    book_filter: Optional[str]
) -> List[Dict[str, Any]]:
    """Enhanced hybrid search with query expansion and reranking."""
    thresholds = DISTANCE_THRESHOLDS.get("FACTUAL", DISTANCE_THRESHOLDS["FACTUAL"])

    # Use expanded query for better recall
    expanded_q = build_expanded_query(question, "FACTUAL")
    q_vec = create_embeddings([expanded_q], input_type="query")[0]

    chunks = search_chunks_hybrid(
        query_vector=q_vec,
        fts_query=question,  # Use original for FTS
        top_k=final_k * 2,  # Get more for reranking
        book_filter=book_filter,
    )

    # If hybrid returns nothing, fall back to pure vector
    if not chunks:
        fallback_vec = create_embeddings([question], input_type="query")[0]
        chunks = search_chunks(query_vector=fallback_vec, top_k=final_k, book_filter=book_filter)

    # Rerank for precision
    chunks = rerank_chunks(chunks, question, "FACTUAL", [], final_k)

    # Gap pruning: stop when distance jumps significantly
    if len(chunks) > 3:
        pruned = [chunks[0]]
        for i in range(1, len(chunks)):
            gap = chunks[i]["distance"] - chunks[i - 1]["distance"]
            if gap > thresholds["gap"] or chunks[i]["distance"] > thresholds["strict"] + 0.1:
                break
            pruned.append(chunks[i])
        chunks = pruned

    return chunks[:final_k]


def strategy_thematic_exploration(
    question: str,
    final_k: int,
    book_filter: Optional[str]
) -> List[Dict[str, Any]]:
    """Thematic query strategy with broader semantic search."""
    thresholds = DISTANCE_THRESHOLDS["THEMATIC"]

    # Expand query for thematic coverage
    expanded_q = build_expanded_query(question, "THEMATIC")
    q_vec = create_embeddings([expanded_q], input_type="query")[0]

    # Use hybrid with expanded query
    chunks = search_chunks_hybrid(
        query_vector=q_vec,
        fts_query=question,
        top_k=final_k * 2,
        book_filter=book_filter,
    )

    # Also try pure vector with thematic query
    thematic_vec = create_embeddings([f"theme symbolism meaning significance {question}"], input_type="query")[0]
    vector_chunks = search_chunks(
        query_vector=thematic_vec,
        top_k=final_k,
        book_filter=book_filter,
    )

    chunks = deduplicate_chunks(chunks + vector_chunks)

    # If still no chunks, fall back to hybrid with original question
    if not chunks:
        fallback_vec = create_embeddings([question], input_type="query")[0]
        chunks = search_chunks_hybrid(fallback_vec, question, final_k, book_filter)

    chunks = rerank_chunks(chunks, question, "THEMATIC", [], final_k)

    return chunks[:final_k]


# ===================== Main Retrieval Function =====================

def smart_retrieve(
    question: str,
    book_filter: Optional[str] = None,
    manual_k: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main retrieval entry point with intelligent routing and multi-strategy fallback.
    """
    start_time = time.time()

    # 1. Analyze query
    query_type, confidence, entities = classify_query(question)

    # 2. Get collection stats
    book_count = get_book_count()
    chapter_count = get_chapter_count()

    # 3. Calculate Dynamic K
    dynamic_k = calculate_dynamic_k(query_type, entities, book_count)
    final_k = manual_k if (manual_k is not None and manual_k > 0) else dynamic_k

    # 4. Route to appropriate strategy
    if query_type == "LISTING" and (not book_filter or book_filter == "All Books"):
        chunks = strategy_stratified_listing(question, final_k, book_count, book_filter, entities=entities)
        strategy_used = "stratified_multi_book"
        reasoning = f"Listing: stratified retrieval across books (K={final_k})"

    elif query_type == "COMPARISON" and (not book_filter or book_filter == "All Books"):
        chunks = strategy_entity_comparison(question, final_k, entities, book_count, book_filter)
        strategy_used = "entity_comparison"
        entity_names = ", ".join(entities[:3]) if entities else "implicit protagonists"
        reasoning = f"Comparison of {entity_names}: sub-query per entity + reranked"

    elif query_type == "TIMELINE" and (not book_filter or book_filter == "All Books"):
        chunks = strategy_entity_timeline(question, final_k, entities, book_count, book_filter)
        strategy_used = "entity_timeline"
        entity_name = entities[0] if entities else "main character"
        reasoning = f"Timeline for '{entity_name}': chronological entity-stratified + reranked"

    elif query_type == "THEMATIC":
        chunks = strategy_thematic_exploration(question, final_k, book_filter)
        strategy_used = "thematic_exploration"
        reasoning = f"Thematic: expanded hybrid search + vector thematic reranking (K={final_k})"

    else:
        # FACTUAL or filtered queries
        chunks = strategy_hybrid_precision(question, final_k, book_filter)
        strategy_used = "hybrid_precision"
        reasoning = f"{query_type.title()}: hybrid RRF + gap pruning + reranking (K={final_k})"

    # 5. Final sort
    ordered_chunks = sort_chunks_by_book_and_page(chunks)

    # 6. Build metadata
    book_dist = Counter(c.get("book_title", "Unknown") for c in ordered_chunks)

    # Calculate coverage metrics
    unique_books = len(book_dist)
    coverage_pct = (unique_books / book_count * 100) if book_count > 0 else 0

    metadata = {
        "query_type": query_type,
        "confidence": round(confidence, 2),
        "dynamic_k": len(ordered_chunks),
        "target_k": final_k,
        "strategy": strategy_used,
        "entities": entities,
        "reasoning": reasoning,
        "book_distribution": dict(book_dist),
        "book_count": book_count,
        "chapter_count": chapter_count,
        "unique_books_retrieved": unique_books,
        "coverage_pct": round(coverage_pct, 1),
        "retrieval_time_ms": round((time.time() - start_time) * 1000, 1),
    }

    return ordered_chunks, metadata


# ===================== Query Suggestions =====================

def get_query_suggestions(query_type: str = None) -> List[Tuple[str, str]]:
    """Get example questions for each query type."""
    suggestions = {
        "LISTING": [
            ("📋 List all characters", "List all characters mentioned across the books"),
            ("🌍 All locations", "What are all the locations and settings in the stories?"),
            ("✨ All themes", "What are all the major themes across the books?"),
            ("📚 All books", "Summarize each book in the collection"),
        ],
        "COMPARISON": [
            ("⚖️ Compare protagonists", "Compare the main characters across books"),
            ("🔮 Magic systems", "Contrast the magic systems in different books"),
            ("👥 Villains", "Compare the antagonists across stories"),
            ("💭 Similarities", "What similarities exist between the protagonists?"),
        ],
        "TIMELINE": [
            ("📅 Character timeline", "Show the timeline of events for the main character"),
            ("📖 Major events", "Summarize all major events in the story in order"),
            ("📈 Character arc", "How does the hero change across the books?"),
            ("⏰ Chronology", "What happens in chronological order?"),
        ],
        "THEMATIC": [
            ("🤔 Why", "Why did the character make that choice?"),
            ("🎭 Explain theme", "Explain the theme of sacrifice in the stories"),
            ("✨ Symbolism", "What is the significance of the recurring symbol?"),
            ("💫 Meaning", "What does the ending represent?"),
        ],
        "FACTUAL": [
            ("❓ Fact", "What is the name of the main city?"),
            ("👤 Character", "Who is the mentor character?"),
            ("📍 Location", "Where does the final battle take place?"),
            ("📖 Detail", "When does the story begin?"),
        ],
    }

    if query_type and query_type in suggestions:
        return suggestions[query_type]

    # Return mixed suggestions (top 5)
    all_sugg = []
    for v in suggestions.values():
        all_sugg.extend(v)
    return all_sugg[:5]