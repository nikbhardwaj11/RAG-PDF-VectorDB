import os
import re
from typing import List, Dict, Any, Optional

import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()

# ===================== Configuration =====================
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDATABASE = os.getenv("PGDATABASE", "ragdb")
PGUSER = os.getenv("PGUSER", "raguser")
PGPASSWORD = os.getenv("PGPASSWORD", "ragpassword")

# Search configuration
DEFAULT_CANDIDATE_MULTIPLIER = 4
MAX_CANDIDATES = 400
RRF_CONSTANT = 60

# Stop words to filter out when generating keyword-based FTS search queries
COMMON_STOP_WORDS = {
    "what", "who", "where", "when", "why", "how", "the", "a", "an", "in", "on", "at",
    "to", "for", "of", "and", "or", "but", "is", "are", "was", "were", "does", "did",
    "has", "have", "can", "could", "will", "would", "should", "list", "compare",
    "explain", "summarize", "describe", "tell", "give", "show", "find", "all", "every",
    "across", "between", "each", "any", "some", "with", "from", "into", "about",
    "mention", "mentioned", "main", "major", "different", "various"
}


def _build_fts_query(text: str) -> str:
    """
    Extract meaningful search keywords joined with OR for PostgreSQL websearch_to_tsquery.
    Avoids zero-result drops caused by strict boolean AND queries on natural language questions.
    """
    cleaned = re.sub(r"[^\w\s-]", " ", text or "")
    tokens = [t.strip("-") for t in cleaned.split() if len(t.strip("-")) >= 3]
    keywords = [t for t in tokens if t.lower() not in COMMON_STOP_WORDS]
    if not keywords:
        keywords = tokens if tokens else ["story"]
    # websearch_to_tsquery supports 'OR' syntax
    return " or ".join(keywords[:10])


def get_connection():
    """Establish a connection to PostgreSQL with pgvector support."""
    conn = psycopg.connect(
        host=PGHOST,
        port=PGPORT,
        dbname=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
    )
    register_vector(conn)
    return conn


# ===================== Schema =====================
def init_db():
    """Create pgvector extension, table, and indexes."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Enable pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Main chunks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    source_id VARCHAR(64) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    book_title VARCHAR(255) NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(2048),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # B-tree indexes for filtering
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_source ON document_chunks(source_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_filename ON document_chunks(filename);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_book_title ON document_chunks(book_title);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_page ON document_chunks(page_number);")

            # GIN index for full-text search
            try:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_doc_chunks_fts
                    ON document_chunks USING gin(to_tsvector('english', content));
                """)
            except Exception as fts_err:
                print(f"Notice: FTS index error: {fts_err}")

            # HNSW index for approximate vector search (halfvec for memory)
            try:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_doc_chunks_halfvec
                    ON document_chunks USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
                    WITH (m = 16, ef_construction = 128);
                """)
            except Exception as idx_err:
                print(f"Notice: Vector index could not be created ({idx_err}), using exact search.")

        conn.commit()
    finally:
        conn.close()


# ===================== Existence Checks =====================
def is_source_indexed(source_id: str) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM document_chunks WHERE source_id = %s LIMIT 1;", (source_id,))
            return cur.fetchone() is not None
    finally:
        conn.close()


def is_book_indexed(book_title: str) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM document_chunks WHERE book_title = %s LIMIT 1;", (book_title,))
            return cur.fetchone() is not None
    finally:
        conn.close()


def is_filename_indexed(filename: str) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM document_chunks WHERE filename = %s LIMIT 1;", (filename,))
            return cur.fetchone() is not None
    finally:
        conn.close()


# ===================== Inserts =====================
def insert_chunks(chunks_records: List[tuple]) -> int:
    """Batch insert chunks with embeddings in chunks of 500 records."""
    if not chunks_records:
        return 0

    conn = get_connection()
    try:
        batch_size = 500
        with conn.cursor() as cur:
            for i in range(0, len(chunks_records), batch_size):
                batch = chunks_records[i : i + batch_size]
                cur.executemany("""
                    INSERT INTO document_chunks
                    (source_id, filename, book_title, page_number, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector);
                """, batch)
        conn.commit()
        return len(chunks_records)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ===================== Search Helpers =====================
def _build_where_clause(book_filter: Optional[str]) -> tuple[str, dict]:
    """Build WHERE clause and params for book filtering with case-insensitive matching."""
    if book_filter and book_filter != "All Books":
        return (
            "WHERE (book_title ILIKE %(b_filter)s OR filename ILIKE %(b_filter)s)",
            {"b_filter": f"%{book_filter}%"}
        )
    return ("", {})


def _build_filter_conditions(book_filter: Optional[str], extra_conditions: Optional[List[str]] = None) -> tuple[str, dict]:
    """
    Build a clean, unified WHERE clause combining book filtering and extra SQL expressions.
    Guarantees a single valid WHERE keyword with AND-joined expressions.
    """
    conditions = []
    params = {}
    if book_filter and book_filter != "All Books":
        conditions.append("(book_title ILIKE %(b_filter)s OR filename ILIKE %(b_filter)s)")
        params["b_filter"] = f"%{book_filter}%"
    if extra_conditions:
        for c in extra_conditions:
            if c and c.strip():
                conditions.append(c.strip())
    if conditions:
        return "WHERE " + " AND ".join(conditions), params
    return "", params


def _row_to_dict(row: tuple) -> Dict[str, Any]:
    """Convert a database row to a chunk dictionary."""
    return {
        "id": row[0],
        "filename": row[1],
        "book_title": row[2],
        "page": row[3],
        "chunk_index": row[4],
        "text": row[5],
        "content": row[5],
        "distance": float(row[6]),
        "score": float(row[7]) if len(row) > 7 else None,
    }


# ===================== Search Functions =====================
def search_chunks(
    query_vector: List[float],
    top_k: int = 5,
    book_filter: Optional[str] = None,
    max_distance: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Pure vector cosine similarity search with proper parameter binding.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            extra_conds = []
            params = {
                "q_vec": query_vector,
                "top_k": top_k,
            }

            if max_distance is not None:
                params["max_dist"] = max_distance
                extra_conds.append("(embedding <=> %(q_vec)s::vector) <= %(max_dist)s")

            where_sql, filter_params = _build_filter_conditions(book_filter, extra_conds)
            params.update(filter_params)

            query = f"""
                SELECT id, filename, book_title, page_number, chunk_index, content,
                       (embedding <=> %(q_vec)s::vector) AS distance
                FROM document_chunks
                {where_sql}
                ORDER BY embedding <=> %(q_vec)s::vector
                LIMIT %(top_k)s;
            """
            cur.execute(query, params)
            rows = cur.fetchall()
            return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def search_chunks_hybrid(
    query_vector: List[float],
    fts_query: str,
    top_k: int = 5,
    book_filter: Optional[str] = None,
    max_distance: Optional[float] = None,
    candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
) -> List[Dict[str, Any]]:
    """
    Hybrid search with Reciprocal Rank Fusion (RRF).
    Combines vector similarity and full-text search with resilient OR-fallback.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            clean_fts = _build_fts_query(fts_query)
            params = {
                "q_vec": query_vector,
                "fts_q": clean_fts,
                "cand_limit": min(max(top_k * candidate_multiplier, 50), MAX_CANDIDATES),
                "top_k": top_k,
                "rrf_k": RRF_CONSTANT,
            }

            # Candidate filter: chunks matching vector distance threshold OR full-text query
            candidate_conds = [
                "((embedding <=> %(q_vec)s::vector) < 0.85 OR to_tsvector('english', content) @@ websearch_to_tsquery('english', %(fts_q)s))"
            ]
            if max_distance is not None:
                params["max_dist"] = max_distance
                candidate_conds.append("(embedding <=> %(q_vec)s::vector) <= %(max_dist)s")

            where_sql, filter_params = _build_filter_conditions(book_filter, candidate_conds)
            params.update(filter_params)

            query = f"""
                WITH candidates AS (
                    SELECT id, filename, book_title, page_number, chunk_index, content,
                           (embedding <=> %(q_vec)s::vector) AS dist,
                           ts_rank_cd(to_tsvector('english', content), websearch_to_tsquery('english', %(fts_q)s)) AS fts_score,
                           ROW_NUMBER() OVER (ORDER BY (embedding <=> %(q_vec)s::vector) ASC) AS vec_rank,
                           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), websearch_to_tsquery('english', %(fts_q)s)) DESC) AS fts_rank
                    FROM document_chunks
                    {where_sql}
                    LIMIT %(cand_limit)s
                )
                SELECT id, filename, book_title, page_number, chunk_index, content, dist,
                       (1.0 / (%(rrf_k)s + vec_rank)) + (1.0 / (%(rrf_k)s + fts_rank)) AS rrf_score
                FROM candidates
                ORDER BY rrf_score DESC
                LIMIT %(top_k)s;
            """
            cur.execute(query, params)
            rows = cur.fetchall()

            # Fallback to pure vector if hybrid returns nothing
            if not rows:
                return search_chunks(query_vector, top_k, book_filter, max_distance)

            results = []
            for row in rows:
                d = _row_to_dict(row)
                d["score"] = float(row[7])
                results.append(d)
            return results
    finally:
        conn.close()


def search_chunks_stratified(
    query_vector: List[float],
    per_book_k: int = 3,
    fts_query: Optional[str] = None,
    book_filter: Optional[str] = None,
    max_distance: float = 0.82,
) -> List[Dict[str, Any]]:
    """
    Stratified multi-book retrieval using window functions.
    Guarantees balanced, high-relevance representation across books.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            clean_fts = _build_fts_query(fts_query or "")

            if book_filter and book_filter != "All Books":
                # Single book: use hybrid search
                return search_chunks_hybrid(
                    query_vector, clean_fts or "story",
                    top_k=per_book_k * 5,
                    book_filter=book_filter,
                    max_distance=max_distance,
                )

            params = {
                "q_vec": query_vector,
                "fts_q": clean_fts,
                "max_dist": max_distance,
                "per_book_k": per_book_k,
            }

            query = """
                WITH ranked_chunks AS (
                    SELECT id, filename, book_title, page_number, chunk_index, content,
                           (embedding <=> %(q_vec)s::vector) AS dist,
                           ROW_NUMBER() OVER (
                               PARTITION BY book_title
                               ORDER BY (embedding <=> %(q_vec)s::vector) ASC
                           ) AS rank_in_book
                    FROM document_chunks
                )
                SELECT id, filename, book_title, page_number, chunk_index, content, dist
                FROM ranked_chunks
                WHERE rank_in_book <= %(per_book_k)s
                  AND dist <= %(max_dist)s
                ORDER BY book_title ASC, page_number ASC;
            """
            cur.execute(query, params)
            rows = cur.fetchall()

            # Fallback: check if we have good book coverage (unique books represented)
            unique_books_in_result = len(set(r[2] for r in rows)) if rows else 0
            # Get total unique books in DB for comparison
            cur.execute("SELECT COUNT(DISTINCT book_title) FROM document_chunks;")
            total_unique_books = cur.fetchone()[0]

            # Trigger fallback if: no results OR few unique books represented (< 50% of total)
            should_fallback = (not rows or
                               unique_books_in_result < max(3, total_unique_books // 2))

            if should_fallback:
                fallback_params = {"q_vec": query_vector, "per_book_k": per_book_k}
                cur.execute("""
                    WITH ranked_chunks AS (
                        SELECT id, filename, book_title, page_number, chunk_index, content,
                               (embedding <=> %(q_vec)s::vector) AS dist,
                               ROW_NUMBER() OVER (
                                   PARTITION BY book_title
                                   ORDER BY (embedding <=> %(q_vec)s::vector) ASC
                               ) AS rank_in_book
                        FROM document_chunks
                    )
                    SELECT id, filename, book_title, page_number, chunk_index, content, dist
                    FROM ranked_chunks
                    WHERE rank_in_book <= %(per_book_k)s
                    ORDER BY book_title ASC, page_number ASC;
                """, fallback_params)
                rows = cur.fetchall()

            return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def search_chunks_by_entity(
    entity_name: str,
    query_vector: List[float],
    per_book_k: int = 4,
    max_distance: float = 0.78,
    top_k: int = 20,
    book_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Entity-focused search prioritizing books and passages where the entity actually appears,
    avoiding contamination from unrelated books. Supports optional book filtering.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            clean_entity = re.sub(r"[^\w\s-]", " ", entity_name).strip()
            entity_like = f"%{clean_entity}%" if clean_entity else "%"
            clean_fts = _build_fts_query(entity_name)

            params = {
                "entity_like": entity_like,
                "fts_q": clean_fts,
                "q_vec": query_vector,
                "max_dist": max_distance,
                "per_book_k": per_book_k,
                "top_k": top_k,
            }

            book_filter_sql = ""
            if book_filter and book_filter != "All Books":
                params["b_filter"] = f"%{book_filter}%"
                book_filter_sql = "AND (book_title ILIKE %(b_filter)s OR filename ILIKE %(b_filter)s)"

            query = f"""
                WITH entity_matches AS (
                    SELECT id, filename, book_title, page_number, chunk_index, content,
                           (embedding <=> %(q_vec)s::vector) AS dist,
                           (CASE WHEN content ILIKE %(entity_like)s THEN 1 ELSE 0 END) AS has_exact_entity,
                           ROW_NUMBER() OVER (
                               PARTITION BY book_title
                               ORDER BY 
                                   (CASE WHEN content ILIKE %(entity_like)s THEN 0 ELSE 1 END) ASC,
                                   (embedding <=> %(q_vec)s::vector) ASC
                           ) AS rank_in_book
                    FROM document_chunks
                    WHERE (content ILIKE %(entity_like)s
                       OR to_tsvector('english', content) @@ websearch_to_tsquery('english', %(fts_q)s)
                       OR (embedding <=> %(q_vec)s::vector) <= %(max_dist)s)
                       {book_filter_sql}
                )
                SELECT id, filename, book_title, page_number, chunk_index, content, dist
                FROM entity_matches
                WHERE rank_in_book <= %(per_book_k)s
                ORDER BY rank_in_book ASC, has_exact_entity DESC, dist ASC
                LIMIT %(top_k)s;
            """
            cur.execute(query, params)
            rows = cur.fetchall()

            # Fallback if no exact entity matches or sparse
            if not rows:
                return search_chunks_hybrid(query_vector, entity_name, top_k=top_k, book_filter=book_filter)

            return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def search_multi_hop(
    query_vector: List[float],
    initial_query: str,
    followup_queries: List[str],
    top_k: int = 5,
    book_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Multi-hop retrieval: run multiple searches and combine.
    Useful for complex queries requiring multiple aspects.
    """
    conn = get_connection()
    all_chunks = []

    try:
        # Initial search
        initial = search_chunks_hybrid(query_vector, initial_query, top_k, book_filter)
        all_chunks.extend(initial)

        # Followup searches
        for fq in followup_queries:
            fq_vec = query_vector  # Reuse or could embed separately
            fq_chunks = search_chunks_hybrid(fq_vec, fq, max(3, top_k // 2), book_filter)
            all_chunks.extend(fq_chunks)

        # Deduplicate
        seen = set()
        unique = []
        for c in all_chunks:
            if c["id"] not in seen:
                seen.add(c["id"])
                unique.append(c)

        return unique[:top_k]
    finally:
        conn.close()


# ===================== Stats =====================
def get_db_stats() -> Dict[str, Any]:
    """Get database statistics."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            total_chunks = cur.fetchone()[0]

            cur.execute("""
                SELECT book_title, filename, COUNT(*), MIN(page_number), MAX(page_number)
                FROM document_chunks
                GROUP BY book_title, filename
                ORDER BY book_title ASC;
            """)
            book_rows = cur.fetchall()

            books = [
                {
                    "book_title": row[0],
                    "filename": row[1],
                    "chunk_count": row[2],
                    "min_page": row[3],
                    "max_page": row[4],
                }
                for row in book_rows
            ]

            return {
                "total_chunks": total_chunks,
                "total_books": len(books),
                "books": books,
            }
    finally:
        conn.close()


# ===================== Book Management =====================
def get_all_books() -> List[Dict[str, Any]]:
    """Get all unique books with stats."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT book_title, filename, COUNT(*) as chunk_count,
                       MIN(page_number) as min_page, MAX(page_number) as max_page
                FROM document_chunks
                GROUP BY book_title, filename
                ORDER BY book_title ASC;
            """)
            rows = cur.fetchall()
            return [
                {
                    "book_title": row[0],
                    "filename": row[1],
                    "chunk_count": row[2],
                    "min_page": row[3],
                    "max_page": row[4],
                }
                for row in rows
            ]
    finally:
        conn.close()


def delete_book(book_title: str) -> int:
    """Delete all chunks for a book title."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM document_chunks WHERE book_title = %s;", (book_title,))
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def delete_chunks_by_filename(filename: str) -> int:
    """Delete all chunks for a filename."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM document_chunks WHERE filename = %s;", (filename,))
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def clear_all_chunks() -> bool:
    """Clear all data."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE document_chunks RESTART IDENTITY;")
        conn.commit()
        return True
    finally:
        conn.close()