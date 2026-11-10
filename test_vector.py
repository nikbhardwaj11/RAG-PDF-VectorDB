from database import get_connection, init_db, search_chunks, get_db_stats

init_db()
conn = get_connection()

try:
    with conn.cursor() as cur:
        # Sample 2048-dimensional embedding
        sample_embedding = [0.01] * 2048
        sample_embedding[0] = 0.5
        sample_embedding[1] = 0.2

        cur.execute("""
            INSERT INTO document_chunks
            (content, filename, book_title, page_number, source_id, chunk_index, embedding)
            VALUES
            (%s, %s, %s, %s, %s, %s, %s::vector)
        """, (
            "A young wizard discovers his magical heritage on his eleventh birthday.",
            "sample.pdf",
            "Test Book",
            1,
            "sample-source-id",
            0,
            sample_embedding
        ))

    conn.commit()
    print("Vector inserted successfully into PostgreSQL (pgvector)!")

    # Test search
    results = search_chunks(sample_embedding, top_k=1)
    print(f"Vector search test: found {len(results)} match!")
    if results:
        print(f"Matched content: {results[0]['content'][:60]}... (distance: {results[0]['distance']:.4f})")

    # Clean up test row
    with conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunks WHERE source_id = 'sample-source-id';")
    conn.commit()
    print("Test cleanup completed successfully!")

except Exception as e:
    conn.rollback()
    print("Error:", e)

finally:
    conn.close()