"""
Test script for the generic retrieval system.
Run this after indexing some books to verify retrieval works correctly.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from retrieval import smart_retrieve, classify_query, extract_entities


def test_query_analysis():
    """Test the query analysis function with various question types."""
    test_cases = [
        # Listing queries
        ("List all characters mentioned across the books.", "LISTING"),
        ("What are all the themes in the series?", "LISTING"),
        ("Enumerate all the magical creatures.", "LISTING"),

        # Comparison queries
        ("Compare the protagonist and antagonist.", "COMPARISON"),
        ("What are the differences between the two main characters?", "COMPARISON"),
        ("Contrast the magic systems in the books.", "COMPARISON"),
        # Comparison with actual entities
        ("Compare Mowgli and Harry Potter.", "COMPARISON"),
        ("Compare the characters Baloo and Bagheera.", "COMPARISON"),
        ("How do Voldemort and Shere Khan compare?", "COMPARISON"),

        # Timeline queries
        ("Summarize the major events in the hero's life.", "TIMELINE"),
        ("What is the timeline of the main character's journey?", "TIMELINE"),
        ("Tell me the history of the kingdom.", "TIMELINE"),
        # Timeline with actual entities
        ("What is the timeline of Mowgli's life?", "TIMELINE"),
        ("Summarize Harry Potter's journey across the books.", "TIMELINE"),

        # Thematic queries
        ("Why did the hero make that choice?", "THEMATIC"),
        ("Explain how the magic system works.", "THEMATIC"),
        ("What led to the war between the kingdoms?", "THEMATIC"),

        # Factual queries (default)
        ("What is the capital city?", "FACTUAL"),
        ("Who is the main character's mentor?", "FACTUAL"),
        ("When does the story take place?", "FACTUAL"),
    ]

    print("=" * 60)
    print("QUERY ANALYSIS TESTS")
    print("=" * 60)

    for question, expected_type in test_cases:
        query_type, confidence, entities = classify_query(question)
        dynamic_k = len(entities) * 10 if query_type == "COMPARISON" else 10  # Approximate
        status = "[OK]" if query_type == expected_type else "[FAIL]"
        print(f"{status} [{query_type}] {question}")
        print(f"    Entities: {entities}")
        print()


def test_smart_retrieve(question: str, book_filter: str = None, manual_k: int = None):
    """Test smart retrieval for a question."""
    print("=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)

    try:
        chunks, meta = smart_retrieve(question, book_filter=book_filter, manual_k=manual_k)
        print(f"Query Type: {meta['query_type']}")
        print(f"Dynamic K: {meta['dynamic_k']} (target: {meta['target_k']})")
        print(f"Strategy: {meta['strategy']}")
        print(f"Entities: {meta.get('entities', [])}")
        print(f"Reasoning: {meta['reasoning']}")
        print(f"Book Distribution: {meta['book_distribution']}")
        print(f"\nRetrieved {len(chunks)} chunks:")

        for i, chunk in enumerate(chunks, 1):
            print(f"\n  [{i}] Book: {chunk['book_title']} | Page: {chunk['page']} | Distance: {chunk['distance']:.4f}")
            print(f"      {chunk['text'][:120]}...")

    except Exception as e:
        print(f"ERROR: {e}")

    print()


if __name__ == "__main__":
    # Test query analysis first
    test_query_analysis()

    # Test retrieval (requires indexed books)
    # Uncomment and add your own test questions after indexing books:
    """
    test_smart_retrieve("List all characters mentioned across the books.")
    test_smart_retrieve("Compare the protagonist and antagonist.")
    test_smart_retrieve("Summarize the major events in the hero's life.")
    test_smart_retrieve("Why did the hero make that choice?")
    test_smart_retrieve("What is the capital city?")
    """

    print("To test retrieval, uncomment the test_smart_retrieve() calls in this file")
    print("and run after indexing some books with: python ingest.py")