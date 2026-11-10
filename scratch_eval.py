from retrieval import smart_retrieve
from app import generate_answer

def test_q(question):
    print("=" * 60)
    print("Testing Question:", question)
    chunks, meta = smart_retrieve(question)
    print(f"Dynamic K: {meta['dynamic_k']} chunks | Strategy: {meta['strategy']}")
    print("Book distribution:", meta["book_distribution"])
    answer = generate_answer(question, chunks, meta)
    print("\nANSWER:\n" + answer[:600] + "...\n")

test_q("Compare the character development of Neville Longbottom and Draco Malfoy across multiple books.")
test_q("Summarize all major events involving Severus Snape across the books.")

