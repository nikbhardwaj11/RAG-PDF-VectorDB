import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable

from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

from database import (
    init_db,
    is_book_indexed,
    insert_chunks,
)

load_dotenv()

# ===================== Configuration =====================
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b")

# Chunking parameters - tune these for your content type
CHUNK_SIZE = 1200      # max characters per chunk (smaller = more precise, larger = more context)
CHUNK_OVERLAP = 300    # characters overlapped between consecutive chunks
MAX_EMBED_WORKERS = 8  # parallel embedding requests (adjust based on API rate limits)

client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)


# ===================== Embeddings =====================
def create_embeddings(
    texts: List[str],
    input_type: str = "passage",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Embeddings are created concurrently (one per text) to avoid server-side
    batch-inference errors on the NVIDIA endpoint.

    Args:
        texts: list of text strings to embed.
        input_type: hint passed to the embedding model ("passage" for documents, "query" for questions).
        progress_callback: optional callback(completed, total) for progress bars.

    Returns:
        list of embedding vectors (one per input text), in the same order as input.
    """
    total = len(texts)
    vectors = [None] * total  # Pre-allocate to maintain order

    def _embed_one(idx: int, text: str):
        """Embed a single text with retry logic."""
        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                response = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=[text],
                    extra_body={"input_type": input_type},
                )
                return idx, response.data[0].embedding
            except Exception:
                # Exponential backoff on transient failures (rate limits, network issues)
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Embedding failed after {max_attempts} attempts for chunk {idx}")

    # Use ThreadPoolExecutor for parallel embedding requests
    with ThreadPoolExecutor(max_workers=MAX_EMBED_WORKERS) as executor:
        # Submit all embedding tasks
        futures = {executor.submit(_embed_one, i, t): i for i, t in enumerate(texts)}
        completed = 0
        # Collect results as they complete
        for future in as_completed(futures):
            idx, vec = future.result()
            vectors[idx] = vec  # Place in correct position
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    return vectors


# ===================== PDF Text Extraction & Chunking =====================
def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split extracted page text into overlapping chunks of CHUNK_SIZE characters.

    Overlapping helps preserve context that might be cut at chunk boundaries.

    Args:
        pages: list of {"page": int, "text": str} dicts from PDF extraction.

    Returns:
        list of {"text": str, "page": int} chunk dicts.
    """
    chunks = []
    for page_record in pages:
        text = page_record["text"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "page": page_record["page"]})
            if end >= len(text):
                break
            # Move start position, accounting for overlap
            start = end - CHUNK_OVERLAP
    return chunks


def extract_pdf_pages_range(
    file_path: str,
    start_page: int,
    end_page: int,
) -> List[Dict[str, Any]]:
    """
    Extract text from a range of pages (1-based, inclusive) in a PDF.
    Skips pages with no extractable text (common in scanned PDFs).

    Args:
        file_path: path to the PDF file.
        start_page: first page to extract (1-based).
        end_page: last page to extract (1-based, inclusive).

    Returns:
        list of {"page": int, "text": str} dicts.
    """
    reader = PdfReader(file_path)
    pages = []
    total = len(reader.pages)

    # Convert to 0-based indices
    start_idx = max(0, start_page - 1)
    end_idx = min(total, end_page)

    empty_pages = 0
    for idx in range(start_idx, end_idx):
        page_number = idx + 1
        try:
            raw_text = reader.pages[idx].extract_text() or ""
            # Normalize whitespace: collapse multiple spaces/newlines
            text = " ".join(raw_text.split())
            if text:
                pages.append({"page": page_number, "text": text})
            else:
                empty_pages += 1
        except Exception as e:
            print(f"Warning extracting page {page_number}: {e}")
            empty_pages += 1
            continue

    if empty_pages > 0:
        print(f"  Note: {empty_pages} page(s) had no extractable text (skipped)")

    return pages


def get_compilation_book_ranges(reader: PdfReader) -> List[Dict[str, Any]]:
    """
    Detect book boundaries inside a single PDF compilation using the PDF outline
    (bookmarks/table of contents). Returns a list of {book_title, start_page, end_page}.

    Returns an empty list if no outline/bookmarks are found — in that case the
    caller should treat the whole PDF as a single book.

    This is useful for PDF files that contain multiple books (e.g., an anthology).
    """
    found_markers = []
    try:
        outline = reader.outline
        if outline:
            for item in outline:
                # PDF outlines can be nested lists; we only handle flat items with titles
                if not isinstance(item, list) and hasattr(item, "title"):
                    try:
                        page_idx = reader.get_destination_page_number(item)
                        found_markers.append({
                            "book_title": item.title,
                            "start_page": page_idx + 1,  # convert to 1-based
                        })
                    except Exception:
                        pass
    except Exception as e:
        print(f"Notice reading PDF outline: {e}")

    if len(found_markers) >= 1:
        # Sort by page number to get correct order
        found_markers.sort(key=lambda x: x["start_page"])
        total_pages = len(reader.pages)
        ranges = []
        for i, m in enumerate(found_markers):
            end_p = (
                found_markers[i + 1]["start_page"] - 1
                if i + 1 < len(found_markers)
                else total_pages
            )
            ranges.append({
                "book_title": m["book_title"],
                "start_page": m["start_page"],
                "end_page": end_p,
            })
        return ranges

    return []


# ===================== Ingestion =====================
def ingest_book_range(
    file_path: str,
    book_title: str,
    start_page: int,
    end_page: int,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """
    Ingest one book (a page range of a PDF) into the vector database.

    Steps:
      1. Skip if the book is already indexed (deduplication).
      2. Extract text from the page range.
      3. Chunk the text into overlapping pieces.
      4. Embed every chunk.
      5. Insert chunks + embeddings into PostgreSQL.

    Returns a result dict summarising what was done.
    """
    filename = os.path.basename(file_path)
    source_id = f"{filename}_{start_page}_{end_page}"

    # Skip books that are already in the database (deduplication by book title)
    if is_book_indexed(book_title):
        if progress_callback:
            progress_callback(f"'{book_title}' already indexed.", 1.0)
        return {
            "filename": filename,
            "book_title": book_title,
            "source_id": source_id,
            "already_indexed": True,
            "chunks_count": 0,
        }

    if progress_callback:
        progress_callback(f"Extracting pages {start_page}–{end_page} for '{book_title}'...", 0.08)

    pages = extract_pdf_pages_range(file_path, start_page, end_page)
    if not pages:
        # Handle scanned/image-based PDFs gracefully
        if progress_callback:
            progress_callback(f"'{book_title}': No extractable text (likely scanned PDF). Skipping.", 1.0)
        return {
            "filename": filename,
            "book_title": book_title,
            "source_id": source_id,
            "already_indexed": False,
            "chunks_count": 0,
            "skipped": True,
            "reason": "No extractable text (scanned/image-based PDF)",
        }

    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError(f"No chunks created for '{book_title}'.")

    total_chunks = len(chunks)
    texts = [c["text"] for c in chunks]

    if progress_callback:
        progress_callback(f"Generating embeddings for {total_chunks} chunks...", 0.15)

    def on_embed_progress(completed: int, total: int):
        if progress_callback:
            # Scale progress: 15% to 90%
            pct = 0.15 + (0.75 * (completed / total))
            progress_callback(f"[{book_title}] Embedded {completed}/{total}...", pct)

    all_vectors = create_embeddings(texts, input_type="passage", progress_callback=on_embed_progress)

    if progress_callback:
        progress_callback(f"Saving {total_chunks} chunks to Vector DB...", 0.93)

    # Prepare records for batch insert
    records = [
        (
            source_id,
            filename,
            book_title,
            chunk["page"],
            index,
            chunk["text"],
            vector,
        )
        for index, (chunk, vector) in enumerate(zip(chunks, all_vectors))
    ]

    insert_chunks(records)

    if progress_callback:
        progress_callback(f"Stored {total_chunks} chunks for '{book_title}'!", 1.0)

    return {
        "filename": filename,
        "book_title": book_title,
        "source_id": source_id,
        "already_indexed": False,
        "chunks_count": total_chunks,
    }


def scan_and_ingest_books(
    books_dir: str = "books",
    progress_callback: Optional[Callable[[str, float], None]] = None,
    target_book_title: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Scan the books/ folder and index every PDF found.

    Supports two PDF kinds:
      - Compilation PDFs with a PDF outline (multiple books in one file).
      - Single-book PDFs (the whole file becomes one book, titled from the filename).

    Args:
        books_dir: folder containing PDF files.
        progress_callback: optional callback(msg, pct) for progress bars.
        target_book_title: if set, only index this one book title.

    Returns:
        list of result dicts, one per book ingested.
    """
    init_db()

    if not os.path.exists(books_dir):
        os.makedirs(books_dir, exist_ok=True)
        return []

    pdf_files = glob.glob(os.path.join(books_dir, "*.pdf"))
    if not pdf_files:
        return []

    results = []

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        reader = PdfReader(pdf_path)
        book_ranges = get_compilation_book_ranges(reader)

        if book_ranges:
            # Compilation PDF: multiple books detected via outline/bookmarks.
            filtered_ranges = book_ranges
            if target_book_title and target_book_title != "All Books":
                filtered_ranges = [r for r in book_ranges if r["book_title"] == target_book_title]

            total_to_process = len(filtered_ranges)
            for idx, r in enumerate(filtered_ranges):
                def step_callback(msg: str, pct: float, idx: int = idx, total: int = total_to_process):
                    overall = (idx + pct) / total
                    if progress_callback:
                        progress_callback(f"[{idx + 1}/{total}] {msg}", overall)

                res = ingest_book_range(
                    file_path=pdf_path,
                    book_title=r["book_title"],
                    start_page=r["start_page"],
                    end_page=r["end_page"],
                    progress_callback=step_callback,
                )
                results.append(res)
        else:
            # Single-book PDF: derive title from the filename.
            book_title = (
                os.path.splitext(filename)[0]
                .replace("_", " ")
                .replace("-", " ")
                .title()
            )
            if target_book_title and target_book_title != "All Books" and book_title != target_book_title:
                continue

            total_pages = len(reader.pages)
            res = ingest_book_range(
                file_path=pdf_path,
                book_title=book_title,
                start_page=1,
                end_page=total_pages,
                progress_callback=progress_callback,
            )
            results.append(res)

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Book RAG — Books Ingestion into Vector DB")
    print("=" * 60)

    init_db()
    books_folder = os.path.join(os.path.dirname(__file__), "books")
    pdfs = glob.glob(os.path.join(books_folder, "*.pdf"))

    print(f"\nScanning directory: {books_folder}")
    print(f"Found {len(pdfs)} PDF file(s).")

    all_ranges = []  # collect every book range across all PDFs

    for p in pdfs:
        reader = PdfReader(p)
        ranges = get_compilation_book_ranges(reader)
        print(f"\nFile: {os.path.basename(p)} ({len(reader.pages)} pages)")
        if ranges:
            print(f"  Detected compilation containing {len(ranges)} book(s):")
            for r in ranges:
                status = "[OK] In DB" if is_book_indexed(r["book_title"]) else "[ ] Not Indexed"
                print(f"    • {r['book_title']}: Pages {r['start_page']}–{r['end_page']} [{status}]")
                all_ranges.append(r)
        else:
            title = (
                os.path.splitext(os.path.basename(p))[0]
                .replace("_", " ")
                .replace("-", " ")
                .title()
            )
            status = "[OK] In DB" if is_book_indexed(title) else "[ ] Not Indexed"
            print(f"  Detected single book: '{title}' [{status}]")
            all_ranges.append({"book_title": title})

    unindexed = [r for r in all_ranges if not is_book_indexed(r["book_title"])] if all_ranges else []

    if unindexed:
        print(f"\nFound {len(unindexed)} unindexed book(s). Starting automatic ingestion...")

        def cli_progress(msg, pct):
            print(f"[{int(pct * 100):3d}%] {msg}")

        results = scan_and_ingest_books(books_folder, progress_callback=cli_progress)
        print("\nAll pending books successfully processed!")
    else:
        print("\nAll detected books are already fully indexed in the Vector DB!")