import os
import glob
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Tuple

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
MAX_EMBED_WORKERS = 12 # parallel embedding requests (optimized concurrency)
MAX_PDF_PAGES = 4000   # Page limit threshold to prevent runaway processing and OOM

client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)


def validate_pdf_pages(file_path_or_stream, max_pages: int = MAX_PDF_PAGES) -> Tuple[bool, int, Optional[str]]:
    """
    Validate whether a PDF is readable and within the maximum page count limit.
    Returns: (is_valid: bool, page_count: int, error_message: Optional[str])
    """
    try:
        if isinstance(file_path_or_stream, str):
            with open(file_path_or_stream, "rb") as fh:
                reader = PdfReader(fh)
                page_count = len(reader.pages)
        else:
            reader = PdfReader(file_path_or_stream)
            page_count = len(reader.pages)

        if page_count > max_pages:
            return False, page_count, f"PDF exceeds maximum page limit of {max_pages:,} pages (found {page_count:,} pages)."
        if page_count == 0:
            return False, 0, "PDF contains 0 pages or could not be read."
        return True, page_count, None
    except Exception as e:
        return False, 0, f"Invalid or unreadable PDF: {str(e)}"



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
            if progress_callback and (completed % 10 == 0 or completed == total):
                progress_callback(completed, total)
                time.sleep(0.001)  # Cooperative sleep to yield GIL

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
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract text from a range of pages (1-based, inclusive) in a PDF.
    Skips pages with no extractable text (common in scanned PDFs).
    Cooperatively yields GIL to prevent freezing background threads.

    Args:
        file_path: path to the PDF file.
        start_page: first page to extract (1-based).
        end_page: last page to extract (1-based, inclusive).
        progress_callback: optional callback(completed_pages, total_pages).

    Returns:
        list of {"page": int, "text": str} dicts.
    """
    with open(file_path, "rb") as fh:
        reader = PdfReader(fh)
        pages = []
        total = len(reader.pages)

        # Convert to 0-based indices
        start_idx = max(0, start_page - 1)
        end_idx = min(total, end_page)
        total_range = max(1, end_idx - start_idx)

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

            # Periodically report progress and release Python GIL so UI stays responsive
            if progress_callback and (idx % 15 == 0 or idx == end_idx - 1):
                done_pages = idx - start_idx + 1
                progress_callback(done_pages, total_range)
                time.sleep(0.001)

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

    NOTE: Currently disabled to treat each PDF as a single book.
    To enable: remove the 'return []' line below.
    """
    # Disable auto-splitting - treat each PDF as a single book
    return []

    # Original outline detection code (disabled):
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


def safe_call_progress(cb, msg: str, pct: float, stage: str = "", current: int = 0, total: int = 0):
    """Safely invoke progress callback supporting either 2 or 5 parameters."""
    if not cb:
        return
    try:
        cb(msg, pct, stage, current, total)
    except TypeError:
        try:
            cb(msg, pct)
        except Exception:
            pass


# ===================== Ingestion =====================
def ingest_book_range(
    file_path: str,
    book_title: str,
    start_page: int,
    end_page: int,
    progress_callback: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """
    Ingest one book (a page range of a PDF) into the vector database.

    Steps:
      1. Skip if the book is already indexed (deduplication).
      2. Extract text from the page range with real-time page progress and GIL yielding.
      3. Chunk the text into overlapping pieces.
      4. Embed every chunk with concurrent workers.
      5. Insert chunks + embeddings into PostgreSQL in batches of 500.

    Returns a result dict summarising what was done.
    """
    filename = os.path.basename(file_path)
    source_id = f"{filename}_{start_page}_{end_page}"
    total_pages_in_range = max(1, end_page - start_page + 1)
    t_start = time.time()

    # Skip books that are already in the database (deduplication by book title)
    if is_book_indexed(book_title):
        if progress_callback:
            safe_call_progress(progress_callback, f"'{book_title}' already indexed.", 1.0, "Completed", total_pages_in_range, total_pages_in_range)
        return {
            "filename": filename,
            "book_title": book_title,
            "source_id": source_id,
            "already_indexed": True,
            "chunks_count": 0,
        }

    def on_extract_progress(done_p: int, total_p: int):
        pct = 0.04 + 0.20 * (done_p / total_p)
        elapsed = time.time() - t_start
        speed = f" • {done_p / max(0.1, elapsed):.1f} pages/s" if elapsed > 1 else ""
        msg = f"📄 Extracting text: Page {done_p:,} / {total_p:,}{speed}"
        safe_call_progress(progress_callback, msg, pct, "Extraction", done_p, total_p)

    pages = extract_pdf_pages_range(file_path, start_page, end_page, progress_callback=on_extract_progress)
    if not pages:
        # Handle scanned/image-based PDFs gracefully
        safe_call_progress(progress_callback, f"'{book_title}': No extractable text (likely scanned PDF). Skipping.", 1.0, "Skipped", 0, 0)
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

    safe_call_progress(progress_callback, f"✂️ Created {total_chunks:,} chunks. Generating embeddings...", 0.24, "Chunking", 0, total_chunks)

    t_embed_start = time.time()
    def on_embed_progress(completed: int, total: int):
        # Scale progress: 25% to 92%
        pct = 0.25 + (0.68 * (completed / total))
        elapsed_embed = time.time() - t_embed_start
        speed = f" • {completed / max(0.1, elapsed_embed):.1f} chunks/s" if elapsed_embed > 1 else ""
        msg = f"🧠 Generating embeddings: {completed:,} / {total:,} chunks{speed}"
        safe_call_progress(progress_callback, msg, pct, "Embedding", completed, total)

    all_vectors = create_embeddings(texts, input_type="passage", progress_callback=on_embed_progress)

    safe_call_progress(progress_callback, f"💾 Saving {total_chunks:,} records into pgvector database...", 0.94, "Saving", total_chunks, total_chunks)

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

    dur = int(time.time() - t_start)
    mins, secs = divmod(dur, 60)
    dur_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
    safe_call_progress(progress_callback, f"✅ Indexed in {dur_str} ({total_chunks:,} chunks, {total_pages_in_range:,} pages)", 1.0, "Completed", total_chunks, total_chunks)

    return {
        "filename": filename,
        "book_title": book_title,
        "source_id": source_id,
        "already_indexed": False,
        "chunks_count": total_chunks,
        "duration_seconds": dur,
        "duration_str": dur_str,
    }


def ingest_single_pdf(
    file_path: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """
    Ingest a single PDF file (validating page limit, handling compilations or single books).
    """
    init_db()
    filename = os.path.basename(file_path)

    # Validate file and page limit
    is_valid, total_pages, err = validate_pdf_pages(file_path, max_pages=MAX_PDF_PAGES)
    if not is_valid:
        raise ValueError(err)

    with open(file_path, "rb") as fh:
        reader = PdfReader(fh)
        book_ranges = get_compilation_book_ranges(reader)

    if book_ranges:
        # Compilation PDF: multiple books detected via outline/bookmarks
        total_ranges = len(book_ranges)
        total_chunks = 0
        for idx, r in enumerate(book_ranges):
            def range_cb(msg: str, pct: float, i=idx, t=total_ranges):
                overall = (i + pct) / t
                if progress_callback:
                    progress_callback(f"[{i + 1}/{t}] {msg}", overall)

            res = ingest_book_range(
                file_path=file_path,
                book_title=r["book_title"],
                start_page=r["start_page"],
                end_page=r["end_page"],
                progress_callback=range_cb,
            )
            total_chunks += res.get("chunks_count", 0)

        return {
            "filename": filename,
            "book_title": f"Compilation ({len(book_ranges)} books)",
            "chunks_count": total_chunks,
            "already_indexed": False,
        }
    else:
        # Single-book PDF: derive title from filename
        book_title = (
            os.path.splitext(filename)[0]
            .replace("_", " ")
            .replace("-", " ")
            .title()
        )
        return ingest_book_range(
            file_path=file_path,
            book_title=book_title,
            start_page=1,
            end_page=total_pages,
            progress_callback=progress_callback,
        )


class IngestionTaskManager:
    """
    Thread-safe background task manager for non-blocking PDF ingestion.
    Allows Streamlit to upload and index documents in the background without freezing the UI.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None

    def _ensure_worker(self):
        with self._lock:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
                self._worker_thread.start()

    def _worker_loop(self):
        while True:
            try:
                task_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with self._lock:
                if task_id not in self._tasks:
                    self._queue.task_done()
                    continue
                task = self._tasks[task_id]
                task["status"] = "processing"
                task["started_at"] = time.time()
                task["progress"] = 0.04
                task["stage"] = "Extraction"
                task["step"] = f"Initializing extraction for {task['total_pages']} pages..."

            try:
                def progress_cb(msg: str, pct: float, stage: str = "", current: int = 0, total: int = 0):
                    with self._lock:
                        if task_id in self._tasks:
                            t = self._tasks[task_id]
                            t["step"] = msg
                            t["progress"] = round(min(max(pct, 0.02), 0.99), 3)
                            if stage:
                                t["stage"] = stage
                            if current:
                                t["current_item"] = current
                            if total:
                                t["total_items"] = total

                res = ingest_single_pdf(
                    file_path=task["file_path"],
                    progress_callback=progress_cb,
                )

                with self._lock:
                    task["status"] = "completed"
                    task["progress"] = 1.0
                    task["finished_at"] = time.time()
                    chunks = res.get("chunks_count", 0)
                    task["total_chunks"] = chunks
                    dur = int(task["finished_at"] - task["started_at"])
                    mins, secs = divmod(dur, 60)
                    dur_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
                    task["duration_str"] = dur_str
                    task["stage"] = "Completed"
                    if res.get("already_indexed"):
                        task["step"] = f"Already indexed ({chunks} chunks in DB)"
                    elif res.get("skipped"):
                        task["step"] = f"Skipped: {res.get('reason', 'No extractable text')}"
                    else:
                        task["step"] = f"Indexed in {dur_str} ({chunks:,} chunks, {task.get('total_pages', 0):,} pages)"

            except Exception as e:
                with self._lock:
                    task["status"] = "failed"
                    task["error"] = str(e)
                    task["stage"] = "Failed"
                    task["step"] = f"Failed: {str(e)[:90]}"
                    task["finished_at"] = time.time()
            finally:
                self._queue.task_done()

    def queue_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Validate page limit and queue file for background processing."""
        is_valid, page_count, err = validate_pdf_pages(file_path, max_pages=MAX_PDF_PAGES)
        book_title = (
            os.path.splitext(filename)[0]
            .replace("_", " ")
            .replace("-", " ")
            .title()
        )

        task_id = filename
        task = {
            "id": task_id,
            "filename": filename,
            "file_path": file_path,
            "book_title": book_title,
            "total_pages": page_count,
            "status": "pending" if is_valid else "rejected",
            "progress": 0.0,
            "step": "Queued for background indexing" if is_valid else f"Rejected: {err}",
            "error": err,
            "total_chunks": 0,
            "started_at": None,
            "finished_at": time.time() if not is_valid else None,
        }

        with self._lock:
            self._tasks[task_id] = task

        if is_valid:
            self._queue.put(task_id)
            self._ensure_worker()

        return task

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._tasks.values())

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [t for t in self._tasks.values() if t["status"] in ("pending", "processing")]

    def is_busy(self) -> bool:
        with self._lock:
            return any(t["status"] in ("pending", "processing") for t in self._tasks.values())

    def clear_task(self, task_id: str):
        with self._lock:
            if task_id in self._tasks and self._tasks[task_id]["status"] not in ("pending", "processing"):
                del self._tasks[task_id]

    def clear_all_finished(self):
        with self._lock:
            to_remove = [k for k, t in self._tasks.items() if t["status"] in ("completed", "failed", "rejected")]
            for k in to_remove:
                del self._tasks[k]


# Global singleton instance
ingestion_task_manager = IngestionTaskManager()


def get_task_timing(task: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Calculate formatted elapsed time, estimated remaining time (ETA), and progress string.
    Returns: (elapsed_str, eta_str, pct_str)
    """
    started = task.get("started_at")
    if not started:
        return "0s", "--", "0%"

    now = task.get("finished_at") or time.time()
    elapsed = max(0, int(now - started))
    mins, secs = divmod(elapsed, 60)
    elapsed_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

    pct = task.get("progress", 0.0)
    pct_str = f"{int(pct * 100)}%"

    if task.get("status") == "completed":
        dur = task.get("duration_str") or elapsed_str
        return dur, "Done", "100%"
    if task.get("status") in ("failed", "rejected"):
        return elapsed_str, "Stopped", pct_str

    if pct > 0.04 and elapsed > 2:
        est_total = elapsed / pct
        remaining = max(0, int(est_total - elapsed))
        r_mins, r_secs = divmod(remaining, 60)
        eta_str = f"~{r_mins}m {r_secs:02d}s" if r_mins > 0 else f"~{r_secs}s"
    else:
        eta_str = "Calculating..."

    return elapsed_str, eta_str, pct_str


def scan_and_ingest_books(
    books_dir: str = "books",
    progress_callback: Optional[Callable[[str, float], None]] = None,
    target_book_title: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Scan the books/ folder and index every PDF found.
    Validates that each PDF does not exceed MAX_PDF_PAGES (4,000 pages).

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
        is_valid, total_pages, err = validate_pdf_pages(pdf_path, max_pages=MAX_PDF_PAGES)
        if not is_valid:
            if progress_callback:
                progress_callback(f"Skipping '{filename}': {err}", 0.0)
            continue

        with open(pdf_path, "rb") as fh:
            reader = PdfReader(fh)
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