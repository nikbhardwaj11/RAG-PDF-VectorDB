# Quick Start Guide - Multi-Book RAG Assistant

## Prerequisites
- Python 3.10+
- Docker (for PostgreSQL + pgvector)
- NVIDIA API Key (get from https://build.nvidia.com)

---

## 1. Start Database

```bash
docker run -d \
  --name pgvector \
  -e POSTGRES_DB=ragdb \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=ragpassword \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Verify:
```bash
docker logs pgvector
# Should show "database system is ready to accept connections"
```

---

## 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
NVIDIA_API_KEY=your_actual_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_CHAT_MODEL=nvidia/nemotron-3.5-bolt-e4b
NVIDIA_EMBED_MODEL=nvidia/nemotron-3.5-embed-1b

PGHOST=localhost
PGPORT=5432
PGDATABASE=ragdb
PGUSER=raguser
PGPASSWORD=ragpassword
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Add Books

**Option A: Via UI** (after starting app)
- Drag & drop PDFs in sidebar "Upload Books"

**Option B: Via filesystem**
```bash
mkdir -p books
# Copy your PDFs to books/
cp ~/my_books/*.pdf books/
```

**Supported formats**:
- Single-book PDFs (title from filename)
- Compilation PDFs with bookmarks/outline (auto-split into books)

---

## 5. Run Application

```bash
streamlit run app.py
```

Opens at: http://localhost:8501

---

## 6. Index Books

1. Open sidebar (left panel)
2. Verify PDFs appear in "PDF Files in ./books"
3. Click **"⚡ Sync / Index Books into Vector DB"**
4. Watch progress bar complete
5. Stats update: "Chunks", "PDFs Indexed", "Chapters"

---

## 7. Ask Questions!

### Example Questions by Type

**📋 LISTING** (broad coverage)
- "List all characters mentioned across the books"
- "What are all the locations in the stories?"
- "Enumerate all magical creatures"

**⚖️ COMPARISON** (entity-focused, balanced)
- "Compare Harry Potter and Mowgli"
- "Contrast the villains across books"
- "What are the differences between the magic systems?"

**📅 TIMELINE** (chronological)
- "Show the timeline of Harry Potter's life"
- "Summarize major events in Mowgli's journey"
- "What happened in chronological order?"

**🎭 THEMATIC** (deep analysis)
- "Why did Harry choose to sacrifice himself?"
- "Explain the theme of friendship in the books"
- "What does the ring symbolize?"

**❓ FACTUAL** (precise, fast)
- "Who is Harry's mentor?"
- "Where is Hogwarts located?"
- "When does the story take place?"

---

## UI Features Tour

### Sidebar
- **Stats**: Live chunk/book/chapter counts
- **Upload**: Drag & drop PDFs
- **Scope**: "All Books" or specific PDF
- **K Mode**: Dynamic (auto) vs Manual slider
- **Sync Button**: Re-index with progress
- **Book Manager**: View/delete individual books

### Main Chat
- **Suggestion Pills**: Click to try example questions
- **Query Type Buttons**: Force specific strategy
- **Metadata Pills**: See strategy, K, confidence, entities
- **Cross-Book Tabs**: Side-by-side for comparisons
- **Expandable Sources**: Grouped by book with page numbers

---

## Testing

```bash
# Database connection
python test_db.py

# Vector operations
python test_vector.py

# Query analysis (no DB needed)
python test_retrieval.py
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| "Failed to connect to vector database" | Check Docker container running: `docker ps` |
| "Missing NVIDIA_API_KEY" | Add to `.env` file |
| "No extractable text" | PDF is scanned/image-based - use OCR first |
| "0 chunks retrieved" | Lower distance threshold or add more books |
| Slow indexing | Reduce `MAX_EMBED_WORKERS` in `ingest.py` |

---

## Project Structure

```
RAG-PDF-VectorDB/
├── app.py              # Streamlit UI (main entry)
├── database.py         # PostgreSQL + pgvector
├── ingest.py           # PDF → chunks → embeddings → DB
├── retrieval.py        # Query analysis + 4 strategies
├── requirements.txt    # Dependencies
├── .env                # Your config (create from example)
├── books/              # Your PDFs (gitignored)
├── test_*.py          # Test scripts
└── DOCS/              # This documentation
```

---

## Next Steps

1. **Add your PDFs** and index them
2. **Try different question types** - notice strategy changes in metadata pills
3. **Compare "All Books" vs specific PDF** scope
4. **Check cross-book tabs** on comparison questions
5. **Explore book manager** to see per-book stats

---

*Happy querying! 📚*