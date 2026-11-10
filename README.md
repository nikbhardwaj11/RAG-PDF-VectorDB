# Multi-Book RAG Assistant

A generic Retrieval-Augmented Generation system for querying multiple fiction/non-fiction books. Built with Streamlit, PostgreSQL + pgvector, and NVIDIA Nemotron models.

## Features

- **Multi-Book Support**: Index and query across any number of PDF books
- **Dynamic K Retrieval**: Automatically adjusts chunk count based on query type
- **Multiple Search Strategies**:
  - **Stratified Multi-Book** - Balanced retrieval across all books (for listings)
  - **Entity Comparison** - Sub-query decomposition for comparing entities
  - **Entity Timeline** - Chronological retrieval for life stories
  - **Hybrid Precision** - Vector + full-text search with RRF (default)
- **Book Management**: Upload, index, view, and delete books from UI
- **Source Citations**: Every answer cites book titles and page numbers

## Quick Start

### 1. Prerequisites

- Python 3.10+
- PostgreSQL with pgvector extension
- NVIDIA API key (for Nemotron models)

### 2. Database Setup

```bash
# Using Docker (recommended)
docker run -d \
  --name pgvector \
  -e POSTGRES_DB=ragdb \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=ragpassword \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Or install pgvector locally and create database manually
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_CHAT_MODEL=nvidia/nemotron-3.5-bolt-e4b
NVIDIA_EMBED_MODEL=nvidia/nemotron-3.5-embed-1b

PGHOST=localhost
PGPORT=5432
PGDATABASE=ragdb
PGUSER=raguser
PGPASSWORD=ragpassword
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add Books

Place PDF files in the `books/` folder, or upload via the UI.

### 6. Run the App

```bash
streamlit run app.py
```

## Usage

1. **Upload Books**: Use the sidebar to upload PDF files
2. **Index Books**: Click "Sync / Index Books into Vector DB" to process and embed
3. **Ask Questions**: Type questions in the chat interface

### Example Questions

- **Listing**: "List all characters mentioned across the books"
- **Comparison**: "Compare the protagonist and antagonist"
- **Timeline**: "Summarize the major events in the hero's life"
- **Thematic**: "Explain how the magic system works"
- **Factual**: "What is the capital city?"

## Architecture

```
app.py              # Streamlit UI, chat interface, book management
├── retrieval.py    # Query analysis, Dynamic K, search strategies
├── ingest.py       # PDF extraction, chunking, embedding, indexing
└── database.py     # PostgreSQL + pgvector operations
```

## Retrieval Strategies

| Query Type | Dynamic K | Strategy | Description |
|------------|-----------|----------|-------------|
| LISTING | 21 | stratified_multi_book | ~3 chunks/book across all books |
| COMPARISON | 20 | entity_comparison | Sub-queries per entity, merged |
| TIMELINE | 18 | entity_timeline | Entity-stratified chronological |
| THEMATIC | 10 | hybrid_precision | Vector + FTS with RRF |
| FACTUAL | 5 | hybrid_precision | High precision + gap pruning |

## Book Manager

Click "🗂️ Manage Indexed Books" in the sidebar to:
- View all indexed books with stats
- Delete individual books
- Clear all data (with confirmation)

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── database.py         # Database layer (PostgreSQL + pgvector)
├── ingest.py           # PDF ingestion pipeline
├── retrieval.py        # Smart retrieval with Dynamic K
├── requirements.txt    # Python dependencies
├── .env                # Configuration (create from .env.example)
├── books/              # PDF storage (gitignored)
├── chroma_db/          # Legacy ChromaDB (unused)
└── test_*.py          # Test scripts
```

## Testing

```bash
# Test database connection
python test_db.py

# Test vector operations
python test_vector.py

# Test query analysis (no DB needed)
python test_retrieval.py
```

## Customization

### Chunking Parameters (ingest.py)
```python
CHUNK_SIZE = 1200      # Characters per chunk
CHUNK_OVERLAP = 300    # Overlap between chunks
MAX_EMBED_WORKERS = 8  # Parallel embedding requests
```

### Search Parameters (database.py)
```python
# In search_chunks_hybrid: distance threshold for candidate selection
(embedding <=> %(q_vec)s::vector) < 0.68

# In search_chunks_stratified: max distance for relevance
max_distance=0.68
```

### Dynamic K Values (retrieval.py)
Modify the `dynamic_k` values in `analyze_query()` for each query type.

## License

MIT