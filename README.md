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

## Architecture & LangChain Engine

```
app.py                  # Streamlit UI (Chat interface, Confidence Meter, Citations)
├── rag_engine.py       # Core LangChain Engine (VectorStore, LCEL Chain, Ingestion)
│   ├── NemotronEmbeddings    # LangChain Embeddings with NVIDIA input_type support
│   ├── PostgresPGVectorStore # LangChain VectorStore adapter (Zero re-index needed)
│   ├── rag_chain             # LCEL Prompt | ChatOpenAI | StrOutputParser
│   └── ingest_pdf_file       # PyPDFLoader + RecursiveCharacterTextSplitter
├── database.py         # PostgreSQL connection & pgvector schema
└── test_retrieval.py   # Regression tests & LangChain compatibility checks
```

### LangChain Components
- **Unified VectorStore Adapter (`rag_engine.py`)**: `PostgresPGVectorStore` wraps your existing `document_chunks` table, allowing LangChain to query all 10,850+ indexed chunks instantly without re-indexing.
- **LCEL Streaming Chain (`rag_engine.py`)**: `ChatPromptTemplate | ChatOpenAI | StrOutputParser` streaming responses with citation grounding.
- **Recursive Text Splitting (`rag_engine.py` / `ingest.py`)**: `RecursiveCharacterTextSplitter` splitting documents on semantic boundaries (`\n\n`, `\n`, sentence endings).
- **LangChain Retriever (`retrieval.py`)**: `MultiBookRetriever(BaseRetriever)` returning standardized `Document` objects with rich metadata for use across any LangChain or LangGraph pipeline.

## Project Structure

```
.
├── app.py                      # Main Streamlit application + LangChain LCEL
├── rag_engine.py               # Core LangChain RAG pipeline (Option 2 Hybrid Adapter)
├── database.py                 # Database layer (PostgreSQL + pgvector)
├── ingest.py                   # PDF ingestion & LangChain text splitter
├── retrieval.py                # Smart retrieval & LangChain MultiBookRetriever
├── test_retrieval.py           # Verification script for query understanding
├── requirements.txt            # Python dependencies (Streamlit, pgvector, LangChain)
├── .env                        # Configuration (NVIDIA NIM & PostgreSQL credentials)
├── books/                      # PDF storage directory
├── README.md                   # Quickstart and overview guide
└── RAG_SYSTEM_DOCUMENTATION.md # Full architecture & system reference
```

## Testing

```bash
# Test query analysis and LangChain retriever instantiation (no DB connection required)
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