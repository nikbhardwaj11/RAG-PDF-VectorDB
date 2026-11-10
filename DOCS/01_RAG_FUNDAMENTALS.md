# RAG Fundamentals - Understanding the Core Concept

## What is RAG?

**Retrieval-Augmented Generation (RAG)** combines two AI capabilities:
1. **Retrieval** - Finding relevant information from a knowledge base
2. **Generation** - Using an LLM to synthesize answers from retrieved context

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  USER QUESTION                                                  │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐    RETRIEVED CHUNKS    ┌─────────────────┐   │
│  │   SEARCH    │ ─────────────────────▶ │   LLM PROMPT    │   │
│  │  (VectorDB) │                        │                 │   │
│  └─────────────┘                        │ Context: [chunks]│   │
│                                         │ Question: [user] │   │
│                                         └────────┬────────┘   │
│                                                  │            │
│                                                  ▼            │
│                                         ┌─────────────────┐   │
│                                         │   LLM GENERATES │   │
│                                         │   GROUNDED ANSWER│   │
│                                         └─────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Why RAG Matters

| Problem | RAG Solution |
|---------|--------------|
| **Hallucination** | Answers grounded in retrieved sources |
| **Knowledge cutoff** | Add new documents anytime |
| **Private data** | Your documents, your control |
| **Citation needs** | Automatic source attribution |
| **Domain specificity** | Custom knowledge base per use case |

## Our RAG Implementation

### Retrieval Layer (PostgreSQL + pgvector)
- **Semantic search**: Vector embeddings for meaning-based matching
- **Keyword search**: Full-text search for exact terms
- **Hybrid fusion**: RRF combines both for best results
- **Stratified search**: Balanced across multiple books

### Augmentation Layer (Context Building)
- Chunks organized by book with page numbers
- Query-type specific prompt templates
- Cross-book synthesis instructions

### Generation Layer (NVIDIA Nemotron)
- Streaming token-by-token response
- Low temperature (0.1) for factual consistency
- Type-specific max tokens (factual=1100, listing=1800)

## Key RAG Metrics

| Metric | Target | Our Approach |
|--------|--------|--------------|
| **Recall@K** | >80% | Dynamic K + hybrid search |
| **Precision@K** | >70% | Gap pruning + reranking |
| **Latency** | <2s | HNSW index + parallel embed |
| **Citation accuracy** | 100% | Page-level chunk metadata |

---

## For Your Presentation

> "RAG is the difference between an LLM that guesses and an LLM that knows. Our system retrieves the exact passages from your books, then generates answers with page-level citations. It's like having a research assistant who has read every book in your library and can quote exact pages."