# UI/UX & Production Features - Streamlit Interface

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT APP STRUCTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SESSION STATE                                                  │
│  ├── messages          → Chat history with full metadata       │
│  ├── pending_question  → From suggestion pills                 │
│  ├── show_book_manager → Sidebar toggle                        │
│  └── selected_query_type → Quick-selector state                │
│                                                                 │
│  SIDEBAR (Always open on desktop)                              │
│  ├── Knowledge Base Stats     → Chunks, PDFs, Chapters         │
│  ├── Upload PDFs             → Drag & drop, multi-file         │
│  ├── Search Scope            → All Books / Single PDF          │
│  ├── Dynamic K Toggle        → Auto vs Manual                  │
│  ├── PDF Files List          → Index status per file           │
│  ├── Sync/Index Button       → Progress bar + callback         │
│  ├── Book Manager            → View/Delete per book            │
│  └── Clear Buttons           → Chat & Database                 │
│                                                                 │
│  MAIN AREA                                                    │
│  ├── Title + Subtitle                                           │
│  ├── Suggestion Pills       → Dynamic per query type           │
│  ├── Query Type Selector    → 5 buttons (LISTING...FACTUAL)   │
│  ├── Chat History           → Streaming + metadata pills       │
│  │   ├── User messages                                          │
│  │   └── Assistant messages                                     │
│  │       ├── Retrieval metadata pill                            │
│  │       ├── Cross-book tabs (COMPARISON/TIMELINE)             │
│  │       ├── Streaming answer                                   │
│  │       └── Expandable sources (grouped by book)              │
│  └── Chat Input                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key UI Components

### 1. Retrieval Metadata Pills (Transparency)
Every answer shows exactly how it was retrieved:

```
🎯 Dynamic K: 20/20 chunks | ⚡ Strategy: entity_comparison | 
🎯 Confidence: High (67%) | ⏱️ Retrieval: 145ms
📚 Cross-book (2 books) | entity-badge: Harry Potter | Mowgli
✅ High Confidence — Coverage: 100% (2/2 books) • Retrieval Confidence: 67%
```

**Visual Coding**:
- Confidence: Green (>70%) / Yellow (>40%) / Red
- Distance: Green (<0.4) / Yellow (<0.55) / Red
- Query type badges: Color-coded per type

### 2. Cross-Book View (Comparison/Timeline)
For COMPARISON and TIMELINE queries, tabs show per-book breakdown:

```
📊 Cross-Book View
┌─────────────────┬─────────────────┐
│ 📘 Harry Potter │ 📘 Jungle Book  │
├─────────────────┼─────────────────┤
│ Page 12 ● 0.234 │ Page 5  ● 0.198 │
│ "Harry learned.."│ "Mowgli met..." │
├─────────────────┼─────────────────┤
│ Page 45 ● 0.312 │ Page 18 ● 0.267 │
│ "He faced..."   │ "The tiger..."  │
└─────────────────┴─────────────────┘
```

### 3. Expandable Source Citations
```
📜 View Retrieved Passages (20 chunks)
📊 Book Distribution:
  ████████████ Harry Potter (10 chunks, 50%)
  ████████████ Jungle Book  (10 chunks, 50%)

📘 Harry Potter (10 chunks)
  ┌─────────────────────────────────────┐
  │ Page 12        Distance: 0.234 ✓   │
  │ "Harry learned he was a wizard..."  │
  └─────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │ Page 45        Distance: 0.312 ✓   │
  │ "He faced Voldemort in the..."      │
  └─────────────────────────────────────┘

📘 Jungle Book (10 chunks)
  ...
```

### 4. Suggestion Pills (Discovery)
Dynamic examples per query type:
```
💡 Suggested Questions
[📋 List all characters] [⚖️ Compare protagonists] [📅 Character timeline]
[🎭 Explain theme] [❓ What is the capital?]
```

### 5. Query Type Quick-Selector
Force a specific strategy + see examples:
```
🎯 Query Type Quick-Select
[📋 Listing] [⚖️ Comparison] [📅 Timeline] [🎭 Thematic] [❓ Factual]

Click "Comparison" → Shows:
• Compare the protagonist and antagonist
• Contrast the magic systems in the books
• How do Voldemort and Shere Khan compare?
```

---

## Book Management UI

### Sidebar PDF List
```
📁 PDF Files in ./books
┌────────────────────────────────────────┐
│ 📄 harry_potter.pdf          400 pages │
│    ✅ Indexed                        │
│    7 chapters: Harry Potter 1-7     │
├────────────────────────────────────────┤
│ 📄 classics_compilation.pdf  800 pages │
│    ✅ Indexed                        │
│    3 chapters: Jungle Book, Dracula..│
├────────────────────────────────────────┤
│ 📄 new_book.pdf              200 pages │
│    ⏳ Not indexed                    │
└────────────────────────────────────────┘
```

### Book Manager (Expandable)
```
🗂️ Book Manager
┌────────────────────────────────────────┐
│ ▼ 📄 harry_potter.pdf (7 chapters)     │
│   ┌──────────────────────────────────┐ │
│   │ Harry Potter and the Sorcerer's  │ │
│   │ Chunks: 1,247 • Pages: 1-223    │ │
│   │ [🗑️ Delete]                      │ │
│   └──────────────────────────────────┘ │
│   ┌──────────────────────────────────┐ │
│   │ Harry Potter and the Chamber...  │ │
│   │ Chunks: 1,156 • Pages: 224-412  │ │
│   │ [🗑️ Delete]                      │ │
│   └──────────────────────────────────┘ │
├────────────────────────────────────────┤
│ [🗑️ Clear ALL Data] (with confirmation)│
└────────────────────────────────────────┘
```

---

## Ingestion UX

### Progress Tracking
```
⚡ Sync / Index Books into Vector DB
████████████████████████████████ 100%

[001/010] Extracting pages 1-223 for 'Harry Potter 1'...
[002/010] Generating embeddings for 1,247 chunks...
[003/010] [Harry Potter 1] Embedded 450/1247...
[004/010] Saving 1,247 chunks to Vector DB...
✅ Indexing complete!
```

### Idempotent Re-runs
- Already indexed books show "already indexed" → skipped
- Safe to click "Sync" multiple times
- Only new/changed books processed

---

## Streaming Answer Generation

### Token-by-Token Display
```python
answer_placeholder = st.empty()
full_answer = ""
for chunk in generate_answer_streaming(question, chunks, meta):
    full_answer += chunk
    answer_placeholder.markdown(full_answer + "▌")  # Cursor effect
answer_placeholder.markdown(full_answer)  # Final
```

### Type-Specific Prompts
Each query type gets specialized instructions:

| Type | Max Tokens | Special Instructions |
|------|-----------|---------------------|
| LISTING | 1800 | "Identify ALL items... group logically... cite every entry" |
| COMPARISON | 1700 | "Structured cross-book comparison... parallels, differences, arcs" |
| TIMELINE | 1600 | "Chronologically trace... Beginning/Middle/Climax/Resolution" |
| THEMATIC | 1500 | "Explore deeper meanings... symbolism, motifs, WHY and HOW" |
| FACTUAL | 1100 | "Answer directly... cite specifics... admit if not in excerpts" |

---

## CSS/Styling Highlights

### Custom Components (via `st.markdown(unsafe_allow_html=True)`)
- **Metric cards**: Gradient backgrounds, gold values
- **Source cards**: Hover effects, distance badges
- **Distribution bars**: Gradient progress bars per book
- **Query type pills**: Color-coded badges
- **Quality indicators**: Traffic light confidence

### Responsive Sidebar
```css
@media (min-width: 768px) {
    /* Hide collapse button */
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    
    /* Fixed width, always open */
    section[data-testid="stSidebar"] {
        width: 380px !important;
        min-width: 380px !important;
        max-width: 380px !important;
        transform: none !important;
    }
}
```

---

## Production Features

### Error Handling
```python
# Database connection
try:
    init_db()
except Exception as db_err:
    st.error(f"❌ Failed to connect: {db_err}")
    st.info("💡 Make sure PostgreSQL + pgvector is running")
    st.stop()

# API failures
try:
    response = client.chat.completions.create(...)
except Exception as e:
    st.error(f"Generation failed: {e}")
    # Graceful fallback message
```

### State Persistence
- Chat history in `st.session_state.messages`
- Survives reruns, not browser refresh
- Clear button for manual reset

### Security
- `.env` for secrets (gitignored)
- No hardcoded credentials
- SQL injection prevention via parameterized queries

---

## For Your Presentation

> "The UI is designed for transparency - every answer shows exactly which strategy was used, how many chunks from which books, confidence scores, and distance metrics. Users can see the retrieval logic, not just the answer. The cross-book tabs for comparisons are particularly powerful - you can instantly see how Harry Potter and Mowgli passages align side by side. The suggestion pills and query type selector make the advanced features discoverable without documentation."