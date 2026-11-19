# DAKI5_GR1 – IT Support RAG

A retrieval-augmented generation system for internal IT support.

## Current implementation
- Document conversion and image extraction
- Structure-aware and table-aware chunking
- BM25 retrieval
- Retrieval evaluation
- Initial BGE-M3 embedding experiments

## In development
- Semantic retrieval
- Hybrid retrieval
- LLM-based answer generation
- Extended evaluation


## Current pipeline architecture

```mermaid
flowchart TD
    A["Source documents"] --> B["Document conversion"]
    B --> C["Markdown conversion"]
    B --> D["Image extraction"]

    D --> E["Image size filtering"]
    E --> F["Image IDs and metadata"]

    C --> G["Markdown cleaning"]
    F --> G

    G --> H["Markdown documents"]
    H --> I["Heading-based section splitting"]
    I --> J["Paragraph and table-aware chunking"]
    J --> K["chunks.jsonl"]

    K --> L["Chunk analysis"]
    K --> M["Filter unwanted chunks"]
    M --> N["Combine heading and text"]
    N --> O["Lowercase and regex tokenization"]
    O --> P["BM25 index"]

    Q["User query"] --> R["Lowercase and regex tokenization"]
    R --> S["BM25 scoring"]
    P --> S

    S --> T["Ranked document chunks"]

    U["Synthetic test cases"] --> V["Retrieval evaluation"]
    T --> V
    V --> W["MRR and Recall@5"]
```
