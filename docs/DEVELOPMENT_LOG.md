# Development Log – RAG Retrieval Pipeline

This document describes the development milestones, technical decisions, and preliminary experimental results of the document preprocessing and retrieval pipeline. Dates refer to commit dates, which may differ from when the code was written.

## October 20, 2026 – Initial Document Conversion

**Commit:** `27ad5b8`

Implemented the first version of the document conversion pipeline.

The purpose was to standardize heterogeneous IT documentation into a format suitable for subsequent processing and retrieval.

## November 18, 2026 – Preprocessing and BM25 Baseline

**Commit:** `2ec3825`

### Document preprocessing
- Conversion of PDF/DOCX/ODT source documents into Markdown (PPTX is enumerated but conversion has not been verified).
- Image extraction and filtering based on image dimensions.
- Unique image identifiers and metadata.
- Markdown text cleaning.

### Chunking
- Heading-based document segmentation.
- Paragraph-aware chunking with a soft target of 350 words (not a hard limit; a few chunks exceed it).
- Table-aware chunk splitting with repeated header rows.
- Preservation of document references and chunk identifiers.
- Analysis of chunk size distributions (255 documents, 3,230 raw chunks).

### Lexical retrieval
- BM25 indexing using `rank-bm25`.
- Lowercase regex-based tokenization (no stemming, lemmatization or stopword removal).
- Indexing of combined chunk headings and chunk text.
- Ranking document chunks based on lexical similarity.

### Evaluation
- Initial synthetic retrieval test cases.
- Implementation of reciprocal rank and top-k hit metrics.

## November 22, 2026 – Semantic Retrieval

**Commit:** `95a761a`

### Chunk embeddings
- Integration of the multilingual `BAAI/bge-m3` embedding model (dense embeddings only; BGE-M3's sparse and multi-vector modes are not used).
- GPU-based batch embedding generation.
- Generation of 3,224 chunk embeddings with 1,024 dimensions.
- L2 normalization and persistent NumPy storage.
- JSONL metadata aligned with embedding rows (contains full chunk text; treated as confidential and excluded from Git).

### Semantic retrieval
- Query embedding generation.
- Cosine similarity search using normalized embeddings (exact dot product, no approximate index).
- Ranking and retrieval of the most similar document chunks.

### Data filtering
- Exclusion of chunks from documents with sensitive filename markers (1 chunk) and of a known PDF boilerplate text (5 chunks), leaving 3,224 of 3,230 chunks.
- Identical filtering and chunk order in BM25 and semantic retrieval.
- Filename filtering does not guarantee that the remaining content is safe to process. Previously generated Markdown may still contain the excluded document; source-level cleanup is outstanding.

## November 22, 2026 – Preliminary Retrieval Comparison

**Commit:** `88bb6c1` (evaluation script committed November 24, 2026)

Evaluated BM25 and BGE-M3 on the same ordered collection of 3,224 chunks using six synthetic queries (`evaluation/compare_retrievers.py`). The script aborts if the two retrievers do not index identical chunk IDs in identical order.

| Metric | BM25 | BGE-M3 |
|---|---:|---:|
| MRR@5 | 0.472 | 1.000 |
| Hit@5 | 0.667 | 1.000 |

BGE-M3 retrieved a chunk from the expected document at rank 1 for all six queries. BM25 missed the expected document entirely for the two queries without product names and performed worse on queries with limited lexical overlap.

**Limitations:**
- Six queries, covering primarily PDF/Adobe-related problems and only two expected documents.
- The queries were written with knowledge of the target documents.
- Relevance is judged by filename only: a hit does not guarantee that the retrieved chunk contains the solution, and other relevant documents may exist.
- The results are preliminary and not representative of performance on real IT tickets.

## Next Steps

- Establish a larger, manually annotated evaluation dataset based on real ticket descriptions (pooled candidates, graded relevance).
- Compare lexical and semantic retrieval across different IT issue categories.
- Evaluate hybrid retrieval using Reciprocal Rank Fusion (RRF) once the evaluation protocol exists.
- Integrate retrieval with the answer generation component once its interface is agreed.