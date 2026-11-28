"""Single entry point for retrieval, used by the generator and evaluation.

Usage:
    from retrieval.retriever import Retriever
    retriever = Retriever(method="bge-m3")       # load once
    chunks = retriever.retrieve(query, top_k=5)  # per query

BGE-M3 requires the GPU container. BM25 runs on CPU.
"""

from retrieval import retrieve_docs as bm25


class Retriever:
    def __init__(self, method="bge-m3", device="cuda"):
        if method not in ("bm25", "bge-m3"):
            raise ValueError(f"Unknown method: {method}")
        self.method = method

        if method == "bge-m3":
            # Imported here so BM25 works without the GPU environment
            from sentence_transformers import SentenceTransformer
            from retrieval.retrieve_semantic import MODEL_NAME, load_embeddings, search
            self._semantic_search = search
            self.embeddings, self.metadata = load_embeddings()
            self.model = SentenceTransformer(MODEL_NAME, device=device)

    def retrieve(self, query, top_k=5):
        """Return a ranked list of chunks with source information."""
        if self.method == "bm25":
            results = bm25.search(query, top_k=top_k)
        else:
            results = self._semantic_search(query, self.model, self.embeddings, self.metadata, top_k=top_k)
        return [
            {
                "rank": rank,
                "chunk_id": r["chunk_id"],
                "filename": r["filename"],
                "heading": r.get("heading"),
                "text": r["text"],
                "score": float(r["score"]),
            }
            for rank, r in enumerate(results, start=1)
        ]