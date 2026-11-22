import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embedded" / "full"
MODEL_NAME = "BAAI/bge-m3"


def load_embeddings():
    embeddings = np.load(
        EMBEDDINGS_DIR / "embeddings.npy",
        allow_pickle=False
    )

    metadata = []

    with open(
        EMBEDDINGS_DIR / "metadata.jsonl",
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            metadata.append(json.loads(line))

    # Check that vectors and metadata correspond
    if embeddings.ndim != 2 or embeddings.shape[1] != 1024:
        raise ValueError("Unexpected embedding dimensions")

    if len(embeddings) != len(metadata):
        raise ValueError("Embeddings and metadata do not match")

    return embeddings, metadata


def search(query, model, embeddings, metadata, top_k=5):
    if not query.strip():
        raise ValueError("Query cannot be empty")

    # Generate an embedding for the query
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Cosine similarity using normalized vectors
    scores = embeddings @ query_embedding

    # Find indices of the highest scores
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):
        chunk = metadata[idx]

        results.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "heading": chunk["heading"],
            "score": float(scores[idx]),
            "text": chunk["text"]
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    # Load the document embedding index
    embeddings, metadata = load_embeddings()
    print(f"Loaded {len(embeddings)} document embeddings")

    # Load BGE-M3
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    # Perform semantic retrieval
    results = search(
        args.query,
        model,
        embeddings,
        metadata,
        top_k=args.top_k
    )

    for result in results:
        print("-" * 50)
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Filename: {result['filename']}")
        print(f"Heading: {result['heading']}")
        print(f"Chunk ID: {result['chunk_id']}")


if __name__ == "__main__":
    main()