import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from retrieval import retrieve_docs as bm25
from retrieval.retrieve_semantic import (
    MODEL_NAME,
    load_embeddings,
    search as semantic_search
)


TEST_CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"
TOP_K = 5


def find_relevant_rank(results, target_filename):
    """Return the rank of the first chunk from the expected document."""
    for rank, result in enumerate(results, start=1):
        if result["filename"] == target_filename:
            return rank

    return None


def main():
    # Load test cases
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Load saved document embeddings and metadata
    embeddings, metadata = load_embeddings()

    # Verify both retrievers use exactly the same chunks
    bm25_ids = [chunk["chunk_id"] for chunk in bm25.filtered_chunks]
    semantic_ids = [chunk["chunk_id"] for chunk in metadata]

    if bm25_ids != semantic_ids:
        raise ValueError(
            "BM25 and semantic retrieval do not use the same chunks. "
            "Regenerate embeddings or check the filtering."
        )

    print(f"Both retrievers use {len(metadata)} identical chunks.")

    # Load BGE-M3 once
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    metrics = {
        "BM25": {"reciprocal_ranks": [], "hits": []},
        "BGE-M3": {"reciprocal_ranks": [], "hits": []}
    }

    for case in test_cases:
        query = case["query"]
        target_filename = case["target_filename"]

        # Run both retrieval methods
        bm25_results = bm25.search(query, top_k=TOP_K)

        semantic_results = semantic_search(
            query,
            model,
            embeddings,
            metadata,
            top_k=TOP_K
        )

        print("\n" + "=" * 60)
        print(f"Query: {query}")

        for method, results in [
            ("BM25", bm25_results),
            ("BGE-M3", semantic_results)
        ]:
            rank = find_relevant_rank(results, target_filename)

            # MRR@5: reciprocal rank of the first relevant result
            rr = 1 / rank if rank is not None else 0

            # Hit@5: whether at least one relevant chunk was retrieved
            hit = int(rank is not None)

            metrics[method]["reciprocal_ranks"].append(rr)
            metrics[method]["hits"].append(hit)

            print(
                f"{method}: "
                f"Rank@{TOP_K} = {rank}, "
                f"RR = {rr:.3f}, "
                f"Hit = {hit}"
            )

    # Calculate average metrics
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for method, values in metrics.items():
        mrr = sum(values["reciprocal_ranks"]) / len(test_cases)
        hit_rate = sum(values["hits"]) / len(test_cases)

        print(f"\n{method}")
        print(f"MRR@{TOP_K}: {mrr:.3f}")
        print(f"Hit@{TOP_K}: {hit_rate:.3f}")


if __name__ == "__main__":
    main()