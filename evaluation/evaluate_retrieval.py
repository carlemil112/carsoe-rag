import json
from pathlib import Path
from retrieval.retrieve_docs import find_document_rank

# Load evaluation test cases
test_cases_path = Path(__file__).resolve().parent / "test_cases.json"

with open(test_cases_path, "r", encoding="utf-8") as f:
    test_cases = json.load(f)


# Run test query and calculate MRR and Recall@5
reciprocal_ranks = []
recall_scores = []

for case in test_cases:
    query = case["query"]
    target_filename = case["target_filename"]

    result = find_document_rank(query, target_filename)

    rr = 0
    recall = 0

    if result is not None:
        rank, score = result

        print(f"Query: {query}")
        print(f"Rank: {rank}")
        print(f"Score: {score:.2f}")

        if score > 0:
            rr = 1 / rank
            if rank <= 5:
                recall = 1
        

    reciprocal_ranks.append(rr)

    recall_scores.append(recall)

    print(f"Reciprocal Rank: {rr:.3f}")
    print(f"Recall@5: {recall}")
    print("-" * 50)


# Calculate MRR
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
print(f"MRR: {mrr:.3f}")

# Calculate Recall@5
recall_5 = sum(recall_scores) / len(recall_scores)
print(f"Recall@5: {recall_5:.3f}")