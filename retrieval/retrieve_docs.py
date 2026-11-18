import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = PROJECT_ROOT / "preprocessing" / "docs_md_v3" / "chunks.jsonl"


# Load the chunks
chunks_path = Path(CHUNKS_PATH)

chunks = []
filtered_chunks = []


with open(chunks_path, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

#Filter the chunks
for chunk in chunks:
    if "Du får den bedste oplevelse" not in chunk["text"]:
        filtered_chunks.append(chunk)

print(f"Loaded {len(filtered_chunks)} chunks")

# Tokenization function
def tokenize(text):
    text = text.lower()
    text = re.findall(r"\b\w+\b", text)
    return text



# Tokenize chunks INCLUDING heading context
tokenized_corpus = []

for chunk in filtered_chunks:
    heading = chunk["heading"] or ""
    text = chunk["text"]

    combined_text = f"{heading}\n{text}"

    tokens = tokenize(combined_text)
    tokenized_corpus.append(tokens)

# Create BM25 index
bm25 = BM25Okapi(tokenized_corpus)

print("BM25 index created!")


def search(query, top_k=5):
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for i in ranked_indices[:top_k]:
        if scores[i] <= 0:
            continue

        results.append({
            "chunk_id": filtered_chunks[i]["chunk_id"],
            "filename": filtered_chunks[i]["filename"],
            "score": float(scores[i]),
            "text": filtered_chunks[i]["text"]
        })

    return results


def find_document_rank(query, target_filename):
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    for rank, i in enumerate(ranked_indices, start=1):
        if filtered_chunks[i]["filename"] == target_filename:
            return rank, float(scores[i])

    return None


