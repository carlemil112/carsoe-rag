import json
from pathlib import Path
from collections import Counter
import re

BASE_DIR = Path(__file__).resolve().parent
chunks_path = BASE_DIR / "docs_md_v3" / "chunks.jsonl"

chunks = []

# Load chunks
with open(chunks_path, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))


# Count words in each chunk
word_counts = []

for chunk in chunks:
    words = chunk["text"].split()
    words_amount = len(words)
    word_counts.append(words_amount)


# Basic statistics
total = len(word_counts)

print("\n--- BASIC STATISTICS ---")
print(f"Total chunks: {total}")
print(f"Minimum words: {min(word_counts)}")
print(f"Maximum words: {max(word_counts)}")
print(f"Average words: {sum(word_counts) / total:.2f}")


# Categorize chunks by size
long_chunks = []
short_chunks = []
empty_chunks = []

for chunk in chunks:
    word_count = len(chunk["text"].split())

    if word_count > 350:
        long_chunks.append(chunk)

    elif word_count == 0:
        empty_chunks.append(chunk)

    elif word_count < 20:
        short_chunks.append(chunk)


# Print chunk size distribution
print("\n--- CHUNK SIZE DISTRIBUTION ---")

print(f"Chunks over 350 words: {len(long_chunks)} "
      f"({len(long_chunks) / total * 100:.2f}%)")

print(f"Chunks between 1 and 19 words: {len(short_chunks)} "
      f"({len(short_chunks) / total * 100:.2f}%)")

print(f"Empty chunks: {len(empty_chunks)} "
      f"({len(empty_chunks) / total * 100:.2f}%)")


# Inspect empty chunks
print("\n--- EMPTY CHUNK EXAMPLES ---")

for chunk in empty_chunks[:10]:
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Filename: {chunk['filename']}")
    print(f"Heading: {chunk['heading']}")
    print(f"Text: {chunk['text']!r}")
    print(f"Image IDs: {chunk.get('image_ids', [])}")
    print("-" * 50)


# Find documents with the most empty chunks
print("\n--- DOCUMENTS WITH MOST EMPTY CHUNKS ---")

empty_by_document = Counter(
    chunk["filename"] for chunk in empty_chunks
)

for filename, count in empty_by_document.most_common(10):
    print(f"{filename}: {count} empty chunks")

print(f"Affected documents: {len(empty_by_document)}")


# Sort long chunks by word count
long_chunks.sort(
    key=lambda chunk: len(chunk["text"].split()),
    reverse=True
)


# Inspect the five largest chunks
print("\n--- LARGEST CHUNKS ---")

for chunk in long_chunks:
    text = chunk["text"]

    whitespace_count = len(text.split())
    regex_count = len(re.findall(r"\b\w+\b", text))

    print(f"Filename: {chunk['filename']}")
    print(f"Heading: {chunk['heading']}")
    print(f"Whitespace count: {whitespace_count}")
    print(f"Regex count: {regex_count}")
    print(f"Text preview: {text[:200]}")
    print("-" * 50)


# Inspect short chunks
print("\n--- SHORT CHUNK EXAMPLES ---")

for chunk in short_chunks[:10]:
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Filename: {chunk['filename']}")
    print(f"Heading: {chunk['heading']}")
    print(f"Word count: {len(chunk['text'].split())}")
    print(f"Text: {chunk['text']!r}")
    print(f"Image IDs: {chunk.get('image_ids', [])}")
    print("-" * 50)