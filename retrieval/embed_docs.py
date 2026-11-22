import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "preprocessing" / "docs_md_v3" / "chunks.jsonl"

MODEL_NAME = "BAAI/bge-m3"


def load_chunks():

    chunks = []
    
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)

            filename = chunk["filename"].lower()

            # Exclude documents marked as sensitive
            if any(term in filename for term in
                   ("sensitiv", "fortrolig", "confidential")):
                continue

            # Filter unwanted PDF boilerplate
            if "Du får den bedste oplevelse" in chunk["text"]:
                continue

            chunks.append(chunk)

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "embedded" / "full"
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")

    # 1. Load document chunks
    chunks = load_chunks()

    if args.limit is not None:
        chunks = chunks[:args.limit]

    if not chunks:
        raise RuntimeError("No document chunks found")

    print(f"Loaded {len(chunks)} chunks")

    # 2. Prepare text for embedding
    texts = []

    for chunk in chunks:
        heading = chunk["heading"] or ""
        text = chunk["text"]

        combined_text = f"{heading}\n{text}"
        texts.append(combined_text)

    # 3. Load embedding model
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda"
    )

    # 4. Generate embeddings
    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print(f"Embeddings shape: {embeddings.shape}")

    # 5. Save embeddings
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", embeddings)

    # 6. Save metadata in the same order
    with open(output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    print(f"Saved {len(embeddings)} embeddings to {output_dir}")


if __name__ == "__main__":
    main()