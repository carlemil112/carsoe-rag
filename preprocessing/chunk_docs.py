import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
folder = BASE_DIR / "docs_md_v3"
docs = []

# Read documents and save them as dictionaries
for file in folder.rglob("*.md"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    document = {
        "filename": file.relative_to(folder).as_posix(),
        "text": text
    }

    docs.append(document)


# Check whether a line is a Markdown heading
def is_heading(line):
    for i in range(1, 7):
        prefix = "#" * i + " "

        if line.startswith(prefix):
            return True

    return False


# Splits tables between rows, and repeat column heading
def split_table(table, max_words=350):
    rows = table.splitlines()

    header = []

    # Detect Markdown table header
    if len(rows) >= 3 and re.fullmatch(
        r"\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*", rows[1]
    ):
        header = rows[:2]
        rows = rows[2:]

    chunks = []
    current_rows = header.copy()

    header_words = len(" ".join(header).split())
    current_words = header_words

    for row in rows:
        row_words = len(row.split())

        if (
            len(current_rows) > len(header)
            and current_words + row_words > max_words
        ):
            chunks.append("\n".join(current_rows))
            current_rows = header.copy()
            current_words = header_words

        current_rows.append(row)
        current_words += row_words

    if len(current_rows) > len(header):
        chunks.append("\n".join(current_rows))

    return chunks


# Split a section into smaller chunks
def chunk_section(section, max_words=350):
    text = section["text"]

    # Remove repeated dot leaders
    text = re.sub(r"\.(?:[ \t]*\.){4,}", " ", text)

    # Skip empty sections
    if not text.strip():
        return []

    words = text.split()

    if len(words) <= max_words:
        return [text]

    # Split text into paragraphs
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph_words = len(paragraph.split())

        # Split oversized Markdown tables
        if paragraph_words > max_words and all(
            line.strip().startswith("|")
            for line in paragraph.splitlines()
        ):
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_words = 0

            chunks.extend(split_table(paragraph, max_words))
            continue

        if current_chunk and current_words + paragraph_words > max_words:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_words = 0

        current_chunk.append(paragraph)
        current_words += paragraph_words

    # Save the final chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# Split documents into sections based on headings
for document in docs:
    sections = []
    current_heading = None
    current_text = []

    for line in document["text"].splitlines():

        if is_heading(line):

            # Save previous section
            if current_heading is not None or current_text:
                section = {
                    "heading": current_heading,
                    "text": "\n".join(current_text).strip()
                }
                sections.append(section)

            # Start a new section
            current_heading = line
            current_text = []

        else:
            current_text.append(line)

    # Save the final section
    if current_heading is not None or current_text:
        section = {
            "heading": current_heading,
            "text": "\n".join(current_text).strip()
        }
        sections.append(section)

    document["sections"] = sections

    print(f"{document['filename']}: {len(sections)} sections")


# Create chunks from all sections
total_chunks = 0

for document in docs:
    chunks = []

    for section in document["sections"]:
        section_chunks = chunk_section(section)

        for chunk_text in section_chunks:
            chunk_id = f"{len(chunks) + 1:04d}"

            image_ids = re.findall(
                r"\[image:\s*([a-fA-F0-9]+_\d+)\]",
                chunk_text
            )

            chunk = {
                "chunk_id": f"{document['filename']}::{chunk_id}",
                "filename": document["filename"],
                "heading": section["heading"],
                "text": chunk_text,
                "image_ids": image_ids
            }

            chunks.append(chunk)

    document["chunks"] = chunks
    total_chunks += len(chunks)

    print(f"{document['filename']}: {len(chunks)} chunks")

print(f"\nTotal documents: {len(docs)}")
print(f"Total chunks: {total_chunks}")


# Save all chunks to a JSONL file
output_file = folder / "chunks.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for document in docs:
        for chunk in document["chunks"]:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"Chunks saved to: {output_file}")