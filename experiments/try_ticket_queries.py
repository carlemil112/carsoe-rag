import json
import random
import re

from sentence_transformers import SentenceTransformer

from retrieval import retrieve_docs as bm25
from retrieval.retrieve_semantic import MODEL_NAME, load_embeddings, search as semantic_search

SPLIT_PATH = "evaluation/private/split.json"
TICKETS_PATH = "Preprocessing/output/rag_docs.jsonl"
LLM_PATH = "Data/topdesk_annotated_llm.json"
N_TICKETS = 20
TOP_K = 5
SEED = 42
SKIP_ANSWERS = {"", "-", "ja", "nej", "yes", "no"}
PLACEHOLDERS = r"\[(?:PERSON_\w+|EMAIL|FJERNET)\]"
CLEAN = False


def is_answer_line(line):
    #Form answers start with '- ' or are just '-' (empty answer).
    return line == "-" or line.startswith("- ")

# query cleaning as forms are messing with BM25 and BGE-M3
def clean_query(text):
    # Remove placeholders like [EMAIL], [PERSON_6], [FJERNET]
    text = re.sub(PLACEHOLDERS, " ", text)

    lines = [line.strip() for line in text.splitlines()]

    # Free-text ticket: keep everything, just join non-empty lines
    if not any(is_answer_line(line) for line in lines):
        return " ".join(line for line in lines if line)

    # Form ticket: drop question lines, keep meaningful answers
    kept = []
    for i, line in enumerate(lines):
        # Find the next non-empty line
        next_line = next((l for l in lines[i + 1:] if l), "")

        # A question is a non-answer line directly followed by an answer
        if not is_answer_line(line) and is_answer_line(next_line):
            continue

        # Strip the leading "-" from answers
        if is_answer_line(line):
            line = line[1:].strip()

        # Drop empty and uninformative answers
        if line.lower() in SKIP_ANSWERS:
            continue

        kept.append(line)

    return " ".join(kept)


# get dev ticket numbers
with open(SPLIT_PATH, encoding="utf-8") as f:
    split = json.load(f)
dev_numbers = split["dev"]

# load pseudomized tickets from "rag_docs.jsonl"
tickets = {}
with open(TICKETS_PATH, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        ticket = json.loads(line)
        tickets[ticket["number"]] = ticket


# load LLM annotations (category, solution_summary)
with open(LLM_PATH, encoding="utf-8") as f:
    data = json.load(f)
annotations = {}
for number, ticket in data.items():
        if "annotation" in ticket:
            annotations[number] = ticket["annotation"]



# Check if loaded correctly
print(len(dev_numbers))
print(len(tickets))
print(len(annotations))


# Choose tickets
rng = random.Random(SEED)
selected = rng.sample(dev_numbers, N_TICKETS)

embeddings, metadata = load_embeddings()
model = SentenceTransformer(MODEL_NAME, device="cuda")
print(f"Loaded {len(metadata)} chunk embeddings")


for number in selected:
    ticket = tickets[number]
    annotation = annotations[number]

    problem = ticket["problem"].strip()
    query = clean_query(problem) if CLEAN else problem
    
    print(f"QUERY USED: {query[:150]}")
    solution = annotation.get("solution_summary", "")

    print("=" * 80)
    print(f"Ticket: {number}   Category: {annotation['category']}")
    print(f"PROBLEM: {problem[:150]}")
    print(f"QUERY USED: {query[:150]}")
    print(f"IT SOLUTION (for judging only): {solution[:150]}")

    bm25_results = bm25.search(query, top_k=TOP_K)
    semantic_results = semantic_search(query, model, embeddings, metadata, top_k=TOP_K)

    for method, results in [("BM25", bm25_results), ("BGE-M3", semantic_results)]:
        print(f"\n  --- {method} ---")
        for rank, result in enumerate(results, start=1):
            print(f"  {rank}. [{result['score']:.3f}] {result['filename']}")


