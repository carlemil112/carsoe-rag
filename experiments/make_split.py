# experiments/make_split.py
import json
import random
from collections import defaultdict

TEST_PER_CATEGORY = 8   # ~80 tickets total across ~10 categories
SEED = 42

with open("Data/topdesk_annotated_llm.json", encoding="utf-8") as f:
    llm = {n: t["annotation"] for n, t in json.load(f).items()
           if isinstance(t.get("annotation"), dict)}
with open("Preprocessing/output/rag_docs.jsonl", encoding="utf-8") as f:
    rag = {json.loads(l)["number"] for l in f if l.strip()}

eligible = {
    n: a for n, a in llm.items()
    if n in rag
    and a["problem_source"] == "Tekst"
    and a["problem_clear"] == "Ja"
    and a["solution"] == "Ja"
    and a["category"] != "HR"
    and not a.get("llm_inconsistent") and not a.get("llm_invalid_pair")
}

by_category = defaultdict(list)
for n, a in sorted(eligible.items()):
    by_category[a["category"]].append(n)

rng = random.Random(SEED)
test = []
for cat, numbers in sorted(by_category.items()):
    test += rng.sample(numbers, min(TEST_PER_CATEGORY, len(numbers)))

dev = sorted(set(eligible) - set(test))
split = {"seed": SEED, "test": sorted(test), "dev": dev}

with open("evaluation/private/split.json", "w", encoding="utf-8") as f:
    json.dump(split, f, indent=2)

print(f"Eligible: {len(eligible)}  Test: {len(test)}  Dev: {len(dev)}")