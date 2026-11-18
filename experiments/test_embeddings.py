import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "BAAI/bge-m3",
    device="cuda"
)

texts = [
    "Jeg kan ikke åbne PDF filer",
    "I cannot open PDF files",
    "Repair Adobe Acrobat installation",
    "Hvordan ændrer jeg min adgangskode?"
]

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

print(embeddings.shape)

# cosine similarity between queries
query_embedding = embeddings[0]

for i, text in enumerate(texts):
    similarity = np.dot(query_embedding, embeddings[i])

    print(f"{text}: {similarity:.4f}")