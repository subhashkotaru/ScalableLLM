"""
Smoke test for the vector store.
Loads model once, runs two queries, prints results.
"""
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH    = "vectorstore/index.faiss"
METADATA_PATH = "vectorstore/metadata.pkl"
EMBED_MODEL   = "nomic-ai/nomic-embed-text-v1.5"

print("Loading index...")
index = faiss.read_index(INDEX_PATH)
print(f"  {index.ntotal} vectors, dim={index.d}")

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Loading model...")
model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)

def search(query, top_k=3, filter_city=None):
    vec = model.encode([f"search_query: {query}"], normalize_embeddings=True)
    vec = np.array(vec, dtype=np.float32)
    faiss.normalize_L2(vec)
    scores, indices = index.search(vec, 100)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        meta = metadata[idx]
        if filter_city and meta.get("city") not in filter_city:
            continue
        results.append((score, meta))
        if len(results) >= top_k:
            break
    return results

print("\n--- seafood restaurants Portland Maine ---")
for score, meta in search("seafood restaurants Portland Maine"):
    print(f"  [{score:.3f}] {meta['city']} / {meta.get('section_type','?')} — {meta['text'][:120]}")

print("\n--- hiking Maine (filtered) ---")
for score, meta in search("hiking outdoor activities", filter_city=["Maine", "Greater Portland"]):
    print(f"  [{score:.3f}] {meta['city']} / {meta.get('section_type','?')} — {meta['text'][:120]}")
