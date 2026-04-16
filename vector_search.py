"""
FAISS vector search over the Wikivoyage index.

On Apple Silicon (Python 3.13), loading torch + faiss-cpu in the same process
as other heavy libs causes a segfault. We work around it by running the
SentenceTransformer encode in a subprocess via multiprocessing, while keeping
the FAISS index in the main process (pure C, no torch).
"""

import os
import pickle
import subprocess
import json
import sys
import tempfile
from pathlib import Path
from functools import lru_cache

import numpy as np
import faiss
from langsmith import traceable

INDEX_PATH    = Path(__file__).parent / "vectorstore/index.faiss"
METADATA_PATH = Path(__file__).parent / "vectorstore/metadata.pkl"
EMBED_MODEL   = "nomic-ai/nomic-embed-text-v1.5"
TOP_K         = 5
FETCH_K       = 100


# ---------------------------------------------------------------------------
# FAISS index + metadata — loaded once, no torch involved
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_index_and_metadata():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Vector store not found at {INDEX_PATH}. "
            "Run: conda run -n scalablellm python build_vectorstore.py"
        )
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata: list[dict] = pickle.load(f)
    return index, metadata


# ---------------------------------------------------------------------------
# Embedding via subprocess — isolates torch from the main process
# ---------------------------------------------------------------------------

_EMBED_SCRIPT = """
import sys, json, numpy as np
from sentence_transformers import SentenceTransformer

query = sys.argv[1]
model = SentenceTransformer("{model}", trust_remote_code=True)
vec = model.encode(["search_query: " + query], normalize_embeddings=True)
print(json.dumps(vec[0].tolist()))
""".format(model=EMBED_MODEL)


def _embed_query(query: str) -> np.ndarray:
    """Run embedding in a subprocess, return float32 numpy vector."""
    python = sys.executable
    result = subprocess.run(
        [python, "-c", _EMBED_SCRIPT, query],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Embedding subprocess failed:\n{result.stderr}")
    vec = json.loads(result.stdout.strip())
    return np.array(vec, dtype=np.float32).reshape(1, -1)


# ---------------------------------------------------------------------------
# Metadata filter (MongoDB-style)
# ---------------------------------------------------------------------------

def _matches_filter(meta: dict, filter: dict | None) -> bool:
    if not filter:
        return True
    if "$and" in filter:
        return all(_matches_filter(meta, sub) for sub in filter["$and"])
    if "$or" in filter:
        return any(_matches_filter(meta, sub) for sub in filter["$or"])
    for field, condition in filter.items():
        value = meta.get(field)
        if isinstance(condition, dict):
            if "$eq" in condition and value != condition["$eq"]:
                return False
            if "$in" in condition and value not in condition["$in"]:
                return False
        else:
            if value != condition:
                return False
    return True


# ---------------------------------------------------------------------------
# Public search function
# ---------------------------------------------------------------------------

@traceable(name="search_destinations", run_type="tool")
def search_destinations(
    query: str,
    filter: dict | None = None,
    top_k: int = TOP_K,
    fetch_k: int = FETCH_K,
) -> list[dict]:
    """
    Semantic search over Wikivoyage chunks.
    Returns list of {passage, destination, section, source, score}.
    """
    index, metadata = _load_index_and_metadata()

    query_vec = _embed_query(query)
    faiss.normalize_L2(query_vec)

    k = min(fetch_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        meta = metadata[idx]
        if not _matches_filter(meta, filter):
            continue
        results.append({
            "passage":     meta["text"],
            "destination": meta.get("city", ""),
            "section":     meta.get("section_type", ""),
            "source":      meta.get("source", ""),
            "score":       float(score),
        })
        if len(results) >= top_k:
            break

    return results
