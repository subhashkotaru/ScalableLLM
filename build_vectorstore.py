"""
Build the FAISS vector store from Wikivoyage PDFs in docs_data/.

Run once:
    conda run -n scalablellm python build_vectorstore.py

Outputs:
    vectorstore/index.faiss
    vectorstore/index.pkl     (LangChain FAISS metadata store)
"""

import os
import re
import json
import pickle
from pathlib import Path

import numpy as np
import pymupdf as fitz
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import faiss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOCS_DIR = Path("docs_data")
OUTPUT_DIR = Path("vectorstore")
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DIMENSION = 768
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 150   # discard header-only fragments below this
BATCH_SIZE = 16

HEADERS_TO_SPLIT_ON = [
    ("#",   "article_title"),
    ("##",  "section_type"),   # Understand, See, Do, Eat, Sleep, etc.
    ("###", "subsection"),     # Budget, Mid-range, Splurge
]

# ---------------------------------------------------------------------------
# PDF → Markdown-ish text
# ---------------------------------------------------------------------------

def pdf_to_text(pdf_path: Path) -> str:
    """Extract all text from a PDF, preserving page breaks as double newlines."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def text_to_markdown(text: str) -> str:
    """
    Heuristically promote Wikivoyage section headings to Markdown headers.
    Wikivoyage PDFs use ALL-CAPS or title-cased section names on their own line.
    """
    known_sections = {
        "Understand", "Get in", "Get around", "See", "Do",
        "Eat", "Drink", "Sleep", "Stay safe", "Go next",
        "Buy", "Connect", "Cope", "Work",
    }
    lines = text.split("\n")
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped in known_sections:
            out.append(f"## {stripped}")
        elif re.match(r"^[A-Z][a-z].*\[edit\]$", stripped):
            # e.g. "Portland Head Light[edit]" → subsection
            out.append(f"### {stripped.replace('[edit]', '').strip()}")
        else:
            out.append(line)
    return "\n".join(out)


def geo_metadata_from_filename(pdf_path: Path) -> dict:
    """Derive city/region from filename, e.g. 'Cape_Cod.pdf' → {city: 'Cape Cod', ...}"""
    name = pdf_path.stem.replace("_", " ").replace(" (1)", "").strip()
    return {
        "city":   name,
        "region": "New England / Mid-Atlantic",
        "source": pdf_path.name,
    }

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False,
)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
    length_function=len,
)


def chunk_wikivoyage_article(markdown_text: str, geo_metadata: dict) -> list[Document]:
    header_splits = md_splitter.split_text(markdown_text)
    final_chunks: list[Document] = []

    for doc in header_splits:
        if len(doc.page_content) > CHUNK_SIZE:
            final_chunks.extend(recursive_splitter.split_documents([doc]))
        else:
            final_chunks.append(doc)

    annotated = []
    for chunk in final_chunks:
        # Drop header-only fragments with no real content
        if len(chunk.page_content.strip()) < MIN_CHUNK_CHARS:
            continue
        chunk.metadata.update(geo_metadata)
        section = chunk.metadata.get("section_type", "General")
        city = geo_metadata.get("city", "")
        chunk.metadata["context_prefix"] = (
            f"From the '{section}' section of {city} travel guide"
        )
        annotated.append(chunk)

    return annotated

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build():
    OUTPUT_DIR.mkdir(exist_ok=True)
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs in {DOCS_DIR}/")

    # Step 1: parse + chunk all PDFs
    all_chunks: list[Document] = []
    for pdf_path in pdf_paths:
        text = pdf_to_text(pdf_path)
        md = text_to_markdown(text)
        geo = geo_metadata_from_filename(pdf_path)
        chunks = chunk_wikivoyage_article(md, geo)
        all_chunks.extend(chunks)
        print(f"  {pdf_path.name}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Step 2: embed
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)

    texts = [
        "search_document: "
        + chunk.metadata.get("context_prefix", "")
        + "\n"
        + chunk.page_content
        for chunk in all_chunks
    ]

    print(f"Encoding {len(texts)} chunks (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    # Step 3: build FAISS index (inner-product = cosine on normalised vecs)
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(embeddings)
    print(f"FAISS index: {index.ntotal} vectors, dim={DIMENSION}")

    # Step 4: save index + metadata
    faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))

    metadata = [chunk.metadata | {"text": chunk.page_content} for chunk in all_chunks]
    with open(OUTPUT_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # Also save a JSON summary for inspection
    summary = {
        "total_chunks": len(all_chunks),
        "embed_model": EMBED_MODEL,
        "dimension": DIMENSION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "sources": [p.name for p in pdf_paths],
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  index.faiss  ({index.ntotal} vectors)")
    print(f"  metadata.pkl ({len(metadata)} entries)")
    print(f"  summary.json")


if __name__ == "__main__":
    build()
