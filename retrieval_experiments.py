"""
retrieval_experiments.py

A standalone script to validate and experiment with your FAISS-based retrieval pipeline.

Usage:
  # 1. Activate your virtual environment
  .\.venv\Scripts\Activate.ps1   # (Windows PowerShell)

  # 2. Install dependencies if not done
  pip install sentence-transformers faiss-cpu numpy sqlite3

  # 3. Place this script in your project root (beside knowledge_base.db and indexes/)

  # 4. Run the script with:
  python retrieval_experiments.py

This will:
  - Load your FAISS indexes and ID maps
  - Embed example queries
  - Retrieve top-k markdown & discourse chunks
  - Print snippets and metadata for inspection

Ensure the following paths exist relative to this file:
  knowledge_base.db
  indexes/markdown.index
  indexes/discourse.index
  indexes/markdown_map.json
  indexes/discourse_map.json

"""
import os
import json
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Build absolute paths based on script location
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DB_PATH         = os.path.join(BASE_DIR, "knowledge_base.db")
MD_INDEX_PATH   = os.path.join(BASE_DIR, "indexes", "markdown.index")
DIS_INDEX_PATH  = os.path.join(BASE_DIR, "indexes", "discourse.index")
MD_MAP_PATH     = os.path.join(BASE_DIR, "indexes", "markdown_map.json")
DIS_MAP_PATH    = os.path.join(BASE_DIR, "indexes", "discourse_map.json")
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 1

# Initialize embedder and indexes
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
print("Loading FAISS indexes and ID maps...")
md_index = faiss.read_index(MD_INDEX_PATH)
dis_index = faiss.read_index(DIS_INDEX_PATH)
with open(MD_MAP_PATH, 'r', encoding='utf-8') as f:
    md_ids = json.load(f)
with open(DIS_MAP_PATH, 'r', encoding='utf-8') as f:
    dis_ids = json.load(f)

# Retrieval helper
def retrieve(query, top_k=TOP_K):
    vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    Dm, Im = md_index.search(vec, top_k)
    Dd, Id = dis_index.search(vec, top_k)
    hits = []
    # md_ids and dis_ids are lists mapping index -> chunk_id
    for score, idx in zip(Dm[0], Im[0]):
        chunk_id = md_ids[idx]      # use integer index for list lookup
        hits.append(("md", chunk_id, float(score)))
    for score, idx in zip(Dd[0], Id[0]):
        chunk_id = dis_ids[idx]
        hits.append(("dis", chunk_id, float(score)))
    # sort by score descending and return top_k combined
    hits.sort(key=lambda x: x[2], reverse=True)
    return hits[:top_k]

# Print snippet from SQLite
def print_snippet(src, chunk_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if src == "md":
        cur.execute("SELECT doc_title, content FROM markdown_chunks WHERE id=?", (chunk_id,))
        title, text = cur.fetchone()
        print(f"[MD] {title} (chunk {chunk_id}) snippet:\n{text[:200]}...\n")
    else:
        cur.execute("SELECT post_id, content FROM discourse_chunks WHERE id=?", (chunk_id,))
        post_id, text = cur.fetchone()
        print(f"[DIS] https://discourse.onlinedegree.iitm.ac.in/t/{post_id} (chunk {chunk_id}) snippet:\n{text[:200]}...\n")
    conn.close()

# Interactive loop
if __name__ == '__main__':
    print("=== Retrieval Experiments ===")
    while True:
        q = input("Enter query (or 'exit'): ")
        if q.lower() in ('exit', 'quit'):
            break
        hits = retrieve(q)
        print(f"Top {len(hits)} hits for '{q}':")
        for src, cid, score in hits:
            print(f"  - {src.upper()} ID {cid} (score={score:.3f})")
            print_snippet(src, cid)
    print("Goodbye!")
