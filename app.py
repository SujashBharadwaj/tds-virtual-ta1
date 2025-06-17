import os
import json
import sqlite3
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("AIPROXY_KEY")
if not API_KEY:
    raise RuntimeError("API key not set in .env (need API_KEY or OPENAI_API_KEY or AIPROXY_KEY)")

# Initialize FastAPI app
app = FastAPI()

@app.get("/health")
async def health():
    try:
        conn = sqlite3.connect("knowledge_base.db")
        conn.execute("SELECT 1")
        conn.close()
        db_ok = True
    except:
        db_ok = False
    return {"status": "healthy" if db_ok else "unhealthy", "db": db_ok, "api_key_set": True}

# Input/output schemas
class QueryIn(BaseModel):
    question: str
    image: str | None = None

class LinkOut(BaseModel):
    url: str
    text: str

class QueryOut(BaseModel):
    answer: str
    links: list[LinkOut]

# Load embedding and retrieval models
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
MD_INDEX = faiss.read_index("indexes/markdown.index")
DIS_INDEX = faiss.read_index("indexes/discourse.index")

with open("indexes/markdown_map.json", "r", encoding="utf-8") as f:
    MD_MAP = json.load(f)
with open("indexes/discourse_map.json", "r", encoding="utf-8") as f:
    DIS_MAP = json.load(f)

# Load generation model
GEN_TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-small")
GEN_MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Core retrieval logic
def retrieve_topk(query: str, k: int = 5):
    vec = EMBED_MODEL.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    Dm, Im = MD_INDEX.search(vec, k)
    Dd, Id = DIS_INDEX.search(vec, k)
    hits = [("markdown", MD_MAP[str(idx)], float(score)) for score, idx in zip(Dm[0], Im[0])]
