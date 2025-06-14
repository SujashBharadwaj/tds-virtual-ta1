<<<<<<< HEAD
# app.py

import os
import sqlite3
import json
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load env variables
load_dotenv()
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("AIPROXY_KEY")
if not API_KEY:
    raise RuntimeError("API key not set in .env")

# Initialize FastAPI
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

class QueryIn(BaseModel):
    question: str
    image: str | None = None

class LinkOut(BaseModel):
    url: str
    text: str

class QueryOut(BaseModel):
    answer: str
    links: list[LinkOut]

# Load models and indexes
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
MD_INDEX = faiss.read_index("indexes/markdown.index")
DIS_INDEX = faiss.read_index("indexes/discourse.index")
with open("indexes/markdown_map.json", "r", encoding="utf-8") as f:
    MD_MAP = json.load(f)
with open("indexes/discourse_map.json", "r", encoding="utf-8") as f:
    DIS_MAP = json.load(f)
GEN_TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-large")
GEN_MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

def retrieve_topk(query: str, k: int = 5):
    vec = EMBED_MODEL.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    Dm, Im = MD_INDEX.search(vec, k)
    Dd, Id = DIS_INDEX.search(vec, k)
    hits = [("markdown", MD_MAP[str(idx)], float(score)) for score, idx in zip(Dm[0], Im[0])]
    hits += [("discourse", DIS_MAP[str(idx)], float(score)) for score, idx in zip(Dd[0], Id[0])]
    hits.sort(key=lambda x: x[2], reverse=True)
    return hits[:k]

def snippet(source: str, cid: int):
    conn = sqlite3.connect("knowledge_base.db")
    cur = conn.cursor()
    if source == "markdown":
        cur.execute("SELECT doc_title, content FROM markdown_chunks WHERE id=?", (cid,))
        title, txt = cur.fetchone() or ("", "")
        url = f"md://{title}"
    else:
        cur.execute("SELECT post_id, content FROM discourse_chunks WHERE id=?", (cid,))
        pid, txt = cur.fetchone() or ("", "")
        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{pid}"
    conn.close()
    snippet = txt.replace("\n", " ")[:400] + "…"
    return url, snippet

def generate_answer(question: str, contexts: list[str]):
    prompt = (
        "You're a helpful and friendly teaching assistant for a data science course.\n"
        "Based on the following excerpts from course materials and forum posts, answer the student's question in a clear and conversational tone.\n\n"
        + "\n\n---\n\n".join(contexts)
        + f"\n\nStudent: {question}\nTA:"
    )
    inputs = GEN_TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=2048)
    out = GEN_MODEL.generate(**inputs, max_length=256, num_beams=4)
    return GEN_TOKENIZER.decode(out[0], skip_special_tokens=True)

@app.post("/query", response_model=QueryOut)
async def query_api(qin: QueryIn):
    hits = retrieve_topk(qin.question, k=5)
    contexts, links = [], []
    for src, cid, score in hits:
        url, txt = snippet(src, cid)
        contexts.append(txt)
        links.append(LinkOut(url=url, text=txt))

    try:
        answer = generate_answer(qin.question, contexts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return QueryOut(answer=answer, links=links)
=======
# app.py

import os
import sqlite3
import json
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1) Load env
load_dotenv()
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("AIPROXY_KEY")
if not API_KEY:
    raise RuntimeError(
        "API key not set in .env (need API_KEY or OPENAI_API_KEY or AIPROXY_KEY)"
    )

# 2) FastAPI setup
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

class QueryIn(BaseModel):
    question: str
    image: str | None = None

class LinkOut(BaseModel):
    url: str
    text: str

class QueryOut(BaseModel):
    answer: str
    links: list[LinkOut]

# 3) Load embedding models & indexes (once at startup)
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
MD_INDEX    = faiss.read_index("indexes/markdown.index")
DIS_INDEX   = faiss.read_index("indexes/discourse.index")
with open("indexes/markdown_map.json", "r", encoding="utf-8") as f:
    MD_MAP = json.load(f)
with open("indexes/discourse_map.json", "r", encoding="utf-8") as f:
    DIS_MAP = json.load(f)

GEN_TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
GEN_MODEL     = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")

def retrieve_topk(query: str, k: int = 5):
    vec = EMBED_MODEL.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    Dm, Im = MD_INDEX.search(vec, k)
    Dd, Id = DIS_INDEX.search(vec, k)
    hits = [("markdown", MD_MAP[str(idx)], float(score)) for score, idx in zip(Dm[0], Im[0])]
    hits += [("discourse", DIS_MAP[str(idx)], float(score)) for score, idx in zip(Dd[0], Id[0])]
    hits.sort(key=lambda x: x[2], reverse=True)
    return hits[:k]

def snippet(source: str, cid: int):
    conn = sqlite3.connect("knowledge_base.db")
    cur = conn.cursor()
    if source == "markdown":
        cur.execute("SELECT doc_title, content FROM markdown_chunks WHERE id=?", (cid,))
        title, txt = cur.fetchone() or ("", "")
        url = f"md://{title}"
    else:
        cur.execute("SELECT post_id, content FROM discourse_chunks WHERE id=?", (cid,))
        pid, txt = cur.fetchone() or ("", "")
        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{pid}"
    conn.close()
    snippet = txt.replace("\n", " ")[:300] + "…"
    return url, snippet

def generate_answer(question: str, contexts: list[str]):
    prompt = (
        "Answer the question based on these excerpts:\n\n"
        + "\n\n---\n\n".join(contexts)
        + f"\n\nQuestion: {question}\nAnswer:"
    )
    inputs = GEN_TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=4096)
    out = GEN_MODEL.generate(**inputs, max_length=256, num_beams=4)
    return GEN_TOKENIZER.decode(out[0], skip_special_tokens=True)

@app.post("/query", response_model=QueryOut)
async def query_api(qin: QueryIn):
    # 1) retrieval
    hits = retrieve_topk(qin.question, k=5)
    contexts, links = [], []
    for src, cid, score in hits:
        url, txt = snippet(src, cid)
        contexts.append(txt)
        links.append(LinkOut(url=url, text=txt))

    # 2) generation
    try:
        answer = generate_answer(qin.question, contexts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return QueryOut(answer=answer, links=links)
>>>>>>> 46e2de3 (Initial import: scraping, preprocessing, retrieval experiments, FastAPI app)
