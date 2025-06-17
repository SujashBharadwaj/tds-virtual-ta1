# embed_and_index.py
import os
import json
import sqlite3
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# ── CONFIG ──
DISCOURSE_JSON_DIR = "data/discourse_posts"   # individual topic_*.json files
EMBED_MODEL        = "all-MiniLM-L6-v2"       # or your chosen huggingface model
INDEX_PATH         = "indexes/subthread.index"
MAP_PATH           = "indexes/subthread_map.json"
TOPIC_TIMEOUT      = 30                       # seconds for HTTP if needed

# ── HELPERS ──
def load_all_posts():
    posts = []
    for fname in os.listdir(DISCOURSE_JSON_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(DISCOURSE_JSON_DIR, fname)
        j = json.load(open(path, encoding="utf8"))
        # API‐style JSON
        if isinstance(j, dict) and "post_stream" in j:
            posts += j["post_stream"].get("posts", [])
        # Else: skip other formats
    return posts

def build_reply_map(posts):
    by_num = {p["post_number"]: p for p in posts}
    parent_map = defaultdict(list)
    for p in posts:
        parent = p.get("reply_to_post_number")
        parent_map[parent].append(p["post_number"])
    return parent_map, by_num

def collect_subthread(root, parent_map, by_num):
    out = []
    def dfs(num):
        out.append(by_num[num])
        for child in parent_map.get(num, []):
            dfs(child)
    dfs(root)
    return out

def clean_text(html):
    # very basic HTML→text: collapse whitespace
    text = re.sub(r"<[^>]+>", " ", html)
    return " ".join(text.split())

# ── MAIN ──
if __name__ == "__main__":
    import re
    # 1) load & group by topic
    print("Loading posts…")
    posts = load_all_posts()
    topics = defaultdict(list)
    for p in posts:
        topics[p["topic_id"]].append(p)
    print(f"Found {len(posts)} posts across {len(topics)} topics.")

    # 2) build subthread data and embeddings
    print("Building embeddings…")
    model = SentenceTransformer(EMBED_MODEL)
    embedding_vectors = []
    map_info = []

    for topic_id, tposts in tqdm(topics.items(), total=len(topics)):
        # sort posts
        tposts.sort(key=lambda p: p["post_number"])
        parent_map, by_num = build_reply_map(tposts)
        # root threads = those with parent=None
        roots = parent_map[None]
        title = tposts[0].get("topic_title","")

        for root_num in roots:
            sub = collect_subthread(root_num, parent_map, by_num)
            # combine text
            combined = f"Topic: {title}\n\n" + "\n\n---\n\n".join(
                clean_text(p["cooked"]) for p in sub
            )
            # embed
            emb = model.encode(combined, convert_to_numpy=True)
            emb /= np.linalg.norm(emb)
            idx = len(embedding_vectors)
            embedding_vectors.append(emb)
            map_info.append({
                "topic_id": topic_id,
                "root_post_number": root_num,
                "post_numbers": [p["post_number"] for p in sub]
            })

    # 3) build FAISS index
    X = np.vstack(embedding_vectors).astype("float32")
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"Wrote {len(map_info)} subthreads to {INDEX_PATH}")

    with open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(map_info, f, indent=2)
    print(f"Wrote map to {MAP_PATH}")

# embed_and_index.py
import os
import json
import sqlite3
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# ── CONFIG ──
DISCOURSE_JSON_DIR = "data/discourse_posts"   # individual topic_*.json files
EMBED_MODEL        = "all-MiniLM-L6-v2"       # or your chosen huggingface model
INDEX_PATH         = "indexes/subthread.index"
MAP_PATH           = "indexes/subthread_map.json"
TOPIC_TIMEOUT      = 30                       # seconds for HTTP if needed

# ── HELPERS ──
def load_all_posts():
    posts = []
    for fname in os.listdir(DISCOURSE_JSON_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(DISCOURSE_JSON_DIR, fname)
        j = json.load(open(path, encoding="utf8"))
        # API‐style JSON
        if isinstance(j, dict) and "post_stream" in j:
            posts += j["post_stream"].get("posts", [])
        # Else: skip other formats
    return posts

def build_reply_map(posts):
    by_num = {p["post_number"]: p for p in posts}
    parent_map = defaultdict(list)
    for p in posts:
        parent = p.get("reply_to_post_number")
        parent_map[parent].append(p["post_number"])
    return parent_map, by_num

def collect_subthread(root, parent_map, by_num):
    out = []
    def dfs(num):
        out.append(by_num[num])
        for child in parent_map.get(num, []):
            dfs(child)
    dfs(root)
    return out

def clean_text(html):
    # very basic HTML→text: collapse whitespace
    text = re.sub(r"<[^>]+>", " ", html)
    return " ".join(text.split())

# ── MAIN ──
if __name__ == "__main__":
    import re
    # 1) load & group by topic
    print("Loading posts…")
    posts = load_all_posts()
    topics = defaultdict(list)
    for p in posts:
        topics[p["topic_id"]].append(p)
    print(f"Found {len(posts)} posts across {len(topics)} topics.")

    # 2) build subthread data and embeddings
    print("Building embeddings…")
    model = SentenceTransformer(EMBED_MODEL)
    embedding_vectors = []
    map_info = []

    for topic_id, tposts in tqdm(topics.items(), total=len(topics)):
        # sort posts
        tposts.sort(key=lambda p: p["post_number"])
        parent_map, by_num = build_reply_map(tposts)
        # root threads = those with parent=None
        roots = parent_map[None]
        title = tposts[0].get("topic_title","")

        for root_num in roots:
            sub = collect_subthread(root_num, parent_map, by_num)
            # combine text
            combined = f"Topic: {title}\n\n" + "\n\n---\n\n".join(
                clean_text(p["cooked"]) for p in sub
            )
            # embed
            emb = model.encode(combined, convert_to_numpy=True)
            emb /= np.linalg.norm(emb)
            idx = len(embedding_vectors)
            embedding_vectors.append(emb)
            map_info.append({
                "topic_id": topic_id,
                "root_post_number": root_num,
                "post_numbers": [p["post_number"] for p in sub]
            })

    # 3) build FAISS index
    X = np.vstack(embedding_vectors).astype("float32")
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"Wrote {len(map_info)} subthreads to {INDEX_PATH}")

    with open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(map_info, f, indent=2)
    print(f"Wrote map to {MAP_PATH}")

