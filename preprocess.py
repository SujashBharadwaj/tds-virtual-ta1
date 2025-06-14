<<<<<<< HEAD
import os
import json
import sqlite3
import re
import glob
from datetime import datetime
from bs4 import BeautifulSoup
import yaml

# Paths
DATA_DIR = "data"
DISCOURSE_DIR = os.path.join(DATA_DIR, "discourse_posts")
COURSE_MD_DIR = os.path.join(DATA_DIR, "course_content")
DB_PATH = "knowledge_base.db"
CHUNK_SIZE, CHUNK_OVERLAP = 1000, 200

# Clean HTML to text
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()

# Split text into overlapping chunks
def create_chunks(text: str):
    text = text.strip()
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks, current = [], ""
    for word in text.split():
        if len(current) + len(word) + 1 > CHUNK_SIZE:
            chunks.append(current)
            current = current[-CHUNK_OVERLAP:] + " " + word
        else:
            current = f"{current} {word}" if current else word
    if current:
        chunks.append(current)
    return chunks

# Ensure SQLite tables exist
def create_tables(conn):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS discourse_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT,
            chunk_index INTEGER,
            content TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS markdown_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_title TEXT,
            chunk_index INTEGER,
            content TEXT
        )
    """)
    conn.commit()

# Main preprocessing
def main():
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    c = conn.cursor()

    # 1) Process Discourse JSON files
    posts = []
    for path in glob.glob(os.path.join(DISCOURSE_DIR, "*.json")):
        try:
            data = json.load(open(path, encoding='utf8'))
        except Exception:
            continue
        if isinstance(data, dict) and 'post_stream' in data:
            posts.extend(data['post_stream'].get('posts', []))
        elif isinstance(data, list):
            for th in data:
                posts.extend(th.get('post_stream', {}).get('posts', []))
    # Insert chunks
    for idx, post in enumerate(posts):
        content_html = post.get('cooked', '')
        clean = clean_html(content_html)
        if not clean:
            continue
        pid = str(post.get('id') or post.get('topic_id') or idx)
        for i, chunk in enumerate(create_chunks(clean)):
            c.execute(
                "INSERT INTO discourse_chunks (post_id, chunk_index, content) VALUES (?,?,?)",
                (pid, i, chunk)
            )
    conn.commit()
    print("Discourse chunks:", c.execute("SELECT COUNT(*) FROM discourse_chunks").fetchone()[0])

    # 2) Process Course Markdown files
    docs = []
    for md_file in glob.glob(os.path.join(COURSE_MD_DIR, '*.md')):
        text = open(md_file, encoding='utf-8').read()
        parts = text.split('---')
        if len(parts) >= 3:
            front = parts[1]
            body = parts[2]
            try:
                meta = yaml.safe_load(front)
                title = meta.get('title') or os.path.basename(md_file).replace('.md','')
            except Exception:
                title = os.path.basename(md_file).replace('.md','')
        else:
            # No frontmatter
            body = text
            title = os.path.basename(md_file).replace('.md','')
        docs.append({'title': title, 'content': body})
    # Insert chunks
    for doc in docs:
        title = doc['title']
        content = doc['content']
        if not content.strip():
            continue
        for i, chunk in enumerate(create_chunks(content)):
            c.execute(
                "INSERT INTO markdown_chunks (doc_title, chunk_index, content) VALUES (?,?,?)",
                (title, i, chunk)
            )
    conn.commit()
    print("Markdown chunks:", c.execute("SELECT COUNT(*) FROM markdown_chunks").fetchone()[0])

    conn.close()

if __name__ == '__main__':
    main()

import os
import json
import sqlite3
import re
import glob
from datetime import datetime
from bs4 import BeautifulSoup
import yaml

# Paths
DATA_DIR = "data"
DISCOURSE_DIR = os.path.join(DATA_DIR, "discourse_posts")
COURSE_MD_DIR = os.path.join(DATA_DIR, "course_content")
DB_PATH = "knowledge_base.db"
CHUNK_SIZE, CHUNK_OVERLAP = 1000, 200

# Clean HTML to text
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()

# Split text into overlapping chunks
def create_chunks(text: str):
    text = text.strip()
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks, current = [], ""
    for word in text.split():
        if len(current) + len(word) + 1 > CHUNK_SIZE:
            chunks.append(current)
            current = current[-CHUNK_OVERLAP:] + " " + word
        else:
            current = f"{current} {word}" if current else word
    if current:
        chunks.append(current)
    return chunks

# Ensure SQLite tables exist
def create_tables(conn):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS discourse_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT,
            chunk_index INTEGER,
            content TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS markdown_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_title TEXT,
            chunk_index INTEGER,
            content TEXT
        )
    """)
    conn.commit()

# Main preprocessing
def main():
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    c = conn.cursor()

    # 1) Process Discourse JSON files
    posts = []
    for path in glob.glob(os.path.join(DISCOURSE_DIR, "*.json")):
        try:
            data = json.load(open(path, encoding='utf8'))
        except Exception:
            continue
        if isinstance(data, dict) and 'post_stream' in data:
            posts.extend(data['post_stream'].get('posts', []))
        elif isinstance(data, list):
            for th in data:
                posts.extend(th.get('post_stream', {}).get('posts', []))
    # Insert chunks
    for idx, post in enumerate(posts):
        content_html = post.get('cooked', '')
        clean = clean_html(content_html)
        if not clean:
            continue
        pid = str(post.get('id') or post.get('topic_id') or idx)
        for i, chunk in enumerate(create_chunks(clean)):
            c.execute(
                "INSERT INTO discourse_chunks (post_id, chunk_index, content) VALUES (?,?,?)",
                (pid, i, chunk)
            )
    conn.commit()
    print("Discourse chunks:", c.execute("SELECT COUNT(*) FROM discourse_chunks").fetchone()[0])

    # 2) Process Course Markdown files
    docs = []
    for md_file in glob.glob(os.path.join(COURSE_MD_DIR, '*.md')):
        text = open(md_file, encoding='utf-8').read()
        parts = text.split('---')
        if len(parts) >= 3:
            front = parts[1]
            body = parts[2]
            try:
                meta = yaml.safe_load(front)
                title = meta.get('title') or os.path.basename(md_file).replace('.md','')
            except Exception:
                title = os.path.basename(md_file).replace('.md','')
        else:
            # No frontmatter
            body = text
            title = os.path.basename(md_file).replace('.md','')
        docs.append({'title': title, 'content': body})
    # Insert chunks
    for doc in docs:
        title = doc['title']
        content = doc['content']
        if not content.strip():
            continue
        for i, chunk in enumerate(create_chunks(content)):
            c.execute(
                "INSERT INTO markdown_chunks (doc_title, chunk_index, content) VALUES (?,?,?)",
                (title, i, chunk)
            )
    conn.commit()
    print("Markdown chunks:", c.execute("SELECT COUNT(*) FROM markdown_chunks").fetchone()[0])

    conn.close()

if __name__ == '__main__':
    main()
>>>>>>> 46e2de3 (Initial import: scraping, preprocessing, retrieval experiments, FastAPI app)
