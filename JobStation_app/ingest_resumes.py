"""
What it does:
  1. Reads Resume.csv
  2. Computes professionalism level via regex
  3. Inserts structured fields into MySQL
  4. Embeds resume text with OpenAI
  5. Upserts vectors into Qdrant Cloud

Run:
  python ingest_resumes.py

Requirements:
  pip install pandas mysql-connector-python openai qdrant-client python-dotenv tqdm
"""

import os
import re
import uuid
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import mysql.connector
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter,
    FieldCondition, MatchValue
)

load_dotenv()

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH = "data\\Resume.csv"
COLLECTION_NAME = "resumes"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536       
BATCH_SIZE      = 50         

# MySQL — fill in your credentials or set in .env
MYSQL_CONFIG = {
    "host":     os.getenv("MYSQL_HOST"),
    "port":     os.getenv("MYSQL_PORT"),
    "user":     os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
}

# Qdrant Cloud — fill in your cluster URL and API key
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# ─── PROFESSIONALISM LOGIC ─────────────────────────────────────────────────────
def extract_years(text: str) -> int | None:
    """
    Extract years of experience from raw resume text using regex.
    Returns None if no explicit mention found.
    """
    patterns = [
        r"(\d+)\+?\s*years?\s*of\s*experience",
        r"(\d+)\+?\s*years?\s*experience",
        r"over\s*(\d+)\s*years?",
        r"more\s*than\s*(\d+)\s*years?",
        r"(\d+)\+\s*years?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def compute_level(years: int | None) -> str:
    """
    Map years of experience to professionalism level.

    Rules (from business logic):
      <= 2 years  → junior
      <= 3 years  → senior
      > 3 years   → specialist
      None        → junior (default, no explicit mention in resume)
    """
    if years is None:
        return "junior"
    if years <= 2:
        return "junior"
    if years <= 3:
        return "senior"
    return "specialist"


# ─── MYSQL SETUP ───────────────────────────────────────────────────────────────
def get_mysql_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)


def create_mysql_table(conn):
    cursor = conn.cursor()

    # Create database if it doesn't exist
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
    cursor.execute(f"USE {MYSQL_CONFIG['database']}")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id          INT PRIMARY KEY,
            category    VARCHAR(100),
            prof_level  ENUM('junior', 'senior', 'specialist'),
            state       ENUM('available', 'interviewed', 'placed', 'inactive')
                        DEFAULT 'available',
            qdrant_id   VARCHAR(100),
            updated_at  TIMESTAMP DEFAULT NOW() ON UPDATE NOW(),
            created_at  TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cursor.close()
    print("✅ MySQL table ready.")


def insert_candidates_mysql(conn, rows: list[dict]):
    """
    rows: list of dicts with keys: id, category, prof_level, qdrant_id
    Uses INSERT IGNORE so re-running the script is safe (no duplicates).
    """
    cursor = conn.cursor()
    sql = """
        INSERT IGNORE INTO candidates
            (id, category, prof_level, state, qdrant_id)
        VALUES
            (%(id)s, %(category)s, %(prof_level)s, 'available', %(qdrant_id)s)
    """
    cursor.executemany(sql, rows)
    conn.commit()
    cursor.close()


# ─── QDRANT SETUP ──────────────────────────────────────────────────────────────
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def create_qdrant_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Qdrant collection '{COLLECTION_NAME}' created.")
    else:
        print(f"ℹ️  Qdrant collection '{COLLECTION_NAME}' already exists.")


# ─── EMBEDDING ─────────────────────────────────────────────────────────────────
def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts. OpenAI ada-002 accepts up to 2048 inputs per call
    but we keep batches small to avoid rate limits.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  RESUME INGESTION — MySQL + Qdrant")
    print("=" * 55)

    # 1. Load CSV
    print(f"\n📂 Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["Resume_str", "Category"])
    print(f"   {len(df)} resumes loaded across {df['Category'].nunique()} categories.")

    # 2. Compute professionalism level
    print("\n🔍 Computing professionalism levels...")
    df["years"]      = df["Resume_str"].apply(extract_years)
    df["prof_level"] = df["years"].apply(compute_level)

    level_counts = df["prof_level"].value_counts()
    print(f"   junior={level_counts.get('junior',0)} | "
          f"senior={level_counts.get('senior',0)} | "
          f"specialist={level_counts.get('specialist',0)}")

    # 3. Generate Qdrant point IDs (one per row, stable across re-runs)
    #    We use the CSV's original ID column cast to string as the qdrant_id.
    df["qdrant_id"] = df["ID"].astype(str)

    # 4. MySQL — create table and insert structured rows
    print("\n🗄️  Inserting into MySQL...")
    conn = get_mysql_connection()
    create_mysql_table(conn)

    mysql_rows = df[["ID", "Category", "prof_level", "qdrant_id"]].rename(
        columns={"ID": "id", "Category": "category"}
    ).to_dict(orient="records")

    insert_candidates_mysql(conn, mysql_rows)
    conn.close()
    print(f"   ✅ {len(mysql_rows)} rows inserted into MySQL.")

    # 5. Qdrant — embed and upsert
    print("\n🧠 Embedding and upserting into Qdrant...")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    qdrant_client = get_qdrant_client()
    create_qdrant_collection(qdrant_client)

    texts      = df["Resume_str"].tolist()
    ids        = df["ID"].tolist()
    categories = df["Category"].tolist()
    levels     = df["prof_level"].tolist()

    points = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="   Batches"):
        batch_texts  = texts[i : i + BATCH_SIZE]
        batch_ids    = ids[i : i + BATCH_SIZE]
        batch_cats   = categories[i : i + BATCH_SIZE]
        batch_levels = levels[i : i + BATCH_SIZE]

        embeddings = embed_texts(openai_client, batch_texts)

        for j, embedding in enumerate(embeddings):
            points.append(
                PointStruct(
                    id=batch_ids[j],          # integer ID from CSV
                    vector=embedding,
                    payload={
                        "category":   batch_cats[j],
                        "prof_level": batch_levels[j],
                        "resume_text": batch_texts[j][:500]  # preview only
                    }
                )
            )

        # Upsert this batch immediately to keep memory low
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[-BATCH_SIZE:]   # only the current batch
        )

    print(f"\n✅ Done! {len(df)} resumes ingested.")
    print("   MySQL  → structured metadata (category, level, state)")
    print("   Qdrant → semantic vectors (for RAG retrieval)")
    print("\nYou can now launch the chatbot app.")


if __name__ == "__main__":
    main()
