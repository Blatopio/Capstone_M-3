import os
import re

import pandas as pd
from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from qdrant_client import QdrantClient

import mysql.connector

from JobStation_app.tools import *


load_dotenv()

# CONFIGURATION
CSV_PATH = r"C:\Alghi\Boothcamp\Purwadhika\Capstone\Capstone_M-3\JobStation_app\data\Resume.csv"
COLLECTION_NAME = "resumes_job_candidates"
BATCH_SIZE = 100


# mysql connection config from environment variables
MYSQL_CONFIG = {
    "host":     os.getenv("MYSQL_HOST"),
    "port":     int(os.getenv("MYSQL_PORT")),
    "user":     os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    #"database": os.getenv("MYSQL_DATABASE"),
}
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# qdrant config
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# openai config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# extract information about years of experience from resume text using regex patterns
def extract_years(text: str) -> int | None:
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
    if years is None:
        return "junior"
    if years <= 2:
        return "junior"
    if years <= 3:
        return "senior"
    return "specialist"


# mysql database setup and insertion functions
def get_mysql_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)


def create_mysql_table(conn):
    cursor = conn.cursor()

    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}")
    cursor.execute(f"USE {MYSQL_DATABASE}")

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
    #conn.close()  # close this temporary connection
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


# Qdrant setup
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

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
    print("Making a documents list for Qdrant ingestion...")
    if not qdrant_client.get_collection(collection_name=COLLECTION_NAME):
        print(f"   Collection '{COLLECTION_NAME}' does not exist. It will be created on first upsert.")
    else:
        print(f"   Collection '{COLLECTION_NAME}' already exists. delete exist one.")
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
    
    documents = []
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        
        if i == 0:
            # First batch — creates the collection
            vectorstore = QdrantVectorStore.from_documents(
                batch,
                embedding=embedding_model(),
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                collection_name=COLLECTION_NAME,
            )
        else:
            # Subsequent batches — add to existing collection
            vectorstore.add_documents(batch)
        
        print(f"   ✅ Uploaded {min(i + BATCH_SIZE, len(documents))}/{len(documents)}")

    print(f"   ✅ {len(documents)} resumes upserted into Qdrant.")

    # Finished step
    print(f"\n✅ Done! {len(df)} resumes ingested.")
    print("   MySQL  → structured metadata (category, level, state)")
    print("   Qdrant → semantic vectors (for RAG retrieval)")
    print("\nYou can now launch the chatbot app.")


if __name__ == "__main__":
    main()
