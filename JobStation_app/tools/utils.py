import os
import mysql.connector
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from JobStation_app.config import *

COLLECTION_NAME = "resumes_job_candidates"

MYSQL_CONFIG = {
    "host":     os.getenv("MYSQL_HOST"),
    "port":     int(os.getenv("MYSQL_PORT", "3306")),
    "user":     os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
}

MYSQL_URI = (
    f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:"
    f"{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:"
    f"{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
)

def get_mysql_connection():
    """Returns a live MySQL connection using env config."""
    return mysql.connector.connect(**MYSQL_CONFIG)


def get_qdrant_vectorstore() -> QdrantVectorStore:
    """Returns a QdrantVectorStore connected to the existing collection."""
    return QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )


def get_cv_text_by_username(username: str) -> str | None:
    """
    Looks up a jobseeker's qdrant_id from MySQL users table,
    then fetches the CV text directly from Qdrant by point ID.
    Returns the CV text string, or None if not found.
    """
    try:
        # Step 1: get qdrant_id from MySQL
        conn   = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT qdrant_id FROM users WHERE username = %s", (username,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row or not row.get("qdrant_id"):
            return None

        qdrant_id = row["qdrant_id"]

        # Step 2: fetch point from Qdrant by UUID
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[qdrant_id],
            with_payload=True,
        )

        if not points:
            return None

        # LangChain stores the text in payload["page_content"]
        payload = points[0].payload or {}
        return payload.get("page_content") or payload.get("content") or None

    except Exception as e:
        print(f"get_cv_text_by_username error: {e}")
        return None