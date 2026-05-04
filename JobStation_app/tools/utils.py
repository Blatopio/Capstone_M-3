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


def _extract_text_from_payload(payload: dict) -> str | None:
    """Extracts CV text from a Qdrant point payload."""
    if not payload:
        return None
    return (
        payload.get("page_content")
        or payload.get("content")
        or payload.get("text")
        or None
    )


def get_cv_text_by_username(username: str) -> str | None:
    """
    Fetches CV text for a jobseeker from Qdrant.

    Strategy 1: retrieve by qdrant_id from MySQL (direct, fast)
    Strategy 2: scroll ALL points, match metadata.username in Python
                (no Qdrant index needed)

    Returns CV text string or None.
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # ── Strategy 1: direct point retrieval via MySQL qdrant_id ────────────────
    try:
        conn   = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT qdrant_id FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row and row.get("qdrant_id"):
            qdrant_id = row["qdrant_id"]
            points    = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[qdrant_id],
                with_payload=True,
            )
            if points:
                text = _extract_text_from_payload(points[0].payload)
                if text:
                    return text
    except Exception:
        pass

    # ── Strategy 2: scroll all points, match username in Python ───────────────
    try:
        all_points = []
        offset     = None
        while True:
            batch, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                with_payload=True,
                with_vectors=False,
                limit=100,
                offset=offset,
            )
            all_points.extend(batch)
            if offset is None:
                break

        for point in all_points:
            payload  = point.payload or {}
            metadata = payload.get("metadata", {})
            if metadata.get("username") == username:
                text = _extract_text_from_payload(payload)
                if text:
                    # Auto-fix MySQL qdrant_id to the actual point ID
                    try:
                        actual_id = str(point.id)
                        conn      = get_mysql_connection()
                        cursor    = conn.cursor()
                        cursor.execute(
                            "UPDATE users SET qdrant_id = %s WHERE username = %s",
                            (actual_id, username)
                        )
                        conn.commit()
                        cursor.close()
                        conn.close()
                    except Exception:
                        pass
                    return text
    except Exception:
        pass

    return None