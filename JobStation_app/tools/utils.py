import os
import mysql.connector
from langchain_qdrant import QdrantVectorStore
from JobStation_app.config import *
 
COLLECTION_NAME = "resumes_job_candidates"
 
MYSQL_CONFIG = {
    "host":     os.getenv("MYSQL_HOST"),
    "port":     int(os.getenv("MYSQL_PORT", "3306")),
    "user":     os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
}
 
 
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