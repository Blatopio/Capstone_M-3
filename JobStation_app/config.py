from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient

from dotenv import load_dotenv

import os

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    )

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    )

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)