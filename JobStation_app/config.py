import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# ─── Langfuse ─────────────────────────────────────────────────────────────────
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

load_dotenv(override=False)

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ─── Langfuse handler ─────────────────────────────────────────────────────────
# Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from .env
# automatically — no kwargs needed for this Langfuse version.
langfuse_handler = LangfuseCallbackHandler()

# ─── LLM & Embeddings ─────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model       = "gpt-4o-mini",
    temperature = 0,
    callbacks   = [langfuse_handler],   # ← Langfuse traces every LLM call
)

embedding_model = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    # callbacks not supported on embeddings — tracing handled via LLM calls
)

qdrant_client = QdrantClient(
    url     = QDRANT_URL,
    api_key = QDRANT_API_KEY,
)
