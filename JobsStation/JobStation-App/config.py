from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities.sql_database import SQLDatabase
from langfuse import get_client
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#LLM Configuration
def llm(temperature=0, model="gpt-4o-mini"):
    llm = ChatOpenAI(
        model=model, 
        temperature=temperature
        )
    logger.info("LLM initialized successfully.")
    return llm

#Embeddings Configuration
def embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

# #langchain SQL Database Configuration
# def sql_database():
#     db = SQLDatabase.from_uri(os.getenv("DATABASE_URL"))
#     logger.info("SQL Database connected successfully.")
#     return db

#Langfuse Client Configuration
def langfuse():
    return get_client()

