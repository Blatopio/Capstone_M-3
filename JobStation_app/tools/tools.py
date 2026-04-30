from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def embedding_model():
    return OpenAIEmbeddings(model="text-embedding-3-small")