# retrieval.py
from langchain_chroma import Chroma
from ollama import Client
from config import initialize_embeddings, CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K


def load_vectorstore() -> Chroma:
    embeddings = initialize_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def get_retriever(client: Client):
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )