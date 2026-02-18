# config.py
import gin
from ollama import Client
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer
from typing import List

# ── Models ────────────────────────────────────────────────
LLM_MODEL = "llama3.3"  
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Chunking ──────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Retrieval ─────────────────────────────────────────────
TOP_K = 5

# ── ChromaDB ──────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "hsmw_lectures"


@gin.configurable
def create_client(host: str, api_key: str) -> Client:
    """Creates the Ollama API client."""
    return Client(
        host,
        headers={"Authorization": f"Bearer {api_key}"},
    )


def initialize_llm(host: str, api_key: str, model_name: str = LLM_MODEL) -> OllamaLLM:
    """Initialize the LLM model."""
    return OllamaLLM(
        model=model_name,
        base_url=host,
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
    )


# class OllamaClientEmbeddings(Embeddings):
#     """Custom embeddings using the ollama Client directly."""

#     def __init__(self, client: Client, model: str = EMBEDDING_MODEL):
#         self.client = client
#         self.model = model

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return [self.client.embed(model=self.model, input=t).embeddings[0] for t in texts]

#     def embed_query(self, text: str) -> List[float]:
#         return self.client.embed(model=self.model, input=text).embeddings[0]


# def initialize_embeddings(client: Client, model_name: str = EMBEDDING_MODEL) -> OllamaClientEmbeddings:
#     """Initialize embeddings using the remote Ollama client."""
#     return OllamaClientEmbeddings(client=client, model=model_name)

class SentenceTransformerEmbeddings:
    """Custom embeddings using SentenceTransformers."""
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

def initialize_embeddings(model_name: str = EMBEDDING_MODEL) -> SentenceTransformerEmbeddings:
    """Initialize embeddings using SentenceTransformers."""
    return SentenceTransformerEmbeddings(model=model_name)