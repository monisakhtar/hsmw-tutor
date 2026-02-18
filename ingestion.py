# ingestion.py
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from ollama import Client
from config import initialize_embeddings, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PERSIST_DIR, COLLECTION_NAME


def extract_text_from_pdf(pdf_file) -> str:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


def store_chunks(chunks: list[str]) -> Chroma:
    embeddings = initialize_embeddings()
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def ingest_pdf(pdf_file, client: Client) -> Chroma:
    text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(text)
    return store_chunks(chunks)