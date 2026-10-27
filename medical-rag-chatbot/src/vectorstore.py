from collections.abc import Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_vectorstore(chunks: Sequence[Document]) -> FAISS:
    if not chunks:
        raise ValueError("At least one document chunk is required to create a vector store.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(list(chunks), embeddings)
