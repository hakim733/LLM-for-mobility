"""
Vector store utilities using ChromaDB.

Responsibilities:
- Create/load a persistent ChromaDB collection
- Embed document chunks
- Store and retrieve chunks by semantic similarity
"""

from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_DB_DIR,
    EMBEDDING_MODEL_NAME,
    TOP_K,
)


class VectorStore:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR)
        )

        # Create or load collection
        self.collection = self.client.get_or_create_collection(
            name="skane_mobility_docs"
        )

    def add_chunks(self, chunks: List[dict]) -> None:
        """
        Add document chunks to the vector store.
        """
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        embeddings = self.embedding_model.encode(
            texts, show_progress_bar=True
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_text: str, n_results: int = TOP_K):
        """
        Perform a semantic similarity search.
        """
        query_embedding = self.embedding_model.encode(
            query_text
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        return results
