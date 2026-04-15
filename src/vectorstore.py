"""
rag/vectorstore.py — ChromaDB persistence layer.

Responsibilities:
  - Initialise (or reopen) a persisted ChromaDB collection
  - Ingest document chunks + pre-computed embeddings
  - Expose the raw collection for querying (used by search.py)
  - Provide a reset_collection() helper for re-ingestion

Design note: VectorStore holds NO embedding logic. Embeddings are
computed by EmbeddingManager and passed in. This keeps each class
single-purpose and makes testing straightforward.
"""

import os
import uuid
from typing import List

import numpy as np
import chromadb
from langchain_core.documents import Document

import config


class VectorStore:
    """
    Manages a persisted ChromaDB collection for document embeddings.

    Usage:
        vs = VectorStore()                   # default settings from config
        vs.add_documents(chunks, embeddings) # ingest once
        # Then pass vs to RAGRetriever for querying
    """

    def __init__(
        self,
        collection_name: str  = config.COLLECTION_NAME,
        persist_directory: str = config.VECTOR_STORE_DIR,
    ):
        self.collection_name   = collection_name
        self.persist_directory = persist_directory
        self.client: chromadb.PersistentClient | None = None
        self.collection = None
        self._init_store()

    # ── Private ───────────────────────────────────────────────────────────────

    def _init_store(self) -> None:
        """Create the persist directory, connect client, get-or-create collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "AI PM Bootcamp document embeddings",
                    "hnsw:space": "cosine",   # cosine similarity; distances ∈ [0, 2]
                },
            )
            print(
                f"✅ Vector store ready  "
                f"[collection='{self.collection_name}'  "
                f"docs={self.collection.count()}]"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise VectorStore: {exc}") from exc

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def doc_count(self) -> int:
        """Number of vectors currently stored."""
        return self.collection.count() if self.collection else 0

    def reset_collection(self) -> None:
        """
        Delete and recreate the collection.
        Use this when changing embedding models or re-ingesting from scratch.
        """
        if self.client is None:
            raise RuntimeError("Client not initialised.")
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "AI PM Bootcamp document embeddings",
                    "hnsw:space": "cosine",
                },
            )
            print(f"🔄 Collection '{self.collection_name}' reset.")
        except Exception as exc:
            raise RuntimeError(f"Failed to reset collection: {exc}") from exc

    def add_documents(self, docs: List[Document], embeddings: np.ndarray) -> None:
        """
        Batch-insert document chunks and their embeddings.

        Args:
            docs:       Chunks from data_loader.split_documents().
            embeddings: Corresponding numpy array from EmbeddingManager.
                        Shape must be (len(docs), embedding_dim).

        Raises:
            ValueError: If lengths do not match.
            RuntimeError: On ChromaDB insertion failure.
        """
        if len(docs) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(docs)} docs vs {len(embeddings)} embeddings."
            )
        if self.collection is None:
            raise RuntimeError("Collection not initialised.")

        ids, metadatas, texts, vecs = [], [], [], []

        for i, (doc, vec) in enumerate(zip(docs, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")

            meta = dict(doc.metadata)
            meta["doc_index"]      = i
            meta["content_length"] = len(doc.page_content)
            metadatas.append(meta)

            texts.append(doc.page_content)
            vecs.append(vec.tolist())

        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=texts,
                embeddings=vecs,
            )
            print(f"📥 Inserted {len(docs)} chunks → total: {self.collection.count()}")
        except Exception as exc:
            raise RuntimeError(f"Failed to add documents to ChromaDB: {exc}") from exc