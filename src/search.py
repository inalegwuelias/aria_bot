"""
rag/search.py — Semantic retrieval from the vector store.

Responsibilities:
  - Accept a natural-language query
  - Embed it with EmbeddingManager
  - Query ChromaDB and post-filter by similarity threshold
  - Return ranked, typed results ready for the LLM context

This module is the ONLY place that touches collection.query().
"""

from typing import List, Dict, Any

from .embedding import EmbeddingManager
from .vectorstore import VectorStore
import config


# Typed alias for a single retrieved document
RetrievedDoc = Dict[str, Any]


class RAGRetriever:
    """
    Retrieves the most relevant document chunks for a query.

    Usage:
        retriever = RAGRetriever(vector_store, embedding_manager)
        docs = retriever.retrieve("What is the team size?")
    """

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self._store  = vector_store
        self._embedder = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int           = config.TOP_K,
        score_threshold: float = config.SCORE_THRESHOLD,
    ) -> List[RetrievedDoc]:
        """
        Find the *top_k* most similar chunks for *query*.

        Args:
            query:           The user's question string.
            top_k:           Maximum chunks to return before threshold filtering.
            score_threshold: Minimum cosine similarity (0–1). Chunks below
                             this score are discarded.

        Returns:
            List of dicts, each containing:
                id             – ChromaDB document id
                text           – Raw chunk text (sent to LLM as context)
                metadata       – Source file, index, etc.
                similarity_score – float in [0, 1]; higher = more relevant
                rank           – 1-based position after sorting

        Returns [] on any retrieval error (so the bot degrades gracefully).
        """
        if not query.strip():
            return []

        # 1 — Embed the query (shape: (1, dim) → take [0])
        query_vec = self._embedder.generate_embeddings([query])[0]

        # 2 — Query ChromaDB
        try:
            results = self._store.collection.query(
                query_embeddings=[query_vec.tolist()],
                n_results=top_k,
            )
        except Exception as exc:
            print(f"❌ ChromaDB query failed: {exc}")
            return []

        # 3 — Unpack and filter
        retrieved: List[RetrievedDoc] = []

        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]
        ids       = results.get("ids",        [[]])[0]

        for rank, (doc_id, text, meta, dist) in enumerate(
            zip(ids, docs, metas, distances), start=1
        ):
            # ChromaDB cosine distance ∈ [0, 2]; similarity = 1 − distance
            # (for normalised vectors distance ∈ [0, 1], but we keep the
            #  formula general)
            similarity = 1.0 - dist
            if similarity >= score_threshold:
                retrieved.append(
                    {
                        "id":               doc_id,
                        "text":             text,
                        "metadata":         meta,
                        "similarity_score": similarity,
                        "rank":             rank,
                    }
                )

        # 4 — Sort descending by score (ChromaDB returns them ordered, but
        #     explicit sort ensures correctness after threshold filtering)
        retrieved.sort(key=lambda d: d["similarity_score"], reverse=True)

        print(
            f"🔍 '{query[:60]}' → "
            f"{len(retrieved)} chunks above threshold {score_threshold}"
        )
        return retrieved