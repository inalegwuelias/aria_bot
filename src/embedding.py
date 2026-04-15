"""
rag/embedding.py — Text → vector conversion.

Responsibilities:
  - Load and cache the SentenceTransformer model (once per process)
  - Expose a clean generate_embeddings() method
  - Keep embedding logic completely decoupled from storage

Design note: EmbeddingManager is a lightweight class rather than
a module-level function so that the model is only loaded once and
can be easily swapped (e.g. for testing with a mock).
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

import config


class EmbeddingManager:
    """
    Wraps a SentenceTransformer model and provides a stable embedding interface.

    Usage:
        em = EmbeddingManager()                          # default model from config
        em = EmbeddingManager("all-MiniLM-L6-v2")       # override model
        vectors = em.generate_embeddings(["text one", "text two"])
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._load_model()

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Download (first run) or load from cache, then verify dimensions."""
        try:
            print(f"⏳ Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            dim = self._model.get_embedding_dimension()
            print(f"✅ Embedding model ready — output dim: {dim}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {exc}") from exc

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Return the vector dimension of the loaded model."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        return self._model.get_sentence_embedding_dimension()

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode a list of strings into dense vectors.

        Args:
            texts:         Strings to embed. Must be non-empty.
            show_progress: Show a tqdm progress bar (useful during ingestion).
            batch_size:    How many texts to encode per GPU/CPU batch.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim).

        Raises:
            ValueError:  If texts is empty.
            RuntimeError: If the model is not loaded.
        """
        if not texts:
            raise ValueError("texts list is empty — nothing to embed.")
        if self._model is None:
            raise RuntimeError("Embedding model is not loaded.")

        embeddings = self._model.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
        return embeddings