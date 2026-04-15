"""
rag/__init__.py — Public API for the RAG pipeline.

External modules (bot.py, app.py) should only import from here.
Internal implementation details stay inside the submodules.
"""

from .data_loader import load_documents, split_documents
from .embedding import EmbeddingManager
from .vectorstore import VectorStore
from .search import RAGRetriever

__all__ = [
    "load_documents",
    "split_documents",
    "EmbeddingManager",
    "VectorStore",
    "RAGRetriever",
]