"""
rag/data_loader.py — Document loading and chunking.

Responsibilities:
  - Discover and load all .md files from a directory tree
  - Chunk documents into overlapping windows for embedding
  - Attach clean, normalised metadata to every chunk

No embedding or vector-store logic lives here.
"""

import os
import glob
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Defaults (overridable per call) ───────────────────────────────────────────
DEFAULT_CHUNK_SIZE    = 1000
DEFAULT_CHUNK_OVERLAP = 200


def load_documents(data_dir: str) -> List[Document]:
    """
    Load all Markdown files from *data_dir* into LangChain Document objects.

    Args:
        data_dir: Path to the folder containing .md source files.

    Returns:
        A flat list of Document objects, one per logical element.

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
        RuntimeError:      If no .md files are found.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    md_paths = glob.glob(os.path.join(data_dir, "**", "*.md"), recursive=True)
    if not md_paths:
        raise RuntimeError(f"No .md files found in: {data_dir}")

    all_docs: List[Document] = []
    for path in sorted(md_paths):
        try:
            loader = UnstructuredMarkdownLoader(path)
            docs   = loader.load()
            # Normalise metadata so downstream code has consistent keys
            for doc in docs:
                doc.metadata.setdefault("source_file", os.path.basename(path))
                doc.metadata.setdefault("file_type", "markdown")
            all_docs.extend(docs)
            print(f"  ✅ Loaded {len(docs):>4} elements  ← {os.path.basename(path)}")
        except Exception as exc:
            print(f"  ⚠️  Skipped {os.path.basename(path)}: {exc}")

    print(f"\n📄 Total elements loaded: {len(all_docs)}")
    return all_docs


def split_documents(
    docs: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller, overlapping chunks.

    Why RecursiveCharacterTextSplitter?
      It tries paragraph → sentence → word boundaries before hard-cutting,
      which preserves semantic coherence better than a fixed-width splitter.

    Args:
        docs:          Documents returned by load_documents().
        chunk_size:    Target character count per chunk.
        chunk_overlap: Characters shared between consecutive chunks
                       (helps retrieval not miss cross-boundary answers).

    Returns:
        A new list of Document chunks with inherited metadata.
    """
    if not docs:
        raise ValueError("No documents provided to split.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,   # adds 'start_index' to metadata for debugging
    )

    chunks = splitter.split_documents(docs)
    # Filter out empty chunks that occasionally come from blank pages
    chunks = [c for c in chunks if c.page_content.strip()]

    print(f"✂️  Split {len(docs)} documents → {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks