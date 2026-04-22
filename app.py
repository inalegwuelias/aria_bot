"""
app.py — Entry point. Wires the RAG pipeline into the Discord bot.

Startup sequence
────────────────
1.  Validate env vars (fast-fail before loading heavy models)
2.  Load & chunk documents from DATA_DIR
3.  Build EmbeddingManager  (downloads model on first run)
4.  Build VectorStore       (opens or creates ChromaDB collection)
5.  Ingest documents if the collection is empty
6.  Build RAGRetriever
7.  Build LLM (Groq)
8.  Wrap RAG pipeline in a single answer_fn  ← injected into bot
9.  Create Discord bot with answer_fn
10. Run

No RAG or Discord logic lives here — app.py is pure glue.
"""

import logging
import config
from src import (
    load_documents,
    split_documents,
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
)
from bot import create_bot

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ── Logging Setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aria.app")


# ── Step 1 — Validate env vars ────────────────────────────────────────────────
logger.info("Validating environment variables...")
config.validate()
logger.info("✅ Environment variables OK")


# ── Step 2 — Load and chunk documents ────────────────────────────────────────
logger.info("Loading documents...")
print("\n📂 Loading documents...")
raw_docs = load_documents(config.DATA_DIR)
chunks   = split_documents(raw_docs)
logger.info(f"✅ Loaded {len(chunks)} chunks from {len(raw_docs)} documents")
print(f"✅ Loaded {len(chunks)} chunks")


# ── Step 3 — Embedding model ──────────────────────────────────────────────────
logger.info("Initialising embedding model...")
print("\n🔢 Initialising embedding model...")
embedding_manager = EmbeddingManager()
logger.info("✅ Embedding model ready")


# ── Step 4 — Vector store ─────────────────────────────────────────────────────
logger.info("Connecting to vector store...")
print("\n🗄️  Connecting to vector store...")
vector_store = VectorStore()
logger.info("✅ Vector store connected")


# ── Step 5 — Ingest if empty ──────────────────────────────────────────────────
if vector_store.doc_count == 0:
    logger.info("Collection is empty — ingesting documents...")
    print("\n📥 Collection is empty — ingesting documents...")
    vector_store.reset_collection()
    texts      = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts, show_progress=True)
    vector_store.add_documents(chunks, embeddings)
    logger.info(f"✅ Ingested {len(chunks)} chunks into vector store")
else:
    logger.info(f"✅ Reusing existing collection ({vector_store.doc_count} docs)")
    print(f"\n✅ Reusing existing collection ({vector_store.doc_count} docs).")


# ── Step 6 — Retriever ────────────────────────────────────────────────────────
retriever = RAGRetriever(vector_store, embedding_manager)
logger.info("✅ Retriever ready")


# ── Step 7 — LLM ─────────────────────────────────────────────────────────────
logger.info("Initialising LLM...")
print("\n🤖 Initialising LLM...")
llm = ChatGroq(
    api_key=config.GROQ_API_KEY,
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
    max_tokens=config.MAX_TOKENS,
)
logger.info(f"✅ LLM ready — model: {config.LLM_MODEL}")


# ── Step 8 — RAG answer function (injected into bot) ─────────────────────────
def answer_fn(query: str) -> str:
    """
    Full RAG pipeline: retrieve → build context → generate → return.

    This is the ONLY function bot.py ever calls. It is completely
    synchronous so it can be safely run in asyncio.to_thread().
    """
    logger.info(f"RAG pipeline started | Query: {query[:80]}")

    retrieved = retriever.retrieve(
        query,
        top_k=config.TOP_K,
        score_threshold=config.SCORE_THRESHOLD,
    )

    if not retrieved:
        logger.warning(f"No relevant docs found | Query: {query[:80]}")
        return config.FALLBACK_NO_DOCS

    logger.info(f"Retrieved {len(retrieved)} chunks | Query: {query[:80]}")

    # Build LLM context from top-scored chunks
    context = "\n\n".join(doc["text"] for doc in retrieved)

    messages = [
        SystemMessage(content=config.SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]

    try:
        response = llm.invoke(messages)
        answer   = response.content.strip()
        logger.info(f"LLM response generated | Length: {len(answer)} chars")
        return answer if answer else config.FALLBACK_NO_DOCS
    except Exception as exc:
        logger.error(f"LLM error: {exc}")
        return config.FALLBACK_LLM_ERROR


# ── Step 9 — Discord bot ──────────────────────────────────────────────────────
logger.info("Creating Discord bot...")
print("\n🎮 Creating Discord bot...")
bot = create_bot(answer_fn)

logger.info("✅ Aria is fully initialised and ready")
print("\n✅ Aria is fully initialised and ready.\n")


# ── Step 10 — Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(config.DISCORD_TOKEN)