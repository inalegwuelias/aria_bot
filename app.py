"""
app.py — Entry point. Wires the RAG pipeline into the Discord bot.

Startup sequence
────────────────
1.  Validate env vars (fast-fail before loading heavy models)
2.  Start keep-alive server (Render port detection)
3.  Load & chunk documents from DATA_DIR
4.  Build EmbeddingManager  (downloads model on first run)
5.  Build VectorStore       (opens or creates ChromaDB collection)
6.  Ingest documents if the collection is empty
7.  Build RAGRetriever
8.  Build LLM (Groq)
9.  Wrap RAG pipeline in a single answer_fn  ← injected into bot
10. Create Discord bot with answer_fn
11. Run

No RAG or Discord logic lives here — app.py is pure glue.
"""

import config
from src import (
    load_documents,
    split_documents,
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
)
from bot import create_bot, keep_alive

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


# ── Step 1 — Validate env vars ────────────────────────────────────────────────
config.validate()


# ── Step 2 — Start keep-alive server FIRST (Render port detection) ────────────
keep_alive()


# ── Step 3 — Load and chunk documents ────────────────────────────────────────
print("\n📂 Loading documents...")
raw_docs = load_documents(config.DATA_DIR)
chunks   = split_documents(raw_docs)


# ── Step 4 — Embedding model ──────────────────────────────────────────────────
print("\n🔢 Initialising embedding model...")
embedding_manager = EmbeddingManager()


# ── Step 5 — Vector store ─────────────────────────────────────────────────────
print("\n🗄️  Connecting to vector store...")
vector_store = VectorStore()


# ── Step 6 — Ingest if empty ──────────────────────────────────────────────────
if vector_store.doc_count == 0:
    print("\n📥 Collection is empty — ingesting documents...")
    vector_store.reset_collection()
    texts      = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts, show_progress=True)
    vector_store.add_documents(chunks, embeddings)
else:
    print(f"\n✅ Reusing existing collection ({vector_store.doc_count} docs).")


# ── Step 7 — Retriever ────────────────────────────────────────────────────────
retriever = RAGRetriever(vector_store, embedding_manager)


# ── Step 8 — LLM ─────────────────────────────────────────────────────────────
print("\n🤖 Initialising LLM...")
llm = ChatGroq(
    api_key=config.GROQ_API_KEY,
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
    max_tokens=config.MAX_TOKENS,
)


# ── Step 9 — RAG answer function (injected into bot) ─────────────────────────
def answer_fn(query: str) -> str:
    """
    Full RAG pipeline: retrieve → build context → generate → return.

    This is the ONLY function bot.py ever calls. It is completely
    synchronous so it can be safely run in asyncio.to_thread().
    """
    retrieved = retriever.retrieve(
        query,
        top_k=config.TOP_K,
        score_threshold=config.SCORE_THRESHOLD,
    )

    if not retrieved:
        return config.FALLBACK_NO_DOCS

    # Build LLM context from top-scored chunks
    context = "\n\n".join(doc["text"] for doc in retrieved)

    messages = [
        SystemMessage(content=config.SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]

    try:
        response = llm.invoke(messages)
        answer   = response.content.strip()
        return answer if answer else config.FALLBACK_NO_DOCS
    except Exception as exc:
        print(f"❌ LLM error: {exc}")
        return config.FALLBACK_LLM_ERROR


# ── Step 10 — Discord bot ─────────────────────────────────────────────────────
print("\n🎮 Creating Discord bot...")
bot = create_bot(answer_fn)

print("\n✅ Aria is fully initialised and ready.\n")


# ── Step 11 — Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(config.DISCORD_TOKEN)