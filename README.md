# Aria — AI PM Bootcamp Discord RAG Bot

Aria is a Discord assistant for the AI PM Bootcamp. She answers questions when mentioned with `@Aria`, powered by a RAG pipeline built on ChromaDB and Groq's LLaMA model.

---

## Project Structure

```
aria_bot/
│
├── app.py                  # Entry point — wires RAG pipeline into the Discord bot
├── bot.py                  # Discord client, events, and mention handling
├── config.py               # All env vars and tunable constants (single source of truth)
├── requirements.txt        # Python dependencies
├── .env.example            # Template for your .env file
│
├── src/
│   ├── __init__.py         # Public API — the only rag/ import other files need
│   ├── data_loader.py      # Load .md files and split into chunks
│   ├── embedding.py        # SentenceTransformer wrapper (text → vectors)
│   ├── vectorstore.py      # ChromaDB persistence layer
│   └── search.py           # Semantic retrieval — query → ranked chunks
│
└── data/
    ├── text_files/         # Drop your .md source documents here
    └── vector_store/       # Auto-created by ChromaDB on first run
```

---

## How It Works

```
User @mentions Aria in Discord
        │
        ▼
   bot.py receives message
        │
        ▼
   answer_fn(query)          ← injected from app.py
        │
        ├── RAGRetriever.retrieve(query)
        │       ├── EmbeddingManager.generate_embeddings(query)
        │       └── ChromaDB.query() → top-k chunks
        │
        ├── Build LLM context from chunks
        │
        └── ChatGroq.invoke(system_prompt + context + question)
                │
                ▼
        Answer sent back to Discord channel
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone <your-repo-url>
cd aria_bot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
DISCORD_TOKEN=your_discord_bot_token_here
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Add your source documents

Place your `.md` files inside `data/text_files/`. Subdirectories are supported — the loader discovers files recursively.

```
data/
└── text_files/
    ├── AI Bootcamp Journey & Learning Path.md
    ├── Intern FAQ - AI Bootcamp.md
    └── Training For AI Engineer Interns.md
```

### 4. Run the bot

```bash
python app.py
```

On first run, Aria will:
1. Load and chunk all `.md` files
2. Download the embedding model (cached after first run)
3. Embed all chunks and store them in ChromaDB
4. Start the Discord bot

On subsequent runs, the existing ChromaDB collection is reused — no re-embedding needed.

---

## Configuration

All tunable settings live in `config.py`. You can change these without touching any other file:

| Constant | Default | Description |
|---|---|---|
| `DATA_DIR` | `./data/text_files` | Folder containing source `.md` files |
| `VECTOR_STORE_DIR` | `./data/vector_store` | ChromaDB persistence path |
| `COLLECTION_NAME` | `documents` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `multi-qa-mpnet-base-dot-v1` | SentenceTransformer model |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq LLM model |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `SCORE_THRESHOLD` | `0.2` | Minimum cosine similarity to include a chunk |
| `TEMPERATURE` | `0.1` | LLM temperature (lower = more factual) |
| `MAX_TOKENS` | `1024` | Max tokens in LLM response |

---

## Re-ingesting Documents

If you add new documents or change the embedding model, you need to reset the vector store. The easiest way is to delete the ChromaDB folder and restart:

```bash
rm -rf data/vector_store
python app.py
```

The bot detects an empty collection on startup and automatically re-ingests everything.

---

## Module Responsibilities

Each file has a single, well-defined job. Nothing crosses these boundaries except through `app.py`.

**`config.py`** — reads `.env`, defines all constants. No business logic.

**`rag/data_loader.py`** — loads `.md` files from disk and splits them into overlapping chunks using `RecursiveCharacterTextSplitter`. No embedding or storage logic.

**`rag/embedding.py`** — wraps `SentenceTransformer`. Accepts a list of strings, returns a numpy array. No knowledge of ChromaDB or Discord.

**`rag/vectorstore.py`** — manages the ChromaDB collection. Handles create, reset, and batch insert. Does not embed anything itself.

**`rag/search.py`** — the only file that calls `collection.query()`. Converts a query string into a ranked list of relevant chunks. No LLM logic.

**`bot.py`** — owns all Discord logic. Accepts `answer_fn` as an injected dependency so it has zero knowledge of RAG, ChromaDB, or Groq.

**`app.py`** — the only file that imports from both `rag/` and `bot.py`. Wires everything together and defines `answer_fn`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `discord.py` | Discord bot client |
| `python-dotenv` | Load `.env` file |
| `langchain` / `langchain-core` | Document types, text splitter |
| `langchain-community` | `UnstructuredMarkdownLoader` |
| `langchain-groq` | Groq LLM client |
| `sentence-transformers` | Local embedding model |
| `chromadb` | Persisted vector store |
| `unstructured[md]` | Markdown parsing backend |

---

## Discord Bot Setup

If you haven't created your Discord bot yet:

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications)
2. Create a new application → Bot → copy the token into `.env`
3. Under **Privileged Gateway Intents**, enable **Message Content Intent**
4. Generate an invite URL under **OAuth2 → URL Generator** with scopes: `bot` and permissions: `Send Messages`, `Read Message History`, `View Channels`
5. Invite the bot to your server

---

## Example Usage

Once the bot is running and invited to your server:

```
@Aria what is the team size for the AI PM Bootcamp?
```

```
@Aria Typically, a team in the AI PM Bootcamp consists of 8–10 members. 
Larger-scope projects may be assigned bigger teams. Teams with multiple 
PMs and developers tend to launch more ambitious, successful products! 🚀
```

```
@Aria when are office hours?
```

```
@Aria Office hours are held every Saturday! Make sure to join to discuss 
your thoughts and issues with your projects with the mentors. 🙋
```

---

## License

MIT
