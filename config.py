"""
config.py — Centralised configuration for Aria Discord RAG Bot
All env vars and tunable constants live here. Nothing else should import os.getenv directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Credentials ───────────────────────────────────────────────────────────────
DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
GROQ_API_KEY: str  = os.getenv("GROQ_API_KEY", "")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR: str         = "./data/text_files"       # folder with .md source files
VECTOR_STORE_DIR: str = "./data/vector_store"     # persisted ChromaDB path

# ── RAG ───────────────────────────────────────────────────────────────────────
COLLECTION_NAME: str    = "documents"
EMBEDDING_MODEL: str    = "multi-qa-mpnet-base-dot-v1"
TOP_K: int              = 5
SCORE_THRESHOLD: float  = 0.2

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL: str   = "llama-3.1-8b-instant"
TEMPERATURE: float = 0.1
MAX_TOKENS: int    = 1024

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = """You are Aria, a friendly and knowledgeable Discord assistant for the AI PM Bootcamp.
You help students, interns, and team members get quick answers about the bootcamp.

PERSONALITY:
- Warm, encouraging and professional
- Speak naturally like a helpful team member, not a robot
- Use light formatting (bold, bullet points) to make answers readable in Discord

RULES:
1. Always answer in a conversational, assistant-like tone.
2. Use the context provided to give the most complete answer you can.
3. If the context partially answers the question — answer what you can, then say:
   "For more details, feel free to ask or reach out to a human assistant! 🙋"
4. If the context has absolutely no relevance to the question, say:
   "Hmm, I don't have information on that just yet. Please reach out to a human assistant for help 🙋"
5. Never mention documents, scores, rankings, or that you're searching anything.
6. Never say "based on the context provided" — just answer naturally.
7. Keep answers concise but complete — no walls of text.

ANSWER FORMAT (for Discord):
- Use **bold** for key terms
- Use bullet points for lists
- Keep responses under 10 lines where possible
- Add a relevant emoji occasionally to keep it friendly 😊
"""

FALLBACK_NO_DOCS: str   = "⚠️ I couldn't find any relevant information. Please contact a human assistant for help. 🙋"
FALLBACK_LLM_ERROR: str = "⚠️ Something went wrong on my end. Please try again or contact a human assistant. 🙋"

# ── Validation ────────────────────────────────────────────────────────────────
def validate():
    """Raise early if critical env vars are missing."""
    missing = [k for k, v in {"DISCORD_TOKEN": DISCORD_TOKEN, "GROQ_API_KEY": GROQ_API_KEY}.items() if not v]
    if missing:
        raise EnvironmentError(f"❌ Missing required environment variables: {', '.join(missing)}")