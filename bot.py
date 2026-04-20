"""
bot.py — Discord bot layer.

Responsibilities:
  - Define intents, client, and event handlers
  - Accept the RAG answer function via dependency injection
  - Own ALL Discord-specific logic (mention parsing, typing indicator, etc.)
  - Provide keep_alive() for Render port binding

Design contract:
  - bot.py imports NOTHING from rag/ or app.py
  - The answer_fn is injected at creation time (create_bot)
  - This makes bot.py independently testable without a running RAG pipeline
"""

import os
import asyncio
import discord
from flask import Flask
from threading import Thread


# ── Keep Alive Server ──────────────────────────────────────────────────────────

_app = Flask(__name__)

@_app.route('/')
def home():
    return "Aria is alive!"

def _run():
    port = int(os.environ.get('PORT', 10000))
    print(f"\n🌐 Flask starting on port {port}...")
    _app.run(host='0.0.0.0', port=port, use_reloader=False)

def keep_alive():
    t = Thread(target=_run)
    t.daemon = True
    t.start()
    print(f"\n🌐 Keep-alive thread launched")


# ── Bot Factory ────────────────────────────────────────────────────────────────

def create_bot(answer_fn) -> discord.Client:
    """
    Build and return a configured Discord client.

    Args:
        answer_fn: A callable with signature
                   ``(query: str) -> str``
                   that returns a Discord-ready answer string.
                   In production this is app.py's rag_answer wrapper.

    Returns:
        A discord.Client instance ready to be started with bot.run(token).
    """
    intents = discord.Intents.default()
    intents.message_content = True

    bot = discord.Client(intents=intents)

    # ── Events ──────────────────────────────────────────────────────────────

    @bot.event
    async def on_ready() -> None:
        print(f"\n🤖 Aria is online — logged in as {bot.user} (ID: {bot.user.id})")
        print("─" * 50)

    @bot.event
    async def on_message(message: discord.Message) -> None:
        # Ignore all bot messages (including Aria's own)
        if message.author.bot:
            return

        # Only respond when @Aria is mentioned
        if bot.user not in message.mentions:
            return

        # Strip both mention formats and clean up whitespace
        query = (
            message.content
            .replace(f"<@{bot.user.id}>",  "")
            .replace(f"<@!{bot.user.id}>", "")
            .strip()
        )

        # Empty mention — greet the user
        if not query:
            await message.channel.send(
                f"Hey {message.author.mention}! 👋  "
                f"I'm **Aria**, your AI PM Bootcamp assistant.\n"
                f"Ask me anything about the bootcamp and I'll do my best to help! 😊"
            )
            return

        # Show Discord's typing indicator while processing
        async with message.channel.typing():
            try:
                # Run the blocking RAG call off the event loop
                answer = await asyncio.to_thread(answer_fn, query)
            except Exception as exc:
                print(f"❌ Unexpected error in on_message: {exc}")
                answer = (
                    "⚠️ Something went wrong on my end. "
                    "Please try again or contact a human assistant. 🙋"
                )

        await message.channel.send(f"{message.author.mention} {answer}")

    return bot