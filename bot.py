"""
bot.py — Discord bot layer.

Responsibilities:
  - Define intents, client, and event handlers
  - Accept the RAG answer function via dependency injection
  - Own ALL Discord-specific logic (mention parsing, typing indicator, etc.)
  - Feedback mechanism via 👍 👎 reactions
  - Logging of key events

Design contract:
  - bot.py imports NOTHING from rag/ or app.py
  - The answer_fn is injected at creation time (create_bot)
  - This makes bot.py independently testable without a running RAG pipeline
"""

import asyncio
import logging
import time
import discord

# ── Logging Setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aria.bot")

# ── Metrics (simple in-memory counters) ───────────────────────────────────────

metrics = {
    "total_requests": 0,
    "successful_responses": 0,
    "errors": 0,
    "thumbs_up": 0,
    "thumbs_down": 0,
}


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
    intents.reactions = True        # 👈 needed to listen for reactions

    bot = discord.Client(intents=intents)

    # Track messages Aria responded to for feedback listening
    aria_responses = {}             # {response_message_id: original_query}

    # ── Events ──────────────────────────────────────────────────────────────

    @bot.event
    async def on_ready() -> None:
        logger.info(f"Aria is online — logged in as {bot.user} (ID: {bot.user.id})")
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

        # ── Log incoming request ─────────────────────────────────────────────
        metrics["total_requests"] += 1
        logger.info(
            f"Request #{metrics['total_requests']} | "
            f"User: {message.author} | "
            f"Query: {query[:80]}{'...' if len(query) > 80 else ''}"
        )
        start_time = time.time()

        # Show Discord's typing indicator while processing
        async with message.channel.typing():
            try:
                # Run the blocking RAG call off the event loop
                answer = await asyncio.to_thread(answer_fn, query)
                metrics["successful_responses"] += 1

                # ── Log response time ────────────────────────────────────────
                latency = round(time.time() - start_time, 2)
                logger.info(
                    f"Response #{metrics['successful_responses']} | "
                    f"Latency: {latency}s | "
                    f"User: {message.author}"
                )

            except Exception as exc:
                metrics["errors"] += 1
                logger.error(
                    f"Error #{metrics['errors']} | "
                    f"User: {message.author} | "
                    f"Error: {exc}"
                )
                answer = (
                    "⚠️ Something went wrong on my end. "
                    "Please try again or contact a human assistant. 🙋"
                )

        # Send the response
        response_msg = await message.channel.send(
            f"{message.author.mention} {answer}"
        )

        # ── Add feedback reactions ────────────────────────────────────────────
        await response_msg.add_reaction("👍")
        await response_msg.add_reaction("👎")

        # Track this response for feedback listening
        aria_responses[response_msg.id] = {
            "query": query,
            "user": str(message.author),
        }

        logger.info(
            f"Response sent to {message.author} | "
            f"Message ID: {response_msg.id}"
        )

    @bot.event
    async def on_reaction_add(reaction: discord.Reaction, user: discord.User) -> None:
        # Ignore bot's own reactions
        if user.bot:
            return

        # Only track reactions on Aria's responses
        if reaction.message.id not in aria_responses:
            return

        original = aria_responses[reaction.message.id]

        if str(reaction.emoji) == "👍":
            metrics["thumbs_up"] += 1
            logger.info(
                f"Feedback: 👍 POSITIVE | "
                f"User: {user} | "
                f"Query: {original['query'][:80]}"
            )

        elif str(reaction.emoji) == "👎":
            metrics["thumbs_down"] += 1
            logger.info(
                f"Feedback: 👎 NEGATIVE | "
                f"User: {user} | "
                f"Query: {original['query'][:80]}"
            )

    return bot