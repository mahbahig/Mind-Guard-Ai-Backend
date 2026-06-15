"""
pipeline/state.py — NEURA ChatState and shared helpers
=======================================================
Contains the ChatState TypedDict and utility functions used across nodes.
"""

from pathlib import Path
from typing_extensions import TypedDict

PROMPTS = Path(__file__).parent.parent / "prompts"


# ── State ──────────────────────────────────────────────────────────────────────


class ChatState(TypedDict):
    # ── Input fields ───────────────────────────────────────────────────────────
    query: str
    history: list
    llm_backend: str

    # ── Computed during flow ───────────────────────────────────────────────────
    language: str  # "en" | "ar"
    history_summary: str  # Gemini summary of older turns (> 6 messages)
    history_context: str  # summary + last 3 exchanges, passed to LLMs
    route: str  # "guard"|"crisis"|"conversational"|"emotional"|"advice"|"factual"
    search_query: str
    context: str

    # ── Output ────────────────────────────────────────────────────────────────
    response: str
    sources: list
    responder: (
        str  # "Llama" | "Gemini" | "Gemini (fallback)" | "Gemini (simple)" | "static"
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _prompt(filename: str) -> str:
    """Load a prompt file from the prompts directory."""
    return (PROMPTS / filename).read_text(encoding="utf-8")


def _is_arabic(text: str) -> bool:
    """True if message contains Arabic Unicode characters."""
    return any("؀" <= c <= "ۿ" for c in text)


def _simple_gemini(prompt: str) -> str:
    """Lightweight Gemini call used internally for summarisation and checks."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    content = model.invoke(prompt).content
    # Newer langchain-google-genai versions return a list instead of a str
    if isinstance(content, list):
        content = " ".join(
            part if isinstance(part, str) else part.get("text", "") for part in content
        )
    return content.strip()
