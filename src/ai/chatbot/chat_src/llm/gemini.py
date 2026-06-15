"""
llm/gemini.py — Gemini generation helpers
==========================================
Gemini generation, polish, opener fixing.
"""

from src.ai.chatbot.chat_src.pipeline.state import _prompt

_BANNED_OPENERS = [
    "i'm so sorry to hear",
    "i am so sorry to hear",
    "i'm sorry to hear",
    "i am sorry to hear",
    "it sounds incredibly",
    "it's completely understandable",
    "it is completely understandable",
    "i can imagine how",
    "i can only imagine",
]


def _gemini_generate(prompt: str) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI
    model   = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    content = model.invoke(prompt).content
    if isinstance(content, list):
        content = " ".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )
    return content


def _fix_opener(text: str) -> str:
    """Strips banned formulaic openers from the start of a response."""
    lower = text.lower().strip()
    for phrase in _BANNED_OPENERS:
        if lower.startswith(phrase):
            for i, ch in enumerate(text):
                if ch in ".!?" and i > 10:
                    remainder = text[i+1:].strip()
                    if remainder:
                        return remainder
            break
    return text


def _polish_with_gemini(raw: str, query_text: str) -> str | None:
    """Lightly polishes a Llama draft with Gemini. Returns None if raw is too short."""
    if len(raw.split()) < 15:
        return None

    from langchain_google_genai import ChatGoogleGenerativeAI
    model        = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    polish_prompt = _prompt("gemini_polish.txt").format(query=query_text, raw=raw)
    try:
        content = model.invoke(polish_prompt).content
        if isinstance(content, list):
            content = " ".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in content
            )
        return content.strip()
    except Exception:
        return raw


def _polish_if_needed(raw: str, query_text: str, fallback_prompt: str) -> str:
    """Returns cleaned Llama response, or Gemini fallback if too short."""
    if not raw or len(raw.split()) < 15:
        return _gemini_generate(fallback_prompt)
    return _fix_opener(raw)
