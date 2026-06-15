"""
nodes/language.py — Language detection node
=============================================
Detects whether the user's message is Arabic or English.
"""

from src.ai.chatbot.chat_src.pipeline.state import ChatState, _is_arabic


def language_node(state: ChatState) -> dict:
    lang = "ar" if _is_arabic(state["query"]) else "en"
    return {"language": lang}
