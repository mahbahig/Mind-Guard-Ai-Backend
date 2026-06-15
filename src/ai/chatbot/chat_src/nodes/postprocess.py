"""
nodes/postprocess.py — Postprocess node
=========================================
Final cleanup: safety-filter detection, banned opener stripping,
language-aware closing question enforcement.
"""

import random
import re

from src.ai.chatbot.chat_src.pipeline.state import ChatState, _prompt, _simple_gemini

_SAFETY_FILTER_PHRASES = [
    "i can't continue this conversation",
    "i cannot continue this conversation",
    "i'm not able to continue this conversation",
    "if you are experiencing thoughts of self-harm",
    "if you're experiencing thoughts of self-harm",
    "please seek help from a qualified mental health",
    "seek immediate help from a mental health professional",
    "i'm unable to assist with this request",
    "i am unable to assist with this request",
]

_EN_CLOSINGS = [
    "What's been on your mind lately?",
    "How are you feeling about that?",
    "What's the hardest part right now?",
    "Would you like to talk more about it?",
    "What does that feel like for you?",
    "How long have you been feeling this way?",
]

_AR_CLOSINGS = [
    "ما الذي يشغل بالك مؤخراً؟",
    "كيف تشعر حيال ذلك؟",
    "ما أصعب جزء في هذا الأمر بالنسبة لك؟",
    "هل تريد أن تتحدث عن ذلك أكثر؟",
    "كيف يبدو ذلك بالنسبة لك؟",
    "منذ متى وأنت تشعر بهذا؟",
]


def postprocess_node(state: ChatState) -> dict:
    from src.ai.chatbot.chat_src.llm import _fix_opener, _gemini_generate

    response = state.get("response", "")
    language = state.get("language", "en")

    # ── Safety-filter detection: phrase list first, LLM fallback ─────────────
    is_refusal = any(phrase in response.lower() for phrase in _SAFETY_FILTER_PHRASES)

    if not is_refusal:
        try:
            safety_prompt = _prompt("safety_check.txt").format(response=response)
            verdict = _simple_gemini(safety_prompt).strip().upper()
            if verdict.startswith("YES"):
                is_refusal = True
        except Exception:
            pass

    if is_refusal:
        query = state.get("query", "")
        try:
            regen_prompt = _prompt("gemini_safety_regen.txt").format(query=query)
            response     = _gemini_generate(regen_prompt)
        except Exception:
            response = "yeah, I hear you — thanks for sharing that with me. how are you feeling right now?"

    response = _fix_opener(response)

    # ── Ensure emotional responses end with a question ────────────────────────
    route = state.get("route", "")
    if route in ("emotional", "conversational") and response:
        if not re.search(r'[?؟]\s*$', response.strip()):
            closings = _AR_CLOSINGS if language == "ar" else _EN_CLOSINGS
            response = response.rstrip() + "\n\n" + random.choice(closings)

    return {"response": response}
