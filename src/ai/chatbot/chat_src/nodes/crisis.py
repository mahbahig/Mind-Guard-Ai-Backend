"""
nodes/crisis.py — Crisis detection node
=========================================
Two-layer crisis detection: keyword scan + optional LLM soft-signal check.
"""

from src.ai.chatbot.chat_src.pipeline.state import ChatState, _prompt, _simple_gemini
from src.ai.chatbot.chat_src.crisis import crisis_router

_SOFT_SIGNALS = [
    # Existence / living
    "can't live", "cannot live", "cant live",
    "don't want to live", "dont want to live",
    "don't want to be alive", "dont want to be alive",
    "tired of living", "tired of life",
    "no reason to live", "no reason to be here",
    "not worth living", "life is not worth",
    # Disappearing / ending
    "disappear", "end it all", "end my life", "end everything",
    "feel like ending", "want to stop existing",
    "wish i wasn't here", "wish i was dead",
    "want to die", "wanna die", "ready to die",
    "just wanna die", "just want to die",
    # Hopelessness / giving up
    "give up", "giving up on life",
    "can't go on", "cant go on", "cannot go on",
    "can't continue", "cant continue",
    "can't take it anymore", "cant take it anymore",
    "can't do this anymore", "cant do this anymore",
    # Worthlessness
    "not worth it", "nobody cares", "better off without me",
    "better off dead", "everyone would be better",
    # Meaninglessness
    "what's the point", "whats the point",
    "no point anymore", "nothing matters",
    "pointless", "why am i even here",
    # Self-harm adjacent
    "hurt myself", "harm myself",
    # Vague dark
    "don't want to be here", "dont want to be here",
]


def crisis_node(state: ChatState) -> dict:
    """
    Two-layer crisis detection:
      Layer 1 — Fast keyword detector (English + Arabic) via crisis_router().
      Layer 2 — LLM check for indirect/metaphorical crisis language that
                 keywords miss ("I want to disappear", "what's the point anymore").
                 Only fires when soft-signal words are present and layer 1 passed.

    If either layer triggers, returns a language-appropriate crisis response
    immediately — bypassing routing, retrieval, and generation entirely.
    """
    # ── Layer 1: keyword check ────────────────────────────────────────────────
    is_crisis, reason = crisis_router(state["query"])

    if not is_crisis:
        # ── Layer 2: soft-signal LLM check ───────────────────────────────────
        q_lower = state["query"].lower()
        has_soft_signal = any(sig in q_lower for sig in _SOFT_SIGNALS)

        if has_soft_signal:
            try:
                check_prompt = (
                    f"A user of a mental health chatbot sent this message:\n"
                    f"\"{state['query']}\"\n\n"
                    f"Is this message expressing suicidal ideation, self-harm intent, "
                    f"or a genuine mental health crisis? "
                    f"Reply with ONLY one word: YES or NO."
                )
                verdict = _simple_gemini(check_prompt).upper()
                if verdict.startswith("YES"):
                    is_crisis = True
                    reason    = "LLM soft-signal crisis detection"
            except Exception:
                pass

    if not is_crisis:
        return {}

    lang        = state.get("language", "en")
    crisis_file = "crisis_prompt_ar.txt" if lang == "ar" else "crisis_prompt.txt"
    try:
        response = _prompt(crisis_file)
    except FileNotFoundError:
        response = _prompt("crisis_prompt.txt")

    return {"route": "crisis", "responder": "static", "response": response, "sources": []}
