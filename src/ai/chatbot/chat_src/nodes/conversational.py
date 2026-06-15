"""
nodes/conversational.py — Conversational node
==============================================
Detects short follow-up replies and positive messages that don't need DSM-5
retrieval. Fast phrase check first; LLM fallback if nothing matched.
"""

from src.ai.chatbot.chat_src.pipeline.state import ChatState, _prompt, _simple_gemini


def conversational_node(state: ChatState) -> dict:
    query   = state["query"]
    history = state.get("history", [])
    q_lower = query.strip().lower()

    # ── Positive message detection ────────────────────────────────────────────
    _POSITIVE_SIGNALS = [
        "i feel great", "i'm happy", "i am happy", "i'm doing great",
        "i am doing great", "feeling good", "feeling great", "doing better",
        "i feel good", "something good happened", "happy now", "feel great",
        "things are better", "i feel better", "i'm feeling better",
        "i'm feeling good", "things got better", "good news", "great news",
        "i feel fine", "i'm fine now", "much better now", "feeling much better",
    ]
    if any(sig in q_lower for sig in _POSITIVE_SIGNALS):
        return {"route": "conversational"}

    # ── Emotional content guard ───────────────────────────────────────────────
    _EMOTIONAL_SIGNALS = {
        "depress", "depressed", "depression", "anxious", "anxiety",
        "sad", "sadness", "lonely", "loneliness", "stressed", "stress",
        "hopeless", "hopelessness", "helpless", "worthless", "empty",
        "numb", "scared", "afraid", "terrified", "angry", "rage",
        "overwhelm", "panic", "grief", "grieving", "trauma", "hurt",
        "pain", "suffer", "suffering", "suicid", "die", "death",
        "cry", "crying", "tears", "breakdown", "burnout", "exhaust",
    }
    if any(sig in q_lower for sig in _EMOTIONAL_SIGNALS):
        return {}

    # ── Short follow-up detection ─────────────────────────────────────────────
    _CONV_WORDS = {
        "yes", "no", "yeah", "nope", "yep", "ok", "okay", "sure",
        "maybe", "idk", "hmm", "hm", "alright", "fine", "good",
        "thanks", "thank you", "not really", "i don't know", "i dont know",
        "kind of", "sort of", "i guess", "perhaps", "probably",
    }

    words  = query.strip().split()
    q_norm = query.strip().lower().rstrip("?.!")
    is_conv = (
        (len(words) <= 4 and q_norm in _CONV_WORDS)
        or (
            len(words) <= 6
            and len(history) >= 2
            and "?" not in query
            and not any(
                kw in query.lower()
                for kw in ("what is", "what are", "how do", "how can",
                           "explain", "tell me", "define", "difference")
            )
        )
    )
    if is_conv:
        return {"route": "conversational"}

    # ── LLM fallback — nothing matched in phrase lists ────────────────────────
    try:
        intent_prompt = _prompt("conversational_intent.txt").format(query=query)
        intent = _simple_gemini(intent_prompt).strip().lower()
        if intent == "conversational":
            return {"route": "conversational"}
        if intent == "emotional":
            return {}
    except Exception:
        pass

    return {}
