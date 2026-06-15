"""
nodes/history.py — History context node
=========================================
Builds the conversation history context string, with summarisation for
long conversations.
"""

from src.ai.chatbot.chat_src.pipeline.state import ChatState, _simple_gemini


def history_node(state: ChatState) -> dict:
    """
    Builds the conversation history context string.

    Short conversations (≤ 6 messages): use verbatim last 6 messages.
    Long conversations (> 6 messages): summarise older turns with Gemini,
    then combine summary + last 3 exchanges verbatim.
    """
    history = state.get("history", [])

    # ── Short conversation: use verbatim last 6 messages ─────────────────────
    if len(history) <= 6:
        pairs = []
        for msg in history[-6:]:
            if msg["role"] in ("user", "assistant"):
                role = "User" if msg["role"] == "user" else "NEURA"
                pairs.append(f"{role}: {msg['content'][:300]}")
        context = "\n".join(pairs) if pairs else "None"
        return {"history_summary": "", "history_context": context}

    # ── Long conversation: summarise older turns ──────────────────────────────
    older  = history[:-6]
    recent = history[-6:]

    summary = state.get("history_summary", "")
    if not summary:
        try:
            older_text = "\n".join(
                f"{'User' if m['role'] == 'user' else 'NEURA'}: {m['content'][:200]}"
                for m in older
                if m["role"] in ("user", "assistant")
            )
            summary = _simple_gemini(
                f"Summarise this mental health conversation history in 2–3 sentences. "
                f"Focus on the main topics discussed, the user's emotional state, "
                f"and any key insights shared. Be concise.\n\n"
                f"{older_text}"
            )
        except Exception:
            summary = ""

    recent_pairs = []
    for msg in recent:
        if msg["role"] in ("user", "assistant"):
            role = "User" if msg["role"] == "user" else "NEURA"
            recent_pairs.append(f"{role}: {msg['content'][:300]}")

    context_parts = []
    if summary:
        context_parts.append(f"[Earlier in conversation]: {summary}")
    if recent_pairs:
        context_parts.append("\n".join(recent_pairs))

    context = "\n\n".join(context_parts) if context_parts else "None"
    return {"history_summary": summary, "history_context": context}
