"""
nodes/guard.py — Guard node
============================
First stop. Uses LLM intent classification to catch memory queries, about-me
queries, meta questions, and trivial input — without fragile phrase lists.
"""

from src.ai.chatbot.chat_src.pipeline.state import ChatState, _prompt, _simple_gemini


def guard_node(state: ChatState) -> dict:
    query = state["query"]

    # Trivial input — skip LLM for empty/meaningless strings
    if len(query.strip().strip(".,!?;:-—")) < 2:
        return {
            "route":     "guard",
            "responder": "static",
            "response":  "hey, I'm here — take your time, no rush to find the words.",
            "sources":   [],
        }

    # LLM intent classification
    try:
        intent_prompt = _prompt("guard_intent.txt").format(query=query)
        intent = _simple_gemini(intent_prompt).strip().lower()
    except Exception:
        intent = "none"

    # ── memory_query ─────────────────────────────────────────────────────────
    if intent == "memory_query":
        history = state.get("history", [])
        n = len(history) // 2
        if n == 0:
            response = (
                "I don't have memory of previous sessions — each conversation "
                "starts fresh. But I'm here now and ready to listen. "
                "What's on your mind today?"
            )
        else:
            summary = state.get("history_summary", "").strip()
            if summary:
                response = (
                    f"From what we've covered so far: {summary}\n\n"
                    "Is there something specific from that you'd like to come back to?"
                )
            else:
                # Ask Gemini to summarize the conversation so far
                lines = []
                for msg in history[-10:]:
                    role = "User" if msg.get("role") == "user" else "NEURA"
                    lines.append(f"{role}: {msg.get('content', '')[:200]}")
                transcript = "\n".join(lines)
                try:
                    summary_prompt = (
                        f"Summarize this mental health conversation in 2-3 warm, natural sentences "
                        f"as if you're reminding the user what you discussed. "
                        f"Write in first person as NEURA.\n\n{transcript}"
                    )
                    summary = _simple_gemini(summary_prompt).strip()
                    response = f"{summary}\n\nIs there something specific you'd like to come back to?"
                except Exception:
                    response = (
                        "We've been talking for a bit — I remember everything you've shared. "
                        "Is there something specific you'd like to come back to?"
                    )
        return {"route": "guard", "responder": "static", "response": response, "sources": []}

    # ── about_me_query ───────────────────────────────────────────────────────
    if intent == "about_me_query":
        history = state.get("history", [])
        n = len(history) // 2
        if n == 0:
            response = (
                "We've just started talking, so I don't know much about you yet — "
                "but I'm genuinely here to listen. What would you like to share?"
            )
        else:
            response = (
                "From our conversation so far, I can see you've shared some things "
                "that matter to you — I hold those with care. I don't store anything "
                "between separate sessions, but within our chat right now I'm following "
                "everything you've told me. What's on your mind?"
            )
        return {"route": "guard", "responder": "static", "response": response, "sources": []}

    # ── meta_query ───────────────────────────────────────────────────────────
    if intent == "meta_query":
        return {
            "route":     "guard",
            "responder": "static",
            "response":  (
                "I'm NEURA — a mental health companion here to listen, support, "
                "and offer a safe space whenever you need it. I can help you "
                "work through difficult emotions, suggest coping strategies, or "
                "answer questions about mental health — all without judgment.\n\n"
                "I'm not a therapist or a replacement for professional care, but "
                "I'm here whenever you need someone to talk to. "
                "How are you feeling right now?"
            ),
            "sources":   [],
        }

    # ── trivial (LLM-detected) ───────────────────────────────────────────────
    if intent == "trivial":
        return {
            "route":     "guard",
            "responder": "static",
            "response":  "hey, I'm here — take your time, no rush to find the words.",
            "sources":   [],
        }

    # ── none — pass through to the rest of the pipeline ─────────────────────
    return {}
