"""
nodes/rewrite.py — Query rewrite node
=======================================
Rewrites the user's casual language into precise DSM-5 clinical search terms.
"""

from src.ai.chatbot.chat_src.pipeline.state import ChatState, _prompt, _simple_gemini


def rewrite_node(state: ChatState) -> dict:
    if state.get("route") in ("emotional", "conversational"):
        return {"search_query": state["query"]}

    history = state.get("history", [])

    # Topic shift detection — if user signals a new subject, ignore history
    _TOPIC_SHIFT_SIGNALS = [
        "different question", "switching topics", "new topic",
        "change of topic", "forget that", "never mind",
        "totally different", "unrelated topic",
        "moving on", "let me ask something else",
        "one more thing,", "different thing,",
        "by the way,", "wait,",
    ]
    q_lower = state["query"].strip().lower()

    _starts_with_actually = q_lower.startswith("actually") and (
        len(q_lower) <= 8 or q_lower[8] in (" ", ",", ".")
    )

    topic_shift = _starts_with_actually or any(sig in q_lower for sig in _TOPIC_SHIFT_SIGNALS)

    history_str = "None" if topic_shift else (
        "\n".join(f"{m['role']}: {m['content']}" for m in history[-3:]) or "None"
    )

    try:
        prompt       = _prompt("query_rewrite_prompt.txt").format(
            history  = history_str,
            question = state["query"],
        )
        search_query = _simple_gemini(prompt)

        if not search_query or search_query == "OFF_TOPIC":
            return {"search_query": state["query"]}

        return {"search_query": search_query}

    except Exception:
        return {"search_query": state["query"]}
