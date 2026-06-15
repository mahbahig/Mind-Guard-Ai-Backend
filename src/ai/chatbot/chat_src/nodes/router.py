"""
nodes/router.py — Router node
==============================
Classifies the query as emotional / advice / factual / off_topic using Gemini.
"""

from src.ai.chatbot.chat_src.pipeline.state import ChatState, _prompt, _simple_gemini


def router_node(state: ChatState) -> dict:
    if state.get("route") in ("guard", "crisis", "conversational"):
        return {}

    history_ctx = state.get("history_context", "None")
    history_section = (
        f"Recent conversation:\n{history_ctx}\n\n"
        if history_ctx and history_ctx != "None"
        else ""
    )

    try:
        prompt  = _prompt("router_prompt.txt").format(
            history_section = history_section,
            question        = state["query"],
        )
        verdict = _simple_gemini(prompt).lower()

        if "off_topic"   in verdict: route = "off_topic"
        elif "emotional" in verdict: route = "emotional"
        elif "advice"    in verdict: route = "advice"
        else:                        route = "factual"

        return {"route": route}

    except Exception:
        return {"route": "emotional"}
