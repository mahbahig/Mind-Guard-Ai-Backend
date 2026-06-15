"""
graph.py — LangGraph Orchestration for NEURA
=============================================
Replaces the monolithic query_rag() with a proper graph:
each step is an isolated node, edges handle routing conditionally.

Flow:
  guard → language → crisis → history → conversational
        → router → rewrite → retrieve → generate → postprocess

Public API:
    from graph import run_chat
    response, sources, message_id = run_chat(query, history, llm_backend)

Requires:
    pip install langgraph
"""

import uuid

from langgraph.graph import StateGraph, END

# ── State ──────────────────────────────────────────────────────────────────────
from src.ai.chatbot.chat_src.pipeline.state import ChatState

# ── Nodes ──────────────────────────────────────────────────────────────────────
from src.ai.chatbot.chat_src.nodes import (
    guard_node, language_node, crisis_node, history_node,
    conversational_node, router_node, rewrite_node,
    retrieve_node, generate_node, postprocess_node,
)


# ── Routing conditions ─────────────────────────────────────────────────────────

def _after_guard(state: ChatState) -> str:
    return "end" if state.get("route") == "guard" else "language"

def _after_crisis(state: ChatState) -> str:
    return "end" if state.get("route") == "crisis" else "history"

def _after_router(state: ChatState) -> str:
    route = state.get("route", "factual")
    if route in ("emotional", "conversational", "off_topic", "advice"):
        return "generate"
    return "rewrite"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(ChatState)

    # Register nodes
    g.add_node("guard",          guard_node)
    g.add_node("language",       language_node)
    g.add_node("crisis",         crisis_node)
    g.add_node("history",        history_node)
    g.add_node("conversational", conversational_node)
    g.add_node("router",         router_node)
    g.add_node("rewrite",        rewrite_node)
    g.add_node("retrieve",       retrieve_node)
    g.add_node("generate",       generate_node)
    g.add_node("postprocess",    postprocess_node)

    # Entry point
    g.set_entry_point("guard")

    # Edges
    g.add_conditional_edges(
        "guard", _after_guard,
        {"end": END, "language": "language"},
    )
    g.add_edge("language",       "crisis")
    g.add_conditional_edges(
        "crisis", _after_crisis,
        {"end": END, "history": "history"},
    )
    g.add_edge("history",        "conversational")
    g.add_edge("conversational", "router")
    g.add_conditional_edges(
        "router", _after_router,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    g.add_edge("rewrite",        "retrieve")
    g.add_edge("retrieve",       "generate")
    g.add_edge("generate",       "postprocess")
    g.add_edge("postprocess",    END)

    return g.compile()


# ── Compile once on import ─────────────────────────────────────────────────────
_GRAPH = _build_graph()


# ── Public API ─────────────────────────────────────────────────────────────────

def run_chat(
    query:           str,
    history:         list = [],
    llm_backend:     str  = "auto",
    history_summary: str  = "",
) -> tuple[str, list, str, str]:
    """
    Main entry point for the NEURA chatbot pipeline.

    Args:
        query:           User's raw message
        history:         Full conversation history (list of role/content dicts)
        llm_backend:     "auto" | "gemini" | "vertex_tuned"
        history_summary: Cached summary from the previous turn (avoids re-summarising
                         older messages on every call for long conversations).

    Returns:
        (response_text, sources, message_id, history_summary)
        — caller should persist history_summary and pass it back next turn.
    """
    initial: ChatState = {
        "query":           query,
        "history":         history or [],
        "llm_backend":     llm_backend,
        "language":        "en",
        "history_summary": history_summary,
        "history_context": "",
        "route":           "",
        "search_query":    "",
        "context":         "",
        "response":        "",
        "sources":         [],
        "responder":       "",
    }

    result      = _GRAPH.invoke(initial)
    new_summary = result.get("history_summary", "")

    return result["response"], result["sources"], new_summary, result.get("responder", "")
