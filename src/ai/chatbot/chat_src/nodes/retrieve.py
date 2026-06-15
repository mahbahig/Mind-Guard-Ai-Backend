"""
nodes/retrieve.py — Retrieve node
===================================
Hybrid BM25 + vector retrieval over DSM-5-TR, followed by a keyword reranker.
"""

from src.ai.chatbot.chat_src.pipeline.state import ChatState


def retrieve_node(state: ChatState) -> dict:
    if state.get("route") in ("emotional", "conversational", "guard", "crisis"):
        return {"context": "", "sources": []}

    from src.ai.chatbot.chat_src.rag2 import _RETRIEVER, _BM25_INDEX, retrieve_parent_sections, _rerank_sections
    from src.ai.chatbot.chat_src.hybrid_retriever import hybrid_retrieve

    if _RETRIEVER is None:
        return {"context": "", "sources": []}

    search_query = state.get("search_query") or state["query"]
    _MIN_SCORE   = 0.45

    try:
        if _BM25_INDEX:
            sections = hybrid_retrieve(_RETRIEVER, _BM25_INDEX, search_query)
        else:
            sections = retrieve_parent_sections(_RETRIEVER, search_query)

        sections = _rerank_sections(sections, search_query, top_k=3)

        # Drop sections below confidence threshold
        sections = [s for s in sections if s.get("score", 1.0) >= _MIN_SCORE]

        if not sections:
            return {"context": "", "sources": []}

        context = "\n\n---\n\n".join(s["content"] for s in sections)

        seen, sources = set(), []
        for s in sections:
            meta  = s["metadata"]
            entry = (
                f"{meta.get('category', meta.get('folder', ''))} "
                f"— {meta.get('source', '')}"
            )
            sect = meta.get("section", "")
            if sect:
                entry += f" ({sect[:40]})"
            if entry not in seen:
                seen.add(entry)
                sources.append(entry)

        return {"context": context, "sources": sources}

    except Exception:
        return {"context": "", "sources": []}
