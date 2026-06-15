"""
rag2.py — NEURA DSM-5 RAG Pipeline
====================================
Hybrid BM25 + vector retrieval over DSM-5-TR knowledge base.
Supports three LLM backends:
  gemini       — Gemini 2.5 Flash (default)
  vertex_tuned — Fine-tuned Llama 3.1 on Vertex AI Model Garden
  auto         — Smart mix: Llama primary, Gemini fallback
"""

import os
import pickle
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.ai.chatbot.chat_src.hybrid_retriever import build_bm25_index, hybrid_retrieve

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent.parent
CHROMA_PATH       = str(BASE_DIR / "chroma_db")
PARENT_STORE_PATH = str(BASE_DIR / "parent_store.pkl")
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
TOP_K_CHILDREN    = 6


# ── Retriever setup ────────────────────────────────────────────────────────────
def load_retriever():
    """
    Loads ChromaDB (child vectors) and parent_store.pkl (full sections).
    Called once at module startup — not per query.
    """
    try:
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function,
            collection_metadata={"hnsw:space": "cosine"}
        )
        with open(PARENT_STORE_PATH, "rb") as f:
            parent_store = pickle.load(f)
        return {"vectorstore": vectorstore, "parent_store": parent_store}
    except FileNotFoundError:
        return None
    except Exception:
        return None


def retrieve_parent_sections(retriever: dict, query: str, k: int = TOP_K_CHILDREN) -> list:
    """
    Searches child chunks for precision, returns full parent sections for context.
    LLM always sees complete disorder sections, not fragments.
    """
    try:
        vectorstore  = retriever["vectorstore"]
        parent_store = retriever["parent_store"]

        child_results   = vectorstore.similarity_search(query, k=k)
        seen_parent_ids = {}
        for child in child_results:
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen_parent_ids:
                seen_parent_ids[pid] = child.metadata

        parent_sections = []
        for pid in seen_parent_ids:
            if pid in parent_store:
                data = parent_store[pid]
                parent_sections.append({
                    "content":  data["content"],
                    "metadata": data["metadata"],
                })
        return parent_sections

    except Exception:
        return []


# ── Module-level initialization ────────────────────────────────────────────────
_RETRIEVER  = load_retriever()

_BM25_INDEX = None
if _RETRIEVER:
    try:
        _BM25_INDEX = build_bm25_index(_RETRIEVER["parent_store"])
    except Exception:
        pass


def _rerank_sections(sections: list, query: str, top_k: int = 3) -> list:
    """
    Keyword-overlap reranker. Keeps the top_k most relevant parent sections.
    Title matches are weighted 5x over content matches.
    """
    if len(sections) <= top_k:
        return sections

    query_words = set(query.lower().split())
    stop_words  = {"the", "a", "an", "is", "it", "of", "and", "or", "in",
                   "to", "i", "me", "my", "do", "what", "how", "feel"}
    query_words -= stop_words

    def _score(section: dict) -> float:
        text       = section["content"].lower()
        hits       = sum(text.count(w) for w in query_words)
        title      = section["metadata"].get("section", "").lower()
        title_hits = sum(1 for w in query_words if w in title)
        return hits + title_hits * 5

    return sorted(sections, key=_score, reverse=True)[:top_k]
