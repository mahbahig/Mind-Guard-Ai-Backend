"""
hybrid_retriever.py — Phase 5: BM25 + Vector Hybrid Retrieval
=============================================================
Combines dense vector search (semantic meaning) with sparse BM25
(keyword frequency) using Reciprocal Rank Fusion (RRF).

Why this matters:
  - Vector search: finds semantically similar content ("feeling hopeless" ↔ depression)
  - BM25 search:   finds exact clinical terms ("F32.1", "sertraline", "Criterion A")
  - RRF fusion:    sections appearing in BOTH lists get a strong combined boost

Public API (used by rag2.py):
  build_bm25_index(parent_store) → dict   called once at startup
  hybrid_retrieve(retriever, bm25_data, query, k) → list[dict]   called per query
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


# ── Tokenizer ──────────────────────────────────────────────────────────────────

def tokenise(text: str) -> List[str]:
    """
    Tokenizes text for BM25 indexing and search.

    Preserves:
        - Clinical codes: F32.1, Z63.0, ICD-style alpha-numeric codes
        - Multi-word clinical terms are split naturally (BM25 handles co-occurrence)
        - Term frequency is preserved (no deduplication — BM25 needs repeats)

    Removes:
        - Single characters
        - Pure numbers shorter than 3 digits (page numbers, list markers)

    Args:
        text: Raw text content or query string.

    Returns:
        List of lowercase tokens. Never empty (returns [''] as fallback).
    """
    text = text.lower()

    # Regex captures:
    #   [a-z]\d+\.\d+   → clinical codes like f32.1, z63.0
    #   [a-z0-9]{2,}    → words and alphanumeric tokens ≥2 chars
    tokens = re.findall(r'[a-z]\d+\.\d+|[a-z0-9]{2,}', text)

    # Keep pure numbers only if they look like meaningful codes (3+ digits)
    tokens = [t for t in tokens if not t.isdigit() or len(t) >= 3]

    return tokens if tokens else ['']


# ── BM25 Index Builder ─────────────────────────────────────────────────────────

def build_bm25_index(parent_store: dict) -> dict:
    """
    Builds a BM25Okapi index from all parent sections.

    Called once at module startup (same pattern as loading ChromaDB).
    Stores sections in a parallel list so BM25 score indices map back
    directly to parent section dicts.

    Args:
        parent_store: The loaded parent_store.pkl dict.
                      Keys: parent IDs (e.g. "parent_000042")
                      Values: {"content": str, "metadata": dict}

    Returns:
        dict with:
            "index"    → BM25Okapi object ready for scoring
            "sections" → list of section dicts in the same order as the index
            "size"     → number of sections indexed

    Raises:
        ImportError: If rank_bm25 is not installed.
        Exception:   Any other build failure (caught by caller).
    """
    from rank_bm25 import BM25Okapi

    sections = []
    corpus   = []

    for pid, data in parent_store.items():
        section = {
            "content":  data["content"],
            "metadata": data["metadata"],
        }
        sections.append(section)
        corpus.append(tokenise(data["content"]))

    index = BM25Okapi(corpus)

    logger.info(
        f"BM25 index built: {len(sections):,} sections, "
        f"avg doc length: {index.avgdl:.0f} tokens"
    )

    return {
        "index":    index,
        "sections": sections,
        "size":     len(sections),
    }


# ── BM25 Search ────────────────────────────────────────────────────────────────

def bm25_search(bm25_data: dict, query: str, k: int = 10) -> List[dict]:
    """
    Returns top-k parent sections by BM25 score for the given query.

    Sections with score 0 are excluded (no keyword overlap at all).

    Args:
        bm25_data: The dict returned by build_bm25_index().
        query:     User query or transformed search string.
        k:         Number of results to return.

    Returns:
        List of section dicts, sorted by BM25 score descending.
        May be shorter than k if fewer sections have non-zero scores.
    """
    tokens   = tokenise(query)
    scores   = bm25_data["index"].get_scores(tokens)
    sections = bm25_data["sections"]

    # Sort indices by score descending, keep only positive scores
    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []
    for i in ranked[:k]:
        if scores[i] > 0:
            results.append(sections[i])

    logger.info(
        f"BM25 search | query='{query[:50]}' | "
        f"tokens={tokens[:8]} | results={len(results)}"
    )
    return results


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    vector_results: List[dict],
    bm25_results:   List[dict],
    rrf_k:          int = 60,
) -> List[dict]:
    """
    Merges two ranked lists using Reciprocal Rank Fusion (RRF).

    Formula:  RRF_score(d) = Σ  1 / (rrf_k + rank(d, list_i))

    A section appearing in both lists gets contributions from both,
    making it rank higher than sections appearing in only one.

    rrf_k = 60 is the standard value (Cormack et al. 2009).
    Higher rrf_k → less weight on top ranks (smoother merging).
    Lower  rrf_k → strongly favors top-ranked results.

    Args:
        vector_results: Ranked list of section dicts from vector search.
        bm25_results:   Ranked list of section dicts from BM25 search.
        rrf_k:          RRF constant (default 60).

    Returns:
        Merged list of section dicts sorted by combined RRF score.
    """
    scores   = {}   # parent_id → cumulative RRF score
    sections = {}   # parent_id → section dict (first seen wins)

    def _score_list(results: List[dict], k: int):
        for rank, section in enumerate(results):
            pid = section["metadata"].get("parent_id", f"idx_{rank}")
            scores[pid]   = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            if pid not in sections:
                sections[pid] = section

    _score_list(vector_results, rrf_k)
    _score_list(bm25_results,   rrf_k)

    sorted_pids = sorted(scores, key=lambda pid: scores[pid], reverse=True)

    logger.info(
        f"RRF fusion | vector={len(vector_results)} | "
        f"bm25={len(bm25_results)} | merged={len(sorted_pids)}"
    )

    return [sections[pid] for pid in sorted_pids]


# ── Main Hybrid Function ───────────────────────────────────────────────────────

def hybrid_retrieve(
    retriever: dict,
    bm25_data: dict,
    query:     str,
    k:         int = 6,
) -> List[dict]:
    """
    Runs vector search and BM25 in parallel, then merges with RRF.

    This is a drop-in replacement for retrieve_parent_sections() in rag2.py.
    The return format is identical — a list of dicts with 'content' and 'metadata'.

    Strategy:
        - Fetch k*2 candidates from each search to give RRF a wide pool
        - Merge with RRF → returns up to k*2 combined results
        - Caller (rag2.py) receives top results; prompt uses all of them

    Args:
        retriever: The dict returned by load_retriever() in rag2.py.
                   Must have 'vectorstore' and 'parent_store' keys.
        bm25_data: The dict returned by build_bm25_index().
        query:     The (possibly transformed) search query.
        k:         Number of child vectors to retrieve in vector search.
                   BM25 fetches k*2 candidates.

    Returns:
        List of parent section dicts (content + metadata), merged and ranked.
        Returns empty list if both searches fail.
    """
    vector_sections = []
    bm25_sections   = []

    # ── Vector search ──────────────────────────────────────────────────────────
    try:
        vectorstore  = retriever["vectorstore"]
        parent_store = retriever["parent_store"]

        child_results = vectorstore.similarity_search(query, k=k)
        logger.info(f"Vector search returned {len(child_results)} child chunks")

        seen_pids = {}
        for child in child_results:
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen_pids:
                seen_pids[pid] = True

        for pid in seen_pids:
            if pid in parent_store:
                data = parent_store[pid]
                vector_sections.append({
                    "content":  data["content"],
                    "metadata": data["metadata"],
                })

        logger.info(f"Vector search -> {len(vector_sections)} parent sections")

    except Exception as e:
        logger.error(f"Vector search failed in hybrid_retrieve: {e}")

    # ── BM25 search ────────────────────────────────────────────────────────────
    try:
        bm25_sections = bm25_search(bm25_data, query, k=k * 2)
        logger.info(f"BM25 search -> {len(bm25_sections)} sections")
    except Exception as e:
        logger.error(f"BM25 search failed in hybrid_retrieve: {e}")

    # ── Fallback: if one fails, use the other ──────────────────────────────────
    if not vector_sections and not bm25_sections:
        logger.error("Both vector and BM25 search failed — returning empty")
        return []

    if not vector_sections:
        logger.warning("Vector search returned nothing — using BM25 only")
        return bm25_sections[:k]

    if not bm25_sections:
        logger.warning("BM25 returned nothing — using vector only")
        return vector_sections

    # ── RRF merge ─────────────────────────────────────────────────────────────
    merged = reciprocal_rank_fusion(vector_sections, bm25_sections)
    logger.info(f"Hybrid retrieve done | merged={len(merged)} sections")
    return merged
