"""
crisis.py — Phase 4: Crisis Detection & Response
=================================================
Provides keyword-based crisis detection that runs BEFORE the normal RAG pipeline.

If a user message is detected as a crisis:
  - The normal pipeline is bypassed entirely
  - A warm, resource-focused response is returned

How it works:
  1. Lowercase keyword scan runs instantly on every message
  2. If any keyword matches → crisis path is taken, RAG is skipped
"""


# ── Crisis keyword list ────────────────────────────────────────────────────────
# Add phrases here. All matching is case-insensitive substring matching.
# ⚠️  False negatives (missed crises) are the dangerous ones — when in doubt, add it.
CRISIS_KEYWORDS = [

    # — Suicidal ideation (explicit) —
    "kill myself",
    "killing myself",
    "wanna die",
    "wanna kill myself",
    "wanna end it",
    "wanna hurt myself",
    "want to die",
    "wants to die",
    "want to be dead",
    "end my life",
    "ending my life",
    "take my life",
    "taking my life",
    "commit suicide",
    "committed suicide",
    "attempting suicide",
    "suicidal",
    "suicide",
    "end it all",
    "not want to be here anymore",
    "don't want to be here",
    "dont want to be here",
    "don't want to live",
    "dont want to live",
    "do not want to live",
    "no reason to live",
    "ready to die",
    "better off dead",
    "wish i was dead",
    "wish i were dead",
    "i should be dead",
    "i want to be dead",

    # — Self-harm —
    "hurt myself",
    "hurting myself",
    "cut myself",
    "cutting myself",
    "harm myself",
    "harming myself",
    "self-harm",
    "self harm",
    "selfharm",
    "burn myself",
    "burning myself",
    "starving myself",
    "hit myself",
    "hitting myself",

    # — Hopelessness / passive ideation —
    "no point in living",
    "no point anymore",
    "can't go on",
    "cannot go on",
    "cant go on",
    "can't take it anymore",
    "cannot take it anymore",
    "cant take it anymore",
    "life is not worth living",
    "life isn't worth living",
    "life isnt worth living",
    "isn't worth living",
    "isnt worth living",
    "not worth living",
    "nobody would care if i died",
    "nobody would miss me",
    "no one would miss me",
    "no one would care if i was gone",
    "nobody would notice if i was gone",
    "nobody would notice if i were gone",
    "better off without me",
    "everyone would be better off without me",
    "world would be better without me",
    "don't see the point of anything",
    "dont see the point",
    "nothing to live for",
    "give up on life",
    "giving up on life",
    "tired of living",
    "tired of being alive",
    "i give up",
    "there's no point",
    "theres no point",

    # — Method-specific / active crisis language —
    "overdose",
    "pills to die",
    "jump off a",
    "hang myself",
    "hanging myself",
    "shoot myself",
    "shooting myself",
    "drown myself",
    "drowning myself",
    "slit my wrist",
    "slit my wrists",

    # — Arabic phrases —
    "عايز أموت",
    "عاوز أموت",
    "نفسي أموت",
    "بفكر في الانتحار",
    "مش قادر أكمل",
    "مش قادر أكمّل",
    "تعبت من الحياة",
    "الحياة مش تستاهل",
    "نفسي أنهي حياتي",
    "أنهي حياتي",
    "اقتل نفسي",
    "أقتل نفسي",
    "مش عايز أعيش",
    "مش قادر أعيش",
    "ماعدش لاقي معنى",
    "مافيش معنى للحياة",
    "حياتي خلصت",
    "خلاص مش قادر",
    "مش عارف أكمل",
    "عايز أختفي",
    "نفسي أختفي",
]

# Pre-compute lowercase for fast matching at runtime
_CRISIS_PATTERNS = [kw.lower() for kw in CRISIS_KEYWORDS]


# ── Public API ─────────────────────────────────────────────────────────────────

def crisis_router(
    query_text: str,
) -> tuple:
    """
    Checks whether the user's message contains crisis signals.

    Runs keyword scan (O(n) in keyword list, very fast).

    Args:
        query_text:  The user's raw message.

    Returns:
        Tuple of (is_crisis: bool, reason: str)
        - is_crisis: True if the message triggers crisis detection
        - reason:    Description of what triggered it
    """
    if not query_text or not query_text.strip():
        return False, ""

    text_lower = query_text.lower().strip()

    matched = [kw for kw in _CRISIS_PATTERNS if kw in text_lower]
    if matched:
        reason = f"keyword_match: {matched}"
        return True, reason

    return False, ""




