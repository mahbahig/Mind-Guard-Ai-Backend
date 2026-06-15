"""
title_generator.py — Inference Wrapper for the Title Transformer
================================================================
Loads the trained TitleTransformer + vocab and exposes a single
`generate_title(message: str) -> str` function.

Falls back gracefully if the model hasn't been trained yet —
returns a keyword-extracted title instead.

Usage (imported by app.py):
    from title_generator import generate_title
    title = generate_title("I've been feeling really anxious at work")
    # → "Work Anxiety Struggles"
"""

import re
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models" / "title_model"
MODEL_PT  = MODEL_DIR / "model.pt"
VOCAB_PT  = MODEL_DIR / "vocab.json"

# Transformer hyperparameters — must match train_title_model.py
_D_MODEL  = 64
_N_HEADS  = 4
_N_LAYERS = 2
_D_FF     = 128
_MAX_LEN  = 64
_MAX_TITLE = 12


# ── Lazy loader (cached so it loads once) ──────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model():
    """Returns (model, tokenizer, device) or None if not available."""
    if not MODEL_PT.exists() or not VOCAB_PT.exists():
        logger.warning("Title model not found — using fallback keyword extraction")
        return None

    try:
        import torch
        from src.ai.chatbot.chat_src.title_model import TitleTransformer, WordTokenizer

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = WordTokenizer.load(VOCAB_PT)

        model = TitleTransformer(
            vocab_size = tokenizer.vocab_size,
            d_model    = _D_MODEL,
            n_heads    = _N_HEADS,
            n_layers   = _N_LAYERS,
            d_ff       = _D_FF,
            max_len    = _MAX_LEN,
            dropout    = 0.0,          # no dropout at inference
            pad_idx    = tokenizer.PAD,
        ).to(device)

        state = torch.load(MODEL_PT, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        logger.info("Title model loaded ✓")
        return model, tokenizer, device

    except Exception as e:
        logger.error(f"Failed to load title model: {e}")
        return None


# ── Fallback: keyword-based title ─────────────────────────────────────────────

_STOP_WORDS = {
    # pronouns / articles / prepositions
    "i", "me", "my", "we", "our", "you", "your", "the", "a", "an",
    "it", "its", "this", "that", "these", "those",
    "he", "she", "his", "her", "they", "them", "their",
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "into",
    # conjunctions / connectors
    "and", "or", "but", "so", "if", "as", "about", "than",
    # aux verbs
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "shall", "may", "might",
    "am", "get", "got", "getting",
    # question words / filler
    "what", "whats", "how", "why", "who", "when", "where", "which",
    "difference", "between", "versus", "compared",
    "tell", "explain", "describe",
    # common mental-health filler (too vague to use as title words)
    "feel", "feeling", "felt", "feels",
    "know", "knowing", "think", "thinking",
    "want", "need", "help", "really", "very",
    "just", "like", "also", "even", "still", "now", "more",
    "some", "any", "all", "not", "no", "too",
    "there", "here", "up", "out", "so", "going",
    "time", "one", "day", "lot", "much", "well",
}

def _keyword_fallback(message: str) -> str:
    """
    Extract 2–3 meaningful topic words as a title.
    Prefers longer, more specific words (e.g. 'bipolar' over 'between').
    """
    words = re.sub(r"[^\w\s]", "", message.lower()).split()
    keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 3]

    if not keywords:
        return "New Conversation"

    # Sort: acronyms (all-caps in original) first, then by length descending
    original_lower = message.lower()
    original_words = message.split()
    acronyms = {w.lower() for w in original_words if w.isupper() and len(w) >= 2}
    keywords.sort(key=lambda w: (0 if w in acronyms else 1, -len(w)))

    # Deduplicate while preserving order
    seen, unique = set(), []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    # Title-case the top 3
    return " ".join(w.capitalize() for w in unique[:3])


# ── Public API ─────────────────────────────────────────────────────────────────

def _is_arabic(text: str) -> bool:
    """Returns True if the text contains Arabic characters."""
    return any("؀" <= ch <= "ۿ" for ch in text)


def _gemini_title(message: str) -> str:
    """Generate a title for Arabic messages using Gemini."""
    try:
        import os
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        prompt = (
            "Generate a short Arabic title (2–4 words) for this conversation opening. "
            "Reply with ONLY the title, no punctuation, no explanation.\n\n"
            f"Message: {message[:200]}"
        )
        result = model.invoke(prompt).content.strip()
        return result if result else "محادثة جديدة"
    except Exception as e:
        logger.error(f"Gemini Arabic title failed: {e}")
        return "محادثة جديدة"


def generate_title(message: str) -> str:
    """
    Generate a short conversation title (2–5 words) from the first user message.

    Uses the trained TitleTransformer if available;
    falls back to keyword extraction otherwise.
    Arabic messages use Gemini directly (English-only model).

    Args:
        message: The first user message in the conversation.

    Returns:
        A short title string, e.g. "Workplace Anxiety Struggles".
    """
    message = message.strip()
    if not message:
        return "New Conversation"

    if _is_arabic(message):
        return _gemini_title(message)

    loaded = _load_model()

    if loaded is None:
        return _keyword_fallback(message)

    model, tokenizer, device = loaded

    try:
        import torch

        tokens = tokenizer.encode(message, max_len=_MAX_LEN)
        src    = torch.tensor([tokens], dtype=torch.long).to(device)

        with torch.no_grad():
            generated = model.generate(
                src,
                bos_idx   = tokenizer.BOS,
                eos_idx   = tokenizer.EOS,
                max_steps = _MAX_TITLE,
            )

        title = tokenizer.decode(generated).strip()

        if not title or len(title.split()) < 2:
            return _keyword_fallback(message)

        # Capitalize each word
        return title.title()

    except Exception as e:
        logger.error(f"Title generation failed: {e}")
        return _keyword_fallback(message)


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_messages = [
        "I've been feeling really empty lately, like nothing matters anymore",
        "I have so much anxiety about work, what can I do to cope?",
        "I just went through a bad breakup and I don't know how to move on",
        "I keep having panic attacks and I don't know how to stop them",
        "I feel so lonely all the time, even when I'm around people",
        "مش قادر أكمل",  # Arabic — should fallback gracefully
        ".",             # Edge case
    ]

    print("Title Generator Test")
    print("=" * 50)
    for msg in test_messages:
        title = generate_title(msg)
        print(f"  msg:   {msg[:60]}")
        print(f"  title: {title}")
        print()
