"""
feedback_scorer.py — Learn from Phase 6 feedback to score response quality.

Parses feedback.log, extracts features, and scores new responses 0.0-1.0.

Two modes:
  Heuristic : always available, rule-based from observed patterns
  ML        : trains TF-IDF + LogisticRegression when >= MIN_SAMPLES exist

Usage:
    from feedback_scorer import FeedbackScorer
    scorer = FeedbackScorer(Path("src/feedback.log"))
    score  = scorer.score(response_text)   # 0.0 – 1.0
    info   = scorer.summary()
"""

import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

MIN_SAMPLES = 20   # minimum labeled pairs needed to switch to ML mode

# ── Feature vocabulary ────────────────────────────────────────────────────────

WARM_PHRASES = [
    "i hear you", "that sounds", "yeah", "that makes sense",
    "it's okay", "you're not alone", "i understand",
    "that can be really", "thank you for sharing", "that must be",
    "i can imagine", "that's really", "i'm here", "makes total sense",
]

CLINICAL_DUMP_PHRASES = [
    "according to the dsm", "as per dsm-5", "criterion a", "criterion b",
    "diagnostic and statistical manual", "icd-10", "f32",
    "dsm-5-tr criterion", "the diagnostic criteria state",
    "clinically significant disturbance", "specifier",
]


# ── Feedback log parser ───────────────────────────────────────────────────────

def parse_feedback_log(log_path: Path) -> List[Tuple[str, int]]:
    """
    Parse feedback.log → list of (response_preview, label).
    label: 1 = thumbs up (preferred), 0 = thumbs down (not preferred).
    Skips lines without a clear rating or too-short previews.
    """
    examples: List[Tuple[str, int]] = []
    if not log_path.exists():
        logger.info("feedback.log not found — scorer starting with 0 examples")
        return examples

    pattern = re.compile(r"rating=(\w+).*?response_preview='(.*?)'", re.DOTALL)

    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = pattern.search(line)
                if not m:
                    continue
                rating, preview = m.group(1), m.group(2)
                if rating in ("up", "down") and len(preview.strip()) > 15:
                    examples.append((preview.strip(), 1 if rating == "up" else 0))
    except Exception as e:
        logger.warning(f"Could not parse feedback.log: {e}")

    logger.info(
        f"FeedbackScorer: loaded {len(examples)} examples "
        f"({sum(1 for _,l in examples if l==1)} up / "
        f"{sum(1 for _,l in examples if l==0)} down)"
    )
    return examples


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(text: str) -> dict:
    """Extract interpretable scalar features from a response."""
    lower = text.lower()
    words = text.split()
    return {
        "word_count":          len(words),
        "has_warm_phrase":     int(any(p in lower for p in WARM_PHRASES)),
        "has_clinical_dump":   int(any(p in lower for p in CLINICAL_DUMP_PHRASES)),
        "ends_with_question":  int(text.rstrip().endswith("?")),
        "question_count":      text.count("?"),
        "has_bullets":         int("* " in text or "\n- " in text),
        "short_response":      int(len(words) < 35),
        "very_long_response":  int(len(words) > 220),
        "has_ellipsis":        int("..." in text or "…" in text),
    }


def heuristic_score(text: str) -> float:
    """
    Rule-based preference score derived from observed good/bad response patterns.
    Returns 0.0 (poor) – 1.0 (excellent).
    """
    f = extract_features(text)
    score = 0.50

    if f["has_warm_phrase"]:        score += 0.18
    if f["ends_with_question"]:     score += 0.10
    if f["question_count"] >= 2:    score += 0.05
    if f["has_ellipsis"]:           score += 0.04   # conversational pacing
    if f["has_clinical_dump"]:      score -= 0.25
    if f["has_bullets"]:            score -= 0.06   # less warm
    if f["short_response"]:         score -= 0.12
    if f["very_long_response"]:     score -= 0.10

    return round(max(0.0, min(1.0, score)), 3)


# ── Scorer class ──────────────────────────────────────────────────────────────

class FeedbackScorer:
    """
    Scores a response 0.0 – 1.0 based on learned user feedback preferences.

    Below MIN_SAMPLES  → heuristic mode (feature rules).
    At/above MIN_SAMPLES → ML mode (TF-IDF + LogisticRegression).
    """

    def __init__(self, feedback_log_path: Path):
        self.log_path  = feedback_log_path
        self.examples  = parse_feedback_log(feedback_log_path)
        self._model      = None
        self._vectorizer = None

        if len(self.examples) >= MIN_SAMPLES:
            self._train()
        else:
            remaining = MIN_SAMPLES - len(self.examples)
            logger.info(
                f"FeedbackScorer: heuristic mode "
                f"(need {remaining} more examples to unlock ML model)"
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(self, response_text: str) -> float:
        """Score a response. Returns 0.0 (poor) to 1.0 (excellent)."""
        if self._model is not None:
            return self._ml_score(response_text)
        return heuristic_score(response_text)

    def label(self, score: float) -> str:
        """Human-readable quality label."""
        if score >= 0.75: return "excellent"
        if score >= 0.55: return "good"
        if score >= 0.40: return "fair"
        return "needs work"

    def emoji(self, score: float) -> str:
        if score >= 0.75: return "\U0001f7e2"   # green
        if score >= 0.55: return "\U0001f7e1"   # yellow
        if score >= 0.40: return "\U0001f7e0"   # orange
        return "\U0001f534"                      # red

    def summary(self) -> dict:
        """Scorer status — shown in sidebar."""
        ups   = sum(1 for _, l in self.examples if l == 1)
        downs = sum(1 for _, l in self.examples if l == 0)
        return {
            "mode":          "ML model" if self._model else "Heuristic",
            "total":         len(self.examples),
            "thumbs_up":     ups,
            "thumbs_down":   downs,
            "need_for_ml":   max(0, MIN_SAMPLES - len(self.examples)),
        }

    def reload(self):
        """Re-parse feedback.log and retrain if threshold reached."""
        self.examples = parse_feedback_log(self.log_path)
        self._model      = None
        self._vectorizer = None
        if len(self.examples) >= MIN_SAMPLES:
            self._train()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _train(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            texts  = [e[0] for e in self.examples]
            labels = [e[1] for e in self.examples]

            self._vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            X = self._vectorizer.fit_transform(texts)

            self._model = LogisticRegression(max_iter=1000, C=1.0)
            self._model.fit(X, labels)

            logger.info(
                f"FeedbackScorer: ML model trained on {len(self.examples)} examples"
            )
        except ImportError:
            logger.warning("scikit-learn not installed — staying in heuristic mode")
        except Exception as e:
            logger.warning(f"ML training failed ({e}) — staying in heuristic mode")

    def _ml_score(self, text: str) -> float:
        try:
            X = self._vectorizer.transform([text])
            proba = self._model.predict_proba(X)[0]
            return round(float(proba[1]), 3)   # P(thumbs up)
        except Exception as e:
            logger.warning(f"ML scoring failed ({e}) — falling back to heuristic")
            return heuristic_score(text)
