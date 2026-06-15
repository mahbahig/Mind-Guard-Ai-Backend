"""
generate_title_data.py — Synthetic Title Pair Generator
========================================================
Uses Gemini to generate NEW (message, title) pairs from topic seeds.
Appends to existing title_pairs.jsonl — safe to re-run (deduplicates).

Run:
    python scripts/generate_title_data.py
    python scripts/generate_title_data.py --target 10000
"""

import json
import re
import time
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "training_data" / "title_pairs.jsonl"

TOPICS = [
    "work stress and burnout",
    "relationship breakup and heartbreak",
    "social anxiety and fear of judgment",
    "panic attacks and physical symptoms",
    "depression and feeling empty inside",
    "loneliness and social isolation",
    "grief and loss of a loved one",
    "low self-esteem and self-worth issues",
    "family conflict and toxic relationships",
    "trauma and PTSD flashbacks",
    "sleep problems and insomnia",
    "anger and frustration management",
    "OCD and intrusive thoughts",
    "eating issues and body image",
    "ADHD and difficulty focusing",
    "bipolar disorder and mood swings",
    "generalized anxiety and overthinking",
    "parenting stress and overwhelm",
    "financial stress and money worries",
    "feeling misunderstood by others",
    "college and academic pressure",
    "fear of failure and perfectionism",
    "trust issues in relationships",
    "emotional numbness and disconnection",
    "feeling like a burden to others",
    "childhood trauma and its effects",
    "identity crisis and life direction",
    "job loss and career anxiety",
    "postpartum depression and new parent struggles",
    "addiction recovery and relapse fears",
]

BATCH_SIZE = 20

PROMPT = """You generate training data for a mental health chatbot.

Topic: {topic}

Generate {n} realistic examples. Each has:
1. A "message" — something a real person might type to a mental health chatbot (informal, natural, 1–3 sentences, varied length)
2. A "title" — 2-3 words, Title Cased, that captures the topic (NOT generic)

Strict rules:
- Every message must be different (different wording, different angle, different emotion)
- Titles: exactly 2-3 words, no punctuation, specific (e.g. "Work Burnout Stress" not "Feeling Bad")
- Messages sound like real people: informal, sometimes with typos, fragmented sentences
- Mix short messages (5-10 words) and longer ones (2-3 sentences)

Return ONLY a valid JSON array, no markdown fences:
[
  {{"message": "...", "title": "..."}},
  {{"message": "...", "title": "..."}}
]"""

BATCH_SIZE = 20


def _load_existing() -> set[str]:
    if not DATA_FILE.exists():
        return set()
    seen = set()
    for line in DATA_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                seen.add(json.loads(line)["message"].strip().lower())
            except Exception:
                pass
    return seen


def _call_gemini(model, topic: str, n: int) -> list[dict]:
    prompt = PROMPT.format(topic=topic, n=n)
    raw = model.invoke(prompt).content.strip()
    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    pairs = json.loads(raw)
    valid = []
    for p in pairs:
        msg   = str(p.get("message", "")).strip()
        title = str(p.get("title",   "")).strip()
        if not msg or not title:
            continue
        words = title.split()
        if not (2 <= len(words) <= 4):
            continue
        valid.append({"message": msg, "title": title})
    return valid


def generate(target: int = 10_000):
    existing = _load_existing()
    current  = len(existing)
    logger.info(f"Existing pairs : {current:,}")
    logger.info(f"Target         : {target:,}")

    if current >= target:
        logger.info("Already at target.")
        return

    needed = target - current
    logger.info(f"Generating {needed:,} new pairs...\n")

    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    load_dotenv()

    from langchain_google_genai import ChatGoogleGenerativeAI
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)

    added   = 0
    errors  = 0
    topic_i = 0

    with open(DATA_FILE, "a", encoding="utf-8") as f:
        while added < needed:
            topic   = TOPICS[topic_i % len(TOPICS)]
            topic_i += 1

            try:
                pairs = _call_gemini(model, topic, BATCH_SIZE)
            except Exception as e:
                errors += 1
                logger.warning(f"Error on '{topic}': {e}")
                time.sleep(4)
                if errors >= 8:
                    logger.error("Too many errors — stopping early.")
                    break
                continue

            errors = 0
            batch_new = 0
            for p in pairs:
                if p["message"].strip().lower() in existing:
                    continue
                existing.add(p["message"].strip().lower())
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                batch_new += 1
                added     += 1
                if added >= needed:
                    break

            logger.info(
                f"  [{current + added:>6,} / {target:,}]  "
                f"+{batch_new:<3}  topic: {topic[:40]}"
            )
            time.sleep(0.6)

    logger.info(f"\nDone — total pairs: {current + added:,}")
    logger.info(f"Saved to: {DATA_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=10_000)
    args = parser.parse_args()
    generate(target=args.target)
