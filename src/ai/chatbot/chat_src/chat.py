"""
chat.py — Terminal test for the NEURA pipeline.
Run from the src/ directory:
    python chat.py
"""

import sys
from pathlib import Path

# Ensure src/ is on the path
src = str(Path(__file__).parent)
if src not in sys.path:
    sys.path.insert(0, src)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from src.ai.chatbot.chat_src.graph import run_chat

history         = []
history_summary = ""

print("\n🧠  NEURA — Mental Health Companion")
print("─" * 40)
print("Type your message and press Enter.")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        break

    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye.")
        break

    try:
        response, sources, history_summary, responder = run_chat(
            query           = user_input,
            history         = history,
            llm_backend     = "vertex_tuned",
            history_summary = history_summary,
        )

        print(f"\nNEURA ({responder}): {response}\n")
        if sources:
            print("Sources:")
            for s in sources:
                print(f"  • {s}")
            print()

        # Update history for next turn
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})

        # Keep last 10 messages only
        if len(history) > 10:
            history = history[-10:]

    except Exception as e:
        print(f"\n[Error] {e}\n")
