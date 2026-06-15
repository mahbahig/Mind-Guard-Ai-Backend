"""
llm/prompts.py — Llama prompt builders
========================================
Builds Llama 3.1 chat-format prompts for each route.

System prompt TEXT lives in src/prompts/*.txt — edit those files to change
tone/behaviour without touching any Python code.

This file only handles:
  - loading the .txt files
  - wrapping with Llama 3.1 chat tokens
  - injecting dynamic variables (query, history, context)
"""

from src.ai.chatbot.chat_src.pipeline.state import _prompt


def _build_llama_emotional_prompt(query_text: str, history_context: str, lang_note: str = "") -> str:
    """
    Builds a SHORT, clean prompt for Llama on emotional queries.

    Llama was fine-tuned on mental health conversations — it excels at
    empathetic, conversational responses. The full RAG prompt overwhelms it.
    For emotional support the user needs empathy, not retrieved clinical text.

    System prompt text → src/prompts/llama_emotional.txt
    """
    history_section = (
        f"Recent conversation:\n{history_context}\n\n"
        if history_context and history_context != "None"
        else ""
    )
    system = _prompt("llama_emotional.txt").format(lang_note=lang_note).strip()
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}\n\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{history_section}"
        f"{query_text}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _build_llama_advice_prompt(query_text: str, context: str, history_context: str) -> str:
    """
    Focused Llama prompt for advice/coping queries.
    Includes top DSM-5 context (truncated to 1 500 chars).

    System prompt text → src/prompts/llama_advice.txt
    """
    history_section = (
        f"Recent conversation:\n{history_context}\n\n"
        if history_context and history_context != "None"
        else ""
    )
    context_section = (
        f"Relevant clinical context:\n{context[:1500]}\n\n"
        if context
        else ""
    )
    system = _prompt("llama_advice.txt").strip()
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}\n\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{history_section}"
        f"{context_section}"
        f"{query_text}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _build_llama_factual_prompt(query_text: str, context: str, history_context: str) -> str:
    """
    Focused Llama prompt for clinical / factual queries.
    Includes up to 2 000 chars of DSM-5 context.

    System prompt text → src/prompts/llama_factual.txt
    """
    history_section = (
        f"Recent conversation:\n{history_context}\n\n"
        if history_context and history_context != "None"
        else ""
    )
    context_section = (
        f"DSM-5-TR reference:\n{context[:2000]}\n\n"
        if context
        else ""
    )
    system = _prompt("llama_factual.txt").strip()
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}\n\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{history_section}"
        f"{context_section}"
        f"{query_text}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
