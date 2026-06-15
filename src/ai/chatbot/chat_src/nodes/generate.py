"""
nodes/generate.py — Generation node
=====================================
Generates the final response using the appropriate LLM and prompt.
"""

from src.ai.chatbot.chat_src.pipeline.state import ChatState, _prompt


def generate_node(state: ChatState) -> dict:
    """
    Routing:
      auto + emotional/conversational → Llama emotional prompt
      auto + advice                   → Llama advice prompt + DSM context
      auto + factual                  → Llama factual prompt + DSM context
      auto + off_topic                → Gemini off-topic redirect
      gemini                          → Gemini with full system_prompt.txt
      vertex_tuned                    → Llama with full RAG prompt
      any Llama failure               → Gemini fallback
    """
    from langchain_core.prompts import ChatPromptTemplate
    from src.ai.chatbot.chat_src.llm import (
        _gemini_generate, _vertex_predict,
        vertex_tuned_configured, _normalize_llm_backend,
        _build_llama_emotional_prompt, _build_llama_advice_prompt,
        _build_llama_factual_prompt, _polish_if_needed, _fix_opener,
    )

    route       = state.get("route", "factual")
    llm_backend = _normalize_llm_backend(state.get("llm_backend", "auto"))
    query       = state["query"]
    history_ctx = state.get("history_context", "None")
    language    = state.get("language", "en")

    # ── Relevance guard: drop context with low keyword overlap ────────────────
    raw_context = state.get("context", "")
    if raw_context:
        _query_words = {
            w for w in query.lower().split()
            if len(w) > 3 and w not in {
                "what", "that", "this", "have", "with", "from", "just",
                "feel", "like", "been", "about", "does", "more", "some",
                "when", "will", "know", "dont", "cant", "want", "help",
                "everything", "something", "anything",
            }
        }
        _hits = sum(1 for w in _query_words if w in raw_context.lower())
        context = "" if _query_words and _hits / len(_query_words) < 0.25 else raw_context
    else:
        context = raw_context

    # ── Build Gemini fallback prompt (system_prompt.txt) ─────────────────────
    _MAX_LLAMA_CONTEXT = 3000
    try:
        tmpl_str = _prompt("system_prompt.txt")
        tmpl     = ChatPromptTemplate.from_template(tmpl_str)
        fallback_prompt = tmpl.format(
            context  = context if llm_backend == "gemini" else context[:_MAX_LLAMA_CONTEXT],
            question = query,
            history  = history_ctx,
        )
    except Exception:
        fallback_prompt = f"You are NEURA, a warm mental health companion. Answer: {query}"

    # ── Inner prompt builders ─────────────────────────────────────────────────
    def _simple_emotional_prompt() -> str:
        lang_note = "Respond in Arabic (العربية). " if language == "ar" else ""
        history_snippet = (
            f"Recent conversation:\n{history_ctx}\n\n"
            if history_ctx and history_ctx != "None" else ""
        )
        return _prompt("gemini_emotional.txt").format(
            history=history_snippet, query=query, lang_note=lang_note,
        )

    def _off_topic_prompt() -> str:
        lang_note = "Respond in Arabic (العربية). " if language == "ar" else ""
        return _prompt("gemini_offtopic.txt").format(query=query, lang_note=lang_note)

    def _is_llama_refusal(text: str) -> bool:
        t = (text or "").lower().strip()[:120]
        return any(t.startswith(r) for r in (
            "i can't provide a response", "i cannot provide a response",
            "i'm not able to provide", "i am not able to provide",
            "i'm unable to", "i am unable to",
            "i cannot answer", "i can't answer", "i'm not able to answer",
        ))

    _ar_suffix   = " Respond in Arabic (العربية)." if language == "ar" else ""
    response     = ""
    responder    = "unknown"

    try:
        if llm_backend == "gemini":
            response  = _gemini_generate(fallback_prompt)
            responder = "Gemini"

        elif llm_backend == "vertex_tuned":
            if not vertex_tuned_configured():
                response  = "Tuned Llama is not configured."
                responder = "error"
            else:
                response  = _vertex_predict(fallback_prompt)
                responder = "Llama"

        else:  # auto
            _vertex_ok = vertex_tuned_configured()

            if not _vertex_ok:
                if route in ("emotional", "conversational"):
                    response  = _gemini_generate(_simple_emotional_prompt())
                    responder = "Gemini"
                elif route == "off_topic":
                    response  = _gemini_generate(_off_topic_prompt())
                    responder = "Gemini"
                else:
                    response  = _gemini_generate(fallback_prompt)
                    responder = "Gemini"

            elif route in ("emotional", "conversational"):
                llama_p = _build_llama_emotional_prompt(query, history_ctx, lang_note=_ar_suffix.strip())
                raw = _vertex_predict(llama_p, pre_formatted=True)
                if _is_llama_refusal(raw) or not raw or len(raw.split()) < 15:
                    response  = _gemini_generate(_simple_emotional_prompt())
                    responder = "Gemini"
                else:
                    response  = _fix_opener(raw)
                    responder = "Llama"

            elif route == "off_topic":
                response  = _gemini_generate(_off_topic_prompt())
                responder = "Gemini"

            elif route == "advice":
                llama_p = _build_llama_advice_prompt(query, context, history_ctx)
                raw     = _vertex_predict(llama_p, pre_formatted=True)
                if _is_llama_refusal(raw) or not raw or len(raw.split()) < 10:
                    response  = _gemini_generate(fallback_prompt)
                    responder = "Gemini"
                else:
                    response  = _polish_if_needed(raw, query, fallback_prompt)
                    responder = "Llama"

            else:  # factual
                llama_p = _build_llama_factual_prompt(query, context, history_ctx)
                raw     = _vertex_predict(llama_p, pre_formatted=True)
                if _is_llama_refusal(raw) or not raw or len(raw.split()) < 10:
                    response  = _gemini_generate(fallback_prompt)
                    responder = "Gemini"
                else:
                    response  = _polish_if_needed(raw, query, fallback_prompt)
                    responder = "Llama"

    except Exception:
        try:
            if route in ("emotional", "conversational"):
                response  = _gemini_generate(_simple_emotional_prompt())
            elif route == "off_topic":
                response  = _gemini_generate(_off_topic_prompt())
            else:
                response  = _gemini_generate(fallback_prompt)
            responder = "Gemini"
        except Exception:
            response  = "I'm having trouble connecting right now. Please try again in a moment."
            responder = "error"

    return {"response": response, "responder": responder}
