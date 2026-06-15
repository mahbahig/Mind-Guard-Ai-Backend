"""
llm/vertex.py — Vertex AI (Llama) integration helpers
=======================================================
Project ID resolution, endpoint configuration check, prediction, and
user-facing error messages for Vertex AI Model Garden.
"""

import json
import os
import time
from pathlib import Path

import google.auth
import google.auth.transport.requests
import requests as _http
from src.ai.chatbot.chat_src.pipeline.state import _prompt



def _strip_env_value(val: str | None) -> str:
    if not val:
        return ""
    s = val.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    return s.strip()


def _vertex_project_id() -> str:
    """
    GCP project id for Vertex init + endpoint paths.
    Prefer VERTEX_PROJECT_ID / GOOGLE_CLOUD_PROJECT; else read project_id from the
    service account JSON if GOOGLE_APPLICATION_CREDENTIALS is set.
    """
    env_pid = ""
    for key in ("VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"):
        v = _strip_env_value(os.environ.get(key))
        if v:
            env_pid = v
            break

    cred_pid = ""
    cred = _strip_env_value(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    if cred:
        path = Path(cred).expanduser()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cred_pid = str(data.get("project_id") or "")
        except Exception:
            pass

    if cred_pid and env_pid and env_pid.isdigit() and not cred_pid.isdigit():
        return cred_pid

    if env_pid:
        return env_pid
    if cred_pid:
        return cred_pid
    return ""


def vertex_tuned_configured() -> bool:
    """True if env vars allow calling the deployed tuned model endpoint."""
    full = _strip_env_value(os.environ.get("VERTEX_ENDPOINT_RESOURCE"))
    if full:
        return True
    return bool(_vertex_project_id() and _strip_env_value(os.environ.get("VERTEX_ENDPOINT_ID")))


def _normalize_llm_backend(name: str | None) -> str:
    if not name:
        return "gemini"
    n = str(name).strip().lower()
    if n in ("vertex", "vertex_tuned", "tuned", "llama"):
        return "vertex_tuned"
    if n in ("auto", "mixed", "ensemble", "hybrid"):
        return "auto"
    return "gemini"



def _parse_vertex_prediction(pred) -> str:
    if pred is None:
        return ""
    if isinstance(pred, str):
        text = pred.strip()

        # Model Garden wraps response as "Prompt:\n...\nOutput:\n<actual response>"
        # Extract only the text after the last "Output:\n" marker.
        marker = "Output:\n"
        if marker in text:
            text = text[text.rindex(marker) + len(marker):].strip()

        # Strip any trailing Llama special tokens the model may echo
        # (e.g. "<|eot_id|>", "<|end_of_text|>", repeated headers)
        for tok in ("<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"):
            if tok in text:
                text = text[:text.index(tok)].strip()

        # Strip "Human:" / "Assistant:" prefixes that leak from training format
        for prefix in ("Human:", "Assistant:", "human:", "assistant:"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        return text
    if isinstance(pred, dict):
        for key in ("generated_text", "content", "output", "text", "response"):
            val = pred.get(key)
            if val:
                return str(val).strip()
        choices = pred.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict) and msg.get("content"):
                return str(msg["content"]).strip()
        return str(pred).strip()
    if isinstance(pred, list):
        parts = [_parse_vertex_prediction(p) for p in pred]
        return "".join(parts).strip()
    return str(pred).strip()


def _vertex_predict(prompt: str, pre_formatted: bool = False) -> str:
    """
    Calls the Vertex AI Model Garden dedicated endpoint.

    Parameters
    ----------
    prompt        : The text to send. If pre_formatted=False (default), it is
                    wrapped in Llama 3.1 chat special tokens automatically.
    pre_formatted : Set True when the caller already built the full Llama chat
                    prompt (e.g. _build_llama_emotional_prompt). Skips wrapping
                    to prevent double-formatting.
    """
    dedicated_url = _strip_env_value(os.environ.get("VERTEX_DEDICATED_URL", ""))
    endpoint_id   = _strip_env_value(os.environ.get("VERTEX_ENDPOINT_ID", ""))

    if not dedicated_url:
        raise RuntimeError("VERTEX_DEDICATED_URL is not set.")
    if not endpoint_id:
        raise RuntimeError("VERTEX_ENDPOINT_ID is not set.")

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    token = credentials.token

    max_tokens  = int(os.environ.get("VERTEX_MAX_TOKENS", "350"))
    temperature = float(os.environ.get("VERTEX_TEMPERATURE", "0.7"))
    top_p       = float(os.environ.get("VERTEX_TOP_P", "0.9"))

    if pre_formatted:
        final_prompt = prompt
    else:
        # Wrap in Llama 3.1 chat tokens — required for factual/RAG prompts.
        # System prompt text → src/prompts/llama_generic.txt
        llama_system = _prompt("llama_generic.txt").strip()
        final_prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{llama_system}\n\n"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    project  = _vertex_project_id()
    if not project:
        raise RuntimeError(
            "GCP project ID not found. Set VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT "
            "in your .env, or set GOOGLE_APPLICATION_CREDENTIALS to a service account "
            "JSON that contains project_id."
        )
    location = _strip_env_value(os.environ.get("VERTEX_LOCATION")) or "us-central1"
    url = (
        f"{dedicated_url.rstrip('/')}"
        f"/v1/projects/{project}/locations/{location}"
        f"/endpoints/{endpoint_id}:predict"
    )

    payload = {
        "instances": [{
            "prompt":      final_prompt,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "top_p":       top_p,
            # Note: stop_sequences is NOT supported by Model Garden dedicated
            # endpoints — sending it causes HTTP 500.  Llama 3.1 stops cleanly
            # at <|eot_id|> because that token is baked into the chat template,
            # and _parse_vertex_prediction() strips any leaked tokens afterward.
        }]
    }


    # Retry up to 2 times on 429 (endpoint cold-start / scale-up).
    # Waits: 8 s then 15 s — fast enough to stay within the Streamlit spinner,
    # and short enough that Gemini fallback kicks in quickly if the endpoint
    # is down. The old 30/60/90 s waits caused 3-minute hangs AND the final
    # request after the last sleep never fired (loop exhausted before re-POST).
    _RETRY_WAITS = (8, 15)
    resp = None
    for attempt, wait in enumerate((*_RETRY_WAITS, None), start=1):
        resp = _http.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if resp.status_code != 429 or wait is None:
            break
        time.sleep(wait)

    if not resp.ok:
        raise RuntimeError(f"Vertex endpoint HTTP {resp.status_code}: {resp.text[:500]}")

    data  = resp.json()
    preds = data.get("predictions", [])
    if not preds:
        raise RuntimeError(f"Vertex returned no predictions: {data}")
    return _parse_vertex_prediction(preds[0])
