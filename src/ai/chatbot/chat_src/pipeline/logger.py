"""
pipeline/logger.py — NEURA pipeline logging setup
==================================================
All logging configuration: CleanFilter, silencing noisy external libraries,
ANSI colour palette, phase grouping, and step/banner print helpers.
Applied on import.
"""

import os
import logging

# ── Terminal logging setup ─────────────────────────────────────────────────────
logger = logging.getLogger("graph")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False   # don't double-print if root logger is also configured

# ── Root-level message filter ─────────────────────────────────────────────────
# Blocks known noisy strings from ALL loggers/handlers, including those set up
# before graph.py imported (e.g. Streamlit's root handler).
class _CleanFilter(logging.Filter):
    _BLOCKED = (
        "AFC is enabled",
        "HTTP Request:",
        "Vertex predict | url=",
        "Vector search returned",
        "Vector search ->",
        "BM25 search |",
        "BM25 search ->",
        "RRF fusion |",
        "Hybrid retrieve done",
        "Response generated |",
    )
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(b in msg for b in self._BLOCKED)

# Apply to root logger AND every handler already registered
_clean_filter = _CleanFilter()
logging.root.addFilter(_clean_filter)
for _h in logging.root.handlers:
    _h.addFilter(_clean_filter)

# ── Silence noisy external libraries ─────────────────────────────────────────
# Suppress INFO/DEBUG from every Google SDK sub-logger, LangChain internals,
# and HTTP libraries so they don't bleed into the clean _step() pipeline view.
_SILENT_LOGGERS = [
    # Google AI / Gemini SDK — cover all sub-namespaces
    "google",
    "google.ai",
    "google.auth",
    "google.generativeai",
    "google.generativeai.client",
    "google.ai.generativelanguage",
    "google.api_core",
    "google.cloud",
    # LangChain
    "langchain",
    "langchain_core",
    "langchain_google_genai",
    "langchain_community",
    # HTTP clients
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    # Other SDKs
    "openai",
    "anthropic",
    "vertexai",
]
for _name in _SILENT_LOGGERS:
    logging.getLogger(_name).setLevel(logging.WARNING)

# rag2.py Vertex 429 retries are already surfaced via _step() in generate_node.
# Raise the threshold to ERROR so only genuine failures from rag2 appear.
logging.getLogger("rag2").setLevel(logging.ERROR)

# ── ANSI colour palette ────────────────────────────────────────────────────────
_C = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "italic":  "\033[3m",
    "cyan":    "\033[36m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "red":     "\033[31m",
    "blue":    "\033[34m",
    "purple":  "\033[35m",
    "white":   "\033[97m",
    "bg_blue": "\033[44m",
    "b_cyan":  "\033[96m",   # bright cyan
    "b_green": "\033[92m",   # bright green
}

# ── Phase grouping ─────────────────────────────────────────────────────────────
# Each node belongs to a visual phase.  When the phase changes, _step() prints
# a section header automatically — no extra calls needed from node functions.
_PHASES: dict[str, tuple[str, str]] = {
    "guard":          ("🔐  SAFETY",      _C["blue"]),
    "language":       ("🔐  SAFETY",      _C["blue"]),
    "crisis":         ("🔐  SAFETY",      _C["blue"]),
    "history":        ("📖  CONTEXT",     _C["purple"]),
    "conversational": ("🔀  ROUTING",     _C["cyan"]),
    "router":         ("🔀  ROUTING",     _C["cyan"]),
    "rewrite":        ("🔍  RETRIEVAL",   _C["yellow"]),
    "retrieve":       ("🔍  RETRIEVAL",   _C["yellow"]),
    "generate":       ("🤖  GENERATION",  _C["b_cyan"]),
    "postprocess":    ("✅  CLEANUP",     _C["green"]),
}

_ICONS = {
    "guard":          "🛡️ ",
    "language":       "🌍",
    "crisis":         "🚨",
    "history":        "📜",
    "conversational": "💬",
    "router":         "🔀",
    "rewrite":        "✏️ ",
    "retrieve":       "🔍",
    "generate":       "🤖",
    "postprocess":    "✅",
}

_LEVEL_COLOUR = {
    "info":    _C["white"],
    "success": _C["b_green"],
    "warn":    _C["yellow"],
    "error":   _C["red"],
    "skip":    _C["dim"],
}

_current_phase = ""   # module-level — reset by _banner() at the start of each call
_SILENT = os.environ.get("NEURA_SILENT", "0") == "1"  # suppress trace in eval mode


def _phase_header(node: str) -> None:
    """Print a section header when the pipeline enters a new phase."""
    if _SILENT:
        return
    global _current_phase
    phase_label, phase_colour = _PHASES.get(node, ("", ""))
    if not phase_label or phase_label == _current_phase:
        return
    _current_phase = phase_label
    W = 54
    label_clean = phase_label          # contains emoji + text
    pad = W - len(label_clean) - 2    # rough padding (emoji ≈ 2 chars)
    print(
        f"\n  {phase_colour}{_C['bold']}┌─  {label_clean}  "
        f"{'─' * max(pad, 4)}┐{_C['reset']}",
        flush=True,
    )


def _step(node: str, msg: str = "", *, level: str = "info") -> None:
    """
    Print one pipeline step line under the current phase section.
    Automatically prints a phase header when the phase changes.
    """
    if _SILENT:
        return
    _phase_header(node)

    icon    = _ICONS.get(node, "  ")
    colour  = _LEVEL_COLOUR.get(level, _C["white"])
    r       = _C["reset"]
    dim     = _C["dim"]
    bold    = _C["bold"]

    # Node label: fixed width so details line up
    label = f"{bold}{node:<13}{r}"

    # Detail text: colour depends on level
    if msg:
        detail = f"{colour}{msg}{r}"
    else:
        detail = ""

    # Connector: │ to stay inside the phase box
    print(f"  {dim}│{r}  {icon}  {label}  {detail}", flush=True)


def _banner(query: str) -> None:
    """Print the pipeline header banner."""
    global _current_phase
    _current_phase = ""          # reset phase tracker for this call
    if _SILENT:
        return

    W  = 58
    q_display = (query[:W - 12] + "…") if len(query) > W - 12 else query
    top    = f"╭{'─' * W}╮"
    title  = f"│  {'🧠  NEURA  —  Pipeline Trace':<{W - 2}}│"
    qline  = f"│  {('❝  ' + q_display + '  ❞'):<{W - 2}}│"
    bottom = f"╰{'─' * W}╯"

    bc   = _C["bold"] + _C["blue"]
    r    = _C["reset"]
    dim  = _C["dim"]
    bold = _C["bold"]

    print(f"\n{bc}{top}{r}",    flush=True)
    print(f"{bc}{title}{r}",   flush=True)
    print(f"{bc}│{r}  {dim}{('❝  ' + q_display + '  ❞')}{r}{' ' * max(0, W - 4 - len(q_display) - 6)}{bc}│{r}", flush=True)
    print(f"{bc}{bottom}{r}\n", flush=True)
