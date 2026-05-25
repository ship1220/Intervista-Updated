"""
Application-wide logging configuration.

Terminal output policy:
  - Structured RL/bandit logs (utils.bandit_logger print)
  - Minimal HTTP request lines (FastAPI middleware)
  - Warnings and errors from the app
  - Brief startup confirmation

Suppressed:
  - Uvicorn access logs (replaced by middleware)
  - RAG / vector store / retriever / chain / LLM routine INFO
  - JSON debug-style logs from utils.Logger
"""

from __future__ import annotations

import logging
import sys


# Third-party and internal modules that should not chatter at INFO
_QUIET_LOGGERS = (
    "uvicorn.access",
    "uvicorn",
    "fastapi",
    "httpx",
    "httpcore",
    "sqlalchemy.engine",
    "sqlalchemy",
    "sentence_transformers",
    "transformers",
    "faiss",
    "groq",
    "openai",
    "services.rag",
    "services.rag.rag_pipeline",
    "services.rag.retriever",
    "services.rag.vector_store",
    "services.rag.document_ingester",
    "core.chains",
    "core.chains.base_chain",
    "core.prompts",
    "core.prompts.prompt_manager",
    "core.llm",
    "core.llm.llm_service",
    "core.agents",
    "utils.rl_helpers",
    "services.rl.rl_service",
    "watchfiles",
)


def configure_app_logging() -> None:
    """Apply logging levels before the rest of the app imports noisy modules."""

    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    # Uvicorn errors only (startup / crashes), not per-request access lines
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # App module: warnings/errors only (routine route INFO suppressed)
    logging.getLogger("main").setLevel(logging.WARNING)

    # Ensure one simple stderr handler for warnings/errors if none exist
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(
            logging.Formatter("%(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(handler)


def log_startup_banner(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Single-line startup confirmation for developers."""
    print(f"\n  Intervista API ready — http://{host}:{port}\n", flush=True)
