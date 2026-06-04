"""Application assembly: build settings + services into a ready-to-use ``App``.

This is the single place where the whole system is wired together, so both the
CLI and the notebook UI can share identical initialisation. Nothing here is
Colab-specific; it runs the same in VS Code, a plain shell, or a notebook.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .config import Settings
from .policies import POLICY_DOCS
from .services import database
from .services.external_api import ApiCache
from .services.llm import GeminiService

logger = logging.getLogger(__name__)


@dataclass
class App:
    settings: Settings
    gemini: GeminiService
    engine: object
    api_cache: ApiCache


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    # Quiet noisy third-party loggers.
    for noisy in ("urllib3", "google", "httpx", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_app(settings: Optional[Settings] = None, *, require_llm: bool = True) -> App:
    """Construct the full application.

    Set ``require_llm=False`` to build the DB + access layers without contacting
    Google (useful for tests and for exploring the data layer offline).
    """
    settings = settings or Settings.from_environment()
    configure_logging(settings.log_level)
    logger.info("Initialising application (nkt_mode=%s).", settings.nkt_mode)

    # Database (always available; no network needed).
    engine = database.get_engine(settings.db_path)
    database.setup_schema(engine, settings.table_name)
    inserted = database.populate(
        engine, settings.table_name, settings.num_documents, settings.classification_levels
    )
    logger.info("Database ready (%d new documents inserted).", inserted)

    # Gemini + RAG (optional).
    gemini = GeminiService(settings)
    if require_llm:
        gemini.connect()
        gemini.build_policy_index(POLICY_DOCS)
    else:
        logger.warning("Skipping LLM initialisation (require_llm=False).")

    return App(settings=settings, gemini=gemini, engine=engine, api_cache=ApiCache())
