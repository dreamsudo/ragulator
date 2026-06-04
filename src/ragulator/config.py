"""Central configuration for The RAGulator (Psypher Labs).

Design intent
-------------
Everything a learner might want to tweak lives here in one typed object, and
*nothing* secret is ever stored in code. Secrets (the Google API key) are
resolved at runtime from, in priority order:

    1. Google Colab secrets  (``google.colab.userdata``)
    2. Process environment   (``GOOGLE_API_KEY``)
    3. A local ``.env`` file  (loaded via python-dotenv, git-ignored)

Pydantic is used for validation when available; if it is not installed the
module transparently falls back to a stdlib dataclass with equivalent
validation, so the PoC still imports and runs (e.g. for offline data-layer
exploration). Behaviour is identical either way.

To take this PoC toward production you typically only need to swap the values
flagged with ``# PROD:`` comments.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VALID_NKT_MODES = ("strict", "moderate", "loose")
_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _load_api_key() -> Optional[str]:
    """Resolve the Google API key from Colab secrets, env, or .env.

    Returns ``None`` if no key is found; callers decide whether that is fatal.
    The key is never logged and never written to disk by this project.
    """
    try:
        from google.colab import userdata  # type: ignore

        key = userdata.get("GOOGLE_API_KEY")
        if key:
            logger.info("Loaded GOOGLE_API_KEY from Colab secrets.")
            return key
    except Exception:
        pass

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    key = os.getenv("GOOGLE_API_KEY")
    if key:
        logger.info("Loaded GOOGLE_API_KEY from environment / .env.")
    return key


# --- default field values, shared by both implementations ------------------
def _default_classification_levels() -> Dict[str, int]:
    return {"TS": 4, "S": 3, "C": 2, "U": 1}


def _default_roles_map() -> Dict[str, str]:
    return {
        "administrator": "TS", "senior analyst": "TS", "project manager": "S",
        "analyst": "S", "developer": "C", "support staff": "C",
        "intern/trainee": "U", "public user": "U", "unknown_user": "U",
    }


# Detect pydantic once.
try:
    from pydantic import BaseModel, Field, field_validator

    _HAVE_PYDANTIC = True
except Exception:  # pragma: no cover - exercised only when pydantic absent
    _HAVE_PYDANTIC = False


class _SettingsBehaviour:
    """Methods common to both the pydantic and dataclass implementations."""

    classification_levels: Dict[str, int]
    roles_clearance_map: Dict[str, str]
    data_dir: Path
    db_filename: str
    nkt_mode: str
    log_level: str

    @property
    def db_path(self) -> Path:
        return (Path(self.data_dir) / self.db_filename).resolve()

    def clearance_for_role(self, role: Optional[str]) -> Tuple[str, int]:
        """Map a (possibly unknown) role to (clearance_str, clearance_int).

        Unknown roles deliberately fail safe to the lowest level ('U').
        """
        role_key = (role or "").lower().strip() or "unknown_user"
        cls_str = self.roles_clearance_map.get(role_key, "U")
        cls_int = self.classification_levels.get(cls_str, 1)
        return cls_str, cls_int

    def level_int_to_str(self, level: int, default: str = "?") -> str:
        for name, value in self.classification_levels.items():
            if value == level:
                return name
        return default


def _validate_common(values: dict) -> dict:
    """Validate/normalise a settings dict (used by the dataclass path)."""
    nkt = values.get("nkt_mode", "moderate")
    if nkt not in _VALID_NKT_MODES:
        raise ValueError(f"Invalid nkt_mode: {nkt!r}")
    lvl = str(values.get("log_level", "INFO")).upper()
    if lvl not in _VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log_level: {lvl!r}")
    values["log_level"] = lvl
    num = int(values.get("num_documents", 1000))
    if num <= 0:
        raise ValueError("num_documents must be > 0")
    values["num_documents"] = num
    if not 0.0 <= float(values.get("llm_temperature", 0.1)) <= 1.0:
        raise ValueError("llm_temperature must be in [0, 1]")
    return values


if _HAVE_PYDANTIC:

    class Settings(_SettingsBehaviour, BaseModel):  # type: ignore[misc]
        """Typed, validated application settings (pydantic implementation)."""

        gemini_model: str = Field(default="gemini-2.5-flash")
        embedding_model: str = Field(default="models/gemini-embedding-001")
        llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
        llm_max_retries: int = Field(default=3, ge=0, le=8)

        classification_levels: Dict[str, int] = Field(default_factory=_default_classification_levels)
        roles_clearance_map: Dict[str, str] = Field(default_factory=_default_roles_map)
        nkt_mode: str = Field(default="moderate")

        db_filename: str = Field(default="doc_store_poc.db")
        table_name: str = Field(default="documents")
        num_documents: int = Field(default=1000, gt=0, le=100_000)

        policy_rag_k: int = Field(default=3, ge=1, le=10)
        max_input_length: int = Field(default=2048, gt=0, le=32_000)

        api_allowed_schemes: List[str] = Field(default_factory=lambda: ["https"])
        api_request_timeout_s: int = Field(default=20, gt=0, le=120)
        api_enforce_host_allowlist: bool = Field(default=False)
        api_host_allowlist: List[str] = Field(default_factory=lambda: ["jsonplaceholder.typicode.com"])

        data_dir: Path = Field(default=Path("./data"))
        log_level: str = Field(default="INFO")

        google_api_key: Optional[str] = Field(default=None, repr=False, exclude=True)

        model_config = {"frozen": False, "arbitrary_types_allowed": True}

        @field_validator("nkt_mode")
        @classmethod
        def _v_nkt(cls, v: str) -> str:
            if v not in _VALID_NKT_MODES:
                raise ValueError(f"Invalid nkt_mode: {v!r}")
            return v

        @field_validator("log_level")
        @classmethod
        def _v_log(cls, v: str) -> str:
            v = v.upper()
            if v not in _VALID_LOG_LEVELS:
                raise ValueError(f"Invalid log_level: {v!r}")
            return v

        @classmethod
        def from_environment(cls, **overrides) -> "Settings":
            merged = _collect_env_overrides(cls.model_fields.keys())
            merged.update(overrides)
            merged.setdefault("google_api_key", _load_api_key())
            settings = cls(**merged)
            Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
            return settings

        @classmethod
        def from_yaml(cls, path, **overrides) -> "Settings":
            return cls.from_environment(**{**_read_yaml(path), **overrides})

else:  # pragma: no cover - exercised only when pydantic absent

    from dataclasses import dataclass, field

    @dataclass
    class Settings(_SettingsBehaviour):  # type: ignore[no-redef]
        """Stdlib fallback settings (used when pydantic is not installed)."""

        gemini_model: str = "gemini-2.5-flash"
        embedding_model: str = "models/gemini-embedding-001"
        llm_temperature: float = 0.1
        llm_max_retries: int = 3

        classification_levels: Dict[str, int] = field(default_factory=_default_classification_levels)
        roles_clearance_map: Dict[str, str] = field(default_factory=_default_roles_map)
        nkt_mode: str = "moderate"

        db_filename: str = "doc_store_poc.db"
        table_name: str = "documents"
        num_documents: int = 1000

        policy_rag_k: int = 3
        max_input_length: int = 2048

        api_allowed_schemes: List[str] = field(default_factory=lambda: ["https"])
        api_request_timeout_s: int = 20
        api_enforce_host_allowlist: bool = False
        api_host_allowlist: List[str] = field(default_factory=lambda: ["jsonplaceholder.typicode.com"])

        data_dir: Path = field(default_factory=lambda: Path("./data"))
        log_level: str = "INFO"
        google_api_key: Optional[str] = None

        def __post_init__(self) -> None:
            values = _validate_common(
                {
                    "nkt_mode": self.nkt_mode,
                    "log_level": self.log_level,
                    "num_documents": self.num_documents,
                    "llm_temperature": self.llm_temperature,
                }
            )
            self.nkt_mode = values["nkt_mode"]
            self.log_level = values["log_level"]
            self.num_documents = values["num_documents"]

        @classmethod
        def _field_names(cls):
            return cls.__dataclass_fields__.keys()

        @classmethod
        def from_environment(cls, **overrides) -> "Settings":
            merged = _collect_env_overrides(cls._field_names())
            merged.update(overrides)
            merged.setdefault("google_api_key", _load_api_key())
            merged = _coerce_types(merged)
            settings = cls(**merged)
            Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
            return settings

        @classmethod
        def from_yaml(cls, path, **overrides) -> "Settings":
            return cls.from_environment(**{**_read_yaml(path), **overrides})


# --- helpers shared across both branches -----------------------------------
def _collect_env_overrides(field_names) -> dict:
    """Pull CDP_<FIELD> environment variables into a dict of overrides."""
    out: Dict[str, object] = {}
    for name in field_names:
        env_name = f"CDP_{name.upper()}"
        if env_name in os.environ:
            out[name] = os.environ[env_name]
    return out


def _coerce_types(values: dict) -> dict:
    """Best-effort coercion of stringy config values for the dataclass path."""
    int_fields = {"llm_max_retries", "num_documents", "policy_rag_k",
                  "max_input_length", "api_request_timeout_s"}
    float_fields = {"llm_temperature"}
    bool_fields = {"api_enforce_host_allowlist"}
    for k in list(values):
        v = values[k]
        if isinstance(v, str):
            if k in int_fields:
                values[k] = int(v)
            elif k in float_fields:
                values[k] = float(v)
            elif k in bool_fields:
                values[k] = v.strip().lower() in ("1", "true", "yes", "on")
        if k == "data_dir":
            values[k] = Path(v)
    return values


def _read_yaml(path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    except Exception as exc:  # pragma: no cover - config is best-effort
        logger.warning("Could not read %s (%s); using defaults.", path, exc)
        return {}
