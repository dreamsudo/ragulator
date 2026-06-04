"""The RAGulator - it regulates.  (Psypher Labs)

A self-contained proof-of-concept that demonstrates layered defence for an
LLM-mediated RAG document access-control system: rule-based screening, an LLM
threat check, role-based access control, and need-to-know enforcement, with a
Google-Gemini RAG policy assistant. Runs in Google Colab and in a local IDE.

This is a teaching/demo artifact, NOT a production access-control system. The
identity, role, and data-classification inputs are self-asserted by design to
illustrate exactly which trust boundaries a real system must replace.
"""

from __future__ import annotations

from .app import App, build_app
from .config import Settings

__version__ = "9.0.1"
__all__ = ["App", "build_app", "Settings", "__version__"]
