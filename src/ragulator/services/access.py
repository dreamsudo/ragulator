"""Access-control decisions: clearance checks and need-to-know (NKT).

This is the heart of the PoC's *simulated* security model. Two gates apply to
document retrieval:

1. **Clearance** — the user's level must be >= the document's level.
2. **Need-to-know** — even with clearance, access requires a job-related reason.

NKT modes (configurable):
    * ``loose``    : a clearance gap of >= 1 over the document auto-passes NKT.
    * ``moderate`` : a clearance gap of >= 2 auto-passes NKT.
    * ``strict``   : no automatic clearance-gap pass; NKT must be justified by a
                     topical keyword match between the user's stated purpose and
                     the document's subject/title.

In every mode, if the gap rule does not grant access, the system falls back to
keyword-overlap matching. Unclassified documents always pass NKT.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Words too generic to indicate a genuine topical need-to-know.
_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "in", "on", "of", "for", "to", "view", "document",
    "get", "show", "find", "about", "review", "access", "see", "read", "need",
}

_GAP_BY_MODE = {"loose": 1, "moderate": 2}  # 'strict' intentionally absent


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str


def check_clearance(user_level: int, doc_level: int) -> bool:
    return user_level >= doc_level


def check_need_to_know(
    user_purpose: Optional[str],
    doc_subject: Optional[str],
    doc_title: Optional[str],
    doc_classification_level: int,
    unclassified_level: int,
    user_clearance_level: int,
    nkt_mode: str,
) -> AccessDecision:
    """Decide whether need-to-know is satisfied for one document."""
    if doc_classification_level == unclassified_level:
        return AccessDecision(True, "NKT passed (document is Unclassified).")

    # Clearance-gap auto-pass (loose/moderate only).
    required_gap = _GAP_BY_MODE.get(nkt_mode)
    if required_gap is not None:
        gap = user_clearance_level - doc_classification_level
        if gap >= required_gap:
            return AccessDecision(
                True, f"NKT passed (clearance gap {gap} >= {required_gap} in '{nkt_mode}' mode)."
            )

    # Keyword-overlap fallback (the only path in 'strict' mode).
    if not user_purpose or not user_purpose.strip():
        return AccessDecision(False, "NKT failed: stated purpose is missing or unclear.")

    doc_text = f"{doc_subject or ''} {doc_title or ''}".lower().strip()
    if not doc_text:
        return AccessDecision(False, "NKT failed: document has no subject/title to match against.")

    purpose_words = set(user_purpose.lower().split())
    relevant = purpose_words - _STOPWORDS or purpose_words
    if relevant & set(doc_text.split()):
        return AccessDecision(True, "NKT passed (purpose keywords match document topic).")

    return AccessDecision(
        False,
        f"NKT failed: purpose '{user_purpose[:50]}' does not match document topic "
        f"'{(doc_subject or '')[:50]}'.",
    )
