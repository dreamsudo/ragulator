"""External-API fetch tool, used to demonstrate clearance-tagged data ingest.

A user supplies a URL + query params; the response (if JSON) is normalised to a
pandas DataFrame and cached together with a *user-asserted* classification
label. The self-asserted label is an intentional teaching device — it shows why
client-side classification is unsafe — and is clearly labelled as such in the UI
and docs. The SSRF guard in ``security.validate_outbound_url`` runs before any
request leaves the process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .. import security

logger = logging.getLogger(__name__)


@dataclass
class ApiCache:
    """Holds the most recently fetched API data and its asserted clearance."""

    data: Optional["Any"] = None  # pandas.DataFrame at runtime
    clearance: Optional[str] = None

    def clear(self) -> None:
        self.data = None
        self.clearance = None

    @property
    def is_loaded(self) -> bool:
        try:
            return self.data is not None and not self.data.empty
        except Exception:
            return False


def fetch_api_data(
    url: str,
    params: Dict[str, str],
    asserted_clearance: str,
    settings,
) -> ApiCache:
    """Fetch + normalise external JSON into a clearance-tagged cache entry.

    Raises ``security.SSRFError`` if the URL is disallowed, and propagates
    ``requests`` exceptions for the caller to render.
    """
    import pandas as pd
    import requests

    # SSRF guard FIRST — before any network egress.
    security.validate_outbound_url(
        url,
        allowed_schemes=settings.api_allowed_schemes,
        enforce_host_allowlist=settings.api_enforce_host_allowlist,
        host_allowlist=settings.api_host_allowlist,
    )

    resp = requests.get(url, params=params, timeout=settings.api_request_timeout_s)
    resp.raise_for_status()
    raw = resp.json()  # raises ValueError on non-JSON; caller handles it

    if isinstance(raw, list):
        df = pd.DataFrame.from_records(raw)
    elif isinstance(raw, dict):
        try:
            df = pd.json_normalize(raw)
        except Exception:
            df = pd.DataFrame([raw])
    else:
        raise ValueError(f"API returned non-tabular data of type {type(raw).__name__}.")

    logger.info("Fetched %d API rows; tagged clearance '%s'.", len(df), asserted_clearance)
    return ApiCache(data=df, clearance=asserted_clearance)
