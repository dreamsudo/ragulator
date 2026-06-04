"""Security primitives: input sanitisation, SSRF guarding, rule-based screening.

None of this is a substitute for a real WAF or a real authorization service.
The rule-based threat detector here is intentionally a *coarse pre-filter*: it
catches obvious prompt-injection / SQL / code-injection shapes cheaply before
the more expensive LLM check runs. It will produce false positives and can be
bypassed by determined obfuscation; that trade-off is documented rather than
hidden, because pretending otherwise would be the real security bug.
"""

from __future__ import annotations

import html
import ipaddress
import logging
import re
import socket
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# --- Coarse threat patterns (pre-filter only) ------------------------------
# Grouped by intent so a learner can see what each cluster is meant to catch.
_PROMPT_INJECTION = [
    r"ignore\s+(?:your\s+)?(?:previous|prior|above|following)\s+instructions?\b",
    r"\bforget\s+(?:your\s+)?rules\b",
    r"\bdisregard\s+(?:the\s+)?instructions\b",
    r"\bsystem\s+override\b",
    r"\bshow\s+(?:your\s+)?system\s+prompt\b",
    r"\breveal\s+(?:your\s+)?rules\b",
    r"\bwhat\s+is\s+your\s+initial\s+prompt\b",
    r"\bdeveloper\s+mode\b",
    r"\bignore\s+(?:safety|security|policy)\b",
    r"\b(?:ignore|disregard|override)\s+(?:my\s+)?clearance\b",
    r"\bbypass\s+(?:security|access\s+control)\b",
    r"\belevate\s+privileges\b",
    r"\bgive\s+(?:me\s+)?access\s+to\s+(?:all|everything)\b",
]
_SQL_INJECTION = [
    r"\bunion\s+(?:all\s+)?select\b",
    r"\bdrop\s+table\b",
    r"\bdelete\s+from\b",
    r";\s*--",
    r"'\s*or\s*'?\d+'?\s*=\s*'?\d+",
    r"\bxp_cmdshell\b",
    r"\binformation_schema\.",
]
_POLICY_VIOLATION = [
    r"\b(?:delete|drop|modify|reclassify)\s+(?:document|file|record)\b",
    r"\bchange\s+classification\b",
    r"\b(?:export|exfiltrate|email|forward)\s+(?:document|file|data)\b",
    r"\bdump\s+database\b",
    r"\blist\s+all\s+files\b",
]
_CODE_INJECTION = [
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"__import__\s*\(",
    r"\bos\.(?:system|popen|spawn)\b",
    r"\bsubprocess\.(?:run|call|check_output|Popen)\b",
    r"\brm\s+-rf\b",
    r"`[^`]+`",
    r"\$\([^)]+\)",
    r"<\s*script",
    r"javascript:",
]

ALL_PATTERNS: List[str] = list(
    dict.fromkeys(_PROMPT_INJECTION + _SQL_INJECTION + _POLICY_VIOLATION + _CODE_INJECTION)
)


def _compile() -> Optional[re.Pattern]:
    try:
        # NOTE: no re.DOTALL — we do not want '.' to span newlines here, which
        # previously made several broad patterns far greedier than intended.
        return re.compile("|".join(f"(?:{p})" for p in ALL_PATTERNS), re.IGNORECASE)
    except re.error as exc:  # pragma: no cover - defensive
        logger.critical("Failed to compile threat regex: %s", exc)
        return None


_THREAT_RE = _compile()


def sanitize_user_input(text: object, max_length: int) -> str:
    """HTML-escape and length-cap arbitrary user input.

    Escaping here protects the *display* layer (results are rendered as HTML).
    It is not a semantic security control on its own.
    """
    if not isinstance(text, str):
        return ""
    escaped = html.escape(text)
    if len(escaped) > max_length:
        logger.warning("Input exceeded %d chars; truncating.", max_length)
        escaped = escaped[:max_length] + "..."
    return escaped


def detect_threat_rules(query: str) -> Optional[List[str]]:
    """Return a sorted list of matched threat phrases, or ``None`` if clean.

    Returns ``["REGEX_ERROR"]`` if the compiled pattern is unavailable, so the
    caller can fail safe.
    """
    if not isinstance(query, str) or not query:
        return None
    if _THREAT_RE is None:
        return ["REGEX_ERROR"]
    matches = {
        m.group(0).strip().lower()
        for m in _THREAT_RE.finditer(query)
        if m.group(0) and not m.group(0).isspace()
    }
    if matches:
        phrases = sorted(matches)
        logger.warning("[rules] threat phrases detected: %s", phrases)
        return phrases
    return None


# --- SSRF guard for the external-API tool ----------------------------------
class SSRFError(ValueError):
    """Raised when a requested URL is disallowed by the SSRF policy."""


def validate_outbound_url(
    url: str,
    allowed_schemes: List[str],
    enforce_host_allowlist: bool,
    host_allowlist: List[str],
) -> str:
    """Validate a user-supplied URL before an outbound request.

    Blocks:
      * non-allowed schemes (default: https only),
      * URLs with no host,
      * hosts that resolve to private / loopback / link-local / reserved IPs
        (the classic SSRF path to cloud metadata at 169.254.169.254).

    This is a *best-effort* guard. It does not defend against DNS-rebinding
    (where a host resolves benign now and malicious later); a production system
    needs request-time pinning and an egress proxy. That residual risk is
    called out explicitly rather than glossed over.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {s.lower() for s in allowed_schemes}:
        raise SSRFError(f"Scheme '{scheme or '(none)'}' not allowed. Use one of {allowed_schemes}.")

    host = parsed.hostname
    if not host:
        raise SSRFError("URL has no host component.")

    if enforce_host_allowlist:
        if host.lower() not in {h.lower() for h in host_allowlist}:
            raise SSRFError(f"Host '{host}' is not in the allowlist.")

    # Resolve every address the host maps to and reject any non-public one.
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if scheme == "https" else 80))
    except socket.gaierror as exc:
        raise SSRFError(f"Could not resolve host '{host}': {exc}") from exc

    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise SSRFError(
                f"Host '{host}' resolves to a non-public address ({ip_str}); blocked."
            )

    return url
