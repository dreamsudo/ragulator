"""Request orchestration: the end-to-end pipeline for a single user query.

Order of operations (defence in depth):

    sanitise -> rule pre-filter -> LLM threat check -> intent analysis ->
    action dispatch (search / retrieve / policy / general / show-api)

Each stage can short-circuit with a structured result. The function returns a
plain dict (``ProcessResult``-shaped) that the UI layer knows how to render, so
the core logic has no UI dependency and is straightforward to unit-test.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from . import security
from .services import access, database
from .services.external_api import ApiCache
from .services.llm import GeminiService

logger = logging.getLogger(__name__)

_API_ALLOWED_ROLES = {"administrator", "senior analyst", "project manager", "analyst"}


def process_request(
    *,
    raw_query: str,
    user_role: str,
    settings,
    gemini: GeminiService,
    engine,
    api_cache: Optional[ApiCache] = None,
) -> Dict[str, Any]:
    """Run one query through the full pipeline and return a render-ready dict."""
    start = time.time()
    query = security.sanitize_user_input(raw_query, settings.max_input_length)
    role_key = (user_role or "unknown_user").lower()
    clearance_str, clearance_int = settings.clearance_for_role(role_key)

    out: Dict[str, Any] = {
        "query": query,
        "raw_query": raw_query,
        "user_role": user_role.title() if user_role else "Unknown",
        "user_clearance": clearance_str,
        "intent": "UNKNOWN",
        "purpose": None,
        "results": [],
        "error_type": None,
        "duration": 0.0,
    }

    def _finish(results: List[Dict[str, Any]], error_type: Optional[str] = None) -> Dict[str, Any]:
        out["results"] = results
        out["error_type"] = error_type
        out["duration"] = round(time.time() - start, 2)
        return out

    if not query or not query.strip():
        return _finish([{"error": "Input", "reason": "Query was empty after sanitisation."}], "Input")

    # 1. Rule-based pre-filter.
    rule_hits = security.detect_threat_rules(query)
    if rule_hits:
        if rule_hits == ["REGEX_ERROR"]:
            return _finish([{"error": "System", "reason": "Internal rule-check error."}], "System")
        return _finish(
            [{"error": "Security Block", "reason": f"Rule violation: {', '.join(rule_hits)}"}],
            "Security Block",
        )

    # 2. LLM threat analysis (fails closed).
    is_threat, threat_reason = gemini.analyze_threat(query, role_key)
    if is_threat:
        etype = "System" if threat_reason.startswith("SYSTEM_ERROR") else "Security Block"
        return _finish([{"error": etype, "reason": threat_reason}], etype)

    # 3. Intent analysis.
    intent_res = gemini.analyze_intent(query, role_key, clearance_str)
    if intent_res.error:
        return _finish([{"error": "System", "reason": intent_res.error}], "System")
    out["intent"] = intent_res.intent
    out["purpose"] = intent_res.purpose

    # 4. Dispatch.
    try:
        results = _dispatch(intent_res, clearance_int, settings, gemini, engine, api_cache, role_key)
    except Exception as exc:  # noqa: BLE001 - last-resort guard
        logger.critical("Action execution error: %s", exc, exc_info=True)
        return _finish([{"error": "System", "reason": "Critical error during execution."}], "System")

    # Derive a top-level error_type from any access denials for the UI banner.
    error_type = None
    for item in results:
        if item.get("access") == "DENY" or item.get("error") == "ACCESS_DENIED":
            error_type = "Access Denied"
            break
    return _finish(results, error_type)


def _dispatch(intent_res, clearance_int, settings, gemini, engine, api_cache, role_key) -> List[Dict[str, Any]]:
    intent = intent_res.intent

    if intent == "SEARCH_DOCUMENTS":
        terms = intent_res.search_terms
        if not terms:
            return [{"warning": "Could not identify search terms."}]
        matches = database.search_documents(
            engine, settings.table_name, terms, clearance_int, settings.classification_levels
        )
        if not matches:
            return [{"not_found": f"No accessible documents matched '{terms}'."}]
        contents = database.fetch_contents(engine, settings.table_name, [m["DocID"] for m in matches])
        enriched = []
        for m in matches:
            row = dict(m)
            row["access"] = "ALLOW"
            row["need_to_know"] = "NOT_CHECKED"
            row["Content"] = contents.get(m["DocID"])
            enriched.append(row)
        return enriched

    if intent == "RETRIEVE_DOCUMENT":
        doc_id = intent_res.doc_id
        if doc_id is None:
            return [{"error": "Input", "reason": "Could not identify a valid document ID."}]
        return [_retrieve_one(doc_id, clearance_int, intent_res.purpose, settings, engine)]

    if intent == "POLICY_QUESTION":
        answer = gemini.answer_policy_question(intent_res.policy_query or "")
        return [{"policy_info": answer}]

    if intent == "GENERAL_QUESTION":
        return [{"general_response": gemini.general_response(intent_res.purpose or "")}]

    if intent == "SHOW_API_DATA":
        return [_show_api(api_cache, clearance_int, settings, role_key)]

    if intent == "UNKNOWN":
        return [{"clarification_needed": "Could not determine the action. Please rephrase."}]

    return [{"error": "System", "reason": f"Unhandled intent: {intent}"}]


def _retrieve_one(doc_id, clearance_int, purpose, settings, engine) -> Dict[str, Any]:
    doc = database.get_document(engine, settings.table_name, doc_id)
    if not doc:
        return {"not_found": f"Document ID {doc_id} does not exist."}

    doc_cls = doc.get("Classification")
    doc_level = settings.classification_levels.get(doc_cls, 99)
    user_cls = settings.level_int_to_str(clearance_int)

    if not access.check_clearance(clearance_int, doc_level):
        return {
            "DocID": doc.get("DocID"), "Title": doc.get("Title"), "Classification": doc_cls,
            "Subject": doc.get("Subject"), "access": "DENY", "need_to_know": "NOT_APPLICABLE",
            "error": "ACCESS_DENIED",
            "reason": f"Clearance ({user_cls}) insufficient for document ({doc_cls}).",
        }

    decision = access.check_need_to_know(
        user_purpose=purpose,
        doc_subject=doc.get("Subject"),
        doc_title=doc.get("Title", ""),
        doc_classification_level=doc_level,
        unclassified_level=settings.classification_levels.get("U", 1),
        user_clearance_level=clearance_int,
        nkt_mode=settings.nkt_mode,
    )
    if decision.allowed:
        doc["access"] = "ALLOW"
        doc["need_to_know"] = "PASS"
        return doc
    return {
        "DocID": doc.get("DocID"), "Title": doc.get("Title"), "Classification": doc_cls,
        "Subject": doc.get("Subject"), "access": "DENY", "need_to_know": "FAIL",
        "error": "ACCESS_DENIED", "reason": f"Access denied: {decision.reason}",
    }


def _show_api(api_cache, clearance_int, settings, role_key) -> Dict[str, Any]:
    if api_cache is None or not api_cache.is_loaded:
        return {"info": "No API data loaded. Use the 'Call API' tool first."}
    api_level = settings.classification_levels.get(api_cache.clearance, 99)
    role_ok = role_key in _API_ALLOWED_ROLES
    clearance_ok = clearance_int >= api_level
    if role_ok and clearance_ok:
        try:
            # escape=True (default) prevents HTML injection from API cell values.
            html_table = api_cache.data.to_html(index=False, max_rows=50, max_cols=20, border=1)
            return {"api_data": html_table}
        except Exception as exc:  # noqa: BLE001
            logger.error("API DataFrame render error: %s", exc)
            return {"error": "System", "reason": "Could not format API data."}
    user_cls = settings.level_int_to_str(clearance_int)
    reason = ""
    if not role_ok:
        reason += f"Role '{role_key.title()}' is not authorised for API data. "
    if not clearance_ok:
        reason += f"Clearance ('{user_cls}') insufficient for data ('{api_cache.clearance}')."
    return {"access": "DENY", "error": "ACCESS_DENIED", "reason": reason.strip()}
