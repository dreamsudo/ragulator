"""ipywidgets interactive UI for Colab / Jupyter.

Importing this module does nothing on its own; call ``launch(app)`` from a
notebook cell. All HTML is escaped at the boundary, and the rendering helpers
are split out so the markup stays readable.
"""

from __future__ import annotations

import html
import logging
import time
import urllib.parse
from typing import Any, Dict

from .. import security
from ..app import App
from ..pipeline import process_request
from ..services.external_api import fetch_api_data

logger = logging.getLogger(__name__)


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "N/A"))


def _render_item(item: Dict[str, Any]) -> str:
    """Turn one result dict into an HTML snippet (all values escaped)."""
    if "DocID" in item and "error" not in item:
        access = item.get("access", "?")
        border, bg = ("#28a745", "#d4edda") if access == "ALLOW" else ("#dc3545", "#f8d7da")
        nkt = _esc(item.get("need_to_know", "?"))
        if access == "ALLOW":
            content = (
                "<details style='margin-top:8px;'><summary style='cursor:pointer;'>Show content"
                f"</summary><pre style='white-space:pre-wrap;max-height:200px;overflow:auto;"
                f"background:#fff;border:1px solid #ccc;padding:8px;'>{_esc(item.get('Content'))}</pre></details>"
            )
        else:
            content = "<span style='color:#6c757d;'>Content not accessible.</span>"
        reason = (
            f"<br><strong style='color:#dc3545;'>Reason:</strong> {_esc(item.get('reason'))}"
            if access == "DENY" and item.get("reason")
            else ""
        )
        return (
            f"<div style='border:2px solid {border};background:{bg};padding:12px;margin:8px 0;"
            f"border-radius:5px;'><strong>DocID {_esc(item.get('DocID'))}</strong> | "
            f"<strong>Cls:</strong> {_esc(item.get('Classification'))} | <strong>NKT:</strong> {nkt}"
            f"<br><strong>Title:</strong> {_esc(item.get('Title'))}"
            f"<br><strong>Subject:</strong> {_esc(item.get('Subject'))}{reason}<br>{content}</div>"
        )

    simple = {
        "policy_info": ("#17a2b8", "#d1ecf1", "Policy"),
        "general_response": ("#6c757d", "#f8f9fa", "Assistant"),
        "info": ("#17a2b8", "#d1ecf1", "Info"),
        "warning": ("#ffc107", "#fff3cd", "Warning"),
        "not_found": ("#6c757d", "#e9ecef", "Not found"),
        "clarification_needed": ("#ffc107", "#fff3cd", "Clarification"),
    }
    for key, (border, bg, label) in simple.items():
        if key in item:
            return (
                f"<div style='border:1px solid {border};background:{bg};padding:10px;margin:8px 0;"
                f"border-radius:4px;'><strong>{label}:</strong><p style='white-space:pre-wrap;'>"
                f"{_esc(item.get(key))}</p></div>"
            )

    if "api_data" in item:
        # api_data is already-escaped HTML produced by DataFrame.to_html(escape=True).
        return (
            "<div style='border:1px solid #007bff;background:#cce5ff;padding:10px;margin:8px 0;"
            "border-radius:4px;'><strong>API Data:</strong><div style='max-height:400px;overflow:auto;"
            f"background:#fff;padding:5px;'>{item['api_data']}</div></div>"
        )

    if "error" in item:
        return (
            f"<div style='border:2px solid #dc3545;background:#f8d7da;padding:10px;margin:8px 0;"
            f"border-radius:4px;'><strong>Error ({_esc(item.get('error'))}):</strong>"
            f"<p style='white-space:pre-wrap;'>{_esc(item.get('reason'))}</p></div>"
        )

    return f"<div style='border:1px dashed #ccc;padding:5px;'>{_esc(item)}</div>"


def launch(app: App) -> None:  # pragma: no cover - requires a notebook frontend
    """Build and display the interactive UI for the given application."""
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display

    settings = app.settings
    role_options = sorted(
        (r for r in settings.roles_clearance_map if r != "unknown_user"),
        key=lambda r: (-settings.classification_levels.get(settings.roles_clearance_map[r], 0), r),
    )

    role_dd = widgets.Dropdown(
        options=[r.title() for r in role_options], value="Analyst", description="Role:"
    )
    query_box = widgets.Textarea(placeholder="Enter your query...", description="Query:",
                                 layout={"width": "98%", "height": "90px"})
    submit_btn = widgets.Button(description="Submit Query", button_style="success", icon="paper-plane")

    api_url = widgets.Text(value="https://jsonplaceholder.typicode.com/posts", description="API URL:",
                           layout={"width": "98%"})
    api_params = widgets.Text(value="userId=1", description="Params:", layout={"width": "98%"})
    api_cls = widgets.Dropdown(options=list(settings.classification_levels), value="S",
                               description="Assert Cls:")
    api_btn = widgets.Button(description="Call API & Set Cls", button_style="info", icon="cloud-download")

    results_out = widgets.Output(layout={"border": "1px solid #ccc", "padding": "10px"})
    api_status = widgets.Output(layout={"padding": "5px"})

    def on_submit(_):
        submit_btn.disabled = True
        try:
            out = process_request(
                raw_query=query_box.value, user_role=role_dd.value, settings=settings,
                gemini=app.gemini, engine=app.engine, api_cache=app.api_cache,
            )
            with results_out:
                clear_output(wait=True)
                header = (
                    f"<div style='background:#e9ecef;padding:10px;border-radius:4px;'>"
                    f"<strong>Role:</strong> {_esc(out['user_role'])} | "
                    f"<strong>Clearance:</strong> {_esc(out['user_clearance'])} | "
                    f"<strong>Intent:</strong> {_esc(out['intent'])} | "
                    f"<strong>Purpose:</strong> {_esc(out['purpose'])}</div>"
                )
                body = "".join(_render_item(i) for i in out["results"]) or "<i>No results.</i>"
                footer = f"<p style='text-align:right;color:#6c757d;font-size:.8em;'>{out['duration']}s</p>"
                display(HTML(header + body + footer))
        except Exception as exc:  # noqa: BLE001
            logger.critical("Submit handler error: %s", exc, exc_info=True)
            with results_out:
                clear_output(wait=True)
                display(HTML(f"<b>Critical error:</b> {_esc(exc)}"))
        finally:
            submit_btn.disabled = False

    def on_api(_):
        with api_status:
            clear_output(wait=True)
            params: Dict[str, str] = {}
            if api_params.value.strip():
                params = {k: v[0] for k, v in urllib.parse.parse_qs(api_params.value.strip()).items()}
            try:
                app.api_cache = fetch_api_data(api_url.value.strip(), params, api_cls.value, settings)
                n = len(app.api_cache.data) if app.api_cache.is_loaded else 0
                display(HTML(
                    f"<p style='color:green;'>Fetched {n} rows; asserted clearance "
                    f"<strong>{_esc(api_cls.value)}</strong>. Now ask 'show API data'.</p>"
                    "<p style='color:#856404;font-size:.85em;'>Note: client-asserted classification "
                    "is a teaching device and is NOT how real systems classify data.</p>"
                ))
            except security.SSRFError as exc:
                display(HTML(f"<p style='color:red;'>Blocked by SSRF policy: {_esc(exc)}</p>"))
            except Exception as exc:  # noqa: BLE001
                display(HTML(f"<p style='color:red;'>API error: {_esc(exc)}</p>"))

    submit_btn.on_click(on_submit)
    api_btn.on_click(on_api)

    display(widgets.VBox([
        widgets.HTML("<h2 style='color:#0056b3;'>The RAGulator <span style='color:#00FF41;'>&#9670;</span> <small>it regulates &middot; Psypher Labs</small></h2>"),
        widgets.HTML("<h3>Document / Policy Query</h3>"), role_dd, query_box, submit_btn,
        widgets.HTML("<hr><h3>External API Tool</h3>"), api_url, api_params, api_cls, api_btn, api_status,
        widgets.HTML("<hr><h3>Results</h3>"), results_out,
    ]))
