# The RAGulator  ◆ 

**A Psypher Labs proof-of-concept.** Layered defence for an LLM-mediated RAG document access-control system.

A compact, **teaching-grade** demonstration of layered defence for an
LLM-mediated document assistant. A user picks a role, asks a question in plain
language, and the system decides *what* they're asking for and *whether they're
allowed to have it* — using Google Gemini for reasoning, FAISS for policy
retrieval (RAG), and a simulated role/clearance/need-to-know model for access
control.

It runs unchanged in **Google Colab** and in a **local IDE / VS Code**.

---

## ⚠️ Read this first: what this is and is not

This is a **proof of concept for learning**, not a real security system. By
deliberate design, the following inputs are *self-asserted* so you can see
exactly which trust boundaries a production system must replace:

| Input | In this PoC | In production you must replace it with |
|---|---|---|
| **Identity / role** | Chosen from a dropdown | Real authentication (SSO/IdP) |
| **Clearance** | A static dictionary lookup | Authoritative directory / entitlement service |
| **Data classification** | Asserted by whoever fetches it | A trusted labelling/classification pipeline |
| **Documents** | Synthetic Faker text in SQLite | Your real, access-controlled data store |

The regex threat filter is a **coarse pre-filter**, not a WAF: it produces false
positives and can be bypassed. The SSRF guard blocks the dangerous common cases
(private/loopback/metadata IPs) but does not defend against DNS-rebinding. These
limitations are stated plainly in code comments too — being honest about them is
part of the lesson.

What *is* solid here and carries toward production: the layered pipeline shape,
parameterised DB access, the SSRF guard, fail-closed threat handling, retry/
backoff on LLM calls, a typed/validated config, clean module boundaries, and a
test suite.

---

## Quickstart

### Option A — Google Colab
1. Upload this project folder (or `pip install` it from your repo).
2. Add your key under the **🔑 Secrets** panel as `GOOGLE_API_KEY`.
3. Open `notebooks/demo.ipynb` and run all cells. The widget UI appears at the end.

### Option B — Local / VS Code
```bash
# 1. Create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install
pip install -r requirements.txt        # or:  pip install -e ".[notebook,dev]"

# 3. Provide your key
cp .env.example .env                   # then edit .env and paste your key

# 4. Run the interactive CLI
python -m ragulator --role analyst

# ...or a single query:
python -m ragulator --role administrator --query "find documents about budget"

# ...or explore the data layer with no Google calls at all:
python -m ragulator --no-llm --query "anything"
```

Get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## How a request flows

```
        user query + role
               │
        sanitise input                 (HTML-escape, length-cap)
               │
        rule pre-filter                (cheap regex; fail-closed on error)
               │
        LLM threat check  ── threat ──▶ blocked   (Gemini; fails CLOSED)
               │ clean
        intent analysis                (Gemini → SEARCH / RETRIEVE / POLICY / …)
               │
        ┌──────┴───────────────────────────────────────┐
        ▼              ▼            ▼          ▼         ▼
     search        retrieve      policy    general   show-api
   (clearance     (clearance +    (RAG)   (redirect) (role+clearance
    filter)        need-to-know)                       on cached data)
```

Each stage can short-circuit with a structured result. The pipeline returns a
plain dict, so the core has no UI dependency and is easy to test.

---

## Project layout

```
ragulator/
├── config.yaml                 # friendly settings (no secrets)
├── .env.example                # secret template (copy to .env)
├── requirements.txt
├── pyproject.toml              # installable package + tooling config
├── Makefile                    # make install / test / run / lint
├── README.md
├── notebooks/
│   └── demo.ipynb              # Colab entrypoint
├── src/ragulator/
│   ├── __init__.py
│   ├── __main__.py             # CLI entrypoint
│   ├── config.py               # typed Settings + secret resolution
│   ├── security.py             # sanitise, threat regex, SSRF guard
│   ├── policies.py             # policy text (display + RAG)
│   ├── pipeline.py             # request orchestration
│   ├── app.py                  # wires settings+services into an App
│   ├── services/
│   │   ├── database.py         # SQLite schema, population, search/retrieve
│   │   ├── access.py           # clearance + need-to-know logic
│   │   ├── llm.py              # Gemini client, threat/intent/RAG, retries
│   │   └── external_api.py     # SSRF-guarded fetch + clearance-tagged cache
│   └── ui/
│       └── notebook.py         # ipywidgets UI
└── tests/
    └── test_core.py            # offline unit tests (no network/key needed)
```

---

## Configuration

Defaults live in code; override via `config.yaml`, or via `CDP_*` environment
variables (e.g. `CDP_NKT_MODE=strict`), or programmatically:

```python
from ragulator import Settings, build_app
settings = Settings.from_yaml("config.yaml", nkt_mode="strict")
app = build_app(settings)
```

The **only** secret, `GOOGLE_API_KEY`, is resolved at runtime from Colab secrets
→ environment → `.env`, and is never logged or serialised.

---

## Testing

```bash
pip install -e ".[dev]"
pytest          # offline; no API key or network required
ruff check .    # lint
```

Two test modules:

* ``tests/test_core.py`` — unit tests: input sanitisation, the threat regex,
  the SSRF guard (including the cloud-metadata IP), clearance + all three
  need-to-know modes, config fail-safes, and synthetic DB population/search.
* ``tests/test_pipeline.py`` — full end-to-end pipeline tests that drive every
  branch (search, retrieve allow/deny, policy, general, rule-block, LLM-threat
  block, empty input, and all show-API paths) using a scripted ``FakeGemini``
  test double, so the entire request flow is exercised with no network or key.
  It also includes a suite of **live** Gemini tests (search, retrieve
  allow+deny, policy, general, and threat-block) that auto-skip unless
  ``GOOGLE_API_KEY`` is set. Run them with ``pytest -k live -v -s`` to prove
  the entire pipeline end-to-end against real Google Gemini.

Because ``config`` (pydantic) and ``database`` (SQLAlchemy) both degrade to
standard-library fallbacks when those packages are absent, the full pipeline
test runs even in a bare Python environment — the real packages are used
automatically when installed.

---

## Taking it to production (the short list)

1. Replace the role dropdown with real authentication; derive clearance from an
   authoritative source, not a dictionary.
2. Move data classification to a trusted pipeline; never let the data consumer
   assert its own label.
3. Put the external-API tool behind an egress proxy with allowlisting and
   request-time IP pinning (defeats DNS-rebinding).
4. Treat the regex filter as telemetry, not enforcement; rely on the access
   layer and on provider-side safety for real control.
5. Add authz audit logging, rate limiting, and a managed database.

## License

MIT.
