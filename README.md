# llm-api-testing

> Part of a [5-project AI/QA testing portfolio](https://github.com/sbezjak/sbezjak) - all projects and write-ups.

A FastAPI service that proxies questions to a local LLM (Ollama), paired with a
pytest suite (23 tests, 100% coverage) that demonstrates how to test a
non-deterministic system without relying on a live model.

**Live reports:** [test report](https://sbezjak.github.io/llm-api-testing/reports/report.html) · [coverage](https://sbezjak.github.io/llm-api-testing/reports/htmlcov/index.html)

## What it teaches

- Testing a **non-deterministic** system with threshold assertions instead of
  exact-match equality
- Using **`respx`** to mock outgoing `httpx` calls so error paths run without
  a live dependency
- Exercising a FastAPI app in-process via **`ASGITransport`** - no live HTTP
  server, no port juggling
- Structuring pytest markers so CI can run a **fast, hermetic** slice
  (`-m "not ollama"`) and a **full integration** slice on demand
- Tracking known false-positives with **`xfail(strict=False)`** so they flip
  to passing the day the classifier is upgraded

## What it tests

- **Input validation** - empty question, missing field, very long prompt
- **Moderation policy** - harmful prompts refused, benign prompts allowed,
  known false-positives tracked with `xfail`
- **Consistency** - same prompt 10×, ≥70% of answers must contain the
  expected token (threshold assertion for a non-deterministic model)
- **Latency** - response under 30s
- **Concurrency** - 5 parallel requests, no 500s
- **Error paths** - Ollama unreachable / 5xx / empty response, mocked with
  `respx` so they run without a live LLM

## Run it

Requires Python 3.10+.

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fast suite - no Ollama needed, ~1 second
pytest -m "not ollama"
```

For the full suite (integration tests against a real LLM), start Ollama
**in a separate terminal tab**:

```bash
ollama serve
```

Then, in this terminal:

```bash
ollama pull llama3.2
pytest --html=reports/report.html --self-contained-html \
       --cov=app --cov-report=html:reports/htmlcov \
       --cov-report=term-missing --durations=10
```

Sample reports are committed and hosted live on GitHub Pages so you can preview without running:

- [Test report](https://sbezjak.github.io/llm-api-testing/reports/report.html) - pytest-html with per-test logs
- [Coverage report](https://sbezjak.github.io/llm-api-testing/reports/htmlcov/index.html) - line-level coverage

Open them after a run:

```bash
open reports/report.html          # pytest-html report
open reports/htmlcov/index.html   # coverage report
```

(Linux: `xdg-open`; Windows: `start`.)

## Run the API locally

Start the FastAPI server with uvicorn (requires Ollama running for `/ask`):

```bash
uvicorn app.main:app --reload
```

Then visit:

- http://localhost:8000/health - health check
- http://localhost:8000/docs - interactive Swagger UI (try `/ask` from the browser)
- http://localhost:8000/redoc - alternative API docs

### Hardened mode (red-team target for project 3)

This flag exists only as a punching bag: a sibling project,
[llm-red](https://github.com/sbezjak/llm-red), attacks this API and needs a
version with guardrails to show that defenses reduce but don't eliminate bypasses.

Set `HARDENED=1` to layer three minimal, deliberately-imperfect defenses on top
of the as-is app: an input filter with normalization (L1), a system prompt
carrying a secret canary (L2), and a rule-based output filter (L3). Off by
default, so plain behaviour is unchanged. The red-team suite
(https://github.com/sbezjak/llm-red) expects it on `:8001` while the plain
target stays on `:8000`, so both can run side by side:

```bash
HARDENED=1 uvicorn app.main:app --port 8001
```

The canary defaults to `CANARY-7f3a9c2e1b8d`; override both repos in lockstep
with `LLMRED_CANARY` so the planted secret and the leak detector can't drift.

## Test markers

| Marker | Runtime | Meaning |
|---|---|---|
| `ollama` | 1–60 s | Requires a real Ollama at `localhost:11434` |
| `mocked` | <50 ms | Uses `respx` to mock Ollama (error-path tests) |

Select with `pytest -m "ollama"`, `pytest -m mocked`, or `pytest -m "not ollama"`.

## Project layout

```
llm-api-testing/
├── app/
│   └── main.py           # FastAPI app + Ollama client + structured logging
├── tests/
│   └── test_api.py       # 23 tests, sectioned by pattern (app / mocked / integration)
├── reports/              # sample HTML reports, committed for preview
│   ├── report.html       #   pytest-html report
│   └── htmlcov/          #   coverage-html report
├── pyproject.toml        # pytest config + marker registration
├── requirements.txt      # runtime + test dependencies
└── README.md
```

## Tech

FastAPI · httpx · pytest · pytest-asyncio · pytest-html · pytest-cov · respx · Ollama (llama3.2)
