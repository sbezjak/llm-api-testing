# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FastAPI service that proxies questions to a **local** Ollama LLM, with an async pytest suite exercising the endpoint via `ASGITransport` (no live HTTP server needed for app-layer tests). Integration tests require Ollama; error-path tests use `respx` to mock it.

## Commands

```bash
# Activate venv
source .venv/bin/activate

# Fast suite - no Ollama needed (app-layer + respx-mocked tests)
pytest -m "not ollama"

# Full suite with HTML + coverage reports (requires `ollama serve` + `ollama pull llama3.2`)
pytest --html=reports/report.html --self-contained-html \
       --cov=app --cov-report=html:reports/htmlcov --cov-report=term-missing

# Run a single parametrized case
pytest "tests/test_api.py::test_moderation_harmful[harmful-bomb]"
```

Tests that require Ollama call `_needs_ollama(resp)` and `pytest.skip` gracefully when the local server at `localhost:11434` is unreachable - so the suite passes without Ollama, just with skips. These tests are also tagged `@pytest.mark.ollama` for explicit selection/deselection.

## Architecture

- `app/main.py` - single-file FastAPI app. Two endpoints: `POST /ask` and `GET /health`.
  - Request flow in `/ask`: empty-string guard (422) → keyword-based harmful-prompt guard (400, see `HARMFUL_KEYWORDS`) → async `httpx` POST to `OLLAMA_URL` (`http://localhost:11434/api/generate`, `stream: false`) → returns `AnswerResponse(answer, model, elapsed_seconds)`.
  - `httpx.ConnectError` is mapped to 503 with the exact string `"Cannot reach Ollama"` - test helpers key off this string, don't change it without updating `_needs_ollama`.
  - `MODEL` and `TIMEOUT_SECONDS` are module-level constants; to switch models edit `app/main.py`.
  - Each request emits a structured log line via `logger` (`app.main`): `verdict=<allowed|refused|...>` with status, elapsed, prompt/answer previews. pytest captures these into the HTML report.
- `tests/test_api.py` - async tests using `httpx.AsyncClient` with `ASGITransport(app=app)`. `pyproject.toml` sets `asyncio_mode = "auto"`, `log_level = "INFO"`, and registers the `ollama` / `mocked` markers. The `client` fixture is async; tests are marked `@pytest.mark.anyio`.
  - Moderation policy is expressed as two parametrized lists (`HARMFUL_CASES`, `BENIGN_CASES`). Known false-positives of the keyword guard are individually marked `xfail` with `strict=False` and a reason - do not remove them; they're regression tests for the day the classifier is swapped in.
- Harmful-prompt detection is a deliberate simple keyword list, not a classifier. Test prompts in `test_moderation_harmful` must match a substring in `HARMFUL_KEYWORDS`; benign prompts in `test_moderation_benign` that happen to contain a trigger substring are the xfail cases.
