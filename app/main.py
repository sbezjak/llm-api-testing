"""
FastAPI endpoint that proxies questions to a local Ollama LLM.

Prerequisites (macOS):
    brew install ollama
    ollama serve          # starts the local server on :11434
    ollama pull llama3.2  # downloads a small, fast model (~2 GB)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import logging
import time

logger = logging.getLogger("app.main")

app = FastAPI(title="Local LLM API")


def _preview(s: str, n: int = 60) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"
TIMEOUT_SECONDS = 120  # generous for first-run cold starts

# Simple keyword list for the "refuse harmful prompts" requirement.
# A real system would use a classifier or moderation model.
HARMFUL_KEYWORDS = [
    "how to make a bomb",
    "how to hack",
    "how to kill",
    "create malware",
    "synthesize drugs",
    "build a weapon",
]


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    model: str
    elapsed_seconds: float


def _is_harmful(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in HARMFUL_KEYWORDS)


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    preview = _preview(req.question)

    if not req.question or not req.question.strip():
        logger.info("POST /ask prompt=%r verdict=empty status=422", preview)
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    if _is_harmful(req.question):
        logger.warning("POST /ask prompt=%r verdict=refused status=400", preview)
        raise HTTPException(status_code=400, detail="This prompt has been refused.")

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            resp = await client.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": req.question, "stream": False},
            )
            resp.raise_for_status()
    except httpx.ConnectError:
        logger.error("POST /ask prompt=%r verdict=ollama_unreachable status=503", preview)
        raise HTTPException(
            status_code=503,
            detail="Cannot reach Ollama. Is `ollama serve` running?",
        )
    except httpx.HTTPStatusError as exc:
        logger.error("POST /ask prompt=%r verdict=ollama_error status=502 body=%r",
                     preview, exc.response.text[:120])
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc.response.text}")

    elapsed = time.perf_counter() - start
    data = resp.json()
    answer_text = data.get("response", "").strip()

    if not answer_text:
        logger.error("POST /ask prompt=%r verdict=empty_answer status=502 elapsed=%.3fs",
                     preview, elapsed)
        raise HTTPException(status_code=502, detail="Ollama returned an empty response.")

    logger.info(
        "POST /ask prompt=%r verdict=allowed status=200 elapsed=%.3fs answer=%r",
        preview, elapsed, _preview(answer_text, 80),
    )
    return AnswerResponse(answer=answer_text, model=MODEL, elapsed_seconds=round(elapsed, 3))


@app.get("/health")
async def health():
    """Quick liveness check."""
    return {"status": "ok"}
