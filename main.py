import json
import logging

import requests
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from models import BuildingData, HealthResponse, ScheduleResponse
from llm import SYSTEM_PROMPT, build_schedule_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"


# --------------------------------------------------------------------------- #
# Startup — warm up the model so it's in VRAM before the first real request
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up model '%s' — this may take a minute on first run...", MODEL_NAME)
    try:
        requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": "Привет",
                "stream": False,
                "options": {"num_predict": 1},  # generate just 1 token — fast
            },
            timeout=300,
        )
        logger.info("Model is loaded and ready.")
    except Exception as exc:
        logger.warning("Could not pre-warm model: %s. First request may be slow.", exc)
    yield  # server runs here


app = FastAPI(
    title="Construction Schedule AI API",
    description="RAG-LLM service for automated calendar planning in construction projects",
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def call_ollama(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Send a prompt to the local Ollama instance and return the response text."""
    payload = {
        "model": MODEL_NAME,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 8192,      # increased from default 4096
            "temperature": 0.3,   # lower = more deterministic JSON output
        },
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot reach Ollama. Make sure it is running on localhost:11434.",
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Ollama took too long to respond. Try a shorter prompt or restart Ollama.",
        )
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code
        if status == 502:
            raise HTTPException(
                status_code=503,
                detail="Ollama is still loading the model into VRAM. Wait a few seconds and try again.",
            )
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an unexpected error: {status}.",
        )

def parse_llm_json(raw: str) -> dict:
    """Strip markdown fences if present, then parse JSON."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"LLM returned invalid JSON: {exc}. Raw response: {raw[:300]}",
        )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check whether the API and Ollama are reachable and the model is loaded."""
    try:
        # Hit /api/generate with 0 tokens — confirms model is actually in VRAM
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": "",
                "stream": False,
                "options": {"num_predict": 0},
            },
            timeout=10,
        )
        ollama_ok = resp.status_code == 200
    except requests.exceptions.RequestException:
        ollama_ok = False

    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        model=MODEL_NAME,
        ollama_reachable=ollama_ok,
    )


@app.post("/generate-schedule", response_model=ScheduleResponse)
def generate_schedule(building: BuildingData) -> ScheduleResponse:
    """
    Main endpoint. Accepts building data from the Renga plugin,
    calls the local LLM, and returns a structured construction schedule.
    """
    prompt = build_schedule_prompt(building)
    raw_response = call_ollama(prompt)
    data = parse_llm_json(raw_response)

    try:
        return ScheduleResponse(**data)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"LLM JSON did not match expected schema: {exc}",
        )