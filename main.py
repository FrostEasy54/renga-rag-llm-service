import json

import requests
from fastapi import FastAPI, HTTPException

from models import BuildingData, HealthResponse, ScheduleResponse

app = FastAPI(
    title="Construction Schedule AI API",
    description="RAG-LLM service for automated calendar planning in construction projects",
    version="0.1.0",
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
SYSTEM_PROMPT = """Ты — эксперт по календарному планированию строительных проектов.
Ты помогаешь составлять календарные планы на основе данных о здании.
Всегда отвечай только на русском языке.
Когда тебя просят составить календарный план, возвращай ответ строго в формате JSON.
"""


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
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
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


def build_schedule_prompt(building: BuildingData) -> str:
    """Convert BuildingData into a detailed Russian prompt for the LLM."""
    if building.elements:
        lines = [
            f"  - {el.count}x {el.type} ({el.material}, объём: {el.volume} м³)"
            for el in building.elements
        ]
        elements_text = "Элементы здания:\n" + "\n".join(lines)
    else:
        elements_text = "Элементы здания: данные не предоставлены."

    return f"""Составь календарный план строительства следующего объекта.

Название проекта: {building.name}
Тип здания: {building.building_type}
Количество этажей: {building.floors}
Общая площадь: {building.total_area} м²
{elements_text}
Дополнительные заметки: {building.notes or 'нет'}

Верни ответ ТОЛЬКО в формате JSON без дополнительного текста, строго по следующей схеме:
{{
  "project_name": "...",
  "total_duration_days": <число>,
  "tasks": [
    {{
      "task_name": "...",
      "duration_days": <число>,
      "depends_on": ["..."],
      "workers_required": <число>,
      "description": "..."
    }}
  ],
  "notes": "..."
}}

Задачи должны идти в логическом строительном порядке (фундамент → стены → перекрытия → кровля → отделка → инженерные сети).
"""


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
    """Check whether the API and Ollama are reachable."""
    try:
        requests.get("http://localhost:11434", timeout=3)
        ollama_ok = True
    except requests.exceptions.ConnectionError:
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