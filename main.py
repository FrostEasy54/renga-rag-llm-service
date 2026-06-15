import json
import logging
from pathlib import Path
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager

from models import BuildingData, HealthResponse, ScheduleResponse, ScheduleTask
from llm import SYSTEM_PROMPT, build_schedule_prompt, build_bare_schedule_prompt
from rag.vector_store import VectorStore
from rag.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
KNOWLEDGE_BASE_DIR = Path("knowledge_base")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Предзагрузка модели '%s' — может занимать некоторое время при первом запуске...", MODEL_NAME)
    try:
        requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": "Привет",
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=300,
        )
        logger.info("Модель загружена и готова.")
    except Exception as exc:
        logger.warning("Не удалось предзагрузить модель: %s. Первый запрос может быть медленным.", exc)

    vector_store = VectorStore(persist_dir=Path("chroma_db"))

    if vector_store.is_empty():
        if KNOWLEDGE_BASE_DIR.is_dir():
            logger.info("Векторная база данных пуста — индексирую документы из '%s'...", KNOWLEDGE_BASE_DIR)
            count = vector_store.index_documents(KNOWLEDGE_BASE_DIR)
            logger.info("Проиндексировано %d чанков в ChromaDB.", count)
        else:
            logger.warning(
                "Директория knowledge_base/ не найдена. RAG-контекст будет отключён."
                "Создайте директорию и добавьте в неё документы (.pdf или .txt) для активации."
            )
    else:
        logger.info("Векторная база данных уже заполнена — повторная индексация не требуется.")
    app.state.retriever = Retriever(vector_store)

    yield 


app = FastAPI(
    title="Construction Schedule AI API",
    description="RAG-LLM service for automated calendar planning in construction projects",
    version="0.4.0",
    lifespan=lifespan,
)

def call_ollama(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    payload = {
        "model": MODEL_NAME,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "temperature": 0.3,
        },
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama недоступен. Убедитесь, что он запущен на localhost:11434.",
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Ollama не ответил вовремя. Попробуйте сократить промпт или перезапустить Ollama.",
        )
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code
        if status == 502:
            raise HTTPException(
                status_code=503,
                detail="Ollama ещё загружает модель в VRAM. Подождите несколько секунд и повторите запрос.",
            )
        raise HTTPException(
            status_code=502,
            detail=f"Ollama вернул неожиданную ошибку: {status}.",
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
            detail=f"Модель вернула некорректный JSON: {exc}. Ответ модели: {raw[:300]}",
        )

def calculate_critical_path_days(tasks: list[ScheduleTask]) -> int:
    if not tasks:
        return 0

    by_name = {task.task_name: task for task in tasks}
    earliest_finish: dict[str, int] = {}
    visiting: set[str] = set()

    def finish_of(name: str) -> int:
        if name in earliest_finish:
            return earliest_finish[name]
        if name in visiting:
            logger.warning("Обнаружен цикл в зависимостях задачи '%s' — игнорирую.", name)
            return 0
        task = by_name.get(name)
        if task is None:
            logger.warning("Задача '%s' указана как зависимость, но отсутствует в списке задач.", name)
            return 0

        visiting.add(name)
        max_dep_finish = 0
        for dep in task.depends_on:
            dep_finish = finish_of(dep)
            if dep_finish > max_dep_finish:
                max_dep_finish = dep_finish
        visiting.remove(name)

        result = task.duration_days + max_dep_finish
        earliest_finish[name] = result
        return result

    for task in tasks:
        finish_of(task.task_name)

    return max(earliest_finish.values())

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check whether the API and Ollama are reachable and the model is loaded."""
    try:
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
def generate_schedule(
    building: BuildingData,
    mode: Literal["bare", "structured", "rag"] = Query(
        default="rag",
        description=(
            "Режим генерации ():\n"
            "- bare: минимальный промпт без структурного знания о строительстве и без RAG;\n"
            "- structured: структурный промпт с поэтапной схемой и правилами, без RAG;\n"
            "- rag: структурный промпт + RAG-контекст из базы знаний (продакшен-режим, по умолчанию)."
        ),
    ),
) -> ScheduleResponse:
    if mode == "bare":
        rag_context = ""
        prompt = build_bare_schedule_prompt(building)
        logger.info(
            "Режим BARE для '%s': минимальный промпт без структурного знания и без RAG.",
            building.name,
        )
    elif mode == "structured":
        rag_context = ""
        prompt = build_schedule_prompt(building, rag_context="")
        logger.info(
            "Режим STRUCTURED для '%s': структурный промпт со схемой этапов, без RAG.",
            building.name,
        )
    else:
        rag_context = app.state.retriever.get_context(building)
        prompt = build_schedule_prompt(building, rag_context=rag_context)
        if rag_context:
            logger.info(
                "Режим RAG для '%s': структурный промпт + контекст из БЗ (%d символов).",
                building.name, len(rag_context),
            )
        else:
            logger.warning(
                "Режим RAG для '%s' запрошен, но контекст не найден — "
                "ответ будет идентичен режиму STRUCTURED.",
                building.name,
            )

    logger.info("Полный промпт, отправленный в LLM:\n%s\n%s", "─" * 60, prompt)

    raw_response = call_ollama(prompt)
    logger.info("Необработанный ответ LLM:\n%s\n%s", "─" * 60, raw_response)

    data = parse_llm_json(raw_response)

    try:
        schedule = ScheduleResponse(**data, rag_context_used=bool(rag_context))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"JSON от LLM не соответствует ожидаемой схеме: {exc}",
        )

    critical_path_days = calculate_critical_path_days(schedule.tasks)
    if critical_path_days != schedule.total_duration_days:
        logger.info(
            "Скорректирована общая продолжительность проекта: LLM указал %d дней, "
            "расчёт по критическому пути дал %d дней.",
            schedule.total_duration_days, critical_path_days,
        )
        schedule.total_duration_days = critical_path_days

    return schedule