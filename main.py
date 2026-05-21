import json
import logging
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager

from models import BuildingData, HealthResponse, ScheduleResponse
from llm import SYSTEM_PROMPT, build_schedule_prompt
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


# --------------------------------------------------------------------------- #
# Startup — warm up the model and initialise the RAG pipeline
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Warm up the LLM
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

    # 2. Initialise ChromaDB vector store
    vector_store = VectorStore(persist_dir=Path("chroma_db"))

    # Index documents only when the collection is empty (i.e. first run).
    # On subsequent starts the persisted DB is reused — no re-indexing needed.
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

    # 3. Create retriever and attach to app state so endpoints can access it
    app.state.retriever = Retriever(vector_store)

    yield  # server runs here


app = FastAPI(
    title="Construction Schedule AI API",
    description="RAG-LLM service for automated calendar planning in construction projects",
    version="0.3.0",
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


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

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
    rag: bool = Query(
        default=True,
        description=(
            "Использовать ли RAG-контекст из базы знаний. "
            "Передайте rag=false для запуска эксперимента без RAG "
            "(см. главу 2.2 диплома)."
        ),
    ),
) -> ScheduleResponse:
    """
    Main endpoint. Accepts building data from the Renga plugin,
    optionally retrieves relevant construction norm context via RAG
    (controlled by the `rag` query parameter), calls the local LLM,
    and returns a structured construction schedule.
    """
    if rag:
        rag_context = app.state.retriever.get_context(building)
        if rag_context:
            logger.info(
                "RAG включён, контекст получен (%d символов) для объекта '%s'.",
                len(rag_context), building.name,
            )
        else:
            logger.warning(
                "RAG включён, но контекст не найден — план будет сформирован "
                "только на основе знаний модели."
            )
    else:
        rag_context = ""
        logger.info(
            "RAG отключён параметром запроса (rag=false) для объекта '%s' — "
            "план будет сформирован без обращения к базе знаний.",
            building.name,
        )

    prompt = build_schedule_prompt(building, rag_context=rag_context)
    logger.info("Полный промпт, отправленный в LLM:\n%s\n%s", "─" * 60, prompt)

    raw_response = call_ollama(prompt)
    logger.info("Необработанный ответ LLM:\n%s\n%s", "─" * 60, raw_response)

    data = parse_llm_json(raw_response)

    try:
        return ScheduleResponse(**data, rag_context_used=bool(rag_context))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"JSON от LLM не соответствует ожидаемой схеме: {exc}",
        )