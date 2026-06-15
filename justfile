set shell := ["powershell", "-Command"]

default:
    @just --list

install:
    .venv\Scripts\Activate.ps1; pip install -r requirements.txt

run:
    .venv\Scripts\Activate.ps1; $env:TRANSFORMERS_OFFLINE=1; uvicorn main:app --reload --port 8000

setup:
    python -m venv .venv
    .venv\Scripts\Activate.ps1; pip install -r requirements.txt

health:
    Invoke-RestMethod http://localhost:8000/health

ollama:
    Start-Process ollama -ArgumentList "serve"

index:
    .venv\Scripts\Activate.ps1; python -c "from pathlib import Path; from rag.vector_store import VectorStore; import shutil; shutil.rmtree('chroma_db', ignore_errors=True); store = VectorStore(); count = store.index_documents(Path('knowledge_base')); print(f'Done — indexed {count} chunks.')"
