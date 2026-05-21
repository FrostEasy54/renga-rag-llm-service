# Use PowerShell on Windows
set shell := ["powershell", "-Command"]

# Default recipe - shows available commands
default:
    @just --list

# Install dependencies
install:
    .venv\Scripts\Activate.ps1; pip install -r requirements.txt

# Run the API server
run:
    .venv\Scripts\Activate.ps1; $env:TRANSFORMERS_OFFLINE=1; uvicorn main:app --reload --port 8000

# Set up venv and install everything from scratch
setup:
    python -m venv .venv
    .venv\Scripts\Activate.ps1; pip install -r requirements.txt

# Check if Ollama and the API are alive
health:
    Invoke-RestMethod http://localhost:8000/health

# Run Ollama in the background (if not already running)
ollama:
    Start-Process ollama -ArgumentList "serve"

# Force re-index of knowledge_base/ into ChromaDB (run after adding/updating documents)
index:
    .venv\Scripts\Activate.ps1; python -c "from pathlib import Path; from rag.vector_store import VectorStore; import shutil; shutil.rmtree('chroma_db', ignore_errors=True); store = VectorStore(); count = store.index_documents(Path('knowledge_base')); print(f'Done — indexed {count} chunks.')"
