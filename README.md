# AI Job Agent - FastAPI

Automated job search and application system powered by AI agents, RAG, and local LLMs via Ollama.

## 📁 Project Structure

```
AI_Slop_FastAPI/
├── app/
│   ├── main.py              # FastAPI application & routes
│   ├── config.py            # Settings from .env
│   ├── models.py            # Pydantic schemas
│   └── routers/
│       └── basic.py         # Text prompt endpoints
├── utils/
│   └── llm.py              # Async Ollama client
├── templates/
│   └── index.html          # Frontend UI
├── docs/
│   └── plan.md            # Implementation roadmap
├── .env                   # Your config (not in git)
├── .env.example           # Template
├── requirements.txt       # Pip dependencies
└── environment.yml        # Conda/mamba environment
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama**: `ollama serve` → `ollama pull llama3.1:70b`
- **12GB+ VRAM** for llama3.1:70b (or use llama3:8b for smaller cards)

### Why Environment Isolation?

Virtual environments provide:

- **Reproducibility**: Same dependencies across machines
- **Isolation**: No conflicts with system Python or other projects
- **Version Control**: Explicit dependency tracking
- **Safety**: Project changes don't break system tools

This project supports both **venv** (standard Python) and **conda/mamba** (for those managing Python versions).

### Setup Option 1: venv (Recommended)

Best for most users. Uses your system Python 3.11+.

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows Git Bash/WSL)
source venv/Scripts/activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install
```

### Setup Option 2: conda/mamba

Best if you need to manage Python versions or prefer conda.

```bash
# Create environment from file
conda env create -f environment.yml

# Or with mamba (faster)
mamba env create -f environment.yml

# Activate
conda activate ai-job-agent

# Install Playwright browsers
playwright install
```

### Configure & Run

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (OLLAMA_BASE_URL, etc.)

# Run the application
uvicorn app.main:app --reload
```

Visit:

- **App**: <http://localhost:8000/>
- **API Docs**: <http://localhost:8000/docs>
- **Health**: <http://localhost:8000/health>

## 🎯 Current Features

- ✅ FastAPI with async/await
- ✅ Ollama integration (local LLM)
- ✅ Text prompt API endpoints
- ✅ Basic web UI
- ✅ Auto-generated OpenAPI docs
- ✅ Environment-based config
- ✅ Type-safe validation (Pydantic)

## 📡 API Endpoints

### `GET /`

Serves the web UI

### `POST /api/submit`

Submit text prompts to Ollama

```json
{
  "text": "Your prompt here"
}
```

### `GET /api/ask?query=your+question`

Quick query endpoint

### `GET /health`

Health check with Ollama connectivity

## 🔧 Configuration

Edit `.env`:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:70b
DEBUG=True
PORT=8000

# API Keys (for Phase 2)
ADZUNA_APP_ID=your_id
ADZUNA_APP_KEY=your_key
```

## � Next Steps (Phase 2 - See plan.md)

1. **RAG Pipeline** (`utils/rag.py`)
   - Document loaders (PDF, DOCX)
   - ChromaDB integration
   - Resume/JD retrieval

2. **Job Search Tools** (`tools/search_tools.py`)
   - Adzuna API
   - USAJobs API
   - Playwright scraping

3. **AI Agents** (`agents/`)
   - Search Agent: Find jobs
   - Tailor Agent: Customize resume
   - Apply Agent: Automate applications

## 🛠️ Development

### Add New Endpoint

1. Create models in `app/models.py`
2. Create router in `app/routers/my_router.py`
3. Register in `app/main.py`: `app.include_router(my_router.router)`

### Use Ollama Client

```python
from utils.llm import ollama_client

response = await ollama_client.query("Your prompt")
async for chunk in ollama_client.query_stream("Prompt"):
    print(chunk, end="")
```

## 📚 Troubleshooting

**Environment not activated**: You'll see `(venv)` or `(ai-job-agent)` in your prompt when activated

**Wrong Python version**:

```bash
python --version  # Should be 3.11+
# If not, recreate environment with correct Python
```

**Import errors**: Ensure environment is activated, then:

```bash
pip install -r requirements.txt --force-reinstall
```

**Playwright browser missing**:

```bash
playwright install
```

**Ollama not responding**: `ollama serve` then verify with `curl http://localhost:11434/api/tags`

**Port in use**: Change `PORT` in `.env`

## � Environment Management

### Deactivating

```bash
# venv
deactivate

# conda
conda deactivate
```

### Updating Dependencies

```bash
# venv: After updating requirements.txt
pip install -r requirements.txt --upgrade

# conda: After updating environment.yml
conda env update -f environment.yml --prune
```

### Removing Environment

```bash
# venv: Just delete the folder
rm -rf venv

# conda
conda env remove -n ai-job-agent
```

### When to Use Which?

**Use venv if:**

- You already have Python 3.11+ installed
- You want standard Python tooling
- You're deploying to cloud/containers

**Use conda/mamba if:**

- You need to manage Python versions
- You're on Windows without Python installed
- You prefer conda's dependency resolver

## �📚 Resources

- **FastAPI**: <https://fastapi.tiangolo.com/>
- **Ollama**: <https://ollama.com/>
- **LangChain**: <https://python.langchain.com/>
- **Implementation Plan**: See `docs/plan.md`
