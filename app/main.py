"""
Main FastAPI application.
Replaces Django's wsgi.py and urls.py with FastAPI routing.
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import logging
from pathlib import Path

from app.config import settings
from app.routers import basic
from app.models import HealthResponse
from utils.llm import ollama_client

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI Job Agent - Automated job search and application system with LLM assistance",
    version="2.0.0",
    debug=settings.debug,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates (Jinja2 for HTML rendering)
templates = Jinja2Templates(directory=str(settings.templates_dir))

# Mount static files
if settings.static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")

# Include routers
app.include_router(basic.router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render the main page - replaces Django's index view.
    Serves the existing index.html template.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint - verify Ollama connection and model availability.
    """
    health_info = await ollama_client.check_health()

    return HealthResponse(
        status="healthy" if health_info.get("connected") else "unhealthy",
        ollama_connected=health_info.get("connected", False),
        model_available=health_info.get("model_available", False),
        version="2.0.0"
    )


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info(f"Starting {settings.app_name}...")
    logger.info(f"Ollama URL: {settings.ollama_base_url}")
    logger.info(f"Ollama Model: {settings.ollama_model}")

    # Check Ollama connection
    health = await ollama_client.check_health()
    if health.get("connected"):
        logger.info("✓ Ollama is connected")
        if health.get("model_available"):
            logger.info(f"✓ Model {settings.ollama_model} is available")
        else:
            logger.warning(f"⚠ Model {settings.ollama_model} not found. Run: ollama pull {settings.ollama_model}")
    else:
        logger.error("✗ Cannot connect to Ollama. Make sure it's running: ollama serve")

    # Ensure data directories exist
    settings.data_dir.mkdir(exist_ok=True)
    logger.info(f"Data directory: {settings.data_dir}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down application...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="debug" if settings.debug else "info"
    )
