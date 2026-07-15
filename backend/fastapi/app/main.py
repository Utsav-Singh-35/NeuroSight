"""FastAPI main entry point for the NeuraSight ML Service.

Configures the application lifespan (model loading at startup),
registers all routers, and adds a global exception handler that
never exposes stack traces to clients.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import health, predict, gradcam, report
from app.services.inference import load_model

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager.

    On startup: loads the PyTorch model into app.state.model.
    On shutdown: performs cleanup if needed.
    """
    # Startup
    try:
        model = load_model(settings.MODEL_PATH)
        app.state.model = model
        logger.info("Model loaded successfully from %s", settings.MODEL_PATH)
    except Exception as e:
        logger.error("Failed to load model from %s: %s", settings.MODEL_PATH, e)
        sys.exit(1)
    yield
    # Shutdown
    logger.info("Shutting down NeuraSight ML Service")


app = FastAPI(title="NeuraSight ML Service", lifespan=lifespan)

# Register routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(gradcam.router)
app.include_router(report.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and return a generic 500 error.

    Logs the full exception details for debugging but never exposes
    stack traces, file paths, or environment variables to the client.
    """
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "An internal error occurred"},
    )
