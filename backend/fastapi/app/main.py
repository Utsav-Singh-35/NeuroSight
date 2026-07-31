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
from app.services.ensemble import load_ensemble

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager.

    On startup: loads the stacking ensemble (4 base models + meta-learner)
    into app.state.ensemble. Also exposes the EfficientNet-B0 base model as
    app.state.model for single-model fallback and Grad-CAM defaults.
    On shutdown: performs cleanup if needed.
    """
    # Startup
    app.state.ensemble = None
    app.state.model = None

    if settings.USE_ENSEMBLE:
        try:
            ensemble = load_ensemble(settings.MODELS_DIR)
            app.state.ensemble = ensemble
            # Reuse the ensemble's EfficientNet as the single-model handle
            app.state.model = ensemble["base_models"].get("efficientnet")
            logger.info(
                "Ensemble loaded from %s | models=%s | meta=%s",
                settings.MODELS_DIR,
                ensemble["order"],
                ensemble["config"].get("meta_learner"),
            )
        except Exception as e:
            logger.error("Failed to load ensemble from %s: %s", settings.MODELS_DIR, e)

    # Fall back to (or supplement with) the single model if needed
    if app.state.model is None:
        try:
            app.state.model = load_model(settings.MODEL_PATH)
            logger.info("Single model loaded from %s", settings.MODEL_PATH)
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
