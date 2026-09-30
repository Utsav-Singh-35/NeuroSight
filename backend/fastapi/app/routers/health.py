"""Health check router for the FastAPI ML service."""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.engine.registry import get_registry, list_modules
from app.models.schemas import HealthResponse
from app.services import llm
from app.services.knowledge import get_knowledge_base
from app.services.model_cache import get_device, model_cache
from app.services.narratives import get_narrative_store
from app.routers._common import get_calibration, get_detector

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Report service readiness and the state of every downstream layer.

    ``model_loaded`` previously reflected only the brain MRI fallback model, so
    the endpoint could report unhealthy while the chest module was perfectly
    able to serve. Readiness is now per-module, and the top-level status is
    healthy when *any* module can answer.
    """
    registry = get_registry()
    modules = registry.available_modules()
    any_available = any(m["available"] for m in modules)

    bundle = get_calibration()
    detector = get_detector()
    knowledge = get_knowledge_base()
    narratives = get_narrative_store()

    payload = {
        "status": "healthy" if any_available else "unhealthy",
        # Retained for backwards compatibility with the Express gateway.
        "model_loaded": any_available,
        "device": str(get_device()),
        "available_modules": modules,
        "models_cached": model_cache.loaded_keys(),
        "calibration": {
            "available": bundle.available,
            "design": bundle.design,
            "temperature": bundle.temperature,
            "conformal_alpha": bundle.alpha,
            "conformal_q_hat": bundle.q_hat,
            "coverage_guarantee": (
                round(1.0 - bundle.alpha, 4) if bundle.q_hat is not None else None
            ),
        },
        "validation": {
            "available": detector.available,
            "primary_detector": detector.primary_detector,
            "threshold": detector.threshold,
            "quality_gate_enforced": request.app.state.enforce_quality_gate
            if hasattr(request.app.state, "enforce_quality_gate")
            else None,
        },
        "knowledge_base": knowledge.stats(),
        "narratives": narratives.stats(),
        "llm": llm.health(),
    }

    return JSONResponse(
        status_code=200 if any_available else 503,
        content=payload,
    )


@router.get("/health/live")
async def liveness():
    """Cheap liveness probe that never touches models or artefacts."""
    return {"status": "alive"}
