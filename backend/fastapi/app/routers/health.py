"""Health check router for the FastAPI ML service."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Check service health and model readiness.

    Returns 200 with healthy status if model is loaded,
    or 503 with unhealthy status if model is not loaded.
    """
    model = getattr(request.app.state, "model", None)
    if model is not None:
        return HealthResponse(status="healthy", model_loaded=True)
    return JSONResponse(
        status_code=503,
        content={"status": "unhealthy", "model_loaded": False},
    )
