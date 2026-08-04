"""Grad-CAM router for generating explainability heatmaps."""

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse

from app.models.schemas import GradCAMResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/gradcam",
    response_model=GradCAMResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Image could not be processed"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def gradcam(request: Request, image: UploadFile = File(...), module: str = "brain_mri"):
    """Generate Grad-CAM heatmap for a medical image.

    Accepts a JPEG or PNG image via multipart/form-data, runs inference
    through the module's pipeline, and generates an explainability heatmap.

    Args:
        request: FastAPI request object.
        image: Uploaded image file (must be JPEG or PNG).
        module: Module ID to use for classification (default: brain_mri).

    Returns:
        GradCAMResponse with heatmap, prediction, and confidence.
    """
    start_time = time.time()

    # Validate file content type
    if image.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(
            status_code=422,
            content={"error": "File must be JPEG or PNG format"},
        )

    # Read file bytes
    try:
        file_bytes = await image.read()
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "An internal error occurred"},
        )

    if not file_bytes:
        return JSONResponse(
            status_code=422,
            content={"error": "Uploaded file is empty"},
        )

    # Get module from registry
    from app.engine.registry import get_registry

    registry = get_registry()
    mod = registry.get(module)
    if mod is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown module: {module}"},
        )
    if not mod.is_available():
        return JSONResponse(
            status_code=503,
            content={"error": f"Module '{module}' is not available. Model weights may be missing."},
        )

    # Run Grad-CAM via module
    try:
        result = mod.gradcam(file_bytes)
    except Exception:
        logger.exception("Heatmap generation failed for module %s", module)
        return JSONResponse(
            status_code=500,
            content={"error": "Heatmap generation failed"},
        )

    # Log request details
    duration_ms = round((time.time() - start_time) * 1000, 2)
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.info(
        "%s | module=%s | filename=%s | prediction=%s | confidence=%.2f%% | duration=%.2fms",
        timestamp,
        module,
        image.filename,
        result["prediction"],
        result["confidence"],
        duration_ms,
    )

    return GradCAMResponse(
        heatmap=result["heatmap"],
        prediction=result["prediction"],
        confidence=result["confidence"],
        module=module,
    )
