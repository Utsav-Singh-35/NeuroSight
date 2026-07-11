"""Grad-CAM router for generating explainability heatmaps."""

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import GradCAMResponse, ErrorResponse
from app.services import preprocessor, inference

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
async def gradcam(request: Request, image: UploadFile = File(...)):
    """Generate Grad-CAM heatmap for a brain MRI image.

    Accepts a JPEG or PNG image via multipart/form-data, preprocesses it,
    runs inference, generates a Grad-CAM heatmap, and returns the result.

    Args:
        request: FastAPI request object (used to access app.state.model).
        image: Uploaded image file (must be JPEG or PNG).

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

    # Preprocess image
    try:
        tensor = preprocessor.preprocess_image(file_bytes)
    except ValueError as e:
        return JSONResponse(
            status_code=422,
            content={"error": str(e)},
        )

    # Run Grad-CAM
    try:
        model = request.app.state.model
        from app.services.gradcam import generate_gradcam

        result = generate_gradcam(model, tensor, file_bytes)
    except Exception:
        logger.exception("Heatmap generation failed")
        return JSONResponse(
            status_code=500,
            content={"error": "Heatmap generation failed"},
        )

    # Log request details
    duration_ms = round((time.time() - start_time) * 1000, 2)
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.info(
        "%s | filename=%s | prediction=%s | confidence=%.2f%% | duration=%.2fms",
        timestamp,
        image.filename,
        result["prediction"],
        result["confidence"],
        duration_ms,
    )

    return GradCAMResponse(**result)
