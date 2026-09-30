"""Predict router for medical image classification."""

import logging
import time

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.models.schemas import ErrorResponse, PredictionResponse
from app.routers._common import (
    build_ensemble_info,
    build_uncertainty_info,
    build_validation_info,
    log_inference,
    prepare,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        413: {"model": ErrorResponse, "description": "Image too large"},
        422: {"model": ErrorResponse, "description": "Image could not be processed"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Module unavailable"},
    },
)
async def predict(
    request: Request, image: UploadFile = File(...), module: str = "brain_mri"
):
    """Classify a medical image using the specified module.

    Returns the classification alongside the uncertainty, validation and
    ensemble-transparency payloads, so a caller can tell not just *what* the
    model decided but how much to trust it.

    Args:
        request: FastAPI request object.
        image: Uploaded image file (JPEG or PNG).
        module: Module ID to use (default: brain_mri).

    Returns:
        A :class:`PredictionResponse`.
    """
    started = time.time()

    context, error = await prepare(image, module)
    if error is not None:
        return error

    try:
        result = context.module.predict(context.file_bytes)
    except Exception:
        logger.exception("Inference failed for module %s", module)
        return JSONResponse(status_code=500, content={"error": "Inference failed"})

    duration_ms = round((time.time() - started) * 1000, 2)
    log_inference(context, result, duration_ms, "predict")

    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        module=module,
        uncertainty=build_uncertainty_info(context, result),
        validation=build_validation_info(context, result),
        ensemble=build_ensemble_info(result),
        duration_ms=duration_ms,
    )
