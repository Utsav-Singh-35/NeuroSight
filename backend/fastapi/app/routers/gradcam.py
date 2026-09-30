"""Grad-CAM router for explainability heatmaps."""

import logging
import time

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.models.schemas import ErrorResponse, GradCAMResponse
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
    "/gradcam",
    response_model=GradCAMResponse,
    responses={
        413: {"model": ErrorResponse, "description": "Image too large"},
        422: {"model": ErrorResponse, "description": "Image could not be processed"},
        500: {"model": ErrorResponse, "description": "Heatmap generation failed"},
        503: {"model": ErrorResponse, "description": "Module unavailable"},
    },
)
async def gradcam(
    request: Request, image: UploadFile = File(...), module: str = "brain_mri"
):
    """Classify an image and return a Grad-CAM attention overlay.

    ``explanation_faithful`` is the field to check before showing the heatmap:
    when no base model agrees with the ensemble's class, the visualisation
    explains a *different* class than the one reported.

    The heatmap is a base64 PNG with no ``data:`` prefix; the client must add it.
    """
    started = time.time()

    context, error = await prepare(image, module)
    if error is not None:
        return error

    try:
        result = context.module.gradcam(context.file_bytes)
    except Exception:
        logger.exception("Heatmap generation failed for module %s", module)
        return JSONResponse(
            status_code=500, content={"error": "Heatmap generation failed"}
        )

    duration_ms = round((time.time() - started) * 1000, 2)
    log_inference(context, result, duration_ms, "gradcam")

    if result.get("explanation_faithful") is False:
        logger.warning(
            "Unfaithful explanation served for %s/%s: no base model agreed with "
            "the ensemble class.",
            module,
            context.filename,
        )

    return GradCAMResponse(
        heatmap=result["heatmap"],
        prediction=result["prediction"],
        confidence=result["confidence"],
        module=module,
        explanation_model=result.get("explanation_model"),
        explanation_faithful=result.get("explanation_faithful"),
        probabilities=result.get("probabilities"),
        uncertainty=build_uncertainty_info(context, result)
        if result.get("probabilities")
        else None,
        validation=build_validation_info(context, result),
        ensemble=build_ensemble_info(result),
    )
