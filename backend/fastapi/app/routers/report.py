"""Report router for generating full AI diagnostic reports."""

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse

from app.config import settings
from app.services import preprocessor, inference
from app.services.report import generate_report

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/report")
async def get_report(request: Request, image: UploadFile = File(...)):
    """Generate a full AI diagnostic report for a brain MRI image.

    Returns prediction, confidence, risk level, AI summary,
    clinical recommendation, and disclaimer.
    """
    start_time = time.time()

    # Validate
    if image.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(status_code=422, content={"error": "File must be JPEG or PNG"})

    try:
        file_bytes = await image.read()
    except Exception:
        return JSONResponse(status_code=500, content={"error": "An internal error occurred"})

    if not file_bytes:
        return JSONResponse(status_code=422, content={"error": "Uploaded file is empty"})

    # Preprocess
    try:
        tensor = preprocessor.preprocess_image(file_bytes)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

    # Inference
    try:
        model = request.app.state.model
        result = inference.run_inference(model, tensor, settings.class_labels)
    except Exception:
        logger.exception("Inference failed")
        return JSONResponse(status_code=500, content={"error": "Inference failed"})

    # Generate report
    report = generate_report(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )

    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(
        "%s | report | prediction=%s | confidence=%.2f%% | duration=%.2fms",
        datetime.now(timezone.utc).isoformat(),
        result["prediction"],
        result["confidence"],
        duration_ms,
    )

    return report
