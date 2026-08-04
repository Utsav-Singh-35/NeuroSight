"""Report router for generating full AI diagnostic reports."""

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/report")
async def get_report(request: Request, image: UploadFile = File(...), module: str = "brain_mri"):
    """Generate a full AI diagnostic report for a medical image.

    Returns prediction, confidence, risk level, AI summary,
    clinical recommendation, and disclaimer.

    Args:
        request: FastAPI request object.
        image: Uploaded image file (must be JPEG or PNG).
        module: Module ID to use for classification (default: brain_mri).
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

    # Run report via module
    try:
        report = mod.report(file_bytes)
    except Exception:
        logger.exception("Report generation failed for module %s", module)
        return JSONResponse(status_code=500, content={"error": "Report generation failed"})

    # Add module info
    report["module"] = module

    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(
        "%s | module=%s | report | prediction=%s | confidence=%.2f%% | duration=%.2fms",
        datetime.now(timezone.utc).isoformat(),
        module,
        report.get("prediction", "unknown"),
        report.get("confidence", 0),
        duration_ms,
    )

    return report
