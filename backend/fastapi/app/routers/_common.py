"""Shared request handling for the inference routers.

All four inference endpoints previously repeated the same six-step guard
sequence, which is how they drifted: ``/report`` used a slightly different
error string, and only some of them would have picked up the new validation
layer. Centralising it means the quality gate, the OOD scoring and the error
semantics are identical everywhere by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from fastapi import UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.engine.base_module import BaseModule
from app.engine.registry import get_registry
from app.services.uncertainty import CalibrationBundle
from app.services.validation import OODDetector

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

# Loaded once at import; the app also stores these on app.state at startup.
_detector: OODDetector | None = None
_bundle: CalibrationBundle | None = None


def get_detector() -> OODDetector:
    """Process-wide OOD detector, loaded on first use."""
    global _detector
    if _detector is None:
        _detector = OODDetector.load(settings.CALIBRATION_DIR)
    return _detector


def get_calibration() -> CalibrationBundle:
    """Process-wide calibration bundle, loaded on first use."""
    global _bundle
    if _bundle is None:
        _bundle = CalibrationBundle.load(settings.CALIBRATION_DIR)
    return _bundle


@dataclass
class RequestContext:
    """Everything a handler needs after the guards have passed."""

    file_bytes: bytes
    module_id: str
    module: BaseModule
    filename: str
    quality: dict = field(default_factory=dict)


async def prepare(
    image: UploadFile, module: str
) -> tuple[RequestContext | None, JSONResponse | None]:
    """Run the shared guard sequence.

    Order matters: cheap checks first, module resolution before any decoding
    work, and the quality assessment last because it decodes the image.

    Args:
        image: Uploaded file.
        module: Requested module id.

    Returns:
        ``(context, None)`` on success, or ``(None, error_response)``.
    """
    # 1. Declared content type. Client-supplied and therefore spoofable, so it
    #    is a convenience check rather than a security boundary.
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        return None, JSONResponse(
            status_code=422, content={"error": "File must be JPEG or PNG format"}
        )

    # 2. Read.
    try:
        file_bytes = await image.read()
    except Exception:
        logger.exception("Could not read uploaded file")
        return None, JSONResponse(
            status_code=500, content={"error": "An internal error occurred"}
        )

    if not file_bytes:
        return None, JSONResponse(
            status_code=422, content={"error": "Uploaded file is empty"}
        )

    # 3. Upload size. The Express gateway caps this too, but a direct call to
    #    FastAPI would otherwise bypass that limit entirely.
    max_bytes = settings.MAX_UPLOAD_BYTES
    if max_bytes and len(file_bytes) > max_bytes:
        return None, JSONResponse(
            status_code=413,
            content={
                "error": f"Image exceeds the {max_bytes // (1024 * 1024)} MB limit"
            },
        )

    # 4. Module resolution.
    mod = get_registry().get(module)
    if mod is None:
        return None, JSONResponse(
            status_code=400, content={"error": f"Unknown module: {module}"}
        )
    if not mod.is_available():
        return None, JSONResponse(
            status_code=503,
            content={
                "error": f"Module '{module}' is not available. "
                "Model weights may be missing."
            },
        )

    # 5. Image quality. Always assessed and always reported. Whether a failure
    #    blocks the request is a policy decision: rejecting costs ~2% of valid
    #    in-distribution scans, so enforcement is opt-in via ENFORCE_QUALITY_GATE
    #    while the assessment itself is unconditional.
    quality = get_detector().check_quality(file_bytes)

    if not quality["passed"] and settings.ENFORCE_QUALITY_GATE:
        return None, JSONResponse(
            status_code=422,
            content={
                "error": "Image did not meet quality requirements for reliable analysis",
                "reasons": quality["reasons"],
                "validation": quality,
            },
        )

    if not quality["passed"]:
        logger.warning(
            "Quality checks failed for %s but the gate is not enforced: %s",
            image.filename,
            "; ".join(quality["reasons"]),
        )

    return (
        RequestContext(
            file_bytes=file_bytes,
            module_id=module,
            module=mod,
            filename=image.filename or "upload",
            quality=quality,
        ),
        None,
    )


def build_validation_info(context: RequestContext, result: dict) -> dict:
    """Assemble the validation payload, including OOD scoring.

    OOD scores are derived from model outputs, so this runs *after* inference.
    The best-performing detector (mean pairwise Jensen-Shannon divergence
    between base models) needs ``base_probabilities``, which the ensemble
    already produces.
    """
    detector = get_detector()
    payload: dict = {
        "validated": bool(context.quality.get("validated", detector.available)),
        "passed": bool(context.quality.get("passed", True)),
        "reasons": list(context.quality.get("reasons", [])),
        "checks": context.quality.get("checks"),
        "features": context.quality.get("features"),
        "enforced": settings.ENFORCE_QUALITY_GATE,
    }

    base_probs = result.get("base_probabilities") or {}
    meta_proba = result.get("raw_meta_proba")

    if not base_probs or not meta_proba:
        # Single-model path: the disagreement signal does not exist. Say so
        # rather than reporting a misleading zero.
        payload["scores"] = None
        payload["is_ood"] = None
        return payload

    try:
        scores = detector.score(
            np.asarray(meta_proba, dtype=np.float64),
            base_probs_row={
                key: np.asarray(value, dtype=np.float64)
                for key, value in base_probs.items()
            },
        )
    except Exception:  # noqa: BLE001 - novelty scoring must not fail a request
        logger.exception("OOD scoring failed")
        payload["scores"] = None
        payload["is_ood"] = None
        return payload

    payload["scores"] = {
        k: v for k, v in scores.items() if isinstance(v, (int, float))
    }
    payload["is_ood"] = scores.get("is_ood")
    payload["ood_detector"] = scores.get("ood_detector")
    payload["ood_threshold"] = scores.get("ood_threshold")
    payload["ood_reason"] = scores.get("ood_reason")
    if scores.get("ood_detector"):
        payload["ood_score"] = scores.get(scores["ood_detector"])

    return payload


def build_ensemble_info(result: dict) -> dict:
    """Assemble the per-model transparency payload."""
    return {
        "ensemble_active": bool(result.get("ensemble_active", False)),
        "models_used": list(result.get("models_used", [])),
        "models_skipped": list(result.get("models_skipped", [])),
        "agreeing_model": result.get("agreeing_model"),
        "base_predictions": result.get("base_predictions"),
    }


def build_uncertainty_info(context: RequestContext, result: dict) -> dict | None:
    """Calibrate and derive the conformal set, when the module supports it.

    Delegates to the module's own ``describe_uncertainty`` so per-module
    correctness is preserved: the brain artefacts must not be applied to the
    3-class chest output.
    """
    describe = getattr(context.module, "describe_uncertainty", None)
    if describe is None:
        return None
    try:
        return describe(result["probabilities"])
    except Exception:  # noqa: BLE001
        logger.exception("Uncertainty description failed for %s", context.module_id)
        return None


def log_inference(
    context: RequestContext, result: dict, duration_ms: float, endpoint: str
) -> None:
    """Structured success log line."""
    logger.info(
        "%s | module=%s | file=%s | prediction=%s | confidence=%.2f%% | "
        "skipped=%s | duration=%.1fms",
        endpoint,
        context.module_id,
        context.filename,
        result.get("prediction"),
        float(result.get("confidence", 0.0)),
        result.get("models_skipped") or "-",
        duration_ms,
    )
