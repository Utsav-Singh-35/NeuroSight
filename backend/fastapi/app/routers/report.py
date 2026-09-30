"""Report router: structured clinical report and PDF rendering."""

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from app.models.schemas import ErrorResponse, ReportResponse
from app.routers._common import (
    build_ensemble_info,
    build_validation_info,
    log_inference,
    prepare,
)
from app.services.pdf_report import build_report_pdf

logger = logging.getLogger(__name__)

router = APIRouter()


def _enrich(context, report: dict, result: dict) -> dict:
    """Attach validation and ensemble payloads to a report dict."""
    report["module"] = context.module_id
    report["filename"] = context.filename
    report["validation"] = build_validation_info(context, result)
    report["ensemble"] = build_ensemble_info(result)

    # Mirror the module's uncertainty fields into the shape the schema and the
    # PDF renderer both expect.
    report["uncertainty"] = {
        "calibrated": report.get("calibrated", False),
        "calibrated_probabilities": report.get("calibrated_probabilities"),
        "prediction_set": report.get("prediction_set"),
        "uncertainty_band": report.get("uncertainty_band"),
        "coverage_guarantee": report.get("coverage_guarantee"),
    }
    return report


async def _build(image: UploadFile, module: str, endpoint: str):
    """Shared path for both the JSON and PDF report endpoints.

    Returns ``(context, report, result, error_response)``.
    """
    started = time.time()

    context, error = await prepare(image, module)
    if error is not None:
        return None, None, None, error

    try:
        # predict() gives the raw ensemble detail the validation and ensemble
        # payloads need; report() re-runs prediction internally, so call the
        # module's report path and reuse its output for the narrative fields.
        result = context.module.predict(context.file_bytes)
        report = context.module.report(context.file_bytes)
    except Exception:
        logger.exception("Report generation failed for module %s", module)
        return (
            None,
            None,
            None,
            JSONResponse(
                status_code=500, content={"error": "Report generation failed"}
            ),
        )

    duration_ms = round((time.time() - started) * 1000, 2)
    log_inference(context, result, duration_ms, endpoint)

    return context, _enrich(context, report, result), result, None


@router.post(
    "/report",
    response_model=ReportResponse,
    responses={
        413: {"model": ErrorResponse, "description": "Image too large"},
        422: {"model": ErrorResponse, "description": "Image could not be processed"},
        500: {"model": ErrorResponse, "description": "Report generation failed"},
        503: {"model": ErrorResponse, "description": "Module unavailable"},
    },
)
async def report(
    request: Request, image: UploadFile = File(...), module: str = "brain_mri"
):
    """Generate the full structured clinical report.

    ``risk_escalated`` indicates the uncertainty layer overrode the class's
    default risk tier. ``evidence_grounded`` indicates whether the clinical text
    came from the cited knowledge base, and ``narrative_source`` whether it is
    human-reviewed prose or a deterministic template.
    """
    _, report_dict, _, error = await _build(image, module, "report")
    if error is not None:
        return error
    return report_dict


@router.post(
    "/report/pdf",
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "Rendered PDF report",
        },
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def report_pdf(
    request: Request,
    image: UploadFile = File(...),
    module: str = "brain_mri",
    patient_id: str | None = Query(default=None),
    age: str | None = Query(default=None),
    sex: str | None = Query(default=None),
    include_heatmap: bool = Query(default=True),
):
    """Render the report as a downloadable PDF.

    The Grad-CAM overlay is included by default. It costs an extra ensemble pass
    plus a backward pass, so ``include_heatmap=false`` is available when only the
    text is wanted.

    Patient fields are optional and used for the document header only; they are
    never sent anywhere outside this process.
    """
    context, report_dict, result, error = await _build(image, module, "report/pdf")
    if error is not None:
        return error

    heatmap = None
    if include_heatmap:
        try:
            cam = context.module.gradcam(context.file_bytes)
            heatmap = cam.get("heatmap")
            report_dict["explanation_faithful"] = cam.get("explanation_faithful")
            report_dict["explanation_model"] = cam.get("explanation_model")
        except Exception:  # noqa: BLE001 - a missing heatmap must not fail the PDF
            logger.exception("Heatmap unavailable for PDF; continuing without it")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    patient = {
        "report_id": f"NS-{stamp}",
        "patient_id": patient_id,
        "age": age,
        "sex": sex,
    }

    try:
        pdf_bytes = build_report_pdf(
            report_dict,
            heatmap_b64=heatmap,
            original_image_bytes=context.file_bytes,
            patient=patient,
        )
    except Exception:
        logger.exception("PDF rendering failed for module %s", module)
        return JSONResponse(
            status_code=500, content={"error": "PDF rendering failed"}
        )

    filename = f"neurasight-{module}-{stamp}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )
