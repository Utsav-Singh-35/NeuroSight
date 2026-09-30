"""Professional PDF generation for NeuraSight analysis reports.

Built on reportlab's platypus flowables so content paginates naturally rather
than being positioned at fixed coordinates.

Editorial stance
----------------
The PDF is the artefact a human is most likely to read detached from the UI, so
it carries *more* caveats than the JSON response, not fewer. Specifically it
always states:

* that the output is a classification, not a diagnosis;
* the conformal prediction set and its coverage guarantee, not just the top-1;
* that Grad-CAM shows model attention and is **not** tumour segmentation;
* when the explanation is not faithful (no base model agreed with the ensemble);
* when the ensemble ran degraded (``models_skipped``);
* when input validation flagged the image;
* whether the clinical text is human-reviewed or template-generated;
* the resolved source list for every citation shown.

Text handling
-------------
All strings are XML-escaped before reaching ``Paragraph`` (platypus interprets a
small markup dialect, so a stray ``&`` or ``<`` would raise) and coerced to
Latin-1-safe characters because the built-in Type 1 fonts do not cover arbitrary
Unicode. Narratives are already ASCII-normalised upstream; this is the safety
net for anything else.
"""

from __future__ import annotations

import base64
import html
import logging
from datetime import datetime, timezone
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# Palette kept sober: this is a clinical document, not a dashboard.
INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#6b7280")
RULE = colors.HexColor("#d1d5db")
ACCENT = colors.HexColor("#0e7490")
BAND_BG = colors.HexColor("#f3f4f6")

RISK_COLOURS = {
    "High": colors.HexColor("#b91c1c"),
    "Medium": colors.HexColor("#b45309"),
    "Low": colors.HexColor("#15803d"),
    "Indeterminate": colors.HexColor("#6d28d9"),
}

BAND_LABELS = {
    "confident": "Confident - single-class prediction set",
    "borderline": "Borderline - two classes remain plausible",
    "indeterminate": "Indeterminate - the model did not discriminate",
}

MODULE_TITLES = {
    "brain_mri": "Brain MRI AI Analysis Report",
    "chest_xray": "Chest X-ray AI Analysis Report",
}

GRADCAM_CAVEAT = (
    "The heatmap shows which image regions most influenced the model's output. "
    "It is an interpretability aid and must not be read as tumour segmentation, "
    "a boundary, or a measurement."
)


def _safe(text) -> str:
    """Escape for platypus markup and drop characters the base fonts lack."""
    if text is None:
        return ""
    value = str(text)
    value = value.encode("latin-1", "replace").decode("latin-1")
    return html.escape(value, quote=False)


def _styles() -> dict:
    """Build the paragraph styles used throughout the document."""
    base = getSampleStyleSheet()

    return {
        "brand": ParagraphStyle(
            "brand", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=20, leading=23, textColor=ACCENT, spaceAfter=1,
        ),
        "title": ParagraphStyle(
            "title", parent=base["Normal"], fontName="Helvetica",
            fontSize=12.5, leading=15, textColor=INK, spaceAfter=2,
        ),
        "meta": ParagraphStyle(
            "meta", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=11, textColor=MUTED,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=13, textColor=INK,
            spaceBefore=11, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.2, leading=13.2, textColor=INK,
            alignment=TA_JUSTIFY, spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontName="Helvetica",
            fontSize=7.8, leading=10.5, textColor=MUTED, spaceAfter=3,
        ),
        "finding": ParagraphStyle(
            "finding", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=17, leading=20, textColor=INK, spaceAfter=1,
        ),
        "caveat": ParagraphStyle(
            "caveat", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.4, leading=11.6, textColor=colors.HexColor("#7f1d1d"),
            alignment=TA_JUSTIFY, spaceAfter=3,
        ),
        "imgcap": ParagraphStyle(
            "imgcap", parent=base["Normal"], fontName="Helvetica",
            fontSize=7.4, leading=9.5, textColor=MUTED, alignment=TA_CENTER,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=8, leading=11, textColor=INK, alignment=TA_JUSTIFY,
        ),
    }


def _rule(space_before: float = 3, space_after: float = 6) -> HRFlowable:
    return HRFlowable(
        width="100%", thickness=0.6, color=RULE,
        spaceBefore=space_before, spaceAfter=space_after,
    )


def _prose_block(text: str, style) -> list:
    """Split plain text into paragraphs and bullet-ish lines."""
    flows = []
    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ")):
            flows.append(
                Paragraph(
                    f"&bull;&nbsp;&nbsp;{_safe(line[2:])}",
                    ParagraphStyle(
                        "bullet", parent=style, leftIndent=9, spaceAfter=2.5,
                        alignment=0,
                    ),
                )
            )
        else:
            flows.append(Paragraph(_safe(line), style))
    return flows


def _header(report: dict, styles: dict, patient: dict | None) -> list:
    """Brand block, document title and scan metadata."""
    module_id = report.get("module") or report.get("module_id") or "brain_mri"
    title = MODULE_TITLES.get(module_id, "AI Analysis Report")

    generated = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    patient = patient or {}

    rows = [
        ["Report ID", _safe(patient.get("report_id", "-"))],
        ["Patient reference", _safe(patient.get("patient_id", "Not supplied"))],
        ["Modality", _safe(module_id.replace("_", " ").title())],
        ["Source file", _safe(report.get("filename", "-"))],
        ["Generated", _safe(generated)],
    ]
    if patient.get("age"):
        rows.insert(2, ["Age", _safe(patient["age"])])
    if patient.get("sex"):
        rows.insert(3, ["Sex", _safe(patient["sex"])])

    table = Table(
        [[Paragraph(f"<b>{k}</b>", styles["meta"]), Paragraph(v, styles["meta"])]
         for k, v in rows],
        colWidths=[34 * mm, 58 * mm],
        hAlign="RIGHT",
    )
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))

    banner = Table(
        [[
            [Paragraph("NEURASIGHT", styles["brand"]),
             Paragraph(_safe(title), styles["title"]),
             Paragraph("Research decision-support output - not a diagnosis",
                       styles["meta"])],
            table,
        ]],
        colWidths=[90 * mm, 92 * mm],
    )
    banner.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))

    return [banner, _rule(6, 8)]


def _finding_block(report: dict, styles: dict) -> list:
    """Headline finding, confidence, risk and uncertainty band."""
    risk = report.get("risk_level", "-")
    risk_colour = RISK_COLOURS.get(risk, INK)
    band = report.get("uncertainty_band")
    confidence = report.get("confidence")

    left = [
        Paragraph("PRIMARY AI FINDING", styles["meta"]),
        Paragraph(_safe(report.get("prediction", "-")), styles["finding"]),
    ]
    if band:
        left.append(Paragraph(_safe(BAND_LABELS.get(band, band)), styles["small"]))

    conf_text = f"{confidence:.1f}%" if isinstance(confidence, (int, float)) else "-"
    if report.get("calibrated"):
        conf_text += " (calibrated)"

    right_rows = [
        ["Model confidence", conf_text],
        ["Risk level", risk],
    ]
    pset = report.get("prediction_set")
    if pset:
        right_rows.append(["Prediction set", ", ".join(pset)])
    coverage = report.get("coverage_guarantee")
    if coverage:
        right_rows.append(["Coverage guarantee", f"{float(coverage) * 100:.0f}%"])

    right = Table(
        [[Paragraph(f"<b>{_safe(k)}</b>", styles["meta"]),
          Paragraph(_safe(v), styles["meta"])] for k, v in right_rows],
        colWidths=[32 * mm, 52 * mm],
    )
    right.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("TEXTCOLOR", (1, 1), (1, 1), risk_colour),
        ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
    ]))

    block = Table([[left, right]], colWidths=[96 * mm, 86 * mm])
    block.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), BAND_BG),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
    ]))

    return [block, Spacer(1, 4)]


def _caveats_block(report: dict, styles: dict) -> list:
    """Surface every condition that weakens the result."""
    items: list[str] = []

    if report.get("risk_escalated"):
        items.append(
            "Risk level was escalated to Indeterminate because the model's "
            "uncertainty exceeded the reporting threshold. The class above should "
            "not be treated as the answer."
        )

    if report.get("explanation_faithful") is False:
        items.append(
            "No individual base model agreed with the ensemble's class, so the "
            "heatmap explains a different class than the one reported. Treat the "
            "visualisation as unreliable for this case."
        )

    skipped = report.get("models_skipped") or []
    if skipped:
        items.append(
            f"The ensemble ran degraded: {', '.join(skipped)} did not contribute. "
            "Zero vectors were substituted, so the confidence figure is less "
            "trustworthy than usual."
        )

    if report.get("ensemble_active") is False:
        items.append(
            "The stacking ensemble was unavailable; this result came from a "
            "single fallback model."
        )

    validation = report.get("validation") or {}
    if validation.get("passed") is False:
        reasons = "; ".join(validation.get("reasons", [])) or "quality checks failed"
        items.append(f"Input validation rejected this image: {reasons}")
    if validation.get("is_ood"):
        items.append(
            validation.get("ood_reason")
            or "The image does not resemble the training distribution."
        )

    if not items:
        return []

    flows = [Paragraph("IMPORTANT CAVEATS", styles["h2"])]
    for item in items:
        flows.append(Paragraph(f"&bull;&nbsp;&nbsp;{_safe(item)}", styles["caveat"]))

    boxed = Table([[flows]], colWidths=[182 * mm])
    boxed.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#fca5a5")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return [boxed, Spacer(1, 4)]


def _probability_table(report: dict, styles: dict) -> list:
    """Probability distribution with an inline proportional bar."""
    probs = report.get("calibrated_probabilities") or report.get("probabilities") or {}
    if not probs:
        return []

    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    predicted = report.get("prediction")
    pset = set(report.get("prediction_set") or [])

    header = ["Class", "Probability", "", "In set"]
    rows = [header]
    bar_width = 72.0

    for label, value in ranked:
        filled = max(0.6, float(value) / 100.0 * bar_width)
        rows.append([
            _safe(label),
            f"{float(value):.2f}%",
            "\u2588" * max(1, int(filled / 3.2)),
            "yes" if label in pset else ("-" if pset else ""),
        ])

    table = Table(rows, colWidths=[52 * mm, 24 * mm, 84 * mm, 18 * mm])
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("TEXTCOLOR", (2, 1), (2, -1), colors.HexColor("#a5b4fc")),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("ALIGN", (3, 0), (3, -1), "CENTER"),
    ]

    for index, (label, _) in enumerate(ranked, start=1):
        if label == predicted:
            style.append(("FONTNAME", (0, index), (1, index), "Helvetica-Bold"))
            style.append(("TEXTCOLOR", (2, index), (2, index), ACCENT))

    table.setStyle(TableStyle(style))

    return [
        Paragraph("PROBABILITY DISTRIBUTION", styles["h2"]),
        table,
        Paragraph(
            "Probabilities are the stacking meta-learner's posterior. "
            + ("Values are temperature-calibrated. " if report.get("calibrated") else "")
            + ("The 'In set' column marks classes retained by the conformal "
               "prediction set at the stated coverage level."
               if pset else ""),
            styles["small"],
        ),
    ]


def _image_row(original: bytes | None, heatmap_b64: str | None, styles: dict) -> list:
    """Original scan beside its Grad-CAM overlay."""
    cells = []
    captions = []
    side = 62 * mm

    # reportlab's Image flowable takes a path or a file-like object. Passing an
    # ImageReader raises "expected str, bytes or os.PathLike object", so hand it
    # the BytesIO directly.
    if original:
        try:
            cells.append(Image(BytesIO(original), width=side, height=side))
            captions.append("Original scan (resized)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not embed original image: %s", exc)

    if heatmap_b64:
        try:
            raw = base64.b64decode(heatmap_b64)
            cells.append(Image(BytesIO(raw), width=side, height=side))
            captions.append("Grad-CAM attention overlay")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not embed heatmap: %s", exc)

    if not cells:
        return []

    grid = Table(
        [cells, [Paragraph(_safe(c), styles["imgcap"]) for c in captions]],
        colWidths=[side + 6 * mm] * len(cells),
        hAlign="LEFT",
    )
    grid.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 1), (-1, 1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))

    return [
        Paragraph("EXPLAINABILITY", styles["h2"]),
        grid,
        Spacer(1, 3),
        Paragraph(_safe(GRADCAM_CAVEAT), styles["small"]),
    ]


def _clinical_sections(report: dict, styles: dict) -> list:
    """Narrative plus the structured clinical sections."""
    flows: list = []

    narrative = report.get("clinical_narrative")
    if narrative:
        flows.append(Paragraph("CLINICAL INTERPRETATION", styles["h2"]))
        flows.extend(_prose_block(narrative, styles["body"]))
        reviewer = report.get("narrative_reviewed_by")
        model = report.get("narrative_authored_by_model")
        flows.append(Paragraph(
            f"Narrative drafted with {_safe(model or 'an AI model')} from cited "
            f"sources and reviewed by {_safe(reviewer or 'a named reviewer')}.",
            styles["small"],
        ))
    else:
        flows.append(Paragraph("AI SUMMARY", styles["h2"]))
        flows.extend(_prose_block(report.get("ai_summary", ""), styles["body"]))
        flows.append(Paragraph(
            "Generated from deterministic templates. No AI-authored narrative "
            "has been approved for this finding and uncertainty band.",
            styles["small"],
        ))

    for key, heading in (
        ("description", "CONDITION OVERVIEW"),
        ("ai_limitations", "LIMITATIONS FOR THIS FINDING"),
        ("investigations", "POTENTIAL FURTHER EVALUATION"),
        ("clinical_considerations", "CLINICAL CONSIDERATIONS"),
        ("follow_up", "FOLLOW-UP CONSIDERATIONS"),
    ):
        text = report.get(key)
        if text:
            flows.append(Paragraph(heading, styles["h2"]))
            flows.extend(_prose_block(text, styles["body"]))

    warnings = report.get("warning_signs")
    if warnings:
        inner = [Paragraph("WARNING SIGNS - SEEK PROMPT MEDICAL ATTENTION", styles["h2"])]
        inner.extend(_prose_block(warnings, styles["body"]))
        boxed = Table([[inner]], colWidths=[182 * mm])
        boxed.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
            ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#fcd34d")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        flows.extend([KeepTogether(boxed), Spacer(1, 4)])

    treatment = report.get("treatment_information")
    if treatment:
        flows.append(Paragraph("TREATMENT INFORMATION (FOR CLINICIAN REVIEW)", styles["h2"]))
        flows.extend(_prose_block(treatment, styles["body"]))
        flows.append(Paragraph(
            "This is general reference information, not a prescription. No "
            "medicine, dose or schedule is recommended by this system.",
            styles["small"],
        ))

    # Jurisdictional context. Placed after the clinical sections and before the
    # recommendation, because it frames who should act on the result and under
    # what regulatory standing - which is exactly what a reader of a detached
    # PDF needs before acting on anything above.
    india = report.get("india_care_pathway")
    if india:
        flows.append(Paragraph("CARE PATHWAY AND REGULATORY STATUS (INDIA)", styles["h2"]))
        flows.extend(_prose_block(india, styles["body"]))

    recommendation = report.get("recommendation")
    if recommendation and recommendation != report.get("investigations"):
        flows.append(Paragraph("RECOMMENDED NEXT STEP", styles["h2"]))
        flows.extend(_prose_block(recommendation, styles["body"]))

    return flows


def _sources_block(report: dict, styles: dict) -> list:
    """Resolved citation list."""
    sources = report.get("sources") or []
    if not sources:
        if report.get("evidence_grounded") is False:
            return [
                Paragraph("EVIDENCE", styles["h2"]),
                Paragraph(
                    "No cited knowledge base has been authored for this module "
                    "yet, so the clinical text above is not source-linked.",
                    styles["small"],
                ),
            ]
        return []

    rows = [["#", "Source", "Publisher / reference"]]
    for index, src in enumerate(sources, start=1):
        if not src.get("resolved", True):
            rows.append([str(index), _safe(src.get("id")), "UNRESOLVED REFERENCE"])
            continue
        detail = src.get("publisher", "")
        if src.get("url"):
            detail = f"{detail} - {src['url']}" if detail else src["url"]
        rows.append([
            str(index),
            Paragraph(_safe(src.get("title", src.get("id"))), styles["small"]),
            Paragraph(_safe(detail), styles["small"]),
        ])

    table = Table(rows, colWidths=[7 * mm, 78 * mm, 97 * mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
    ]))

    flows = [Paragraph("EVIDENCE AND REFERENCES", styles["h2"]), table]

    if report.get("clinical_review_required"):
        flows.append(Spacer(1, 3))
        flows.append(Paragraph(
            "The clinical content in this report is pending verification against "
            "the primary sources by a qualified reviewer.",
            styles["small"],
        ))

    return flows


def _limitations_block(report: dict, styles: dict) -> list:
    """Closing disclaimer."""
    disclaimer = report.get("disclaimer") or (
        "This AI-generated report is for research and decision-support purposes "
        "only. It is not a diagnosis or a prescription and must not replace "
        "evaluation by a qualified healthcare professional."
    )
    boxed = Table(
        [[Paragraph(_safe(disclaimer), styles["disclaimer"])]],
        colWidths=[182 * mm],
    )
    boxed.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BAND_BG),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return [
        Paragraph("AI LIMITATIONS AND INTENDED USE", styles["h2"]),
        boxed,
    ]


def _decorate_page(canvas, doc) -> None:
    """Footer with page numbers and a standing disclaimer."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)

    y = 12 * mm
    canvas.line(14 * mm, y + 5 * mm, A4[0] - 14 * mm, y + 5 * mm)
    canvas.drawString(
        14 * mm, y,
        "NeuraSight - AI decision support. Not a diagnosis. Validate with a qualified clinician.",
    )
    canvas.drawRightString(A4[0] - 14 * mm, y, f"Page {doc.page}")
    canvas.restoreState()


def build_report_pdf(
    report: dict,
    heatmap_b64: str | None = None,
    original_image_bytes: bytes | None = None,
    patient: dict | None = None,
) -> bytes:
    """Render an analysis report as a PDF.

    Args:
        report: The report dict produced by
            :func:`app.services.report.generate_report`, optionally enriched
            with ``validation``, ``prediction_set``, ``uncertainty_band``,
            ``explanation_faithful`` and ``models_skipped``.
        heatmap_b64: Base64 PNG Grad-CAM overlay, if available.
        original_image_bytes: Raw uploaded image, for side-by-side display.
        patient: Optional non-identifying metadata (``report_id``,
            ``patient_id``, ``age``, ``sex``).

    Returns:
        The PDF as bytes, ready to stream.
    """
    buffer = BytesIO()
    styles = _styles()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=13 * mm,
        bottomMargin=20 * mm,
        title="NeuraSight AI Analysis Report",
        author="NeuraSight",
        subject=str(report.get("prediction", "")),
    )

    story: list = []
    story += _header(report, styles, patient)
    story += _finding_block(report, styles)
    story += _caveats_block(report, styles)
    story += _probability_table(report, styles)

    images = _image_row(original_image_bytes, heatmap_b64, styles)
    if images:
        story.append(Spacer(1, 4))
        story += images

    story.append(Spacer(1, 2))
    story += _clinical_sections(report, styles)

    sources = _sources_block(report, styles)
    if sources:
        story.append(Spacer(1, 3))
        story += sources

    story.append(Spacer(1, 5))
    story += _limitations_block(report, styles)

    doc.build(story, onFirstPage=_decorate_page, onLaterPages=_decorate_page)
    return buffer.getvalue()
