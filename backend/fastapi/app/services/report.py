"""Clinical report generator.

Two behaviours here are deliberate corrections of earlier defects.

**Unknown labels no longer fall back to reassurance.** The previous
implementation did ``TUMOR_INFO.get(prediction, TUMOR_INFO["No Tumor"])``, so any
unrecognised label -- a renamed class, a new module -- silently rendered a "no
abnormality detected" narrative. That is the worst possible direction for a
default. Unknown labels now produce an explicit indeterminate report.

**Risk level is coupled to uncertainty.** Risk used to be a static per-class
attribute, which produced a real observed failure: a tuberculosis scan
classified as Normal at 56.51% confidence generated ``risk_level: "Low"`` and
"No immediate intervention required". Risk is now escalated to
``Indeterminate`` whenever the conformal layer reports a non-singleton
prediction set, so a coin-flip can no longer be presented as low risk.

Brain MRI content comes from the cited knowledge base under ``knowledge/``.
Chest X-ray still uses the legacy inline dictionary, because no cited knowledge
base has been authored for it yet -- that is reported via ``evidence_grounded``
rather than left ambiguous.
"""

from __future__ import annotations

import logging

from app.services.knowledge import get_knowledge_base
from app.services.narratives import get_narrative_store

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This AI-generated report is for clinical decision support only. "
    "It must not replace professional medical diagnosis. All findings should be "
    "validated by a qualified radiologist or specialist physician."
)

# Confidence below which the narrative adds an explicit uncertainty note, used
# when no conformal prediction set is supplied.
LOW_CONFIDENCE_THRESHOLD = 90.0

# Legacy inline content. Retained only for modules without a cited knowledge
# base (currently chest_xray). Brain MRI is served from knowledge/brain_mri/.
TUMOR_INFO = {
    "Normal": {
        "risk": "Low",
        "description": "No significant pulmonary abnormalities detected in the chest X-ray.",
        "characteristics": "The chest X-ray appears within normal limits. Clear lung fields with no "
                          "consolidation, infiltrates, or pleural effusion identified by the AI model.",
        "recommendation": "No immediate intervention required based on AI analysis. If respiratory symptoms "
                         "persist, consult a pulmonologist for further evaluation. This AI result does not "
                         "replace a formal radiological report.",
    },
    "Pneumonia": {
        "risk": "High",
        "description": "Pneumonia is an infection that inflames the air sacs in one or both lungs, "
                       "which may fill with fluid or pus causing cough, fever, and difficulty breathing.",
        "characteristics": "The chest X-ray demonstrates imaging features consistent with pneumonia, "
                          "showing areas of consolidation or ground-glass opacities in the lung fields.",
        "recommendation": "Consultation with a pulmonologist or infectious disease specialist is advised. "
                         "Further workup including blood tests, sputum culture, and possibly CT scan "
                         "may be warranted for pathogen identification and treatment planning.",
    },
    "Tuberculosis": {
        "risk": "High",
        "description": "Tuberculosis (TB) is a serious infectious disease caused by Mycobacterium tuberculosis "
                       "that primarily affects the lungs but can spread to other organs.",
        "characteristics": "The chest X-ray shows features consistent with tuberculosis, which may include "
                          "upper lobe infiltrates, cavitary lesions, or hilar lymphadenopathy.",
        "recommendation": "Referral to an infectious disease specialist is recommended. Confirmatory testing "
                         "including sputum AFB smear, culture, and molecular testing should be performed. "
                         "Management decisions rest with a qualified clinician.",
    },
}

_REFERRAL_RECOMMENDATION = (
    "The AI result is inconclusive and should not be acted upon as a finding. "
    "Refer this scan for review by a qualified radiologist or specialist "
    "physician, who can interpret it alongside the clinical history."
)


def _secondary_consideration(prediction: str, probabilities: dict) -> str:
    """Return a sentence naming the runner-up class, or an empty string."""
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    for label, value in ranked:
        if label != prediction and value > 1.0:
            return (
                f" A secondary consideration of {label} ({value:.1f}%) was noted "
                "but scored lower."
            )
    return ""


def _resolve_risk(base_tier: str, uncertainty: dict | None, confidence: float) -> tuple[str, bool]:
    """Combine the class risk tier with uncertainty into a reported risk level.

    A static per-class tier is not safe on its own: it lets a near-tie between
    two classes inherit the confident narrative of whichever one won.

    Returns:
        ``(risk_level, escalated)``.
    """
    band = (uncertainty or {}).get("uncertainty_band")

    if band in {"borderline", "indeterminate"}:
        return "Indeterminate", True

    # No conformal set available: fall back to a plain confidence threshold so
    # the protection still applies.
    if band is None and confidence < LOW_CONFIDENCE_THRESHOLD:
        return "Indeterminate", True

    return base_tier, False


def _unknown_report(prediction: str, confidence: float, probabilities: dict, module_id: str) -> dict:
    """Report for a label with no knowledge-base or legacy entry.

    Deliberately withholds clinical guidance instead of borrowing another
    class's text.
    """
    logger.error(
        "No knowledge entry for label '%s' in module '%s'. Emitting an "
        "indeterminate report rather than substituting another class.",
        prediction,
        module_id,
    )
    return {
        "prediction": prediction,
        "confidence": confidence,
        "risk_level": "Indeterminate",
        "risk_escalated": True,
        "description": (
            f"The model returned the label '{prediction}', for which this system "
            "holds no reviewed clinical information."
        ),
        "ai_summary": (
            f"The model classified this scan as '{prediction}' with "
            f"{confidence:.1f}% confidence, but no reviewed clinical content is "
            "available for that label, so no interpretation is offered."
        ),
        "recommendation": _REFERRAL_RECOMMENDATION,
        "warning_signs": None,
        "investigations": None,
        "follow_up": None,
        "treatment_information": None,
        "india_care_pathway": None,
        "sources": [],
        "evidence_grounded": False,
        "clinical_review_required": True,
        "content_source": "none",
        "disclaimer": DISCLAIMER,
        "probabilities": probabilities,
    }


def generate_report(
    prediction: str,
    confidence: float,
    probabilities: dict,
    module_id: str = "brain_mri",
    uncertainty: dict | None = None,
) -> dict:
    """Build the clinical report payload for one prediction.

    Args:
        prediction: Predicted class label.
        confidence: Confidence percentage (0-100).
        probabilities: Per-class probability percentages.
        module_id: Owning module, used to select knowledge-base content.
        uncertainty: Optional payload from
            :meth:`app.services.uncertainty.CalibrationBundle.describe`,
            supplying ``uncertainty_band`` and ``prediction_set``.

    Returns:
        Report dict. ``evidence_grounded`` indicates whether the clinical text
        came from the cited knowledge base; ``sources`` carries the resolved
        citations when it did.
    """
    knowledge = get_knowledge_base()
    entry = knowledge.get(module_id, prediction)

    if entry is not None:
        risk_level, escalated = _resolve_risk(entry.risk_tier, uncertainty, confidence)
        summary = (
            f"The model classified this scan as '{prediction}' with "
            f"{confidence:.1f}% confidence."
            f"{_secondary_consideration(prediction, probabilities)}"
        )

        # Attach the human-reviewed narrative for this class and uncertainty
        # band, if one exists. The store withholds unreviewed text, so a missing
        # narrative degrades to template prose rather than showing unvetted
        # clinical claims. No LLM is called here.
        band = (uncertainty or {}).get("uncertainty_band")
        narrative = None
        if band:
            narrative = get_narrative_store().get(module_id, prediction, band)

        report = {
            "prediction": prediction,
            "confidence": confidence,
            "risk_level": risk_level,
            "risk_escalated": escalated,
            "description": entry.plain("overview"),
            "ai_summary": summary,
            "ai_limitations": entry.plain("ai_limitations") or entry.plain("interpretation"),
            "recommendation": (
                _REFERRAL_RECOMMENDATION if escalated else entry.plain("investigations")
            ),
            "investigations": entry.plain("investigations"),
            "clinical_considerations": entry.plain("clinical_considerations"),
            "follow_up": entry.plain("follow_up"),
            "warning_signs": entry.plain("warning_signs"),
            "treatment_information": entry.plain("treatment_information"),
            # Jurisdictional context: Indian care pathway, resource setting and
            # regulatory status. Separate from the clinical fields so the report
            # stays portable if another jurisdiction is added later.
            "india_care_pathway": entry.plain("india_care_pathway"),
            "sources": knowledge.resolve_sources(entry.all_citations()),
            "evidence_grounded": True,
            "clinical_review_required": entry.requires_clinical_review,
            "content_source": "knowledge_base",
            "disclaimer": DISCLAIMER,
            "probabilities": probabilities,
        }

        if uncertainty:
            report["uncertainty_band"] = uncertainty.get("uncertainty_band")
            report["prediction_set"] = uncertainty.get("prediction_set")
            report["coverage_guarantee"] = uncertainty.get("coverage_guarantee")

        if narrative is not None:
            report["clinical_narrative"] = narrative.text
            report["narrative_reviewed_by"] = narrative.reviewed_by
            report["narrative_authored_by_model"] = narrative.model
            report["narrative_source"] = "reviewed_llm_narrative"
            # Narrative citations join the entry's own, so the report's source
            # list covers everything actually shown to the reader.
            merged = sorted(set(entry.all_citations()) | set(narrative.citations))
            report["sources"] = knowledge.resolve_sources(merged)
        else:
            report["clinical_narrative"] = None
            report["narrative_source"] = "template"

        return report

    # Legacy path for modules without an authored knowledge base.
    info = TUMOR_INFO.get(prediction)
    if info is None:
        return _unknown_report(prediction, confidence, probabilities, module_id)

    risk_level, escalated = _resolve_risk(info["risk"], uncertainty, confidence)
    summary = (
        f"{info['characteristics']} The model classified this scan as "
        f"'{prediction}' with {confidence:.1f}% confidence."
        f"{_secondary_consideration(prediction, probabilities)}"
    )

    report = {
        "prediction": prediction,
        "confidence": confidence,
        "risk_level": risk_level,
        "risk_escalated": escalated,
        "description": info["description"],
        "ai_summary": summary,
        "recommendation": (
            _REFERRAL_RECOMMENDATION if escalated else info["recommendation"]
        ),
        "warning_signs": None,
        "investigations": None,
        "follow_up": None,
        "treatment_information": None,
        # Chest has no knowledge-base entry, so no jurisdictional section either.
        "india_care_pathway": None,
        "sources": [],
        # Stated explicitly: this module's text is not yet cited.
        "evidence_grounded": False,
        "clinical_review_required": True,
        "content_source": "legacy_inline",
        "disclaimer": DISCLAIMER,
        "probabilities": probabilities,
    }

    if uncertainty:
        report["uncertainty_band"] = uncertainty.get("uncertainty_band")
        report["prediction_set"] = uncertainty.get("prediction_set")
        report["coverage_guarantee"] = uncertainty.get("coverage_guarantee")

    return report
