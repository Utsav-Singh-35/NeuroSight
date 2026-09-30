"""Pydantic response schemas for the FastAPI ML service.

Design note
-----------
The earlier ``PredictionResponse`` exposed only ``prediction``, ``confidence``
and ``probabilities``, which silently discarded work the pipeline had already
done: per-base-model outputs, which models actually contributed, whether the
ensemble ran degraded. The per-model disagreement signal turned out to be the
*best* out-of-distribution detector available (AUROC 0.954 on off-modality
images), so throwing it away was costing real capability.

Everything the pipeline computes is now surfaced, with optional fields so a
module that cannot produce a signal reports its absence rather than faking it.
"""

from pydantic import BaseModel, ConfigDict, Field


class BasePrediction(BaseModel):
    """Per-base-model top-1 output."""

    prediction: str = Field(..., description="This model's own predicted label")
    confidence: float = Field(..., ge=0, le=100, description="Its own confidence")


class UncertaintyInfo(BaseModel):
    """Calibration and conformal prediction output.

    ``prediction_set`` is the conformal set: the classes that cannot be excluded
    at the stated coverage level. A set larger than one means the model has not
    discriminated, and the UI must not present the top-1 as an answer.
    """

    calibrated: bool = Field(
        ..., description="Whether temperature calibration was applied"
    )
    temperature: float | None = Field(
        default=None, description="Fitted temperature (1.0 means no scaling)"
    )
    calibration_design: str | None = Field(
        default=None, description="Which calibration design is deployed"
    )
    calibrated_probabilities: dict[str, float] | None = Field(
        default=None, description="Post-calibration per-class percentages"
    )
    prediction_set: list[str] | None = Field(
        default=None, description="Conformal prediction set"
    )
    prediction_set_size: int | None = Field(default=None)
    uncertainty_band: str | None = Field(
        default=None, description="confident | borderline | indeterminate"
    )
    coverage_guarantee: float | None = Field(
        default=None, description="Conformal coverage level, e.g. 0.99"
    )
    conformal_threshold: float | None = Field(default=None)
    conformal_fallback: bool | None = Field(
        default=None,
        description="True when an empty set forced a top-1 fallback, which "
        "slightly breaks the formal guarantee for that case",
    )


class ValidationInfo(BaseModel):
    """Input quality and out-of-distribution assessment."""

    validated: bool = Field(
        ..., description="Whether fitted validation thresholds were available"
    )
    passed: bool = Field(..., description="Whether the image passed quality checks")
    enforced: bool | None = Field(
        default=None,
        description="Whether a quality failure would have rejected the request. "
        "When False the assessment is advisory only.",
    )
    reasons: list[str] = Field(
        default_factory=list, description="Human-readable failure reasons"
    )
    checks: dict[str, bool] | None = Field(default=None)
    features: dict[str, float] | None = Field(default=None)
    is_ood: bool | None = Field(
        default=None, description="Whether the input looks out-of-distribution"
    )
    ood_detector: str | None = Field(default=None)
    ood_score: float | None = Field(default=None)
    ood_threshold: float | None = Field(default=None)
    ood_reason: str | None = Field(default=None)
    scores: dict[str, float] | None = Field(
        default=None, description="All computed novelty scores"
    )


class EnsembleInfo(BaseModel):
    """Which models contributed, and how they behaved."""

    ensemble_active: bool = Field(
        ..., description="False means a single fallback model produced this result"
    )
    models_used: list[str] = Field(default_factory=list)
    models_skipped: list[str] = Field(
        default_factory=list,
        description="Models that failed to load. Non-empty means zero vectors "
        "were substituted and the confidence is less trustworthy.",
    )
    agreeing_model: str | None = Field(
        default=None,
        description="Base model whose top-1 matched the ensemble. None means "
        "no model agreed, so any heatmap explains a different class.",
    )
    base_predictions: dict[str, BasePrediction] | None = Field(default=None)


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    prediction: str = Field(..., description="Predicted class label")
    confidence: float = Field(
        ..., ge=0, le=100, description="Confidence percentage (0-100)"
    )
    probabilities: dict[str, float] = Field(
        ..., description="Per-class probability distribution (each value 0-100)"
    )
    module: str = Field(
        default="brain_mri", description="Module ID used for this prediction"
    )
    uncertainty: UncertaintyInfo | None = Field(default=None)
    validation: ValidationInfo | None = Field(default=None)
    ensemble: EnsembleInfo | None = Field(default=None)
    duration_ms: float | None = Field(default=None)


class GradCAMResponse(BaseModel):
    """Response schema for the /gradcam endpoint."""

    heatmap: str = Field(..., description="Base64-encoded PNG heatmap (no data: prefix)")
    prediction: str = Field(..., description="Predicted class label")
    confidence: float = Field(
        ..., ge=0, le=100, description="Confidence percentage (0-100)"
    )
    module: str = Field(default="brain_mri", description="Module ID used")
    explanation_model: str | None = Field(
        default=None, description="Base model the heatmap was computed from"
    )
    explanation_faithful: bool | None = Field(
        default=None,
        description="False means no base model agreed with the ensemble, so the "
        "heatmap explains a different class than the one reported",
    )
    probabilities: dict[str, float] | None = Field(default=None)
    uncertainty: UncertaintyInfo | None = Field(default=None)
    validation: ValidationInfo | None = Field(default=None)
    ensemble: EnsembleInfo | None = Field(default=None)


class SourceRef(BaseModel):
    """A resolved citation."""

    id: str
    resolved: bool = True
    title: str | None = None
    publisher: str | None = None
    url: str | None = None
    type: str | None = None
    licence: str | None = None
    accessed: str | None = None


class ReportResponse(BaseModel):
    """Response schema for the /report endpoint.

    Previously this endpoint had **no** response model, so its shape was
    unvalidated and undocumented in the OpenAPI schema.
    """

    model_config = ConfigDict(extra="allow")

    prediction: str
    confidence: float = Field(..., ge=0, le=100)
    probabilities: dict[str, float]
    module: str = "brain_mri"

    risk_level: str
    risk_escalated: bool = Field(
        default=False,
        description="True when uncertainty forced the risk level to Indeterminate",
    )

    description: str | None = None
    ai_summary: str | None = None
    ai_limitations: str | None = None
    recommendation: str | None = None
    investigations: str | None = None
    clinical_considerations: str | None = None
    follow_up: str | None = None
    warning_signs: str | None = None
    treatment_information: str | None = None
    india_care_pathway: str | None = Field(
        default=None,
        description="Jurisdictional context for India: care pathway, resource "
        "setting, regulatory status under CDSCO and the ICMR ethical framework. "
        "Explicitly states that citing Indian authorities does not imply the "
        "model was validated on an Indian cohort.",
    )

    clinical_narrative: str | None = None
    narrative_source: str | None = Field(
        default=None, description="'reviewed_llm_narrative' or 'template'"
    )
    narrative_reviewed_by: str | None = None
    narrative_authored_by_model: str | None = None

    sources: list[SourceRef] = Field(default_factory=list)
    evidence_grounded: bool = False
    clinical_review_required: bool = True
    content_source: str | None = None

    disclaimer: str

    uncertainty: UncertaintyInfo | None = None
    validation: ValidationInfo | None = None
    ensemble: EnsembleInfo | None = None


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="Service status (healthy or unhealthy)")
    model_loaded: bool = Field(..., description="Whether a model is loaded and ready")
    device: str | None = Field(default=None, description="Inference device")
    available_modules: dict | list | None = Field(default=None)
    calibration: dict | None = Field(default=None)
    validation: dict | None = Field(default=None)
    knowledge_base: dict | None = Field(default=None)
    narratives: dict | None = Field(default=None)
    llm: dict | None = Field(default=None)
    models_cached: list[str] | None = Field(default=None)


class ModuleInfo(BaseModel):
    """Schema for a single module in the /modules response."""

    id: str = Field(..., description="Unique module identifier")
    name: str = Field(..., description="Human-readable module name")
    classes: list[str] = Field(..., description="Classes this module can predict")
    available: bool = Field(..., description="Whether model weights are ready")


class ModulesResponse(BaseModel):
    """Response schema for the /modules endpoint."""

    modules: list[ModuleInfo] = Field(..., description="List of registered AI modules")


class ErrorResponse(BaseModel):
    """Response schema for error responses."""

    error: str = Field(..., description="Error description message")
