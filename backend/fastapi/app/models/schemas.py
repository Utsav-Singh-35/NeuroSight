"""Pydantic response schemas for the FastAPI ML service."""

from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    prediction: str = Field(
        ..., description="Predicted class label (Glioma, Meningioma, No Tumor, Pituitary)"
    )
    confidence: float = Field(
        ..., ge=0, le=100, description="Confidence percentage (0-100)"
    )
    probabilities: dict[str, float] = Field(
        ..., description="Per-class probability distribution (each value 0-100)"
    )


class GradCAMResponse(BaseModel):
    """Response schema for the /gradcam endpoint."""

    heatmap: str = Field(..., description="Base64-encoded PNG heatmap image")
    prediction: str = Field(
        ..., description="Predicted class label (Glioma, Meningioma, No Tumor, Pituitary)"
    )
    confidence: float = Field(
        ..., ge=0, le=100, description="Confidence percentage (0-100)"
    )


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="Service status (healthy or unhealthy)")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded and ready")


class ErrorResponse(BaseModel):
    """Response schema for error responses."""

    error: str = Field(..., description="Error description message")
