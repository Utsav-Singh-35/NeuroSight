"""FastAPI service configuration loaded from environment variables.

Supports multi-module AI engine with per-module model directories.
"""

from functools import cached_property
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the .env file at backend/fastapi/.env.

    Attributes:
        MODEL_PATH: Path to the default PyTorch model weights file (brain MRI EfficientNet).
        MODELS_DIR: Base directory containing all model weight files.
        MODELS_BASE_DIR: Alias for MODELS_DIR — base path for all modules' models.
        USE_ENSEMBLE: Whether to use stacking ensemble (all base models + meta-learner).
        HOST: Host address to bind the FastAPI server.
        PORT: Port number for the FastAPI server.
        CLASS_LABELS: Comma-separated class label names (brain MRI default).
        GRADCAM_OPACITY: Opacity for Grad-CAM heatmap overlay (0.0 to 1.0).
        GRADCAM_COLORMAP: Colormap name for Grad-CAM visualization.
    """

    MODEL_PATH: str = "../../models/BRAIN_MRI_EFFICIENTNET.pth"
    MODELS_DIR: str = "../../models"
    MODELS_BASE_DIR: str = "../../models"
    CHEST_MODELS_DIR: str = "../../chest"
    USE_ENSEMBLE: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CLASS_LABELS: str = "Glioma,Meningioma,No Tumor,Pituitary"
    GRADCAM_OPACITY: float = 0.4
    GRADCAM_COLORMAP: str = "jet"

    # --- Inference runtime -------------------------------------------------
    # Pin to CPU even when a GPU is present (useful for reproducible timing).
    FORCE_CPU: bool = False

    # --- Input validation --------------------------------------------------
    # Image quality is ALWAYS assessed and reported. This flag decides whether a
    # failure also rejects the request. Enforcement costs roughly 2% of valid
    # in-distribution scans at the fitted 1st-percentile thresholds, so it is
    # opt-in rather than on by default.
    ENFORCE_QUALITY_GATE: bool = False
    # Hard upload cap at the FastAPI layer. The Express gateway caps uploads
    # too, but a direct call to FastAPI would otherwise bypass that entirely.
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
    # Comma-separated module ids whose weights are loaded at startup. Modules
    # left out are still served, just cached lazily on first request.
    PRELOAD_MODULES: str = "brain_mri"

    # --- Calibration / conformal artefacts --------------------------------
    # Directory holding temperature + conformal quantile files produced by the
    # offline calibration scripts. Resolved like the other model paths.
    CALIBRATION_DIR: str = "../../models/calibration"
    # Target miscoverage for conformal prediction sets (0.10 -> 90% coverage).
    # 0.01 (99% target coverage), not the textbook 0.10. The brain ensemble is
    # ~97.7% accurate, so any target coverage below that produces only
    # singleton sets and the conformal layer becomes decorative. See
    # models/calibration/conformal.json -> alpha_sweep for the evidence.
    CONFORMAL_ALPHA: float = 0.01

    # --- LLM / evidence layer ---------------------------------------------
    # Groq API key. Lives in the repo-root .env so the whole project shares
    # one value; see model_config below for the load order.
    GROQ_API: str = ""
    # Groq's model lineup changes; `llama-3.3-70b-versatile` returned 404 for
    # this key on 2026-09-29. Verify with scripts/author_narratives.py
    # --list-models before assuming a name is valid.
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def preload_modules(self) -> list[str]:
        """Parse PRELOAD_MODULES into a list of module ids."""
        return [m.strip() for m in self.PRELOAD_MODULES.split(",") if m.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def class_labels(self) -> list[str]:
        """Parse CLASS_LABELS comma-separated string into a list of class names."""
        return [label.strip() for label in self.CLASS_LABELS.split(",")]

    # Two env files, lowest precedence first. The repo-root .env holds values
    # shared across the whole project (the Groq key, Mongo URI); the service
    # -local .env holds FastAPI specifics and wins on conflict.
    model_config = SettingsConfigDict(
        env_file=(
            Path(__file__).resolve().parents[3] / ".env",
            Path(__file__).resolve().parent.parent / ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
