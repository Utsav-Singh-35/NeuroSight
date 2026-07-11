"""FastAPI service configuration loaded from environment variables."""

from functools import cached_property
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the .env file at backend/fastapi/.env.

    Attributes:
        MODEL_PATH: Path to the PyTorch model weights file.
        HOST: Host address to bind the FastAPI server.
        PORT: Port number for the FastAPI server.
        CLASS_LABELS: Comma-separated class label names (parsed via class_labels property).
        GRADCAM_OPACITY: Opacity for Grad-CAM heatmap overlay (0.0 to 1.0).
        GRADCAM_COLORMAP: Colormap name for Grad-CAM visualization.
    """

    MODEL_PATH: str = "../../models/Brain_MRI_scan.pth"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CLASS_LABELS: str = "Glioma,Meningioma,No Tumor,Pituitary"
    GRADCAM_OPACITY: float = 0.4
    GRADCAM_COLORMAP: str = "jet"

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def class_labels(self) -> list[str]:
        """Parse CLASS_LABELS comma-separated string into a list of class names."""
        return [label.strip() for label in self.CLASS_LABELS.split(",")]

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
