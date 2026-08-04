"""Abstract base class for all medical imaging modules.

Every AI module in NeuraSight (brain_mri, chest_xray, etc.) must implement
this interface. The engine uses these methods to run predictions, generate
Grad-CAM heatmaps, and produce clinical reports.
"""

from abc import ABC, abstractmethod


class BaseModule(ABC):
    """Every medical imaging module implements this interface."""

    @property
    @abstractmethod
    def module_id(self) -> str:
        """Unique identifier like 'brain_mri', 'chest_xray'."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable module name."""
        pass

    @property
    @abstractmethod
    def classes(self) -> list[str]:
        """List of class labels this module can predict."""
        pass

    @property
    @abstractmethod
    def models_dir(self) -> str:
        """Directory containing model weight files."""
        pass

    @abstractmethod
    def predict(self, image_bytes: bytes) -> dict:
        """Run full pipeline: preprocess → ensemble/single → result.

        Args:
            image_bytes: Raw image file bytes (JPEG or PNG).

        Returns:
            Dict with keys: prediction, confidence, probabilities.
        """
        pass

    @abstractmethod
    def gradcam(self, image_bytes: bytes) -> dict:
        """Run prediction + generate Grad-CAM heatmap.

        Args:
            image_bytes: Raw image file bytes (JPEG or PNG).

        Returns:
            Dict with keys: heatmap (base64), prediction, confidence.
        """
        pass

    @abstractmethod
    def report(self, image_bytes: bytes) -> dict:
        """Run prediction + generate clinical report.

        Args:
            image_bytes: Raw image file bytes (JPEG or PNG).

        Returns:
            Dict with report fields (prediction, confidence, risk_level,
            description, ai_summary, recommendation, disclaimer, probabilities).
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if model weights exist and module is ready for inference."""
        pass
