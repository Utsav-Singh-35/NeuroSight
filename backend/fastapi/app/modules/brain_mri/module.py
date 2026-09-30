"""Brain MRI module — implements the BaseModule interface.

Stacking ensemble of EfficientNet-B0, ResNet-50, DenseNet-121 and VGG-16 with a
logistic-regression meta-learner over Glioma / Meningioma / No Tumor /
Pituitary.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from app.config import settings
from app.engine.base_module import BaseModule
from app.services import inference, preprocessor
from app.services.ensemble import get_base_model, load_ensemble, run_ensemble_inference
from app.services.gradcam import generate_gradcam
from app.services.report import generate_report
from app.services.uncertainty import CalibrationBundle

logger = logging.getLogger(__name__)


class BrainMRIModule(BaseModule):
    """Brain MRI tumour classification module."""

    def __init__(self) -> None:
        self._metadata = self._load_metadata()
        self._ensemble: dict | None = None
        self._ensemble_failed = False
        self._calibration: CalibrationBundle | None = None

    def _load_metadata(self) -> dict:
        meta_path = os.path.join(os.path.dirname(__file__), "metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def module_id(self) -> str:
        return "brain_mri"

    @property
    def display_name(self) -> str:
        return self._metadata["display_name"]

    @property
    def classes(self) -> list[str]:
        return self._metadata["classes"]

    @property
    def models_dir(self) -> str:
        return settings.MODELS_DIR

    @property
    def timm_names(self) -> dict[str, str]:
        """Arch-key to timm-name mapping declared in metadata.json."""
        return self._metadata["timm_names"]

    @property
    def base_models(self) -> list[str]:
        return self._metadata["base_models"]

    def is_available(self) -> bool:
        """Whether this module can serve requests.

        Only probes EfficientNet, which is also the single-model fallback. A
        true result therefore means "can answer", not "full ensemble intact" --
        ensemble completeness is reported per-request via ``models_skipped``.
        """
        effnet = os.path.join(self.models_dir, "BRAIN_MRI_EFFICIENTNET.pth")
        return os.path.exists(effnet)

    def weight_paths(self) -> dict[str, str]:
        """Map each configured architecture to its weights file path."""
        prefix = self._metadata["model_prefix"]
        return {
            arch: os.path.join(self.models_dir, f"{prefix}_{arch.upper()}.pth")
            for arch in self.base_models
        }

    def preload_specs(self) -> list[dict]:
        """Model-cache specs for every base model with weights on disk."""
        specs = []
        for arch, path in self.weight_paths().items():
            if os.path.exists(path):
                specs.append(
                    {
                        "module_id": self.module_id,
                        "arch": arch,
                        "timm_name": self.timm_names[arch],
                        "weights_path": path,
                        "num_classes": len(self.classes),
                    }
                )
        return specs

    def _get_ensemble(self) -> dict | None:
        """Load and memoise the ensemble, or return None if unavailable."""
        if not settings.USE_ENSEMBLE:
            return None
        if self._ensemble is not None:
            return self._ensemble
        if self._ensemble_failed:
            return None

        try:
            self._ensemble = load_ensemble(self.models_dir)
            return self._ensemble
        except (FileNotFoundError, ValueError) as exc:
            # Cache the failure so a broken config is not re-read per request.
            self._ensemble_failed = True
            logger.error("Brain ensemble unavailable, falling back: %s", exc)
            return None

    def _get_single_model(self):
        """Cached EfficientNet-B0 used when the ensemble is unavailable."""
        return get_base_model(
            "efficientnet",
            os.path.join(self.models_dir, "BRAIN_MRI_EFFICIENTNET.pth"),
            num_classes=len(self.classes),
            module_id=self.module_id,
            timm_names=self.timm_names,
        )

    def _get_calibration(self) -> CalibrationBundle:
        """Lazily load the calibration/conformal artefacts once."""
        if self._calibration is None:
            self._calibration = CalibrationBundle.load(settings.CALIBRATION_DIR)
        return self._calibration

    def describe_uncertainty(self, probabilities: dict) -> dict:
        """Calibrate a probability dict and derive the conformal prediction set.

        Args:
            probabilities: Per-class percentages keyed by display label.

        Returns:
            The payload from :meth:`CalibrationBundle.describe`, or a minimal
            uncalibrated stand-in when no artefacts are present.
        """
        bundle = self._get_calibration()
        vector = np.array(
            [float(probabilities.get(label, 0.0)) / 100.0 for label in self.classes],
            dtype=np.float64,
        )
        return bundle.describe(vector, self.classes)

    def predict(self, image_bytes: bytes) -> dict:
        """Classify a brain MRI image."""
        tensor = preprocessor.preprocess_image(image_bytes)

        ensemble = self._get_ensemble()
        if ensemble is not None:
            return run_ensemble_inference(
                ensemble,
                tensor,
                self.classes,
                module_id=self.module_id,
                timm_names=self.timm_names,
            )

        result = inference.run_inference(
            self._get_single_model(), tensor, self.classes
        )
        result["ensemble_active"] = False
        return result

    def gradcam(self, image_bytes: bytes) -> dict:
        """Classify, then explain the prediction with Grad-CAM.

        The heatmap is produced from the base model that agrees with the
        ensemble, so the explanation reflects a model that actually supports
        the reported class. When no base model agrees, that is reported via
        ``explanation_faithful=False`` rather than silently presenting a
        heatmap for a different class.
        """
        tensor = preprocessor.preprocess_image(image_bytes)
        ensemble = self._get_ensemble()

        if ensemble is None:
            result = generate_gradcam(
                self._get_single_model(),
                tensor,
                image_bytes,
                self.classes,
                "efficientnet",
            )
            result["ensemble_active"] = False
            result["explanation_model"] = "efficientnet"
            result["explanation_faithful"] = True
            return result

        ens = run_ensemble_inference(
            ensemble,
            tensor,
            self.classes,
            module_id=self.module_id,
            timm_names=self.timm_names,
        )

        agree_key = ens["agreeing_model"]
        faithful = agree_key is not None
        if not faithful:
            # Fall back to the first model that ran so the user still gets a
            # visualisation, but flag that it explains a different class.
            agree_key = (ens["models_used"] or ensemble["order"])[0]
            logger.info(
                "No base model agreed with ensemble class %s; heatmap from %s "
                "is not a faithful explanation of the reported class.",
                ens["prediction"],
                agree_key,
            )

        target_index = self.classes.index(ens["prediction"]) if faithful else None

        cam_model = get_base_model(
            agree_key,
            ensemble["model_paths"][agree_key],
            num_classes=len(self.classes),
            module_id=self.module_id,
            timm_names=self.timm_names,
        )
        cam = generate_gradcam(
            cam_model,
            tensor,
            image_bytes,
            self.classes,
            agree_key,
            target_index=target_index,
        )

        return {
            "heatmap": cam["heatmap"],
            "prediction": ens["prediction"],
            "confidence": ens["confidence"],
            "probabilities": ens["probabilities"],
            "explanation_model": agree_key,
            "explanation_faithful": faithful,
            "models_used": ens["models_used"],
            "models_skipped": ens["models_skipped"],
            "ensemble_active": True,
        }

    def report(self, image_bytes: bytes) -> dict:
        """Classify and render the clinical report payload.

        Uncertainty is computed here and passed into the generator, because risk
        escalation depends on the conformal band. Leaving it to the caller meant
        a borderline case could be reported with its class's default risk tier,
        which is exactly the failure this pipeline exists to prevent.
        """
        result = self.predict(image_bytes)
        uncertainty = self.describe_uncertainty(result["probabilities"])

        report = generate_report(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            module_id=self.module_id,
            uncertainty=uncertainty,
        )
        report["ensemble_active"] = result.get("ensemble_active", True)
        report["models_skipped"] = result.get("models_skipped", [])
        report["calibrated"] = uncertainty.get("calibrated", False)
        report["calibrated_probabilities"] = uncertainty.get("calibrated_probabilities")
        report["module"] = self.module_id
        return report
