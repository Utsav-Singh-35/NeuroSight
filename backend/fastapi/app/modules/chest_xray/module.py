"""Chest X-ray module — implements the BaseModule interface.

Stacking ensemble of EfficientNet-B0, ResNet-50 and DenseNet-121 with a
logistic-regression meta-learner over Normal / Pneumonia / Tuberculosis.

This module previously reimplemented ensemble loading and inference locally,
because the shared helper hardcoded ``num_classes=4``. The shared helper now
takes the class count and a timm-name map, so both modules use one code path
and chest inherits model caching and device selection automatically.
"""

from __future__ import annotations

import json
import logging
import os

from app.config import settings
from app.engine.base_module import BaseModule
from app.services import inference, preprocessor
from app.services.ensemble import get_base_model, load_ensemble, run_ensemble_inference
from app.services.gradcam import generate_gradcam
from app.services.report import generate_report

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "chest_xray_ensemble_config.json"
META_FILENAME = "meta_model_Chest_Xray.pkl"


class ChestXrayModule(BaseModule):
    """Chest X-ray disease detection module."""

    def __init__(self) -> None:
        self._metadata = self._load_metadata()
        self._ensemble: dict | None = None
        self._ensemble_failed = False

    def _load_metadata(self) -> dict:
        meta_path = os.path.join(os.path.dirname(__file__), "metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def module_id(self) -> str:
        return "chest_xray"

    @property
    def display_name(self) -> str:
        return self._metadata["display_name"]

    @property
    def classes(self) -> list[str]:
        return self._metadata["classes"]

    @property
    def models_dir(self) -> str:
        # Chest weights live in their own directory, not MODELS_DIR.
        return settings.CHEST_MODELS_DIR

    @property
    def timm_names(self) -> dict[str, str]:
        return self._metadata["timm_names"]

    @property
    def base_models(self) -> list[str]:
        return self._metadata["base_models"]

    def is_available(self) -> bool:
        """Whether this module can serve requests (EfficientNet present)."""
        effnet = os.path.join(self.models_dir, "CHEST_XRAY_EFFICIENTNET.pth")
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
        """Load and memoise the chest ensemble, or None if unavailable.

        Only genuinely-missing files and a mismatched feature width are
        tolerated. Anything else propagates, rather than being swallowed by a
        blanket ``except Exception`` that used to hide corrupt pickles and
        sklearn version errors behind a silent single-model fallback.
        """
        if not settings.USE_ENSEMBLE:
            return None
        if self._ensemble is not None:
            return self._ensemble
        if self._ensemble_failed:
            return None

        try:
            self._ensemble = load_ensemble(
                self.models_dir,
                config_filename=CONFIG_FILENAME,
                meta_filename=META_FILENAME,
            )
            return self._ensemble
        except (FileNotFoundError, ValueError) as exc:
            self._ensemble_failed = True
            logger.error("Chest ensemble unavailable, falling back: %s", exc)
            return None

    def _get_single_model(self):
        """Cached EfficientNet-B0 used when the ensemble is unavailable."""
        return get_base_model(
            "efficientnet",
            os.path.join(self.models_dir, "CHEST_XRAY_EFFICIENTNET.pth"),
            num_classes=len(self.classes),
            module_id=self.module_id,
            timm_names=self.timm_names,
        )

    def predict(self, image_bytes: bytes) -> dict:
        """Classify a chest X-ray image."""
        if not self.is_available():
            raise RuntimeError(
                "Chest X-ray module not available — model weights missing."
            )

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
        """Classify, then explain the prediction with Grad-CAM."""
        if not self.is_available():
            raise RuntimeError(
                "Chest X-ray module not available — model weights missing."
            )

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

        Passes ``module_id`` explicitly: without it the generator defaulted to
        ``brain_mri`` and chest labels only resolved by falling through to the
        legacy inline dictionary, which worked by coincidence rather than by
        design.
        """
        result = self.predict(image_bytes)

        # No uncertainty payload is passed, deliberately. The calibration and
        # conformal artefacts in models/calibration/ were fitted on the brain
        # ensemble's 16-dimensional features and 4 classes; applying that
        # temperature or quantile to 3-class chest output would be meaningless.
        # generate_report therefore falls back to its confidence threshold for
        # risk escalation, and the response reports calibrated=False.
        report = generate_report(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            module_id=self.module_id,
        )
        report["ensemble_active"] = result.get("ensemble_active", True)
        report["models_skipped"] = result.get("models_skipped", [])
        report["calibrated"] = False
        report["module"] = self.module_id
        return report
