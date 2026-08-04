"""Brain MRI module — implements BaseModule interface.

Wraps the existing ensemble/single-model inference pipeline for brain
tumor classification (Glioma, Meningioma, No Tumor, Pituitary).
"""

import gc
import json
import os

from app.engine.base_module import BaseModule
from app.config import settings
from app.services import preprocessor, inference
from app.services.ensemble import load_ensemble, run_ensemble_inference, _load_single_model
from app.services.gradcam import generate_gradcam
from app.services.report import generate_report


class BrainMRIModule(BaseModule):
    """Brain MRI tumor classification module.

    Uses EfficientNet-B0, ResNet-50, DenseNet-121, and VGG-16 base models
    with a Logistic Regression meta-learner (stacking ensemble).
    """

    def __init__(self):
        self._metadata = self._load_metadata()
        self._ensemble = None
        self._single_model = None

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

    def is_available(self) -> bool:
        """Check if at least the EfficientNet model weights exist."""
        effnet_path = os.path.join(self.models_dir, "BRAIN_MRI_EFFICIENTNET.pth")
        return os.path.exists(effnet_path)

    def _get_ensemble(self) -> dict | None:
        """Load ensemble if USE_ENSEMBLE is enabled and files exist."""
        if not settings.USE_ENSEMBLE:
            return None
        if self._ensemble is None:
            try:
                self._ensemble = load_ensemble(self.models_dir)
            except FileNotFoundError:
                return None
        return self._ensemble

    def _get_single_model(self):
        """Load the single EfficientNet model as fallback."""
        if self._single_model is None:
            self._single_model = inference.load_model(settings.MODEL_PATH)
        return self._single_model

    def predict(self, image_bytes: bytes) -> dict:
        """Run brain MRI classification pipeline."""
        tensor = preprocessor.preprocess_image(image_bytes)

        ensemble = self._get_ensemble()
        if ensemble is not None:
            return run_ensemble_inference(ensemble, tensor, self.classes)
        else:
            model = self._get_single_model()
            return inference.run_inference(model, tensor, self.classes)

    def gradcam(self, image_bytes: bytes) -> dict:
        """Run prediction + generate Grad-CAM heatmap."""
        tensor = preprocessor.preprocess_image(image_bytes)

        ensemble = self._get_ensemble()
        if ensemble is not None:
            ens_result = run_ensemble_inference(ensemble, tensor, self.classes)
            agree_key = ens_result["agreeing_model"]
            cam_model = _load_single_model(agree_key, ensemble["model_paths"][agree_key])
            cam_result = generate_gradcam(
                cam_model, tensor, image_bytes, self.classes, agree_key
            )
            del cam_model
            gc.collect()
            return {
                "heatmap": cam_result["heatmap"],
                "prediction": ens_result["prediction"],
                "confidence": ens_result["confidence"],
            }
        else:
            model = self._get_single_model()
            return generate_gradcam(model, tensor, image_bytes, self.classes, "efficientnet")

    def report(self, image_bytes: bytes) -> dict:
        """Run prediction + generate clinical report."""
        result = self.predict(image_bytes)
        return generate_report(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        )
