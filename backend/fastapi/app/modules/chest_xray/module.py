"""Chest X-ray module — implements BaseModule interface.

Full implementation for chest X-ray disease detection.
Classes: Normal, Pneumonia, Tuberculosis
Models: EfficientNet-B0, ResNet-50, DenseNet-121 (3 models)
"""

import gc
import json
import os
import pickle

import numpy as np
import timm
import torch

from app.engine.base_module import BaseModule
from app.config import settings
from app.services import preprocessor
from app.services.gradcam import generate_gradcam
from app.services.report import generate_report

TIMM_NAMES = {
    "efficientnet": "efficientnet_b0",
    "resnet": "resnet50",
    "densenet": "densenet121",
}


class ChestXrayModule(BaseModule):
    """Chest X-ray disease detection module.

    Uses EfficientNet-B0, ResNet-50, DenseNet-121 base models with a
    Logistic Regression meta-learner (stacking ensemble).
    """

    def __init__(self):
        self._metadata = self._load_metadata()
        self._ensemble = None

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
        return settings.CHEST_MODELS_DIR

    def is_available(self) -> bool:
        """Check if chest X-ray model weights exist."""
        effnet_path = os.path.join(self.models_dir, "CHEST_XRAY_EFFICIENTNET.pth")
        return os.path.exists(effnet_path)

    def _load_ensemble(self) -> dict | None:
        """Load the chest X-ray stacking ensemble.

        Uses chest-specific filenames:
        - Config: chest_xray_ensemble_config.json
        - Meta-learner: meta_model_Chest_Xray.pkl
        """
        if self._ensemble is not None:
            return self._ensemble
        if not settings.USE_ENSEMBLE:
            return None
        try:
            models_dir = self.models_dir
            config_path = os.path.join(models_dir, "chest_xray_ensemble_config.json")
            if not os.path.exists(config_path):
                return None

            with open(config_path, "r") as f:
                config = json.load(f)

            order = config["model_order"]
            save_names = config["save_names"]

            model_paths = {}
            for key in order:
                path = os.path.join(models_dir, save_names[key] + ".pth")
                if not os.path.exists(path):
                    return None
                model_paths[key] = path

            meta_path = os.path.join(models_dir, "meta_model_Chest_Xray.pkl")
            if not os.path.exists(meta_path):
                return None

            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            self._ensemble = {
                "model_paths": model_paths,
                "meta": meta,
                "order": order,
                "config": config,
            }
            return self._ensemble
        except (FileNotFoundError, Exception):
            return None

    def _load_single_model(self, key: str, path: str):
        """Load a single timm model from disk."""
        timm_name = TIMM_NAMES[key]
        model = timm.create_model(timm_name, pretrained=False, num_classes=len(self.classes))
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _run_ensemble_inference(self, ensemble: dict, tensor, class_labels: list[str]) -> dict:
        """Run stacking ensemble inference (3 models × 3 classes = 9 features)."""
        order = ensemble["order"]
        meta = ensemble["meta"]
        model_paths = ensemble["model_paths"]

        per_model_probs = {}
        features = []

        for key in order:
            try:
                model = self._load_single_model(key, model_paths[key])
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                per_model_probs[key] = probs
                features.append(probs)
                del model
                gc.collect()
            except Exception:
                features.append(np.zeros(len(class_labels), dtype=np.float32))
                gc.collect()

        X = np.concatenate(features).reshape(1, -1)
        pred_idx = int(meta.predict(X)[0])
        proba = meta.predict_proba(X)[0]
        confidence = round(float(proba[pred_idx]) * 100, 2)
        prediction = class_labels[pred_idx]
        probabilities = {
            class_labels[i]: round(float(proba[i]) * 100, 2)
            for i in range(len(class_labels))
        }

        # Find the agreeing model with highest confidence for the predicted class
        agreeing_model = order[0]
        best_conf = -1.0
        for key in order:
            if key in per_model_probs:
                p = per_model_probs[key]
                if int(p.argmax()) == pred_idx and float(p[pred_idx]) > best_conf:
                    best_conf = float(p[pred_idx])
                    agreeing_model = key

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "agreeing_model": agreeing_model,
        }

    def predict(self, image_bytes: bytes) -> dict:
        """Run chest X-ray classification pipeline."""
        if not self.is_available():
            raise RuntimeError("Chest X-ray module not available — model weights missing.")

        tensor = preprocessor.preprocess_image(image_bytes)
        ensemble = self._load_ensemble()

        if ensemble:
            return self._run_ensemble_inference(ensemble, tensor, self.classes)
        else:
            # Fallback to single EfficientNet model
            path = os.path.join(self.models_dir, "CHEST_XRAY_EFFICIENTNET.pth")
            model = self._load_single_model("efficientnet", path)
            from app.services.inference import run_inference

            result = run_inference(model, tensor, self.classes)
            del model
            gc.collect()
            return result

    def gradcam(self, image_bytes: bytes) -> dict:
        """Run prediction + generate Grad-CAM heatmap."""
        if not self.is_available():
            raise RuntimeError("Chest X-ray module not available — model weights missing.")

        tensor = preprocessor.preprocess_image(image_bytes)
        ensemble = self._load_ensemble()

        if ensemble:
            ens_result = self._run_ensemble_inference(ensemble, tensor, self.classes)
            agree_key = ens_result["agreeing_model"]
            cam_model = self._load_single_model(agree_key, ensemble["model_paths"][agree_key])
            cam_result = generate_gradcam(cam_model, tensor, image_bytes, self.classes, agree_key)
            del cam_model
            gc.collect()
            return {
                "heatmap": cam_result["heatmap"],
                "prediction": ens_result["prediction"],
                "confidence": ens_result["confidence"],
            }
        else:
            path = os.path.join(self.models_dir, "CHEST_XRAY_EFFICIENTNET.pth")
            model = self._load_single_model("efficientnet", path)
            result = generate_gradcam(model, tensor, image_bytes, self.classes, "efficientnet")
            del model
            gc.collect()
            return result

    def report(self, image_bytes: bytes) -> dict:
        """Run prediction + generate clinical report."""
        if not self.is_available():
            raise RuntimeError("Chest X-ray module not available — model weights missing.")

        result = self.predict(image_bytes)
        return generate_report(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        )
