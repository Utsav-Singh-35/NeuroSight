"""Stacking ensemble service for brain tumor classification.

Loads the four base CNN models (EfficientNet-B0, ResNet-50, DenseNet-121,
VGG-16) and the Logistic Regression meta-learner, then produces a stacked
prediction by concatenating each base model's softmax probabilities and
feeding them to the meta-learner.

The concatenation order is defined by ``ensemble_config.json`` (``model_order``)
and MUST match the order used when the meta-learner was trained.
"""

import json
import os
import pickle

import numpy as np
import timm
import torch

# Map ensemble model keys to their timm architecture names
TIMM_NAMES = {
    "efficientnet": "efficientnet_b0",
    "resnet": "resnet50",
    "densenet": "densenet121",
    "vgg": "vgg16",
}


def load_ensemble(models_dir: str) -> dict:
    """Load all base models + meta-learner + config from a directory.

    Args:
        models_dir: Directory containing BRAIN_MRI_*.pth, meta_model.pkl,
            and ensemble_config.json.

    Returns:
        Dict with keys: base_models (dict key->nn.Module in eval mode),
        meta (sklearn estimator), order (list[str]), config (dict).

    Raises:
        FileNotFoundError: If a required file is missing.
    """
    config_path = os.path.join(models_dir, "ensemble_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    order = config["model_order"]
    save_names = config["save_names"]

    # Instead of loading ALL models into memory at once (>2.5GB),
    # we store paths and load each model on-demand during inference.
    model_paths = {}
    for key in order:
        weights_path = os.path.join(models_dir, save_names[key] + ".pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing model weights: {weights_path}")
        model_paths[key] = weights_path

    with open(os.path.join(models_dir, "meta_model.pkl"), "rb") as f:
        meta = pickle.load(f)

    return {
        "model_paths": model_paths,
        "meta": meta,
        "order": order,
        "config": config,
    }


def _load_single_model(key: str, path: str) -> torch.nn.Module:
    """Load a single base model from disk, run inference-ready."""
    timm_name = TIMM_NAMES[key]
    model = timm.create_model(timm_name, pretrained=False, num_classes=4)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_ensemble_inference(
    ensemble: dict,
    tensor: torch.Tensor,
    class_labels: list[str],
) -> dict:
    """Run stacked inference on a preprocessed image tensor.

    Loads each base model one at a time (to save memory), gets its softmax
    probabilities, then frees it before loading the next. Finally feeds the
    concatenated probabilities to the meta-learner.

    Args:
        ensemble: The dict returned by ``load_ensemble``.
        tensor: Preprocessed image tensor of shape (1, 3, 224, 224).
        class_labels: Class label names matching output indices.

    Returns:
        Dict with keys:
            - prediction (str): Final ensemble class label.
            - confidence (float): Meta-learner confidence percentage (0-100).
            - probabilities (dict): Per-class ensemble probability percentages.
            - agreeing_model (str): Base model key whose top-1 matched the
              ensemble prediction (used for Grad-CAM).
            - base_predictions (dict): Per-base-model top-1 label + confidence.
    """
    import gc

    order = ensemble["order"]
    meta = ensemble["meta"]
    model_paths = ensemble["model_paths"]

    per_model_probs = {}
    features = []
    skipped_models = []

    for key in order:
        try:
            model = _load_single_model(key, model_paths[key])
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            per_model_probs[key] = probs
            features.append(probs)
            del model
            gc.collect()
        except (OSError, MemoryError, RuntimeError) as e:
            # Model too large for available RAM — skip it gracefully
            import logging
            logging.getLogger(__name__).warning(
                "Skipping %s due to memory: %s", key, str(e)[:80]
            )
            skipped_models.append(key)
            # Insert zeros so feature vector length stays consistent for meta-learner
            features.append(np.zeros(len(class_labels), dtype=np.float32))
            gc.collect()

    X = np.concatenate(features).reshape(1, -1)  # (1, 16)

    pred_idx = int(meta.predict(X)[0])
    proba = meta.predict_proba(X)[0]
    confidence = round(float(proba[pred_idx]) * 100, 2)
    prediction = class_labels[pred_idx]
    probabilities = {
        class_labels[i]: round(float(proba[i]) * 100, 2)
        for i in range(len(class_labels))
    }

    # Pick the base model that agreed with the ensemble (highest confidence)
    # Only consider models that successfully ran (not skipped)
    agreeing_model = None
    best_conf = -1.0
    for key in order:
        if key in skipped_models:
            continue
        if key not in per_model_probs:
            continue
        p = per_model_probs[key]
        if int(p.argmax()) == pred_idx and float(p[pred_idx]) > best_conf:
            best_conf = float(p[pred_idx])
            agreeing_model = key
    if agreeing_model is None:
        # Fall back to the first model that actually ran
        for key in order:
            if key in per_model_probs:
                agreeing_model = key
                break
        if agreeing_model is None:
            agreeing_model = order[0]

    base_predictions = {
        key: {
            "prediction": class_labels[int(p.argmax())],
            "confidence": round(float(p.max()) * 100, 2),
        }
        for key, p in per_model_probs.items()
    }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "agreeing_model": agreeing_model,
        "base_predictions": base_predictions,
    }
