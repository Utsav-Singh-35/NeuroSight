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

    base_models = {}
    for key in order:
        timm_name = TIMM_NAMES[key]
        model = timm.create_model(timm_name, pretrained=False, num_classes=4)
        weights_path = os.path.join(models_dir, save_names[key] + ".pth")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        base_models[key] = model

    with open(os.path.join(models_dir, "meta_model.pkl"), "rb") as f:
        meta = pickle.load(f)

    return {"base_models": base_models, "meta": meta, "order": order, "config": config}


def run_ensemble_inference(
    ensemble: dict,
    tensor: torch.Tensor,
    class_labels: list[str],
) -> dict:
    """Run stacked inference on a preprocessed image tensor.

    Runs all base models, concatenates their softmax probabilities in the
    configured order, and feeds the feature vector to the meta-learner.

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
              ensemble prediction (used for Grad-CAM). Falls back to the first
              model in the order if none agree.
            - base_predictions (dict): Per-base-model top-1 label + confidence.
    """
    order = ensemble["order"]
    meta = ensemble["meta"]

    per_model_probs = {}
    features = []
    with torch.no_grad():
        for key in order:
            logits = ensemble["base_models"][key](tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            per_model_probs[key] = probs
            features.append(probs)

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
    agreeing_model = None
    best_conf = -1.0
    for key in order:
        p = per_model_probs[key]
        if int(p.argmax()) == pred_idx and float(p[pred_idx]) > best_conf:
            best_conf = float(p[pred_idx])
            agreeing_model = key
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
