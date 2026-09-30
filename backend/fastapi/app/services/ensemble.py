"""Stacking-ensemble loading and inference.

Design notes
------------
Base models emit a softmax probability vector each. Those vectors are
concatenated in the exact order given by ``config["model_order"]`` and fed to a
logistic-regression meta-learner, so the reported confidence is the
meta-learner's own posterior rather than an average of the base softmaxes.

That concatenation order is the single most fragile contract in the pipeline:
the column layout must match what the meta-learner was fitted on. A permuted
``model_order`` still yields the right feature *count*, so sklearn would not
complain -- it would just produce confidently wrong answers. ``load_ensemble``
therefore validates the width against ``meta.n_features_in_`` at load time.

Models are resolved through ``model_cache``, so weights are read from disk once
per process rather than once per request.
"""

from __future__ import annotations

import json
import logging
import os
import pickle

import numpy as np
import torch

from app.services.model_cache import get_device, model_cache

logger = logging.getLogger(__name__)

# Default architecture map for the brain MRI module. Other modules pass their
# own mapping from metadata.json.
TIMM_NAMES = {
    "efficientnet": "efficientnet_b0",
    "resnet": "resnet50",
    "densenet": "densenet121",
    "vgg": "vgg16",
}


def load_ensemble(
    models_dir: str,
    config_filename: str = "ensemble_config.json",
    meta_filename: str = "meta_model.pkl",
) -> dict:
    """Load ensemble config, weight paths and meta-learner from a directory.

    Args:
        models_dir: Directory holding the weights, config JSON and meta pickle.
        config_filename: Ensemble config file name.
        meta_filename: Pickled meta-learner file name.

    Returns:
        Dict with ``model_paths``, ``meta``, ``order``, ``config``,
        ``class_names``, ``num_classes``.

    Raises:
        FileNotFoundError: A required file is missing.
        ValueError: The meta-learner's expected feature width does not match
            ``len(model_order) * num_classes``.
    """
    config_path = os.path.join(models_dir, config_filename)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    order = config["model_order"]
    save_names = config["save_names"]
    class_names = config.get("class_names", [])
    num_classes = int(config.get("num_classes", len(class_names) or 4))

    model_paths: dict[str, str] = {}
    for key in order:
        weights_path = os.path.join(models_dir, save_names[key] + ".pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing model weights: {weights_path}")
        model_paths[key] = weights_path

    meta_path = os.path.join(models_dir, meta_filename)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta-learner: {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # Guard the feature-order contract as far as is mechanically possible.
    expected = len(order) * num_classes
    actual = getattr(meta, "n_features_in_", None)
    if actual is not None and int(actual) != expected:
        raise ValueError(
            f"Meta-learner expects {actual} features but config implies "
            f"{expected} ({len(order)} models x {num_classes} classes). "
            f"model_order or num_classes in {config_filename} is wrong."
        )

    return {
        "model_paths": model_paths,
        "meta": meta,
        "order": order,
        "config": config,
        "class_names": class_names,
        "num_classes": num_classes,
    }


def get_base_model(
    arch: str,
    weights_path: str,
    num_classes: int = 4,
    module_id: str = "brain_mri",
    timm_names: dict[str, str] | None = None,
) -> torch.nn.Module:
    """Return a cached base model for the given architecture.

    Args:
        arch: Short architecture key, e.g. ``"resnet"``.
        weights_path: Path to the state dict.
        num_classes: Classifier width.
        module_id: Cache namespace.
        timm_names: Arch-key to timm-name mapping; defaults to brain's.

    Returns:
        Eval-mode model on the inference device.
    """
    names = timm_names or TIMM_NAMES
    return model_cache.get(
        module_id=module_id,
        arch=arch,
        timm_name=names[arch],
        weights_path=weights_path,
        num_classes=num_classes,
    )


def _load_single_model(key: str, path: str) -> torch.nn.Module:
    """Backwards-compatible shim for the brain MRI 4-class base models.

    Retained because existing callers import this name. New code should prefer
    :func:`get_base_model`, which is explicit about module and class count.
    """
    return get_base_model(key, path, num_classes=4, module_id="brain_mri")


def run_ensemble_inference(
    ensemble: dict,
    tensor: torch.Tensor,
    class_labels: list[str],
    module_id: str = "brain_mri",
    timm_names: dict[str, str] | None = None,
) -> dict:
    """Run stacked inference on a preprocessed image tensor.

    Args:
        ensemble: The dict returned by :func:`load_ensemble`.
        tensor: Preprocessed tensor of shape ``(1, 3, 224, 224)``.
        class_labels: Display labels ordered to match model output indices.
        module_id: Cache namespace for base models.
        timm_names: Arch-key to timm-name mapping.

    Returns:
        Dict with ``prediction``, ``confidence``, ``probabilities``,
        ``agreeing_model``, ``base_predictions``, ``base_probabilities``,
        ``models_used``, ``models_skipped`` and ``ensemble_active``.

        ``models_skipped`` is non-empty when a base model failed to load and a
        zero vector was substituted to preserve feature width. Callers should
        surface that, because the meta-learner is then reading an input region
        it never saw in training while still returning a confident-looking
        number.
    """
    order = ensemble["order"]
    meta = ensemble["meta"]
    model_paths = ensemble["model_paths"]
    num_classes = ensemble.get("num_classes", len(class_labels))

    device = get_device()
    tensor = tensor.to(device)

    per_model_probs: dict[str, np.ndarray] = {}
    features: list[np.ndarray] = []
    skipped: list[str] = []

    for key in order:
        try:
            model = get_base_model(
                key,
                model_paths[key],
                num_classes=num_classes,
                module_id=module_id,
                timm_names=timm_names,
            )
            with torch.inference_mode():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            per_model_probs[key] = probs
            features.append(probs)
        except Exception as exc:  # noqa: BLE001 - one bad model must not 500
            logger.warning("Base model %s unavailable: %s", key, str(exc)[:160])
            skipped.append(key)
            features.append(np.zeros(num_classes, dtype=np.float32))

    feature_vector = np.concatenate(features).reshape(1, -1)

    pred_idx = int(meta.predict(feature_vector)[0])
    proba = meta.predict_proba(feature_vector)[0]
    prediction = class_labels[pred_idx]
    confidence = round(float(proba[pred_idx]) * 100, 2)
    probabilities = {
        class_labels[i]: round(float(proba[i]) * 100, 2)
        for i in range(len(class_labels))
    }

    agreeing_model = _pick_agreeing_model(order, per_model_probs, pred_idx)

    base_predictions = {
        key: {
            "prediction": class_labels[int(p.argmax())],
            "confidence": round(float(p.max()) * 100, 2),
        }
        for key, p in per_model_probs.items()
    }
    base_probabilities = {
        key: [round(float(v), 6) for v in p] for key, p in per_model_probs.items()
    }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "raw_meta_proba": [float(v) for v in proba],
        "agreeing_model": agreeing_model,
        "base_predictions": base_predictions,
        "base_probabilities": base_probabilities,
        "models_used": [k for k in order if k in per_model_probs],
        "models_skipped": skipped,
        "ensemble_active": True,
    }


def _pick_agreeing_model(
    order: list[str],
    per_model_probs: dict[str, np.ndarray],
    pred_idx: int,
) -> str | None:
    """Choose the base model whose explanation best matches the ensemble.

    Among base models whose own top-1 equals the ensemble's class, return the
    one most confident in that class. Grad-CAM then explains a model that
    actually agrees with the reported answer.

    Returns ``None`` when nothing agrees, which is a meaningful signal: the
    caller should not pretend a heatmap from a disagreeing model explains the
    reported class.
    """
    best_key: str | None = None
    best_conf = -1.0

    for key in order:
        probs = per_model_probs.get(key)
        if probs is None:
            continue
        if int(probs.argmax()) == pred_idx and float(probs[pred_idx]) > best_conf:
            best_conf = float(probs[pred_idx])
            best_key = key

    return best_key
