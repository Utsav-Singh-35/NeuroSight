"""Fit and evaluate input-validation thresholds for the brain MRI module.

What this measures
------------------
Five novelty detectors are compared on the same data, rather than one being
asserted to work:

============  ======================================================
MSP           1 - max softmax probability
entropy       Shannon entropy of the ensemble distribution
energy        -log sum exp over the meta-learner's decision scores
disagreement  fraction of base models whose top-1 differs
js_divergence mean pairwise Jensen-Shannon divergence between bases
============  ======================================================

All are oriented so **higher means more out-of-distribution**.

In-distribution scores come from the saved probability matrices restricted to
the **final test split**, so they never touch the rows used to fit the
calibration temperature or the conformal quantile.

Out-of-distribution images are run through the real inference path. Results are
reported **per tier** (near / corrupted / far) because pooling them would let
trivially-separable synthetic images inflate the headline number.

Usage
-----
Run from ``backend/fastapi``::

    python scripts/fit_ood.py
    python scripts/fit_ood.py --primary entropy --target-tpr 0.95
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.services.ensemble import get_base_model, load_ensemble  # noqa: E402
from app.services.model_cache import get_device, model_cache  # noqa: E402
from app.services.preprocessor import preprocess_image  # noqa: E402
from app.services.uncertainty import fit_conformal_quantile  # noqa: E402
from app.services.validation import (  # noqa: E402
    ensemble_disagreement,
    energy_score,
    evaluate_detector,
    fit_quality_thresholds,
    image_quality_features,
    max_softmax_probability,
    mean_pairwise_divergence,
    predictive_entropy,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DETECTORS = ["msp", "entropy", "energy", "disagreement", "js_divergence"]


def collect_images(directory: Path) -> list[Path]:
    """List image files in a directory, sorted."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def infer_base_probs(
    paths: list[Path], ensemble: dict, num_classes: int, batch_size: int, label: str
) -> dict[str, np.ndarray]:
    """Run every base model over a list of images.

    Models are evicted after each pass: this is a batch job on a 7.7 GB machine,
    and holding all four brain checkpoints (~645 MB) alongside the OOD tensors
    causes swapping.
    """
    device = get_device()
    out: dict[str, np.ndarray] = {}

    for arch in ensemble["order"]:
        started = time.time()
        model = get_base_model(
            arch,
            ensemble["model_paths"][arch],
            num_classes=num_classes,
            module_id="brain_mri",
        )
        probs = np.zeros((len(paths), num_classes), dtype=np.float64)

        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            tensors = []
            for path in chunk:
                with open(path, "rb") as handle:
                    tensors.append(preprocess_image(handle.read()))
            batch = torch.cat(tensors, dim=0).to(device)
            with torch.inference_mode():
                probs[start : start + len(chunk)] = (
                    torch.softmax(model(batch), dim=1).detach().cpu().numpy()
                )

        out[arch] = probs
        del model
        model_cache.clear()
        gc.collect()
        print(
            f"    {label}/{arch}: {len(paths)} images in {time.time() - started:.1f}s",
            flush=True,
        )

    return out


def score_all(
    base_probs: dict[str, np.ndarray], meta, order: list[str]
) -> dict[str, np.ndarray]:
    """Compute every detector score from base-model probabilities."""
    features = np.concatenate([base_probs[a] for a in order], axis=1)
    meta_probs = meta.predict_proba(features)

    # decision_function gives the *unnormalised* scores energy needs. Softmax
    # probabilities cannot be used: they sum to 1, making log-sum-exp constant.
    logits = meta.decision_function(features)
    if logits.ndim == 1:  # binary edge case
        logits = np.column_stack([-logits, logits])

    pred = meta_probs.argmax(axis=1)

    return {
        # Oriented so higher = more OOD.
        "msp": 1.0 - max_softmax_probability(meta_probs),
        "entropy": predictive_entropy(meta_probs),
        "energy": energy_score(logits),
        "disagreement": ensemble_disagreement(base_probs, pred),
        "js_divergence": mean_pairwise_divergence(base_probs),
        "_meta_probs": meta_probs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-tpr", type=float, default=0.95)
    parser.add_argument(
        "--primary",
        default=None,
        help="Detector to deploy; default picks best near-OOD AUROC",
    )
    parser.add_argument("--quality-percentile", type=float, default=1.0)
    parser.add_argument("--ood-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    calib_dir = Path(args.out) if args.out else Path(settings.CALIBRATION_DIR)
    calib_dir = calib_dir.resolve()
    ood_root = Path(args.ood_dir) if args.ood_dir else repo_root / "data" / "ood"

    with open(calib_dir / "probs_manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    order = manifest["model_order"]
    num_classes = manifest["num_classes"]

    ensemble = load_ensemble(settings.MODELS_DIR)
    with open(Path(settings.MODELS_DIR) / "meta_model.pkl", "rb") as handle:
        meta = pickle.load(handle)

    # --- in-distribution ---------------------------------------------------
    id_probs = {
        arch: np.load(calib_dir / filename).astype(np.float64)
        for arch, filename in manifest["files"].items()
    }
    test_idx = np.load(calib_dir / "split_test_idx.npy")
    id_probs_split = {arch: mat[test_idx] for arch, mat in id_probs.items()}

    print(f"ID rows (final test split) : {len(test_idx)}")
    id_scores = score_all(id_probs_split, meta, order)

    # --- OOD tiers ---------------------------------------------------------
    tiers = {}
    for tier in ("near_ood", "corrupted", "far_ood"):
        paths = collect_images(ood_root / tier)
        if not paths:
            print(f"\n{tier}: no images found, skipping")
            continue
        print(f"\n{tier}: {len(paths)} images")
        base = infer_base_probs(paths, ensemble, num_classes, args.batch_size, tier)
        tiers[tier] = {"paths": paths, "scores": score_all(base, meta, order)}

    if not tiers:
        print("\nERROR: no OOD images. Run scripts/build_ood_set.py first.", file=sys.stderr)
        return 2

    # --- evaluate ----------------------------------------------------------
    results: dict[str, dict] = {}
    print("\n" + "=" * 74)
    print("OOD detector comparison (higher score = more OOD)")
    print("=" * 74)

    for tier, payload in tiers.items():
        results[tier] = {}
        n_ood = len(payload["paths"])
        print(f"\n--- {tier}  (n_ood = {n_ood}) ---")
        print(f"  {'detector':16s} {'AUROC':>8s} {'FPR@95':>8s} {'ID mean':>10s} {'OOD mean':>10s}")
        for name in DETECTORS:
            evaluation = evaluate_detector(
                id_scores[name], payload["scores"][name], args.target_tpr
            )
            results[tier][name] = evaluation
            print(
                f"  {name:16s} {evaluation['auroc']:>8.4f} "
                f"{evaluation[f'fpr_at_{int(args.target_tpr * 100)}_tpr']:>8.4f} "
                f"{evaluation['id_mean']:>10.4f} {evaluation['ood_mean']:>10.4f}"
            )

    # --- choose the deployed detector --------------------------------------
    # Selected on near-OOD, the hardest and most clinically realistic tier.
    # Optimising for far-OOD would reward detecting noise, which nobody uploads.
    selection_tier = "near_ood" if "near_ood" in results else next(iter(results))
    if args.primary:
        primary = args.primary
    else:
        primary = max(results[selection_tier], key=lambda d: results[selection_tier][d]["auroc"])

    fpr_key = f"fpr_at_{int(args.target_tpr * 100)}_tpr"
    primary_threshold = results[selection_tier][primary]["threshold"]

    print(f"\n=== Selected detector: {primary} ===")
    print(f"  chosen on tier    : {selection_tier}")
    print(f"  AUROC             : {results[selection_tier][primary]['auroc']:.4f}")
    print(f"  FPR@{int(args.target_tpr*100)}TPR        : {results[selection_tier][primary][fpr_key]:.4f}")
    print(f"  threshold         : {primary_threshold:.6f}")

    if len(tiers[selection_tier]["paths"]) < 20:
        print(
            f"\n  CAVEAT: only {len(tiers[selection_tier]['paths'])} images in "
            f"{selection_tier}. AUROC has a wide confidence interval and the "
            "threshold is not publication-grade. Add more real off-modality "
            "images before quoting this number."
        )

    # --- quality thresholds from the ID population -------------------------
    print("\n=== Fitting quality thresholds from ID images ===")
    data_dir = repo_root / "data" / "brainMRI"
    with open(calib_dir / "test_files.json", "r", encoding="utf-8") as handle:
        all_files = json.load(handle)
    id_files = [data_dir / all_files[i] for i in test_idx]

    started = time.time()
    id_features = []
    for path in id_files:
        with open(path, "rb") as handle:
            id_features.append(image_quality_features(handle.read()))
    gates = fit_quality_thresholds(id_features, args.quality_percentile)
    print(f"  measured {len(id_features)} ID images in {time.time() - started:.1f}s")
    print(f"  min_laplacian_variance (p{args.quality_percentile}) : "
          f"{gates.min_laplacian_variance:.4f}")
    print(f"  min_pixel_std          (p{args.quality_percentile}) : "
          f"{gates.min_pixel_std:.4f}")

    # How well do those gates reject the corrupted tier?
    quality_eval = {}
    for tier, payload in tiers.items():
        features = []
        for path in payload["paths"]:
            with open(path, "rb") as handle:
                features.append(image_quality_features(handle.read()))
        rejected = sum(
            1
            for f in features
            if f["laplacian_variance"] < gates.min_laplacian_variance
            or f["pixel_std"] < gates.min_pixel_std
            or min(f["width"], f["height"]) < gates.min_dimension
            or f["aspect_ratio"] > gates.max_aspect_ratio
        )
        quality_eval[tier] = {
            "rejected": rejected,
            "total": len(features),
            "rejection_rate": round(rejected / max(1, len(features)), 4),
        }
        print(f"  quality gate rejects {rejected}/{len(features)} of {tier}")

    id_rejected = sum(
        1
        for f in id_features
        if f["laplacian_variance"] < gates.min_laplacian_variance
        or f["pixel_std"] < gates.min_pixel_std
    )
    print(
        f"  quality gate rejects {id_rejected}/{len(id_features)} of ID "
        f"(expected ~{args.quality_percentile}% by construction)"
    )

    # --- conformal set size as a novelty signal ----------------------------
    cal_idx = np.load(calib_dir / "split_calibration_idx.npy")
    cal_features = np.concatenate([id_probs[a][cal_idx] for a in order], axis=1)
    cal_probs = meta.predict_proba(cal_features)
    labels = np.load(calib_dir / "test_labels.npy")
    q_hat = fit_conformal_quantile(cal_probs, labels[cal_idx], settings.CONFORMAL_ALPHA)
    threshold = 1.0 - q_hat

    set_size_stats = {}
    id_sizes = (id_scores["_meta_probs"] >= threshold).sum(axis=1)
    set_size_stats["in_distribution"] = round(float(id_sizes.mean()), 4)
    for tier, payload in tiers.items():
        sizes = (payload["scores"]["_meta_probs"] >= threshold).sum(axis=1)
        set_size_stats[tier] = round(float(sizes.mean()), 4)

    print("\n=== Mean conformal set size (novelty signal, alpha="
          f"{settings.CONFORMAL_ALPHA}) ===")
    for key, value in set_size_stats.items():
        print(f"  {key:18s} {value:.4f}")

    # --- persist -----------------------------------------------------------
    doc = {
        "primary_detector": primary,
        "primary_threshold": primary_threshold,
        "selection_tier": selection_tier,
        "target_tpr": args.target_tpr,
        "detectors": results,
        "detector_orientation": "higher score = more out-of-distribution",
        "quality_thresholds": {
            "min_dimension": gates.min_dimension,
            "max_aspect_ratio": gates.max_aspect_ratio,
            "min_laplacian_variance": round(gates.min_laplacian_variance, 6),
            "min_pixel_std": round(gates.min_pixel_std, 6),
            "fitted_percentile": args.quality_percentile,
            "fitted_on_n_id_images": len(id_features),
        },
        "quality_gate_rejection": quality_eval,
        "quality_gate_id_rejection": {
            "rejected": id_rejected,
            "total": len(id_features),
            "rate": round(id_rejected / max(1, len(id_features)), 4),
        },
        "conformal_mean_set_size": set_size_stats,
        "tier_sizes": {t: len(p["paths"]) for t, p in tiers.items()},
        "tier_notes": {
            "near_ood": "Real chest X-rays. Hardest and most realistic; headline tier.",
            "corrupted": "Degraded real MRIs. Tests the quality gate, not novelty.",
            "far_ood": "Synthetic patterns. Sanity floor only; inflates AUROC.",
        },
        "id_source": "final test split (never used for calibration or conformal)",
        "n_id": int(len(test_idx)),
        "generated": datetime.now(timezone.utc).isoformat(),
    }

    with open(calib_dir / "ood.json", "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2)

    print(f"\nWritten: {calib_dir / 'ood.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
