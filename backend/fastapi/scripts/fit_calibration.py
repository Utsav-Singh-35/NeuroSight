"""Fit probability calibration and conformal prediction for the brain ensemble.

Split design
------------
The shipped ``models/meta_model.pkl`` was fitted on one half of a 50/50
stratified split of the 1,600-image test set (``random_state=42``). That half is
therefore *spent* -- the meta-learner has seen it.

Rather than retrain and invalidate the deployed artefact, this script:

1. Reconstructs the documented split and **verifies** the existing pickle
   reproduces the published 96.75% on the untouched half. If that check fails,
   nothing downstream can be trusted, so it is reported loudly.
2. Subdivides the untouched half into a calibration split (40%) and a final
   test split (60%), stratified. The temperature and the conformal quantile are
   fitted on calibration; every reported number comes from the final test split,
   which neither the meta-learner nor the calibration has ever seen.

```
1,600 test rows
├── half A (800)  meta-learner training      [already used by meta_model.pkl]
└── half B (800)  untouched by the meta
     ├── calibration (320)  -> temperature T, conformal q̂
     └── final test  (480)  -> all reported metrics
```

Two calibration designs are compared, because the answer is not obvious: the
meta-learner is *itself* a learned recalibrator, so scaling its output may beat
scaling its inputs.

- **Design A**: temperature-scale each base model, then refit the meta-learner.
- **Design B**: leave bases raw, temperature-scale the meta-learner's output.

Design A is run as an experiment only. It does **not** overwrite the deployed
pickle.

Usage
-----
Run from ``backend/fastapi``::

    python scripts/fit_calibration.py
    python scripts/fit_calibration.py --alpha 0.05 --design A
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.services.uncertainty import (  # noqa: E402
    accuracy,
    adaptive_calibration_error,
    apply_temperature,
    brier_score,
    evaluate_conformal,
    expected_calibration_error,
    fit_conformal_quantile,
    fit_temperature,
    negative_log_likelihood,
    reliability_curve,
    reliability_curve_quantile,
    risk_coverage_curve,
    selective_risk_at_coverage,
)

PUBLISHED_ENSEMBLE_ACCURACY = 96.75


def load_matrices(calib_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Load per-model probability matrices, labels and the manifest."""
    manifest_path = calib_dir / "probs_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run scripts/generate_test_probs.py first."
        )

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    labels = np.load(calib_dir / "test_labels.npy")
    probs = {
        arch: np.load(calib_dir / filename).astype(np.float64)
        for arch, filename in manifest["files"].items()
    }
    return probs, labels, manifest


def build_features(probs: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    """Concatenate per-model probabilities in the trained model order.

    Column layout must match what the meta-learner was fitted on; ``order``
    comes from the manifest, which came from ``ensemble_config.json``.
    """
    return np.concatenate([probs[arch] for arch in order], axis=1)


def stratified_split(
    labels: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into two stratified parts.

    Args:
        labels: Class labels used for stratification.
        fraction: Proportion assigned to the first part.
        seed: RNG seed.

    Returns:
        ``(first_idx, second_idx)``, both sorted.
    """
    rng = np.random.default_rng(seed)
    first: list[int] = []
    second: list[int] = []

    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        cut = int(round(fraction * len(idx)))
        first.extend(idx[:cut].tolist())
        second.extend(idx[cut:].tolist())

    return np.sort(np.array(first)), np.sort(np.array(second))


def verify_published_accuracy(
    features: np.ndarray, labels: np.ndarray, meta, calib_dir: Path
) -> dict:
    """Reproduce the documented 50/50 split and check the published accuracy.

    The notebooks used ``sklearn.model_selection.train_test_split`` with
    ``test_size=0.5, stratify=y, random_state=42``. Reproducing that exactly
    requires sklearn's own RNG, so it is imported here rather than reimplemented.
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    _, meta_test_idx = train_test_split(
        indices, test_size=0.5, stratify=labels, random_state=42
    )

    proba = meta.predict_proba(features[meta_test_idx])
    acc = accuracy(proba, labels[meta_test_idx])
    delta = acc - PUBLISHED_ENSEMBLE_ACCURACY

    print("\n=== Reproduction check: documented 50/50 split ===")
    print(f"  meta-test size        : {len(meta_test_idx)}")
    print(f"  published accuracy    : {PUBLISHED_ENSEMBLE_ACCURACY:.2f}%")
    print(f"  reproduced accuracy   : {acc:.2f}%")
    print(f"  difference            : {delta:+.2f} pp")

    if abs(delta) < 0.01:
        print("  VERDICT               : EXACT match")
    elif abs(delta) < 0.5:
        print("  VERDICT               : close match (within 0.5 pp)")
    else:
        print(
            "  VERDICT               : MISMATCH — investigate before trusting "
            "any calibration result below."
        )

    return {
        "published_accuracy": PUBLISHED_ENSEMBLE_ACCURACY,
        "reproduced_accuracy": round(acc, 4),
        "difference_pp": round(delta, 4),
        "meta_test_size": int(len(meta_test_idx)),
        "meta_test_indices_file": "verify_meta_test_indices.npy",
    }


def metric_block(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> dict:
    """Compute the full metric set for one probability matrix."""
    return {
        "accuracy": round(accuracy(probs, labels), 4),
        "ece": round(expected_calibration_error(probs, labels, n_bins), 6),
        "adaptive_ece": round(adaptive_calibration_error(probs, labels, n_bins), 6),
        "brier": round(brier_score(probs, labels), 6),
        "nll": round(negative_log_likelihood(probs, labels), 6),
        "mean_confidence": round(float(probs.max(axis=1).mean()) * 100, 4),
        "aurc": risk_coverage_curve(probs, labels)["aurc"],
        "selective_risk_at_80_coverage": round(
            selective_risk_at_coverage(probs, labels, 0.80), 4
        ),
    }


def save_reliability_plot(
    before: np.ndarray,
    after: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    n_bins: int,
) -> bool:
    """Render a before/after reliability diagram. Returns False if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Row 1: conventional equal-width bins, with sample counts overlaid so the
    # reader can see which points are statistically meaningful. Without the
    # counts this plot looks wildly miscalibrated in the mid range, when in
    # fact those bins hold only a couple of samples each.
    for ax, probs, title in (
        (axes[0][0], before, "Before — equal-width bins"),
        (axes[0][1], after, "After — equal-width bins"),
    ):
        curve = reliability_curve(probs, labels, n_bins)
        ece = expected_calibration_error(probs, labels, n_bins)

        counts = ax.twinx()
        counts.bar(
            curve["bin_confidence"],
            curve["bin_count"],
            width=1.0 / n_bins * 0.85,
            color="#cbd5e1",
            alpha=0.7,
            zorder=1,
        )
        counts.set_ylabel("Samples in bin", color="#64748b")
        counts.tick_params(axis="y", colors="#64748b")

        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, zorder=2)
        ax.plot(
            curve["bin_confidence"],
            curve["bin_accuracy"],
            "o-",
            color="#2563eb",
            zorder=3,
        )
        ax.set_zorder(counts.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Observed accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)

    # Row 2: equal-mass bins. Every point carries the same sample weight, so
    # this is the view to reason about.
    for ax, probs, title in (
        (axes[1][0], before, "Before — equal-mass bins"),
        (axes[1][1], after, "After — equal-mass bins"),
    ):
        curve = reliability_curve_quantile(probs, labels, n_bins)
        ace = adaptive_calibration_error(probs, labels, n_bins)
        per_bin = curve["bin_count"][0] if curve["bin_count"] else 0

        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.plot(
            curve["bin_confidence"],
            curve["bin_accuracy"],
            "s-",
            color="#059669",
        )
        ax.set_title(f"{title}\nAdaptive ECE = {ace:.4f}  (~{per_bin} samples/bin)")
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Observed accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Brain MRI stacking ensemble — reliability "
        f"(n = {len(labels)} held-out images)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=settings.CONFORMAL_ALPHA)
    parser.add_argument(
        "--design",
        choices=["A", "B", "both"],
        default="both",
        help="Which calibration design to fit; the winner is deployed.",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.40)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    calib_dir = Path(args.out) if args.out else Path(settings.CALIBRATION_DIR)
    calib_dir = calib_dir.resolve()

    probs, labels, manifest = load_matrices(calib_dir)
    order = manifest["model_order"]
    class_names = manifest["class_names"]
    n_classes = manifest["num_classes"]

    print(f"Calibration dir : {calib_dir}")
    print(f"Rows            : {len(labels)}")
    print(f"Model order     : {order}")
    print(f"Classes         : {class_names}")

    features = build_features(probs, order)
    print(f"Feature matrix  : {features.shape}")

    with open(Path(settings.MODELS_DIR) / "meta_model.pkl", "rb") as handle:
        meta = pickle.load(handle)

    if features.shape[1] != meta.n_features_in_:
        print(
            f"\nERROR: feature width {features.shape[1]} != meta.n_features_in_ "
            f"{meta.n_features_in_}",
            file=sys.stderr,
        )
        return 2

    verification = verify_published_accuracy(features, labels, meta, calib_dir)

    # --- Splits -----------------------------------------------------------
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    half_a_idx, half_b_idx = train_test_split(
        indices, test_size=0.5, stratify=labels, random_state=42
    )
    half_a_idx = np.sort(half_a_idx)
    half_b_idx = np.sort(half_b_idx)

    rel_cal, rel_test = stratified_split(
        labels[half_b_idx], args.calibration_fraction, args.seed
    )
    cal_idx = half_b_idx[rel_cal]
    test_idx = half_b_idx[rel_test]

    print("\n=== Splits ===")
    print(f"  half A (meta-train, spent) : {len(half_a_idx)}")
    print(f"  half B (untouched)         : {len(half_b_idx)}")
    print(f"    -> calibration           : {len(cal_idx)}")
    print(f"    -> final test            : {len(test_idx)}")

    overlap = set(cal_idx.tolist()) & set(test_idx.tolist())
    if overlap:
        print(f"\nERROR: calibration/test overlap of {len(overlap)} rows", file=sys.stderr)
        return 2

    raw_cal = meta.predict_proba(features[cal_idx])
    raw_test = meta.predict_proba(features[test_idx])

    print("\n=== Uncalibrated (final test split) ===")
    base_metrics = metric_block(raw_test, labels[test_idx], args.bins)
    for key, value in base_metrics.items():
        print(f"  {key:32s} {value}")

    results: dict[str, dict] = {}

    # --- Design B: scale the meta-learner's output -------------------------
    if args.design in ("B", "both"):
        temp_b = fit_temperature(raw_cal, labels[cal_idx])
        test_b = apply_temperature(raw_test, temp_b)
        results["B"] = {
            "temperature": round(temp_b, 6),
            "metrics": metric_block(test_b, labels[test_idx], args.bins),
        }
        print(f"\n=== Design B: temperature on meta output (T = {temp_b:.4f}) ===")
        for key, value in results["B"]["metrics"].items():
            print(f"  {key:32s} {value}")

    # --- Design A: scale each base model, refit the meta-learner -----------
    if args.design in ("A", "both"):
        from sklearn.linear_model import LogisticRegression

        per_model_temps: dict[str, float] = {}
        for arch in order:
            per_model_temps[arch] = fit_temperature(
                probs[arch][cal_idx], labels[cal_idx]
            )

        scaled = {
            arch: apply_temperature(probs[arch], per_model_temps[arch])
            for arch in order
        }
        scaled_features = build_features(scaled, order)

        # Refit on half A only, so the calibration and test splits stay clean.
        meta_a = LogisticRegression(max_iter=1000, C=1.0)
        meta_a.fit(scaled_features[half_a_idx], labels[half_a_idx])
        test_a = meta_a.predict_proba(scaled_features[test_idx])

        results["A"] = {
            "per_model_temperature": {k: round(v, 6) for k, v in per_model_temps.items()},
            "metrics": metric_block(test_a, labels[test_idx], args.bins),
        }
        print("\n=== Design A: temperature on bases + refit meta ===")
        print("  per-model T: " + ", ".join(f"{k}={v:.4f}" for k, v in per_model_temps.items()))
        for key, value in results["A"]["metrics"].items():
            print(f"  {key:32s} {value}")

    # --- Pick the design to deploy ----------------------------------------
    # Only design B can be applied to the *shipped* meta-learner: design A's
    # calibration is baked into a refitted meta, so adopting it means replacing
    # models/meta_model.pkl and re-verifying every published number. That is a
    # decision for a human, not for this script.
    #
    # So: deploy the best *deployable* design, and report A separately as a
    # quantified upgrade path. Silently shipping A's temperature against the old
    # meta-learner would be wrong -- those temperatures were fitted for
    # different inputs.
    best_overall = min(results, key=lambda d: results[d]["metrics"]["ece"])
    deployable = [d for d in results if d == "B"]
    chosen = min(deployable, key=lambda d: results[d]["metrics"]["ece"]) if deployable else "none"

    print(f"\n=== Design selection ===")
    print(f"  lowest ECE overall     : {best_overall} "
          f"(ECE {results[best_overall]['metrics']['ece']:.6f})")
    print(f"  deployed               : {chosen} "
          f"(applies to the shipped meta_model.pkl)")

    upgrade_note = None
    if best_overall != chosen and best_overall in results:
        a_metrics = results[best_overall]["metrics"]
        b_metrics = results[chosen]["metrics"] if chosen in results else base_metrics
        upgrade_note = (
            f"Design {best_overall} reaches ECE {a_metrics['ece']:.6f} vs "
            f"{b_metrics['ece']:.6f} for the deployed design, and improves AURC "
            f"{b_metrics['aurc']:.6f} -> {a_metrics['aurc']:.6f} and selective risk "
            f"@80% coverage {b_metrics['selective_risk_at_80_coverage']:.4f}% -> "
            f"{a_metrics['selective_risk_at_80_coverage']:.4f}%. Cost: accuracy "
            f"{b_metrics['accuracy']:.2f}% -> {a_metrics['accuracy']:.2f}%. Adopting it "
            f"requires refitting and re-exporting meta_model.pkl on "
            f"temperature-scaled base features, then re-verifying all baselines."
        )
        print(f"\n  UPGRADE AVAILABLE: {upgrade_note}")

    if chosen == "B":
        deployed_temp = results["B"]["temperature"]
        calibrated_cal = apply_temperature(raw_cal, deployed_temp)
        calibrated_test = apply_temperature(raw_test, deployed_temp)
    else:
        deployed_temp = 1.0
        calibrated_cal = raw_cal
        calibrated_test = raw_test

    per_model_temps_out = (
        results["A"]["per_model_temperature"] if "A" in results else None
    )

    # --- Conformal --------------------------------------------------------
    # Sweep first. Conformal prediction is *vacuous* whenever the target
    # coverage (1 - alpha) sits below the model's accuracy: every set is the
    # singleton top-1 and coverage simply equals accuracy. For a 97.7%-accurate
    # model that means alpha must drop below ~0.023 before the sets carry any
    # information. Recording the sweep makes that threshold explicit instead of
    # leaving a reader to wonder why alpha=0.10 does nothing.
    test_accuracy = base_metrics["accuracy"]
    vacuous_below = round((100.0 - test_accuracy) / 100.0, 6)

    sweep = []
    print("\n=== Conformal alpha sweep (informative once alpha < "
          f"{vacuous_below:.4f}) ===")
    print(f"  {'alpha':>7} {'q_hat':>9} {'target':>8} {'actual':>8} {'meanSz':>7}  sizes")
    for alpha_candidate in (0.20, 0.10, 0.05, 0.025, 0.02, 0.01, 0.005):
        q_c = fit_conformal_quantile(calibrated_cal, labels[cal_idx], alpha_candidate)
        eval_c = evaluate_conformal(
            calibrated_test, labels[test_idx], q_c, n_classes
        )
        row = {
            "alpha": alpha_candidate,
            "q_hat": round(float(q_c), 6),
            "target_coverage": round(1.0 - alpha_candidate, 4),
            "empirical_coverage": eval_c["empirical_coverage"],
            "mean_set_size": eval_c["mean_set_size"],
            "set_size_distribution": eval_c["set_size_distribution"],
            "degenerate": bool(q_c >= 1.0),
        }
        sweep.append(row)
        flag = "  <- degenerate (calibration set too small)" if row["degenerate"] else ""
        print(
            f"  {alpha_candidate:>7.3f} {q_c:>9.5f} {1 - alpha_candidate:>8.3f} "
            f"{eval_c['empirical_coverage']:>8.4f} {eval_c['mean_set_size']:>7.3f}  "
            f"{eval_c['set_size_distribution']}{flag}"
        )

    # Finite-sample floor: the quantile index rounds up to n below this alpha,
    # which forces q_hat to the maximum score and admits every class.
    alpha_floor = round(1.0 / (len(cal_idx) + 1), 6)
    print(f"  finite-sample floor with n_cal={len(cal_idx)}: alpha >= {alpha_floor:.6f}")

    q_hat = fit_conformal_quantile(calibrated_cal, labels[cal_idx], args.alpha)
    conformal_eval = evaluate_conformal(
        calibrated_test, labels[test_idx], q_hat, n_classes
    )

    print(f"\n=== Conformal prediction (alpha = {args.alpha}) ===")
    print(f"  q_hat                          {q_hat:.6f}")
    print(f"  threshold (1 - q_hat)          {1.0 - q_hat:.6f}")
    print(f"  target coverage                {1.0 - args.alpha:.4f}")
    print(f"  empirical coverage             {conformal_eval['empirical_coverage']:.4f}")
    print(f"  mean set size                  {conformal_eval['mean_set_size']:.4f}")
    print(f"  set size distribution          {conformal_eval['set_size_distribution']}")
    print(f"  band distribution              {conformal_eval['band_distribution']}")
    print("  coverage by class:")
    for cls_idx, cov in conformal_eval["coverage_by_class"].items():
        print(f"    {class_names[int(cls_idx)]:14s} {cov:.4f}")

    # --- Artefacts --------------------------------------------------------
    generated = datetime.now(timezone.utc).isoformat()
    split_info = {
        "total_rows": int(len(labels)),
        "meta_train_half": int(len(half_a_idx)),
        "calibration": int(len(cal_idx)),
        "final_test": int(len(test_idx)),
        "calibration_fraction_of_half_b": args.calibration_fraction,
        "seed": args.seed,
        "documented_split": "train_test_split(test_size=0.5, stratify=y, random_state=42)",
    }

    calibration_doc = {
        "design": chosen,
        "temperature": round(float(deployed_temp), 6),
        "per_model_temperature": per_model_temps_out,
        "n_bins": args.bins,
        "class_names": class_names,
        "model_order": order,
        "uncalibrated_metrics": base_metrics,
        "metrics": results[chosen]["metrics"] if chosen in results else base_metrics,
        "design_comparison": {d: results[d]["metrics"] for d in results},
        "best_overall_design": best_overall,
        "deployed_design": chosen,
        "upgrade_note": upgrade_note,
        "deployability_note": (
            "Design B scales the shipped meta-learner's output and is deployable "
            "as-is. Design A scales the base models and requires a refitted "
            "meta-learner, so it is reported but not deployed."
        ),
        "reproduction_check": verification,
        "split": split_info,
        "reliability_before": reliability_curve(raw_test, labels[test_idx], args.bins),
        "reliability_after": reliability_curve(
            calibrated_test, labels[test_idx], args.bins
        ),
        "generated": generated,
        "sklearn_warning": (
            "Meta-learner pickle was written with scikit-learn 1.6.1; this run used a "
            "different version if InconsistentVersionWarning appeared. Pin 1.6.1 for "
            "publication-grade numbers."
        ),
    }

    conformal_doc = {
        "method": "split conformal, LAC score s = 1 - p_true",
        "alpha": args.alpha,
        "target_coverage": round(1.0 - args.alpha, 6),
        "q_hat": round(float(q_hat), 6),
        "threshold": round(float(1.0 - q_hat), 6),
        "quantile_level": "ceil((n+1)(1-alpha))/n",
        "calibrated_with_design": chosen,
        "temperature": round(float(deployed_temp), 6),
        "class_names": class_names,
        "metrics": conformal_eval,
        "alpha_sweep": sweep,
        "vacuous_below_alpha": vacuous_below,
        "vacuous_note": (
            "Conformal sets are all singletons whenever 1-alpha < model accuracy; "
            "coverage then just equals accuracy and the layer carries no "
            f"information. With test accuracy {test_accuracy:.2f}% that means "
            f"alpha must be below {vacuous_below:.4f} to be informative."
        ),
        "finite_sample_alpha_floor": alpha_floor,
        "finite_sample_note": (
            f"With n_calibration={len(cal_idx)}, the quantile level "
            "ceil((n+1)(1-alpha))/n reaches 1.0 for alpha below "
            f"{alpha_floor:.6f}, forcing q_hat=1 and admitting every class."
        ),
        "band_rules": {
            "confident": "set size 1",
            "borderline": "set size 2",
            "indeterminate": "set size >= 3",
        },
        "split": split_info,
        "generated": generated,
    }

    with open(calib_dir / "calibration.json", "w", encoding="utf-8") as handle:
        json.dump(calibration_doc, handle, indent=2)
    with open(calib_dir / "conformal.json", "w", encoding="utf-8") as handle:
        json.dump(conformal_doc, handle, indent=2)

    np.save(calib_dir / "split_calibration_idx.npy", cal_idx)
    np.save(calib_dir / "split_test_idx.npy", test_idx)
    np.save(calib_dir / "split_meta_train_idx.npy", half_a_idx)

    plotted = save_reliability_plot(
        raw_test,
        calibrated_test,
        labels[test_idx],
        calib_dir / "reliability.png",
        args.bins,
    )

    print("\n=== Written ===")
    print(f"  {calib_dir / 'calibration.json'}")
    print(f"  {calib_dir / 'conformal.json'}")
    print(f"  split_{{calibration,test,meta_train}}_idx.npy")
    if plotted:
        print(f"  {calib_dir / 'reliability.png'}")
    else:
        print("  (matplotlib unavailable — reliability.png skipped)")

    ece_before = base_metrics["ece"]
    deployed_metrics = results[chosen]["metrics"] if chosen in results else base_metrics
    ece_after = deployed_metrics["ece"]
    reduction = (ece_before - ece_after) / ece_before * 100 if ece_before else 0.0
    print(
        f"\nDeployed: ECE {ece_before:.6f} -> {ece_after:.6f} "
        f"({reduction:+.1f}% reduction) | accuracy "
        f"{deployed_metrics['accuracy']:.2f}% (temperature scaling never changes argmax)"
    )
    if best_overall in results and best_overall != chosen:
        print(
            f"Best available ({best_overall}): ECE "
            f"{results[best_overall]['metrics']['ece']:.6f} "
            f"({(ece_before - results[best_overall]['metrics']['ece']) / ece_before * 100:+.1f}% "
            "reduction) — requires refitting the meta-learner."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
