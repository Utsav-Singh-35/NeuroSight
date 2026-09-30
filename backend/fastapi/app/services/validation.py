"""Input validation: image quality checks and out-of-distribution detection.

Two independent layers, deliberately kept separate because they fail in
different ways and have different costs.

**Quality assessment** is cheap, deterministic and runs before any model: is the
file decodable, is it large enough, is it hopelessly blurred, is it a blank
frame? These catch operator error.

**Out-of-distribution detection** is statistical and runs *after* inference,
because the signals are derived from model outputs: a chest X-ray fed to the
brain module produces a confident-looking tumour class, and nothing about the
pixels alone reveals that. This is the layer that addresses the closed-world
softmax problem -- the model's probabilities always sum to 1 over its four
classes, so it can never natively say "this is not a brain MRI".

Five detectors are implemented so they can be compared rather than asserted:

1. **MSP** (max softmax probability) -- the standard baseline.
2. **Predictive entropy** -- spread of the whole distribution, not just the top.
3. **Energy** -- ``-T log sum exp(z/T)`` over the meta-learner's *unnormalised*
   decision scores. Energy cannot be computed from softmax probabilities (they
   sum to 1, so the log-sum-exp is constant); it needs real logits, which
   ``LogisticRegression.decision_function`` provides.
4. **Ensemble disagreement** -- the fraction of base models whose top-1 differs
   from the ensemble's. This signal is *already computed* on every request and
   currently discarded by the response schema, so it is free.
5. **Conformal set size** -- a set containing every class is itself a strong
   novelty indicator.

All thresholds are **fitted from data** (percentiles of the in-distribution
population), never hardcoded guesses.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Hard floor: below this the 224x224 upscale invents almost all of its detail.
MIN_DIMENSION = 64
# Above this an image is so elongated that the squash-resize destroys anatomy.
MAX_ASPECT_RATIO = 3.0

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------

@dataclass
class QualityThresholds:
    """Data-driven quality gates.

    Defaults are permissive placeholders; real values come from
    :func:`fit_quality_thresholds` run over the training distribution and are
    persisted in ``ood.json``.
    """

    min_dimension: int = MIN_DIMENSION
    max_aspect_ratio: float = MAX_ASPECT_RATIO
    min_laplacian_variance: float = 0.0
    min_pixel_std: float = 0.0


def image_quality_features(image_bytes: bytes) -> dict:
    """Extract raw quality measurements from an image.

    Args:
        image_bytes: Raw file bytes.

    Returns:
        Dict with ``width``, ``height``, ``aspect_ratio``,
        ``laplacian_variance``, ``pixel_std``, ``pixel_mean``,
        ``unique_intensity_ratio``.

    Raises:
        ValueError: The image cannot be decoded.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        image = Image.open(BytesIO(image_bytes))
        image = image.convert("L")  # quality is an intensity property
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Image could not be decoded") from exc

    width, height = image.size
    arr = np.asarray(image, dtype=np.float32)

    # Variance of the Laplacian is the standard sharpness proxy: a blurred
    # image has little high-frequency content, so its second derivative is flat.
    laplacian_variance = float(cv2.Laplacian(arr, cv2.CV_32F).var())

    return {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": round(float(max(width, height) / max(1, min(width, height))), 4),
        "laplacian_variance": round(laplacian_variance, 4),
        "pixel_std": round(float(arr.std()), 4),
        "pixel_mean": round(float(arr.mean()), 4),
        "unique_intensity_ratio": round(
            float(len(np.unique(arr.astype(np.uint8))) / 256.0), 4
        ),
    }


def assess_quality(
    image_bytes: bytes, thresholds: QualityThresholds | None = None
) -> dict:
    """Run the quality gate.

    Args:
        image_bytes: Raw file bytes.
        thresholds: Fitted gates; defaults to permissive.

    Returns:
        Dict with ``passed``, ``features``, ``checks`` (per-check booleans) and
        ``reasons`` (human-readable failures).
    """
    gates = thresholds or QualityThresholds()

    try:
        features = image_quality_features(image_bytes)
    except ValueError as exc:
        return {
            "passed": False,
            "features": {},
            "checks": {"decodable": False},
            "reasons": [str(exc)],
        }

    checks = {
        "decodable": True,
        "resolution": min(features["width"], features["height"]) >= gates.min_dimension,
        "aspect_ratio": features["aspect_ratio"] <= gates.max_aspect_ratio,
        "sharpness": features["laplacian_variance"] >= gates.min_laplacian_variance,
        "contrast": features["pixel_std"] >= gates.min_pixel_std,
    }

    reasons = []
    if not checks["resolution"]:
        reasons.append(
            f"Resolution {features['width']}x{features['height']} is below the "
            f"{gates.min_dimension}px minimum."
        )
    if not checks["aspect_ratio"]:
        reasons.append(
            f"Aspect ratio {features['aspect_ratio']:.2f} exceeds "
            f"{gates.max_aspect_ratio:.2f}; the image would be badly distorted "
            "by the square resize."
        )
    if not checks["sharpness"]:
        reasons.append(
            f"Image appears excessively blurred (Laplacian variance "
            f"{features['laplacian_variance']:.1f} < "
            f"{gates.min_laplacian_variance:.1f})."
        )
    if not checks["contrast"]:
        reasons.append(
            f"Image has almost no tonal variation (std "
            f"{features['pixel_std']:.1f} < {gates.min_pixel_std:.1f}); it may "
            "be blank or corrupted."
        )

    return {
        "passed": all(checks.values()),
        "features": features,
        "checks": checks,
        "reasons": reasons,
    }


def fit_quality_thresholds(
    features: list[dict], percentile: float = 1.0
) -> QualityThresholds:
    """Fit quality gates from the in-distribution population.

    Uses a low percentile rather than the minimum so a single pathological
    training image cannot drag the gate to zero.

    Args:
        features: Output of :func:`image_quality_features` for ID images.
        percentile: Percentile of the ID distribution to use as the floor.

    Returns:
        Fitted thresholds.
    """
    lap = np.array([f["laplacian_variance"] for f in features], dtype=np.float64)
    std = np.array([f["pixel_std"] for f in features], dtype=np.float64)

    return QualityThresholds(
        min_dimension=MIN_DIMENSION,
        max_aspect_ratio=MAX_ASPECT_RATIO,
        min_laplacian_variance=float(np.percentile(lap, percentile)),
        min_pixel_std=float(np.percentile(std, percentile)),
    )


# ---------------------------------------------------------------------------
# OOD scores
# ---------------------------------------------------------------------------

def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row, in nats.

    Higher means the model is spreading probability mass, which correlates with
    novelty. ``(n, K) -> (n,)``.
    """
    p = np.clip(probs, _EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def max_softmax_probability(probs: np.ndarray) -> np.ndarray:
    """Maximum class probability per row. Lower means more novel."""
    return probs.max(axis=1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Free-energy OOD score over **unnormalised** logits.

    ``E(x) = -T * log sum_j exp(z_j / T)``

    Lower energy means the input is more in-distribution. This must not be fed
    softmax probabilities: those sum to 1, which makes the log-sum-exp constant
    and the score useless. Use a model's raw decision scores.

    Args:
        logits: ``(n, K)`` unnormalised scores.
        temperature: Softening parameter.

    Returns:
        ``(n,)`` energies.
    """
    scaled = logits / temperature
    # logsumexp with the max subtracted for numerical stability.
    peak = scaled.max(axis=1, keepdims=True)
    lse = peak[:, 0] + np.log(np.exp(scaled - peak).sum(axis=1))
    return -temperature * lse


def ensemble_disagreement(
    base_probs: dict[str, np.ndarray], ensemble_pred: np.ndarray
) -> np.ndarray:
    """Fraction of base models whose top-1 differs from the ensemble's.

    Free novelty signal: the values are already computed during every ensemble
    inference and then discarded by the response schema. On genuinely
    out-of-distribution input the base models tend to scatter, because nothing
    in the image supports a common answer.

    Args:
        base_probs: ``arch -> (n, K)`` probability matrices.
        ensemble_pred: ``(n,)`` ensemble predicted indices.

    Returns:
        ``(n,)`` disagreement fraction in [0, 1].
    """
    if not base_probs:
        return np.zeros(len(ensemble_pred), dtype=np.float64)

    votes = np.stack([p.argmax(axis=1) for p in base_probs.values()], axis=1)
    return (votes != ensemble_pred[:, None]).mean(axis=1)


def mean_pairwise_divergence(base_probs: dict[str, np.ndarray]) -> np.ndarray:
    """Mean pairwise Jensen-Shannon divergence between base model outputs.

    A finer-grained disagreement measure than vote counting: it responds even
    when models agree on the argmax but differ sharply in their distributions.

    Args:
        base_probs: ``arch -> (n, K)`` probability matrices.

    Returns:
        ``(n,)`` mean pairwise JS divergence in nats.
    """
    mats = list(base_probs.values())
    if len(mats) < 2:
        return np.zeros(len(mats[0]) if mats else 0, dtype=np.float64)

    total = np.zeros(len(mats[0]), dtype=np.float64)
    pairs = 0

    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            p = np.clip(mats[i], _EPS, 1.0)
            q = np.clip(mats[j], _EPS, 1.0)
            m = 0.5 * (p + q)
            kl_pm = np.sum(p * np.log(p / m), axis=1)
            kl_qm = np.sum(q * np.log(q / m), axis=1)
            total += 0.5 * (kl_pm + kl_qm)
            pairs += 1

    return total / pairs


# ---------------------------------------------------------------------------
# Detector evaluation
# ---------------------------------------------------------------------------

def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC for separating OOD from ID, where a *higher* score means OOD.

    Computed via the Mann-Whitney U identity, so no sklearn dependency and ties
    are handled correctly by averaging ranks.

    Returns:
        AUROC in [0, 1]. 0.5 is chance.
    """
    combined = np.concatenate([id_scores, ood_scores])
    ranks = _average_ranks(combined)
    n_id = len(id_scores)
    n_ood = len(ood_scores)
    if n_id == 0 or n_ood == 0:
        return float("nan")

    rank_sum_ood = ranks[n_id:].sum()
    u = rank_sum_ood - n_ood * (n_ood + 1) / 2.0
    return float(u / (n_id * n_ood))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]

    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = average
        i = j + 1

    return ranks


def fpr_at_tpr(
    id_scores: np.ndarray, ood_scores: np.ndarray, target_tpr: float = 0.95
) -> dict:
    """False-positive rate on ID data at a target OOD detection rate.

    "How much valid input do we wrongly reject in order to catch 95% of
    garbage?" -- the number that decides whether a gate is deployable.

    Args:
        id_scores: ID scores (higher = more OOD).
        ood_scores: OOD scores.
        target_tpr: Fraction of OOD that must be caught.

    Returns:
        Dict with ``threshold``, ``fpr``, ``achieved_tpr``.
    """
    # To catch target_tpr of OOD, threshold at the (1 - target_tpr) quantile of
    # OOD scores, since higher means more OOD.
    threshold = float(np.quantile(ood_scores, 1.0 - target_tpr))
    fpr = float((id_scores >= threshold).mean())
    achieved = float((ood_scores >= threshold).mean())

    return {
        "threshold": round(threshold, 6),
        "fpr": round(fpr, 6),
        "achieved_tpr": round(achieved, 6),
    }


def evaluate_detector(
    id_scores: np.ndarray, ood_scores: np.ndarray, target_tpr: float = 0.95
) -> dict:
    """Full evaluation of one OOD detector."""
    at_tpr = fpr_at_tpr(id_scores, ood_scores, target_tpr)
    return {
        "auroc": round(auroc(id_scores, ood_scores), 6),
        f"fpr_at_{int(target_tpr * 100)}_tpr": at_tpr["fpr"],
        "threshold": at_tpr["threshold"],
        "achieved_tpr": at_tpr["achieved_tpr"],
        "id_mean": round(float(id_scores.mean()), 6),
        "ood_mean": round(float(ood_scores.mean()), 6),
        "n_id": int(len(id_scores)),
        "n_ood": int(len(ood_scores)),
    }


# ---------------------------------------------------------------------------
# Serving-time detector
# ---------------------------------------------------------------------------

@dataclass
class OODDetector:
    """Serving-time input validation, loaded from ``ood.json``.

    When the artefact is missing the detector stays ``available=False`` and
    callers must report ``validated: false`` rather than implying a check
    happened.
    """

    quality: QualityThresholds = field(default_factory=QualityThresholds)
    primary_detector: str = "entropy"
    threshold: float | None = None
    metrics: dict = field(default_factory=dict)
    # The tier the threshold was selected on. Needed to look up the operating
    # point's measured error rates, since each tier fits its own threshold.
    selection_tier: str | None = None
    available: bool = False

    @property
    def false_positive_rate(self) -> float | None:
        """Measured FPR of the deployed operating point, or None if unknown.

        Read from the fitted artefact rather than hard-coded so the number
        quoted to the user cannot drift away from the number that was measured.
        """
        if not self.selection_tier:
            return None
        tier = self.metrics.get(self.selection_tier, {})
        entry = tier.get(self.primary_detector, {})
        rate = entry.get("fpr_at_95_tpr")
        return float(rate) if rate is not None else None

    @classmethod
    def load(cls, calibration_dir: str | Path) -> OODDetector:
        """Load fitted thresholds, tolerating absence."""
        path = Path(calibration_dir) / "ood.json"
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)

        gates = doc.get("quality_thresholds", {})
        return cls(
            quality=QualityThresholds(
                min_dimension=int(gates.get("min_dimension", MIN_DIMENSION)),
                max_aspect_ratio=float(gates.get("max_aspect_ratio", MAX_ASPECT_RATIO)),
                min_laplacian_variance=float(gates.get("min_laplacian_variance", 0.0)),
                min_pixel_std=float(gates.get("min_pixel_std", 0.0)),
            ),
            primary_detector=str(doc.get("primary_detector", "entropy")),
            threshold=(
                float(doc["primary_threshold"])
                if doc.get("primary_threshold") is not None
                else None
            ),
            metrics=doc.get("detectors", {}),
            selection_tier=(
                str(doc["selection_tier"]) if doc.get("selection_tier") else None
            ),
            available=True,
        )

    def check_quality(self, image_bytes: bytes) -> dict:
        """Run the pre-inference quality gate."""
        result = assess_quality(image_bytes, self.quality)
        result["validated"] = bool(self.available)
        return result

    def score(
        self,
        meta_probs_row: np.ndarray,
        meta_logits_row: np.ndarray | None = None,
        base_probs_row: dict[str, np.ndarray] | None = None,
    ) -> dict:
        """Compute post-inference novelty signals for one prediction.

        Args:
            meta_probs_row: ``(K,)`` ensemble probabilities.
            meta_logits_row: ``(K,)`` unnormalised meta scores, for energy.
            base_probs_row: ``arch -> (K,)`` base model probabilities.

        Returns:
            Dict of scores plus ``is_ood`` and ``ood_reason`` when a threshold
            is configured.
        """
        probs = np.asarray(meta_probs_row, dtype=np.float64).reshape(1, -1)

        scores: dict = {
            "msp": round(float(max_softmax_probability(probs)[0]), 6),
            "entropy": round(float(predictive_entropy(probs)[0]), 6),
        }

        if meta_logits_row is not None:
            logits = np.asarray(meta_logits_row, dtype=np.float64).reshape(1, -1)
            scores["energy"] = round(float(energy_score(logits)[0]), 6)

        if base_probs_row:
            stacked = {k: np.asarray(v, dtype=np.float64).reshape(1, -1)
                       for k, v in base_probs_row.items()}
            pred = probs.argmax(axis=1)
            scores["disagreement"] = round(
                float(ensemble_disagreement(stacked, pred)[0]), 6
            )
            # Key must match the detector names used by scripts/fit_ood.py and
            # persisted as `primary_detector` in ood.json, otherwise the
            # threshold lookup below silently finds nothing and no input is
            # ever flagged.
            scores["js_divergence"] = round(
                float(mean_pairwise_divergence(stacked)[0]), 6
            )

        if self.available and self.threshold is not None:
            value = scores.get(self.primary_detector)
            if value is None:
                logger.warning(
                    "Configured OOD detector '%s' produced no score; available "
                    "keys were %s. Input will not be flagged.",
                    self.primary_detector,
                    sorted(scores.keys()),
                )
            else:
                # All configured detectors are oriented so higher = more OOD.
                scores["is_ood"] = bool(value >= self.threshold)
                scores["ood_detector"] = self.primary_detector
                scores["ood_threshold"] = self.threshold
                if scores["is_ood"]:
                    # Deliberately phrased as a flag, not a verdict. At this
                    # operating point the detector also fires on 7.3% of
                    # in-distribution test images (fpr_at_95_tpr = 0.072917 on
                    # the near-OOD tier), so asserting "this is not a brain MRI"
                    # would overstate what the measurement supports.
                    fpr = self.false_positive_rate
                    rate = (
                        f" At this threshold the detector also fires on "
                        f"{fpr * 100:.1f}% of in-distribution test images, so a "
                        f"flag is a prompt to check the input rather than proof "
                        f"the input is invalid."
                        if fpr is not None
                        else ""
                    )
                    scores["ood_reason"] = (
                        f"Novelty flag: {self.primary_detector} = {value:.4f}, "
                        f"above the fitted threshold {self.threshold:.4f}. The "
                        f"base models disagree more than they typically do on "
                        f"brain MRI the ensemble was trained on.{rate}"
                    )

        return scores
