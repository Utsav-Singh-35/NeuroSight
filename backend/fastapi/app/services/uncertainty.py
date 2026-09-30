"""Probability calibration and conformal prediction.

Two distinct problems are solved here, and they are easy to conflate:

**Calibration** asks whether a reported confidence *means* what it says -- if the
model says 90%, is it right 90% of the time? A model can be accurate and badly
calibrated at the same time. Fixed by temperature scaling, measured by ECE.

**Conformal prediction** asks for a *set* of plausible classes with a coverage
guarantee, rather than a single label plus a number. It is what lets the system
say "indeterminate between glioma and meningioma" with statistical backing
instead of an arbitrary confidence cutoff.

Everything here is pure numpy: no torch, no sklearn. That keeps it trivially
testable and cheap enough to run inside a request.

Note on temperature scaling from probabilities
----------------------------------------------
Temperature scaling is normally defined on logits, but this pipeline persists
*probabilities* (the meta-learner's ``predict_proba`` output). Because softmax
is invariant to an additive constant, ``log(p)`` is a valid logit surrogate:

    softmax(log(p) / T)

recovers exactly the temperature-scaled distribution. No logits required.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Guards log(0) without meaningfully shifting any realistic probability.
_EPS = 1e-12

# Set-size to human-facing band. Drives whether the UI asserts a finding,
# hedges, or abstains.
BAND_CONFIDENT = "confident"
BAND_BORDERLINE = "borderline"
BAND_INDETERMINATE = "indeterminate"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Expected Calibration Error using equal-width confidence bins.

    ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|

    Args:
        probs: ``(n, K)`` probability matrix, rows summing to 1.
        labels: ``(n,)`` integer true class indices.
        n_bins: Number of equal-width bins over [0, 1].

    Returns:
        ECE in [0, 1]. Lower is better; 0 means perfectly calibrated.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for lo, hi in zip(edges[:-1], edges[1:]):
        # Upper-inclusive on the final bin so confidence == 1.0 is counted.
        in_bin = (confidences > lo) & (confidences <= hi)
        if hi == edges[-1]:
            in_bin |= confidences == lo
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_acc = float(correct[in_bin].mean())
        bin_conf = float(confidences[in_bin].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict:
    """Per-bin accuracy vs confidence, for plotting a reliability diagram.

    Returns:
        Dict with ``bin_lower``, ``bin_upper``, ``bin_confidence``,
        ``bin_accuracy``, ``bin_count``. Empty bins are omitted.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    out: dict[str, list] = {
        "bin_lower": [],
        "bin_upper": [],
        "bin_confidence": [],
        "bin_accuracy": [],
        "bin_count": [],
    }

    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        out["bin_lower"].append(round(float(lo), 4))
        out["bin_upper"].append(round(float(hi), 4))
        out["bin_confidence"].append(round(float(confidences[in_bin].mean()), 6))
        out["bin_accuracy"].append(round(float(correct[in_bin].mean()), 6))
        out["bin_count"].append(count)

    return out


def adaptive_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """ECE using equal-**mass** (quantile) bins instead of equal-width bins.

    Why this matters here: a 97.7%-accurate ensemble puts almost all of its
    predictions above 0.95 confidence. With equal-width bins the mid-confidence
    bins hold a handful of samples each, so their observed accuracy swings
    between 0.0 and 1.0 on sampling noise alone and the reliability diagram
    looks alarming for no real reason.

    Equal-mass binning puts the same number of samples in every bin, which
    makes each point statistically meaningful. Report both: equal-width ECE is
    the conventional number, adaptive ECE is the honest one for a
    confidence-concentrated model.

    Args:
        probs: ``(n, K)`` probabilities.
        labels: ``(n,)`` true class indices.
        n_bins: Number of equal-mass bins.

    Returns:
        Adaptive ECE in [0, 1].
    """
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    n = len(labels)
    if n == 0:
        return 0.0

    order = np.argsort(confidences)
    conf_sorted = confidences[order]
    corr_sorted = correct[order]

    # np.array_split handles a non-divisible n by making some bins one larger.
    ace = 0.0
    for conf_chunk, corr_chunk in zip(
        np.array_split(conf_sorted, n_bins), np.array_split(corr_sorted, n_bins)
    ):
        if conf_chunk.size == 0:
            continue
        ace += (conf_chunk.size / n) * abs(
            float(corr_chunk.mean()) - float(conf_chunk.mean())
        )

    return float(ace)


def reliability_curve_quantile(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict:
    """Equal-mass counterpart of :func:`reliability_curve`.

    Every returned point represents the same number of samples, so the curve is
    interpretable rather than noise-dominated.
    """
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)

    order = np.argsort(confidences)
    conf_sorted = confidences[order]
    corr_sorted = correct[order]

    out: dict[str, list] = {
        "bin_confidence": [],
        "bin_accuracy": [],
        "bin_count": [],
    }

    for conf_chunk, corr_chunk in zip(
        np.array_split(conf_sorted, n_bins), np.array_split(corr_sorted, n_bins)
    ):
        if conf_chunk.size == 0:
            continue
        out["bin_confidence"].append(round(float(conf_chunk.mean()), 6))
        out["bin_accuracy"].append(round(float(corr_chunk.mean()), 6))
        out["bin_count"].append(int(conf_chunk.size))

    return out


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against one-hot truth.

    Unlike ECE this is a *proper scoring rule*, so it rewards sharpness as well
    as calibration. Reporting both is the honest thing to do -- a model can cut
    its ECE by becoming uniformly unsure, and Brier would catch that.

    Returns:
        Score in [0, 2]. Lower is better.
    """
    n, n_classes = probs.shape
    onehot = np.zeros((n, n_classes), dtype=np.float64)
    onehot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean NLL (cross-entropy) of the true class.

    Returns:
        NLL in [0, inf). Lower is better.
    """
    true_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true_probs, _EPS, 1.0))))


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy as a percentage."""
    return float((probs.argmax(axis=1) == labels).mean() * 100.0)


def risk_coverage_curve(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Selective-prediction risk as a function of coverage.

    Sorts predictions by confidence descending, then reports the error rate
    among the most-confident fraction. This is what justifies an abstention
    policy: it shows how much error is removed by declining the least-confident
    cases.

    Returns:
        Dict with ``coverage``, ``risk`` and ``aurc`` (area under the
        risk-coverage curve; lower is better).
    """
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)

    order = np.argsort(-confidences)
    errors = 1.0 - correct[order]
    cumulative_errors = np.cumsum(errors)
    counts = np.arange(1, len(errors) + 1)

    risk = cumulative_errors / counts
    coverage = counts / len(errors)

    return {
        "coverage": [round(float(c), 6) for c in coverage],
        "risk": [round(float(r), 6) for r in risk],
        "aurc": round(float(np.mean(risk)), 6),
    }


def selective_risk_at_coverage(
    probs: np.ndarray, labels: np.ndarray, target_coverage: float
) -> float:
    """Error rate among the most-confident ``target_coverage`` fraction.

    Args:
        probs: ``(n, K)`` probabilities.
        labels: ``(n,)`` true indices.
        target_coverage: Fraction of cases to answer, in (0, 1].

    Returns:
        Error rate as a percentage.
    """
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    order = np.argsort(-confidences)

    keep = max(1, int(round(target_coverage * len(labels))))
    kept = correct[order][:keep]
    return float((1.0 - kept.mean()) * 100.0)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Re-scale a probability matrix by temperature ``T``.

    ``T > 1`` softens (less confident), ``T < 1`` sharpens. ``T == 1`` is a
    no-op. Argmax is unchanged for any ``T > 0``, so **accuracy is never
    affected** -- only the confidence values are.

    Args:
        probs: ``(n, K)`` probabilities.
        temperature: Positive scalar.

    Returns:
        ``(n, K)`` re-scaled probabilities.

    Raises:
        ValueError: Non-positive temperature.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    if temperature == 1.0:
        return probs.astype(np.float64, copy=True)

    logits = np.log(np.clip(probs, _EPS, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    exps = np.exp(logits)
    return exps / exps.sum(axis=1, keepdims=True)


def fit_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    bounds: tuple[float, float] = (0.05, 10.0),
    tol: float = 1e-4,
) -> float:
    """Fit a single temperature by minimising NLL on held-out data.

    Uses golden-section search rather than pulling in scipy: the objective is
    smooth and unimodal in ``T`` over a sane range, and this keeps the module
    dependency-free.

    Args:
        probs: ``(n, K)`` uncalibrated probabilities from a held-out split.
        labels: ``(n,)`` true class indices for that split.
        bounds: Search interval for ``T``.
        tol: Interval width at which to stop.

    Returns:
        The fitted temperature.
    """
    low, high = bounds
    invphi = (math.sqrt(5.0) - 1.0) / 2.0  # 1/phi

    def objective(temp: float) -> float:
        return negative_log_likelihood(apply_temperature(probs, temp), labels)

    left = high - invphi * (high - low)
    right = low + invphi * (high - low)
    f_left, f_right = objective(left), objective(right)

    while (high - low) > tol:
        if f_left < f_right:
            high, right, f_right = right, left, f_left
            left = high - invphi * (high - low)
            f_left = objective(left)
        else:
            low, left, f_left = left, right, f_right
            right = low + invphi * (high - low)
            f_right = objective(right)

    return float((low + high) / 2.0)


# ---------------------------------------------------------------------------
# Conformal prediction
# ---------------------------------------------------------------------------

def fit_conformal_quantile(
    probs: np.ndarray, labels: np.ndarray, alpha: float = 0.10
) -> float:
    """Fit the split-conformal threshold quantile.

    Uses the LAC / "least ambiguous" nonconformity score
    ``s_i = 1 - p_{y_i}(x_i)`` and the finite-sample-corrected quantile level
    ``ceil((n + 1)(1 - alpha)) / n``. That correction is what makes the
    coverage guarantee hold at finite ``n`` rather than only asymptotically.

    Args:
        probs: ``(n, K)`` calibrated probabilities from a held-out split.
        labels: ``(n,)`` true class indices for that split.
        alpha: Target miscoverage (0.10 -> 90% coverage).

    Returns:
        ``q_hat`` in [0, 1]. Prediction sets are ``{k : p_k >= 1 - q_hat}``.

    Raises:
        ValueError: ``alpha`` outside (0, 1) or empty calibration set.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(labels)
    if n == 0:
        raise ValueError("Cannot fit a conformal quantile on an empty set")

    scores = 1.0 - probs[np.arange(n), labels]

    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    if level >= 1.0:
        # Too few calibration points to certify this alpha; the only honest
        # threshold admits every class.
        return 1.0

    return float(np.quantile(scores, level, method="higher"))


def prediction_set(
    probs_row: np.ndarray,
    q_hat: float,
    class_labels: list[str] | None = None,
    ensure_non_empty: bool = True,
) -> dict:
    """Build a conformal prediction set for one sample.

    Args:
        probs_row: ``(K,)`` calibrated probabilities for one image.
        q_hat: Threshold from :func:`fit_conformal_quantile`.
        class_labels: Optional display labels for the included classes.
        ensure_non_empty: When the threshold excludes every class, fall back to
            the single most likely one. A clinical UI showing "no possible
            diagnosis" is worse than showing the top-1 flagged as low evidence,
            so this defaults to on. It does slightly break the formal guarantee
            in that rare branch, which is why it is reported via
            ``fallback_applied``.

    Returns:
        Dict with ``indices``, ``labels``, ``size``, ``band``, ``threshold``
        and ``fallback_applied``.
    """
    threshold = 1.0 - q_hat
    indices = np.flatnonzero(probs_row >= threshold)

    fallback = False
    if indices.size == 0 and ensure_non_empty:
        indices = np.array([int(probs_row.argmax())])
        fallback = True

    size = int(indices.size)
    if size <= 1:
        band = BAND_CONFIDENT
    elif size == 2:
        band = BAND_BORDERLINE
    else:
        band = BAND_INDETERMINATE

    labels_out = None
    if class_labels is not None:
        labels_out = [class_labels[i] for i in indices]

    return {
        "indices": [int(i) for i in indices],
        "labels": labels_out,
        "size": size,
        "band": band,
        "threshold": round(float(threshold), 6),
        "fallback_applied": fallback,
    }


def evaluate_conformal(
    probs: np.ndarray,
    labels: np.ndarray,
    q_hat: float,
    n_classes: int | None = None,
) -> dict:
    """Measure empirical coverage and set size on a held-out split.

    Empirical coverage should land close to ``1 - alpha``. Materially below
    means the calibration split was unrepresentative; materially above means
    the sets are wider than necessary.

    Returns:
        Dict with ``empirical_coverage``, ``mean_set_size``,
        ``set_size_distribution``, ``coverage_by_class`` and
        ``band_distribution``.
    """
    threshold = 1.0 - q_hat
    included = probs >= threshold

    # Mirror the serving-time non-empty fallback so measured coverage reflects
    # what the system actually returns.
    empty_rows = ~included.any(axis=1)
    if empty_rows.any():
        included[empty_rows, probs[empty_rows].argmax(axis=1)] = True

    covered = included[np.arange(len(labels)), labels]
    sizes = included.sum(axis=1)

    k = int(n_classes if n_classes is not None else probs.shape[1])

    size_dist = {
        str(s): int((sizes == s).sum()) for s in range(1, k + 1) if (sizes == s).any()
    }

    coverage_by_class = {}
    for cls in range(k):
        mask = labels == cls
        if mask.any():
            coverage_by_class[str(cls)] = round(float(covered[mask].mean()), 6)

    bands = {
        BAND_CONFIDENT: int((sizes <= 1).sum()),
        BAND_BORDERLINE: int((sizes == 2).sum()),
        BAND_INDETERMINATE: int((sizes >= 3).sum()),
    }

    return {
        "empirical_coverage": round(float(covered.mean()), 6),
        "mean_set_size": round(float(sizes.mean()), 6),
        "set_size_distribution": size_dist,
        "coverage_by_class": coverage_by_class,
        "band_distribution": bands,
        "n_evaluated": int(len(labels)),
    }


# ---------------------------------------------------------------------------
# Serving-time bundle
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBundle:
    """Calibration + conformal artefacts used at request time.

    Loaded once at startup. When artefacts are absent the bundle stays
    ``available=False`` and callers must report raw probabilities with
    ``calibrated: false`` rather than silently implying calibration happened.
    """

    temperature: float = 1.0
    q_hat: float | None = None
    alpha: float = 0.10
    design: str = "none"
    available: bool = False
    metrics: dict = field(default_factory=dict)

    @classmethod
    def load(cls, calibration_dir: str | Path) -> CalibrationBundle:
        """Load artefacts from a directory, tolerating their absence.

        Args:
            calibration_dir: Directory holding ``calibration.json`` and
                ``conformal.json``.

        Returns:
            A bundle. ``available`` is False if calibration.json is missing.
        """
        base = Path(calibration_dir)
        cal_path = base / "calibration.json"
        conf_path = base / "conformal.json"

        if not cal_path.exists():
            return cls()

        with open(cal_path, "r", encoding="utf-8") as handle:
            cal = json.load(handle)

        bundle = cls(
            temperature=float(cal.get("temperature", 1.0)),
            design=str(cal.get("design", "unknown")),
            available=True,
            metrics={"calibration": cal.get("metrics", {})},
        )

        if conf_path.exists():
            with open(conf_path, "r", encoding="utf-8") as handle:
                conf = json.load(handle)
            bundle.q_hat = float(conf["q_hat"])
            bundle.alpha = float(conf.get("alpha", 0.10))
            bundle.metrics["conformal"] = conf.get("metrics", {})

        return bundle

    def calibrate(self, probs_row: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to one probability vector."""
        if not self.available or self.temperature == 1.0:
            return np.asarray(probs_row, dtype=np.float64)
        return apply_temperature(
            np.asarray(probs_row, dtype=np.float64).reshape(1, -1), self.temperature
        )[0]

    def describe(self, probs_row: np.ndarray, class_labels: list[str]) -> dict:
        """Produce the full uncertainty payload for one prediction.

        Returns:
            Dict with ``calibrated`` flag, ``calibrated_probabilities``,
            ``confidence``, and (when a conformal quantile is available)
            ``prediction_set``, ``uncertainty_band`` and ``coverage_guarantee``.
        """
        calibrated = self.calibrate(probs_row)
        top = int(calibrated.argmax())

        payload: dict = {
            "calibrated": bool(self.available),
            "temperature": round(float(self.temperature), 6),
            "calibration_design": self.design,
            "calibrated_probabilities": {
                class_labels[i]: round(float(calibrated[i]) * 100, 2)
                for i in range(len(class_labels))
            },
            "confidence": round(float(calibrated[top]) * 100, 2),
            "prediction": class_labels[top],
        }

        if self.q_hat is None:
            payload["prediction_set"] = None
            payload["uncertainty_band"] = None
            return payload

        pset = prediction_set(calibrated, self.q_hat, class_labels)
        payload["prediction_set"] = pset["labels"]
        payload["prediction_set_size"] = pset["size"]
        payload["uncertainty_band"] = pset["band"]
        payload["conformal_threshold"] = pset["threshold"]
        payload["conformal_fallback"] = pset["fallback_applied"]
        payload["coverage_guarantee"] = round(1.0 - self.alpha, 4)
        return payload
