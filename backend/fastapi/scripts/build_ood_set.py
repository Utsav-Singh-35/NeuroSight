"""Build an out-of-distribution evaluation set for the brain MRI module.

Three tiers, because they test different failures and should never be pooled
into one number:

``near_ood/`` — **real chest X-rays.** Genuine medical greyscale images of the
wrong anatomy. This is the clinically realistic failure: a user selects the
wrong scan type. It is also the *hardest* tier, because chest X-rays share the
low-level statistics of radiology images. Treat this tier as the headline
result.

``corrupted/`` — **real brain MRIs, degraded.** Heavy blur, heavy noise, severe
downsampling, near-blank frames. These test the pre-inference quality gate
rather than semantic novelty.

``far_ood/`` — **synthetic patterns.** Noise, gradients, checkerboards, solid
fills, geometric shapes. Trivially separable, so a high AUROC here proves very
little. Included as a sanity floor: a detector that *fails* on this tier is
broken.

Synthetic generation is seeded, so the set is reproducible without any download
or licensing concern. The honest caveat to state in the write-up: synthetic
far-OOD overstates detector performance, which is exactly why the three tiers
are reported separately.

Usage
-----
Run from ``backend/fastapi``::

    python scripts/build_ood_set.py
    python scripts/build_ood_set.py --count 80 --seed 7
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SIZE = 256  # generated before the pipeline's own 224 resize


def _save(arr: np.ndarray, path: Path) -> None:
    """Save a uint8 array as a JPEG."""
    Image.fromarray(arr.astype(np.uint8)).save(path, quality=92)


def make_synthetic(rng: np.random.Generator, kind: str) -> np.ndarray:
    """Generate one synthetic far-OOD image of the requested kind."""
    if kind == "uniform_noise":
        return rng.integers(0, 256, size=(SIZE, SIZE), dtype=np.uint8)

    if kind == "gaussian_noise":
        return np.clip(rng.normal(128, 45, size=(SIZE, SIZE)), 0, 255)

    if kind == "linear_gradient":
        angle = rng.uniform(0, np.pi)
        ys, xs = np.mgrid[0:SIZE, 0:SIZE]
        field = xs * np.cos(angle) + ys * np.sin(angle)
        field -= field.min()
        return field / max(field.max(), 1e-6) * 255.0

    if kind == "radial_gradient":
        cy, cx = rng.uniform(0.3, 0.7, size=2) * SIZE
        ys, xs = np.mgrid[0:SIZE, 0:SIZE]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        return np.clip(255.0 - dist / dist.max() * 255.0, 0, 255)

    if kind == "checkerboard":
        cell = int(rng.integers(8, 40))
        ys, xs = np.mgrid[0:SIZE, 0:SIZE]
        return (((xs // cell) + (ys // cell)) % 2) * 255.0

    if kind == "stripes":
        period = int(rng.integers(4, 24))
        ys, xs = np.mgrid[0:SIZE, 0:SIZE]
        axis = xs if rng.random() < 0.5 else ys
        return ((axis // period) % 2) * 255.0

    if kind == "solid":
        return np.full((SIZE, SIZE), float(rng.integers(0, 256)))

    if kind == "shapes":
        canvas = np.full((SIZE, SIZE), float(rng.integers(0, 80)), dtype=np.float32)
        for _ in range(int(rng.integers(3, 9))):
            shade = float(rng.integers(90, 256))
            if rng.random() < 0.5:
                centre = tuple(int(v) for v in rng.integers(30, SIZE - 30, size=2))
                cv2.circle(canvas, centre, int(rng.integers(10, 50)), shade, -1)
            else:
                pt1 = tuple(int(v) for v in rng.integers(0, SIZE - 60, size=2))
                pt2 = tuple(int(v) for v in (np.array(pt1) + rng.integers(20, 70, size=2)))
                cv2.rectangle(canvas, pt1, pt2, shade, -1)
        return canvas

    if kind == "text_like":
        canvas = np.full((SIZE, SIZE), 245.0, dtype=np.float32)
        for row in range(20, SIZE - 20, 22):
            x = 15
            while x < SIZE - 30:
                width = int(rng.integers(6, 26))
                cv2.rectangle(canvas, (x, row), (x + width, row + 9), 25.0, -1)
                x += width + int(rng.integers(4, 12))
        return canvas

    if kind == "sinusoid":
        freq = rng.uniform(0.02, 0.2)
        ys, xs = np.mgrid[0:SIZE, 0:SIZE]
        return (np.sin(xs * freq) * np.cos(ys * freq) + 1.0) * 127.5

    raise ValueError(f"Unknown synthetic kind: {kind}")


SYNTHETIC_KINDS = [
    "uniform_noise",
    "gaussian_noise",
    "linear_gradient",
    "radial_gradient",
    "checkerboard",
    "stripes",
    "solid",
    "shapes",
    "text_like",
    "sinusoid",
]


def make_corrupted(arr: np.ndarray, rng: np.random.Generator, kind: str) -> np.ndarray:
    """Degrade a real MRI to test the quality gate."""
    if kind == "heavy_blur":
        return cv2.GaussianBlur(arr, (0, 0), sigmaX=9.0)

    if kind == "extreme_blur":
        return cv2.GaussianBlur(arr, (0, 0), sigmaX=20.0)

    if kind == "heavy_noise":
        return np.clip(arr + rng.normal(0, 70, size=arr.shape), 0, 255)

    if kind == "downsampled":
        small = cv2.resize(arr, (14, 14), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, arr.shape[::-1], interpolation=cv2.INTER_NEAREST)

    if kind == "near_blank":
        # Collapse contrast to almost nothing without going fully constant.
        return np.clip(arr.mean() + (arr - arr.mean()) * 0.02, 0, 255)

    raise ValueError(f"Unknown corruption kind: {kind}")


CORRUPTION_KINDS = ["heavy_blur", "extreme_blur", "heavy_noise", "downsampled", "near_blank"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=60, help="Synthetic image count")
    parser.add_argument("--corrupt-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Default: <repo>/data/ood")
    parser.add_argument("--clean", action="store_true", help="Wipe the output dir first")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(args.out) if args.out else repo_root / "data" / "ood"

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    far_dir = out_dir / "far_ood"
    near_dir = out_dir / "near_ood"
    corrupt_dir = out_dir / "corrupted"
    for directory in (far_dir, near_dir, corrupt_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    manifest: dict = {"seed": args.seed, "tiers": {}}

    # --- far OOD: synthetic -------------------------------------------------
    counts: dict[str, int] = {}
    for index in range(args.count):
        kind = SYNTHETIC_KINDS[index % len(SYNTHETIC_KINDS)]
        arr = make_synthetic(rng, kind)
        _save(arr, far_dir / f"{kind}_{index:03d}.jpg")
        counts[kind] = counts.get(kind, 0) + 1

    manifest["tiers"]["far_ood"] = {
        "description": "Synthetic patterns. Trivially separable; sanity floor only.",
        "count": args.count,
        "by_kind": counts,
        "caveat": (
            "A high AUROC on this tier proves little. Report near_ood as the "
            "headline result."
        ),
    }
    print(f"far_ood   : {args.count} synthetic images -> {far_dir}")

    # --- near OOD: real chest X-rays ----------------------------------------
    chest_src = repo_root / "data" / "samples" / "ChestXray"
    near_count = 0
    if chest_src.is_dir():
        for path in sorted(chest_src.iterdir()):
            if path.suffix.lower() in IMAGE_EXTS:
                shutil.copy2(path, near_dir / f"chest_{path.name}")
                near_count += 1

    manifest["tiers"]["near_ood"] = {
        "description": "Real chest X-rays: correct modality family, wrong anatomy.",
        "count": near_count,
        "source": "data/samples/ChestXray",
        "note": (
            "Hardest and most clinically realistic tier -- this is what happens "
            "when a user picks the wrong scan type."
        ),
    }
    print(f"near_ood  : {near_count} real chest X-rays -> {near_dir}")
    if near_count < 20:
        print(
            f"  NOTE: only {near_count} near-OOD images available. AUROC on this "
            "tier will have a wide confidence interval; add more chest X-rays "
            "for a publication-grade number."
        )

    # --- corrupted: degraded real MRIs -------------------------------------
    mri_src = repo_root / "data" / "brainMRI" / "Testing"
    corrupt_counts: dict[str, int] = {}
    made = 0

    if mri_src.is_dir():
        sources: list[Path] = []
        for class_dir in sorted(d for d in mri_src.iterdir() if d.is_dir()):
            files = sorted(
                p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
            )
            sources.extend(files[: max(1, args.corrupt_count // 4)])

        for index, src in enumerate(sources):
            if made >= args.corrupt_count:
                break
            kind = CORRUPTION_KINDS[index % len(CORRUPTION_KINDS)]
            arr = np.asarray(Image.open(src).convert("L"), dtype=np.float32)
            _save(make_corrupted(arr, rng, kind), corrupt_dir / f"{kind}_{index:03d}.jpg")
            corrupt_counts[kind] = corrupt_counts.get(kind, 0) + 1
            made += 1

    manifest["tiers"]["corrupted"] = {
        "description": "Real brain MRIs degraded; tests the quality gate, not novelty.",
        "count": made,
        "by_kind": corrupt_counts,
    }
    print(f"corrupted : {made} degraded MRIs -> {corrupt_dir}")

    with open(out_dir / "ood_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nmanifest  : {out_dir / 'ood_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
