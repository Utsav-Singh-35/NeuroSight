"""Generate per-base-model softmax probability matrices over a test set.

Why this exists
---------------
Calibration (temperature scaling), conformal prediction and every reliability
metric need the base models' probability outputs on held-out data. The original
project produced these as ``*_test_probs.npy`` files on Google Drive, but they
are not in the repository and the notebooks were saved with outputs cleared, so
none of those numbers can currently be reproduced here.

This script regenerates them locally and writes them into the repo so the
downstream calibration work is reproducible.

Critical detail: preprocessing
------------------------------
Probabilities are produced through the *exact* serving preprocessing path
(``app.services.preprocessor.preprocess_image`` on raw file bytes). Using a
different transform here -- even a subtly different resize -- would calibrate
the model against a distribution it never sees in production, which is a silent
way to make every calibration number wrong.

Label order
-----------
Class index order is the sorted directory order
(``glioma, meningioma, notumor, pituitary``), which matches
``torchvision.datasets.ImageFolder`` at training time and the ``class_names``
field in ``models/ensemble_config.json``. That equivalence is asserted at
startup rather than assumed.

Usage
-----
Run from ``backend/fastapi``::

    python scripts/generate_test_probs.py                 # full test set
    python scripts/generate_test_probs.py --limit 8       # quick smoke test
    python scripts/generate_test_probs.py --split Training
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make the app package importable when run as a script from backend/fastapi.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.services.ensemble import get_base_model, load_ensemble  # noqa: E402
from app.services.model_cache import get_device, model_cache  # noqa: E402
from app.services.preprocessor import preprocess_image  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def discover_images(split_dir: Path, limit_per_class: int | None) -> tuple[list[Path], np.ndarray, list[str]]:
    """Collect image paths and integer labels from a split directory.

    Args:
        split_dir: Directory containing one subdirectory per class.
        limit_per_class: Optional cap on images per class, for smoke tests.

    Returns:
        ``(paths, labels, class_dir_names)`` where ``labels[i]`` is the class
        index of ``paths[i]`` and ``class_dir_names`` is sorted.

    Raises:
        FileNotFoundError: The split directory does not exist or has no classes.
    """
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories in {split_dir}")

    paths: list[Path] = []
    labels: list[int] = []

    for idx, class_dir in enumerate(class_dirs):
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if limit_per_class is not None:
            files = files[:limit_per_class]
        paths.extend(files)
        labels.extend([idx] * len(files))

    return paths, np.asarray(labels, dtype=np.int64), [d.name for d in class_dirs]


def batched_probabilities(
    model: torch.nn.Module,
    paths: list[Path],
    num_classes: int,
    batch_size: int,
    device: torch.device,
    label: str,
) -> np.ndarray:
    """Run a model over every image and return its softmax probabilities.

    Args:
        model: Eval-mode model.
        paths: Image file paths, in a fixed order.
        num_classes: Output width.
        batch_size: Images per forward pass.
        device: Device to run on.
        label: Name used in progress output.

    Returns:
        Array of shape ``(len(paths), num_classes)``, dtype float32.
    """
    out = np.zeros((len(paths), num_classes), dtype=np.float32)
    started = time.time()

    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]

        # Preprocess through the serving path, one image at a time, then stack.
        tensors = []
        for path in chunk:
            with open(path, "rb") as handle:
                tensors.append(preprocess_image(handle.read()))
        batch = torch.cat(tensors, dim=0).to(device)

        with torch.inference_mode():
            probs = torch.softmax(model(batch), dim=1)
        out[start : start + len(chunk)] = probs.detach().cpu().numpy()

        done = start + len(chunk)
        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (len(paths) - done) / rate if rate > 0 else 0.0
        print(
            f"    {label}: {done}/{len(paths)} "
            f"({rate:.1f} img/s, ~{remaining:.0f}s left)",
            flush=True,
        )

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="Testing", help="Split directory name")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset root (default: <repo>/data/brainMRI)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, default=None, help="Images per class (smoke test)"
    )
    parser.add_argument(
        "--out", default=None, help=f"Output dir (default: {settings.CALIBRATION_DIR})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if an output file already exists",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data" / "brainMRI"
    split_dir = data_dir / args.split
    out_dir = Path(args.out) if args.out else Path(settings.CALIBRATION_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device        : {device}")
    print(f"Split         : {split_dir}")
    print(f"Output        : {out_dir.resolve()}")

    ensemble = load_ensemble(settings.MODELS_DIR)
    order = ensemble["order"]
    config_classes = ensemble["class_names"]
    num_classes = ensemble["num_classes"]

    paths, labels, class_dirs = discover_images(split_dir, args.limit)
    print(f"Images        : {len(paths)}")
    print(f"Folder classes: {class_dirs}")
    print(f"Config classes: {config_classes}")
    print(f"Model order   : {order}")

    # The whole point of saving these matrices is that column j means the same
    # class everywhere. Verify the folder order matches the trained order
    # instead of trusting it.
    normalised_folders = [c.lower().replace(" ", "").replace("_", "") for c in class_dirs]
    normalised_config = [c.lower().replace(" ", "").replace("_", "") for c in config_classes]
    if normalised_folders != normalised_config:
        print(
            "\nERROR: folder class order does not match ensemble_config.json "
            "class_names.\n"
            f"  folders: {class_dirs}\n"
            f"  config : {config_classes}\n"
            "Saving probabilities now would silently mislabel every column.",
            file=sys.stderr,
        )
        return 2

    manifest: dict = {
        "split": args.split,
        "num_images": len(paths),
        "num_classes": num_classes,
        "class_names": config_classes,
        "model_order": order,
        "batch_size": args.batch_size,
        "device": str(device),
        "limit_per_class": args.limit,
        "preprocessing": "app.services.preprocessor.preprocess_image (serving path)",
        "files": {},
    }

    np.save(out_dir / "test_labels.npy", labels)
    with open(out_dir / "test_files.json", "w", encoding="utf-8") as handle:
        json.dump([str(p.relative_to(data_dir)) for p in paths], handle, indent=2)

    total_started = time.time()
    for arch in order:
        filename = f"BRAIN_MRI_{arch.upper()}_test_probs.npy"
        existing = out_dir / filename

        # Resume support: a completed pass is not repeated. Useful because the
        # full run takes ~12 minutes and VGG-16 alone is a 512 MB checkpoint.
        if existing.exists() and not args.force:
            probs = np.load(existing)
            if probs.shape == (len(paths), num_classes):
                acc = float((probs.argmax(axis=1) == labels).mean() * 100)
                print(f"\n[{arch}] reusing {filename} (top-1 {acc:.2f}%)")
                manifest["files"][arch] = filename
                manifest.setdefault("base_accuracy", {})[arch] = round(acc, 2)
                continue
            print(
                f"\n[{arch}] {filename} has shape {probs.shape}, expected "
                f"{(len(paths), num_classes)} — regenerating."
            )

        print(f"\n[{arch}] loading weights...", flush=True)
        model = get_base_model(
            arch,
            ensemble["model_paths"][arch],
            num_classes=num_classes,
            module_id="brain_mri",
        )
        probs = batched_probabilities(
            model, paths, num_classes, args.batch_size, device, arch
        )

        # Release immediately. This is a batch job: each model is used for
        # exactly one pass and never reused, so keeping it cached would stack
        # all four (~645 MB for brain) in memory and start swapping on a
        # small-RAM machine. The serving path wants the opposite behaviour,
        # which is why eviction lives here and not in ModelCache.
        del model
        model_cache.clear()
        gc.collect()

        # Sanity: every row must be a probability distribution.
        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            print(
                f"  WARNING: {arch} rows do not sum to 1 "
                f"(min={row_sums.min():.5f}, max={row_sums.max():.5f})",
                file=sys.stderr,
            )

        filename = f"BRAIN_MRI_{arch.upper()}_test_probs.npy"
        np.save(out_dir / filename, probs)
        manifest["files"][arch] = filename

        acc = float((probs.argmax(axis=1) == labels).mean() * 100)
        print(f"  saved {filename}  top-1 accuracy on this split: {acc:.2f}%")
        manifest.setdefault("base_accuracy", {})[arch] = round(acc, 2)

    manifest["generation_seconds"] = round(time.time() - total_started, 1)
    with open(out_dir / "probs_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nDone in {manifest['generation_seconds']}s")
    print("Per-model top-1 accuracy:")
    for arch, acc in manifest.get("base_accuracy", {}).items():
        print(f"  {arch:14s} {acc:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
