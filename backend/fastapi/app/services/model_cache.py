"""Process-wide model cache with explicit device selection.

Why this exists
---------------
The original pipeline called ``timm.create_model`` + ``torch.load`` on every
request, then deleted the model and ran ``gc.collect()``. For the brain MRI
ensemble that meant re-reading roughly 645 MB of checkpoints from disk per
prediction (VGG-16 alone is 512 MB), which put a single ``/predict`` call at
about 10-15 seconds on CPU. ``/gradcam`` was worse: it ran the whole ensemble
to find the agreeing model, then loaded that model a *second* time.

Every weight load also hardcoded ``map_location="cpu"`` and nothing ever called
``.to(device)``, so the service could not use a GPU even when one was present.

This module fixes both: models are built once, kept in memory, and placed on a
single chosen device. Inference then costs only the forward pass.

Trade-off
---------
Memory is now held for the lifetime of the process instead of being reclaimed
after each request (~645 MB for brain, ~135 MB for chest). That is the correct
trade for a service: predictable latency beats a low idle footprint. Use
``PRELOAD_MODULES`` to control which modules pay that cost eagerly at startup
versus lazily on first request.
"""

from __future__ import annotations

import logging
import threading

import timm
import torch

from app.config import settings

logger = logging.getLogger(__name__)

# Resolved once, then reused. Kept module-level so every code path agrees on
# the device rather than each call site deciding for itself.
_device: torch.device | None = None
_device_lock = threading.Lock()


def get_device() -> torch.device:
    """Return the torch device used for all inference.

    Honours ``settings.FORCE_CPU`` so a GPU box can still be pinned to CPU for
    reproducible benchmarking.

    Returns:
        The selected ``torch.device``.
    """
    global _device
    if _device is not None:
        return _device

    with _device_lock:
        if _device is None:
            if getattr(settings, "FORCE_CPU", False):
                chosen = torch.device("cpu")
                reason = "FORCE_CPU=true"
            elif torch.cuda.is_available():
                chosen = torch.device("cuda")
                reason = f"cuda available ({torch.cuda.get_device_name(0)})"
            else:
                chosen = torch.device("cpu")
                reason = "no cuda device detected"
            _device = chosen
            logger.info("Inference device selected: %s (%s)", chosen, reason)
    return _device


class ModelCache:
    """Caches eval-mode models keyed by ``"<module_id>:<arch>"``.

    Thread-safe: FastAPI dispatches synchronous endpoints to a worker
    threadpool, so two concurrent requests can race to populate the same entry.
    A re-entrant lock guards insertion while reads stay lock-free once warm.
    """

    def __init__(self) -> None:
        self._models: dict[str, torch.nn.Module] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(module_id: str, arch: str) -> str:
        return f"{module_id}:{arch}"

    def get(
        self,
        module_id: str,
        arch: str,
        timm_name: str,
        weights_path: str,
        num_classes: int,
    ) -> torch.nn.Module:
        """Return a ready-to-use model, building and caching it if needed.

        Args:
            module_id: Owning module, e.g. ``"brain_mri"``.
            arch: Short architecture key, e.g. ``"efficientnet"``.
            timm_name: timm architecture name, e.g. ``"efficientnet_b0"``.
            weights_path: Path to the ``.pth`` state dict.
            num_classes: Classifier output width. Supplied by the caller rather
                than hardcoded, so 3-class chest and 4-class brain both work.

        Returns:
            An eval-mode model on the selected device.

        Raises:
            FileNotFoundError: Weights missing.
            RuntimeError: State dict does not match the architecture.
        """
        key = self._key(module_id, arch)

        cached = self._models.get(key)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._models.get(key)
            if cached is not None:
                return cached

            model = self._build(timm_name, weights_path, num_classes)
            self._models[key] = model
            logger.info(
                "Model cached: %s (%s, %d classes) on %s",
                key,
                timm_name,
                num_classes,
                get_device(),
            )
            return model

    def _build(
        self, timm_name: str, weights_path: str, num_classes: int
    ) -> torch.nn.Module:
        """Construct one model and move it to the inference device."""
        device = get_device()
        model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        # Gradients are only needed for Grad-CAM, which re-enables them on the
        # input tensor rather than on parameters.
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    def preload(self, specs: list[dict]) -> dict:
        """Eagerly build a batch of models, tolerating individual failures.

        Args:
            specs: Dicts with keys ``module_id``, ``arch``, ``timm_name``,
                ``weights_path``, ``num_classes``.

        Returns:
            ``{"loaded": [...], "failed": {key: reason}}``
        """
        loaded: list[str] = []
        failed: dict[str, str] = {}

        for spec in specs:
            key = self._key(spec["module_id"], spec["arch"])
            try:
                self.get(**spec)
                loaded.append(key)
            except Exception as exc:  # noqa: BLE001 - startup must not die
                failed[key] = f"{type(exc).__name__}: {exc}"
                logger.error("Preload failed for %s: %s", key, exc)

        return {"loaded": loaded, "failed": failed}

    def is_loaded(self, module_id: str, arch: str) -> bool:
        """Whether a given model is already resident."""
        return self._key(module_id, arch) in self._models

    def loaded_keys(self) -> list[str]:
        """Sorted list of cached model keys."""
        return sorted(self._models.keys())

    def clear(self) -> None:
        """Drop all cached models. Primarily for tests."""
        with self._lock:
            self._models.clear()


# Single shared instance.
model_cache = ModelCache()


def gradcam_ready(model: torch.nn.Module) -> torch.nn.Module:
    """Re-enable parameter gradients on a cached model for Grad-CAM.

    Cached models have ``requires_grad=False`` on every parameter to keep
    forward passes cheap. Grad-CAM needs a backward pass through the target
    convolutional layer, so gradients must be switched back on. The model is
    shared, so this mutates cache state deliberately and idempotently -- it is
    cheap and leaves the model valid for ordinary inference either way.

    Args:
        model: A cached model about to be used for Grad-CAM.

    Returns:
        The same model, with parameter gradients enabled.
    """
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad_(True)
    return model
