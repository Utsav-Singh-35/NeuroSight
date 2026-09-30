"""Generic inference runner — works for any module.

Loads models, runs ensemble, generates Grad-CAM and reports
based on the module's metadata configuration.
"""

from app.config import settings
from app.engine.registry import get_module_metadata


def get_models_dir_for_module(module_id: str) -> str:
    """Get the models directory path for a specific module.

    All model weights are stored in the root models/ dir with a prefix
    (e.g., BRAIN_MRI_EFFICIENTNET.pth, CHEST_XRAY_EFFICIENTNET.pth).
    """
    return settings.MODELS_DIR


def get_ensemble_config_for_module(module_id: str) -> dict:
    """Build ensemble config dict from module metadata.

    Args:
        module_id: Module identifier (e.g., 'brain_mri', 'chest_xray').

    Returns:
        Dict with module_id, model_prefix, base_models, timm_names,
        num_classes, and class_labels.
    """
    meta = get_module_metadata(module_id)
    return {
        "module_id": module_id,
        "model_prefix": meta["model_prefix"],
        "base_models": meta["base_models"],
        "timm_names": meta["timm_names"],
        "num_classes": meta["num_classes"],
        "class_labels": meta["classes"],
    }
