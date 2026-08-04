"""Module registry — discovers and manages available AI modules.

The registry holds references to all registered BaseModule instances and
provides methods to look them up by ID, list available modules, and
retrieve module metadata.
"""

import json
import logging
from pathlib import Path

from app.engine.base_module import BaseModule

logger = logging.getLogger(__name__)

MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"


class ModuleRegistry:
    """Central registry for all AI modules."""

    def __init__(self):
        self._modules: dict[str, BaseModule] = {}

    def register(self, module: BaseModule):
        """Register a module instance."""
        self._modules[module.module_id] = module
        logger.info(
            "Registered module: %s (%s) | available=%s",
            module.module_id,
            module.display_name,
            module.is_available(),
        )

    def get(self, module_id: str) -> BaseModule | None:
        """Get a module by ID. Returns None if not found."""
        return self._modules.get(module_id)

    def available_modules(self) -> list[dict]:
        """Return list of all modules with their status."""
        return [
            {
                "id": m.module_id,
                "name": m.display_name,
                "classes": m.classes,
                "available": m.is_available(),
            }
            for m in self._modules.values()
        ]

    def list_ids(self) -> list[str]:
        """Return list of all registered module IDs."""
        return list(self._modules.keys())


# --- Singleton-style global registry for backward compatibility ---

_global_registry = ModuleRegistry()


def get_registry() -> ModuleRegistry:
    """Get the global module registry."""
    return _global_registry


def init_registry():
    """Initialize the global registry by discovering and registering modules."""
    from app.modules.brain_mri.module import BrainMRIModule
    from app.modules.chest_xray.module import ChestXrayModule

    _global_registry.register(BrainMRIModule())
    _global_registry.register(ChestXrayModule())
    logger.info(
        "Registry initialized with %d modules: %s",
        len(_global_registry.list_ids()),
        _global_registry.list_ids(),
    )


# --- Convenience functions (used by routers) ---


def discover_modules() -> dict:
    """Scan the modules directory and return a dict of module_id -> metadata."""
    modules = {}
    for entry in MODULES_DIR.iterdir():
        if entry.is_dir() and (entry / "metadata.json").exists():
            with open(entry / "metadata.json", "r") as f:
                meta = json.load(f)
            modules[meta["module_id"]] = meta
    return modules


def get_module_metadata(module_id: str) -> dict:
    """Get metadata for a specific module."""
    modules = discover_modules()
    if module_id not in modules:
        raise ValueError(f"Unknown module: {module_id}. Available: {list(modules.keys())}")
    return modules[module_id]


def list_modules() -> list:
    """Return list of available modules with their display names."""
    modules = discover_modules()
    return [
        {"module_id": mid, "display_name": meta["display_name"], "classes": meta["classes"]}
        for mid, meta in modules.items()
    ]
