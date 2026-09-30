"""AI Router — determines which module to use for an uploaded image.

For now, the user selects the module via a 'module' query parameter.
In the future, this will include a lightweight CNN modality detector
that automatically identifies the scan type.
"""

from app.engine.registry import discover_modules, get_registry


class AIRouter:
    """Routes images to the correct AI module."""

    def __init__(self):
        self.modality_detector = None  # Will be trained later

    def route(self, image_bytes: bytes = None, module_id: str = None) -> tuple[str, float]:
        """Route an image to the correct module.

        If module_id is provided, use it directly (user selected).
        If not, use modality detector (when available) or raise error.

        Args:
            image_bytes: Raw image bytes (for future modality detection).
            module_id: Explicitly selected module ID.

        Returns:
            Tuple of (module_id, confidence).

        Raises:
            ValueError: If no module specified and no detector available.
        """
        if module_id:
            registry = get_registry()
            module = registry.get(module_id)
            if module is None:
                available = registry.list_ids()
                raise ValueError(
                    f"Unknown module: {module_id}. Available: {available}"
                )
            return module_id, 1.0

        if self.modality_detector:
            # Future: run lightweight classifier on image_bytes
            pass

        # No module specified and no detector available
        raise ValueError("Please specify a scan type (module_id)")


# --- Convenience function for backward compatibility ---


def route_image(module_id: str = None) -> str:
    """Determine which module to use for an uploaded image.

    Args:
        module_id: Explicitly specified module. If None, defaults to brain_mri.

    Returns:
        The module_id to use.
    """
    available = discover_modules()

    if module_id and module_id in available:
        return module_id

    # Default to brain_mri if no module specified
    # TODO: Add AI modality detector here in the future
    return "brain_mri"
