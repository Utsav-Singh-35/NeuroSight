"""Model inference service for brain tumor classification.

Provides functions to load the trained EfficientNet-B0 model and run
inference on preprocessed MRI image tensors.

The model was trained using the `timm` (PyTorch Image Models) library's
EfficientNet-B0 implementation with a simple Linear(1280, 4) classifier head.
"""

import torch
import torch.nn as nn
import timm

# Default class labels matching model output indices
DEFAULT_CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


def load_model(model_path: str, num_classes: int = 4) -> torch.nn.Module:
    """Load the trained EfficientNet-B0 model from a state dict file.

    The model uses the timm library's EfficientNet-B0 architecture with a
    Linear(1280, 4) classifier head, matching the training configuration.

    Args:
        model_path: Path to the .pth file containing model state_dict.
        num_classes: Number of output classes (default 4).

    Returns:
        The loaded model in evaluation mode, ready for inference.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If the state dict cannot be loaded into the model architecture.
    """
    # Create timm EfficientNet-B0 with the correct number of classes
    # This matches the architecture used during training
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    class_labels: list[str] = DEFAULT_CLASS_LABELS,
) -> dict:
    """Run inference on a preprocessed image tensor.

    Performs a forward pass without gradient computation, applies softmax
    to get class probabilities, and returns the prediction result.

    Args:
        model: The loaded BrainTumorClassifier in eval mode.
        tensor: Preprocessed image tensor of shape (1, 3, 224, 224).
        class_labels: List of class label names matching output indices.

    Returns:
        Dictionary with keys:
            - prediction (str): The predicted class label.
            - confidence (float): Confidence percentage (0-100), rounded to 2 decimal places.
            - probabilities (dict): Per-class probability percentages, each rounded to 2 decimal places.
    """
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

    # Get predicted class (argmax)
    predicted_index = torch.argmax(probabilities, dim=1).item()
    predicted_label = class_labels[predicted_index]

    # Calculate confidence (max probability as percentage)
    confidence = round(probabilities[0, predicted_index].item() * 100, 2)

    # Build per-class probabilities dict
    probabilities_dict = {
        label: round(probabilities[0, i].item() * 100, 2)
        for i, label in enumerate(class_labels)
    }

    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities_dict,
    }
