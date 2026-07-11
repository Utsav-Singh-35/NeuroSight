"""Grad-CAM explainability service for brain tumor classification.

Generates class activation map heatmaps highlighting regions the model
focused on when making a prediction. Uses pytorch-grad-cam library when
available, with a manual fallback implementation.
"""

import base64
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image

from app.config import settings

# Attempt to import pytorch-grad-cam library
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    GRADCAM_LIB_AVAILABLE = True
except ImportError:
    GRADCAM_LIB_AVAILABLE = False


def _get_target_layer(model: torch.nn.Module):
    """Get the last convolutional layer of the timm EfficientNet-B0.

    For timm's EfficientNet-B0, the last block before the classifier is
    accessed via model.blocks[-1] (the final MBConv block).

    Args:
        model: The timm EfficientNet-B0 model.

    Returns:
        The last convolutional block (model.blocks[-1]).
    """
    return model.blocks[-1]


def _run_inference_with_gradients(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_labels: list[str],
) -> tuple[int, str, float, torch.Tensor]:
    """Run inference and return predicted class info along with logits.

    Args:
        model: The timm EfficientNet-B0 model.
        image_tensor: Preprocessed tensor of shape (1, 3, 224, 224).
        class_labels: List of class label strings.

    Returns:
        Tuple of (predicted_index, predicted_label, confidence, probabilities_tensor).
    """
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

    predicted_index = torch.argmax(probabilities, dim=1).item()
    predicted_label = class_labels[predicted_index]
    confidence = round(probabilities[0, predicted_index].item() * 100, 2)

    return predicted_index, predicted_label, confidence, probabilities


def _prepare_original_image(original_image_bytes: bytes) -> np.ndarray:
    """Prepare the original image as a normalized RGB numpy array at 224x224.

    Args:
        original_image_bytes: Raw image bytes.

    Returns:
        Numpy array of shape (224, 224, 3) with values in [0, 1] as float32.
    """
    image = Image.open(BytesIO(original_image_bytes))
    image = image.convert("RGB")
    image = image.resize((224, 224), Image.BILINEAR)
    return np.array(image, dtype=np.float32) / 255.0


def _generate_cam_library(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    predicted_index: int,
) -> np.ndarray:
    """Generate CAM using the pytorch-grad-cam library.

    Args:
        model: The BrainTumorClassifier model.
        image_tensor: Preprocessed tensor of shape (1, 3, 224, 224).
        predicted_index: Index of the predicted class.

    Returns:
        Grayscale CAM heatmap as numpy array of shape (224, 224) in [0, 1].

    Raises:
        RuntimeError: If heatmap generation fails.
    """
    target_layer = _get_target_layer(model)

    try:
        cam = GradCAM(model=model, target_layers=[target_layer])
        # Use the predicted class as the target
        targets = None  # None defaults to highest scoring category
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        # grayscale_cam shape is (batch, H, W), take first item
        heatmap = grayscale_cam[0, :]
        cam.__del__()  # Clean up hooks
        return heatmap
    except Exception as e:
        raise RuntimeError("Heatmap generation failed") from e


def _generate_cam_manual(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    predicted_index: int,
) -> np.ndarray:
    """Generate CAM using manual gradient computation (fallback).

    Steps:
        1. Get last conv layer output and gradients via hooks
        2. Global average pool the gradients
        3. Weight feature maps by pooled gradients
        4. Apply ReLU
        5. Normalize to [0, 1]
        6. Resize to 224x224

    Args:
        model: The BrainTumorClassifier model.
        image_tensor: Preprocessed tensor of shape (1, 3, 224, 224).
        predicted_index: Index of the predicted class.

    Returns:
        Grayscale CAM heatmap as numpy array of shape (224, 224) in [0, 1].

    Raises:
        RuntimeError: If heatmap generation fails.
    """
    activations = []
    gradients = []

    target_layer = _get_target_layer(model)

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Enable gradients for this forward pass
        model.eval()
        input_tensor = image_tensor.clone().requires_grad_(True)

        # Forward pass
        output = model(input_tensor)

        # Zero all existing gradients
        model.zero_grad()

        # Backward pass for the predicted class
        target = output[0, predicted_index]
        target.backward()

        if not activations or not gradients:
            raise RuntimeError("Heatmap generation failed")

        # Get the activations and gradients
        activation = activations[0]  # Shape: (1, C, H, W)
        gradient = gradients[0]  # Shape: (1, C, H, W)

        # Global average pooling on gradients -> (1, C, 1, 1)
        pooled_gradients = torch.mean(gradient, dim=[2, 3], keepdim=True)

        # Weight feature maps by pooled gradients
        weighted_activations = activation * pooled_gradients

        # Sum across channels -> (1, 1, H, W)
        cam = torch.sum(weighted_activations, dim=1, keepdim=True)

        # Apply ReLU
        cam = torch.relu(cam)

        # Remove batch and channel dims -> (H, W)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to 224x224
        cam = cv2.resize(cam, (224, 224))

        return cam

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError("Heatmap generation failed") from e
    finally:
        forward_handle.remove()
        backward_handle.remove()


def _overlay_heatmap(
    cam: np.ndarray,
    original_image: np.ndarray,
    opacity: float,
    colormap: str,
) -> np.ndarray:
    """Overlay the CAM heatmap on the original image.

    Args:
        cam: Grayscale CAM array of shape (224, 224) in [0, 1].
        original_image: RGB image array of shape (224, 224, 3) in [0, 1].
        opacity: Opacity for the heatmap overlay (0.0 to 1.0).
        colormap: Name of the colormap (e.g., "jet").

    Returns:
        Overlaid image as uint8 numpy array of shape (224, 224, 3) in [0, 255].
    """
    if GRADCAM_LIB_AVAILABLE:
        # Use the library's show_cam_on_image utility
        overlaid = show_cam_on_image(
            original_image, cam, use_rgb=True, image_weight=1.0 - opacity
        )
    else:
        # Manual overlay with JET colormap
        # Convert cam to uint8 for colormap application
        heatmap_uint8 = np.uint8(255 * cam)

        # Apply JET colormap (cv2 uses BGR by default)
        colormap_code = cv2.COLORMAP_JET
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap_code)
        # Convert BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Normalize heatmap_colored to [0, 1] for blending
        heatmap_float = heatmap_colored.astype(np.float32) / 255.0

        # Overlay: blend original image with heatmap
        overlaid_float = (1.0 - opacity) * original_image + opacity * heatmap_float
        overlaid = np.uint8(np.clip(overlaid_float * 255, 0, 255))

    return overlaid


def _encode_image_to_base64(image_array: np.ndarray) -> str:
    """Encode an image numpy array as a base64 PNG string.

    Args:
        image_array: RGB image as uint8 numpy array of shape (H, W, 3).

    Returns:
        Base64-encoded PNG string.
    """
    image = Image.fromarray(image_array, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    original_image_bytes: bytes,
    class_labels: list[str] | None = None,
) -> dict:
    """Generate a Grad-CAM heatmap for the given image.

    Performs inference, computes Grad-CAM for the predicted class,
    overlays the heatmap on the original image, and returns the result
    as a base64-encoded PNG.

    Args:
        model: The loaded BrainTumorClassifier in eval mode.
        image_tensor: Preprocessed image tensor of shape (1, 3, 224, 224).
        original_image_bytes: Raw original image bytes for overlay.
        class_labels: List of class label names. Defaults to settings.class_labels.

    Returns:
        Dictionary with keys:
            - heatmap (str): Base64-encoded PNG of the overlaid heatmap.
            - prediction (str): Predicted class label.
            - confidence (float): Confidence percentage (0-100), rounded to 2 decimal places.

    Raises:
        RuntimeError: If heatmap generation fails.
    """
    if class_labels is None:
        class_labels = settings.class_labels

    # Step 1: Run inference to get predicted class and confidence
    predicted_index, predicted_label, confidence, _ = _run_inference_with_gradients(
        model, image_tensor, class_labels
    )

    # Step 2 & 3: Generate CAM heatmap
    if GRADCAM_LIB_AVAILABLE:
        cam = _generate_cam_library(model, image_tensor, predicted_index)
    else:
        cam = _generate_cam_manual(model, image_tensor, predicted_index)

    # Step 4 & 5: Prepare original image and overlay heatmap
    original_image = _prepare_original_image(original_image_bytes)
    overlaid_image = _overlay_heatmap(
        cam,
        original_image,
        opacity=settings.GRADCAM_OPACITY,
        colormap=settings.GRADCAM_COLORMAP,
    )

    # Step 6: Encode result as base64 PNG
    heatmap_base64 = _encode_image_to_base64(overlaid_image)

    # Step 7: Return result
    return {
        "heatmap": heatmap_base64,
        "prediction": predicted_label,
        "confidence": confidence,
    }
