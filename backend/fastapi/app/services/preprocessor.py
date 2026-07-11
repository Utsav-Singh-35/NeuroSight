"""Image preprocessing service for MRI image classification.

Converts raw image bytes into a PyTorch tensor suitable for EfficientNet-B0 inference.
The model was trained with timm's EfficientNet-B0, which uses ImageNet normalization.
"""

from io import BytesIO

import numpy as np
import torch
from PIL import Image

# ImageNet normalization values (required for timm EfficientNet-B0)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    """Preprocess an image from raw bytes into a model-ready tensor.

    Steps:
        1. Open image from bytes using PIL.Image
        2. Convert to RGB (grayscale → replicate channels, RGBA → discard alpha)
        3. Resize to 224x224 with bilinear interpolation
        4. Normalize pixel values to [0, 1] by dividing by 255.0
        5. Apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        6. Convert to PyTorch float32 tensor with shape (1, 3, 224, 224) — channels first

    Args:
        file_bytes: Raw image file bytes (JPEG or PNG).

    Returns:
        A PyTorch tensor of shape (1, 3, 224, 224) with dtype float32,
        normalized with ImageNet mean and std.

    Raises:
        ValueError: If the image cannot be decoded or processed.
    """
    try:
        image = Image.open(BytesIO(file_bytes))
        image.verify()  # Verify it's a valid image
        # Re-open after verify (verify can leave the file in an unusable state)
        image = Image.open(BytesIO(file_bytes))
    except Exception:
        raise ValueError("Image could not be processed")

    try:
        # Convert to RGB (handles grayscale by replicating channels, RGBA by discarding alpha)
        image = image.convert("RGB")

        # Resize to 224x224 with bilinear interpolation
        image = image.resize((224, 224), Image.BILINEAR)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Apply ImageNet normalization (per-channel)
        img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

        # Convert to channels-first format: (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))

        # Convert to PyTorch tensor and add batch dimension: (C, H, W) -> (1, C, H, W)
        tensor = torch.from_numpy(img_array).unsqueeze(0)

        return tensor
    except Exception:
        raise ValueError("Image could not be processed")
