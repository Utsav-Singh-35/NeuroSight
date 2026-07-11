"""
Create a dummy EfficientNet-B0 model with the correct architecture for NeuraSight.

This script generates a `best_model.pth` file with random weights matching
the trained model's architecture. Use this for development and testing when
the trained model weights are not available.

Usage:
    python create_dummy_model.py
"""

import os

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class BrainTumorClassifier(nn.Module):
    """EfficientNet-B0 with custom classification head for 4-class brain tumor classification."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.base = efficientnet_b0(weights=None)
        # Replace the classifier head with custom layers
        in_features = self.base.classifier[1].in_features  # 1280 for EfficientNet-B0
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.base(x)


def main():
    # Create the model with random weights
    model = BrainTumorClassifier(num_classes=4)

    # Save the state dict
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Dummy model saved to: {save_path}")
    print(f"Architecture: EfficientNet-B0 with custom classifier head")
    print(f"  - Linear(1280, 256) -> ReLU -> Dropout(0.5)")
    print(f"  - Linear(256, 128) -> ReLU -> Dropout(0.3)")
    print(f"  - Linear(128, 4)")
    print(f"Classes: [Glioma, Meningioma, No Tumor, Pituitary]")
    print(f"Note: This model has random weights and will produce random predictions.")


if __name__ == "__main__":
    main()
