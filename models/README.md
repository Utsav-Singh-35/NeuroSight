# NeuraSight Model Weights

## Overview

This directory contains the trained PyTorch model weights used by the FastAPI inference service.

## Model: `best_model.pth`

The model file `best_model.pth` should be placed in this directory. It is not included in version control due to its size.

### Download Location

Download from **Google Drive**: `My Drive/NeuraSight/best_model.pth`

### Architecture

The model is an **EfficientNet-B0** with a custom classification head for 4-class brain tumor classification:

- **Base**: EfficientNet-B0 (torchvision, pretrained backbone)
- **Classifier Head**:
  - `Linear(1280, 256)` → `ReLU` → `Dropout(0.5)`
  - `Linear(256, 128)` → `ReLU` → `Dropout(0.3)`
  - `Linear(128, 4)`

### Classes

The model classifies brain MRI images into 4 categories (mapped by output index):

| Index | Class Label   |
|-------|---------------|
| 0     | Glioma        |
| 1     | Meningioma    |
| 2     | No Tumor      |
| 3     | Pituitary     |

### Input Requirements

- Image size: 224×224 pixels
- Channels: 3 (RGB)
- Normalization: pixel values divided by 255.0 (range [0, 1])
- Tensor shape: `(1, 3, 224, 224)` with dtype `float32`

## Development Without the Trained Model

If you don't have access to the trained model weights, you can generate a dummy model with random weights for development and testing:

```bash
python create_dummy_model.py
```

This creates a `best_model.pth` file with the correct architecture but random (untrained) weights. Predictions will be random, but this allows testing the full inference pipeline.
