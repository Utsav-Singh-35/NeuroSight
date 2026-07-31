# NeuraSight Model Weights

## Overview

This directory holds the trained model weights and the stacking-ensemble meta-learner used by the FastAPI inference service. Large weight files are not committed to version control — download them from Google Drive (`My Drive/NeuraSight/models/`).

## Stacking Ensemble Files

| File | Description |
|------|-------------|
| `BRAIN_MRI_EFFICIENTNET.pth` | EfficientNet-B0 base learner (state_dict) |
| `BRAIN_MRI_RESNET.pth` | ResNet-50 base learner (state_dict) |
| `BRAIN_MRI_DENSENET.pth` | DenseNet-121 base learner (state_dict) |
| `BRAIN_MRI_VGG.pth` | VGG-16 base learner (state_dict) |
| `meta_model.pkl` | Logistic Regression meta-learner (pickle) |
| `ensemble_config.json` | Model order, class names, feature dim |

> Until full ensemble integration lands, the backend runs single-model inference using `BRAIN_MRI_EFFICIENTNET.pth` (the EfficientNet-B0 base learner). The stacking ensemble reached **96.75%** accuracy on the held-out evaluation split.

## Base Model Architecture

All base models use the **`timm`** library with a 4-class head:

```python
model = timm.create_model(TIMM_NAME, pretrained=True, num_classes=4)
```

| Key | timm name | Params |
|-----|-----------|--------|
| efficientnet | `efficientnet_b0` | 5.3M |
| resnet | `resnet50` | 25.6M |
| densenet | `densenet121` | 8.0M |
| vgg | `vgg16` | 138M |

## Meta-Learner (Stacking)

- Each base model outputs a 4-class softmax vector.
- The four vectors are concatenated into a **16-dimensional** feature vector (order defined in `ensemble_config.json`).
- A **Logistic Regression** meta-learner produces the final prediction.

## Classes (output index order)

| Index | Class |
|-------|-------|
| 0 | Glioma |
| 1 | Meningioma |
| 2 | No Tumor |
| 3 | Pituitary |

## Input Requirements

- Image size: 224×224 pixels, 3-channel RGB
- Normalization: divide by 255, then ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- Tensor shape: `(1, 3, 224, 224)`, dtype `float32`

## Training Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/BRAIN_MRI_SCAN.ipynb` | Trains all 4 base models in one run; exports probabilities + metrics |
| `notebooks/BRAIN_MRI_ENSEMBLE.ipynb` | Trains the Logistic Regression meta-learner; saves `meta_model.pkl` |

## Development Without Trained Weights

```bash
python create_dummy_model.py
```

Creates a dummy EfficientNet-B0 file with the correct architecture but random weights, so the inference pipeline can be tested without the real model.
