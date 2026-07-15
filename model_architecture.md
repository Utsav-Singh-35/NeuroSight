# Model Architecture - NeuraSight

## Overview

NeuraSight uses **EfficientNet-B0** via the `timm` (PyTorch Image Models) library for brain tumor classification. The model is pre-trained on ImageNet (1.2M images, 1000 classes) and fine-tuned on 5,600 brain MRI images for 4-class classification.

**Model file:** `models/Brain_MRI_scan.pth` (state_dict, ~20MB)

---

## Why EfficientNet-B0?

| Reason | Details |
|--------|---------|
| **Accuracy-to-size ratio** | 77.1% ImageNet top-1 with only 5.3M params (ResNet-50 needs 25.6M for 76.1%) |
| **Fast inference** | ~1 second on CPU, suitable for real-time web deployment |
| **Transfer learning** | ImageNet features (edges, textures, shapes) transfer well to medical images |
| **Memory efficient** | Runs on machines without dedicated GPU for inference |
| **Well-supported** | Available in `timm` library with pre-trained weights via HuggingFace |

### Comparison with Alternatives

| Model | Parameters | ImageNet Acc | Brain Tumor Acc* | Inference Speed | Choice |
|-------|-----------|-------------|-----------------|-----------------|--------|
| Custom CNN | ~1-5M | N/A | 80-85% | Fast | ❌ Low accuracy |
| VGG-16 | 138M | 71.3% | 88-91% | Slow | ❌ Too heavy |
| ResNet-50 | 25.6M | 76.1% | 90-93% | Medium | ❌ Good but heavy |
| **EfficientNet-B0** | **5.3M** | **77.1%** | **95%** | **Fast** | **✅ Used** |

---

## How the Model Was Trained (From Notebook)

### Step 1: Data Setup

**Dataset:** [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Split | Images | Purpose |
|-------|--------|---------|
| Training | 5,600 | Model learning |
| Testing | 1,600 | Validation & evaluation |

**Classes** (assigned by `ImageFolder` alphabetically):

| Index | Class | Folder Name |
|-------|-------|-------------|
| 0 | Glioma | `glioma/` |
| 1 | Meningioma | `meningioma/` |
| 2 | No Tumor | `notumor/` |
| 3 | Pituitary | `pituitary/` |

**Image sizes:** Vary from 150×168 to 1375×1446 pixels (all resized to 224×224).

---

### Step 2: Preprocessing & Augmentation

**Training transforms** (applied randomly each time an image is loaded):

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # Standardize size
    transforms.RandomHorizontalFlip(),      # 50% chance flip
    transforms.RandomRotation(10),          # ±10° rotation
    transforms.ColorJitter(                 # Random brightness/contrast
        brightness=0.2, contrast=0.2
    ),
    transforms.ToTensor(),                  # Convert to [0,1] tensor
    transforms.Normalize(                   # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**Test transforms** (no augmentation — consistent evaluation):

```python
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**Why ImageNet normalization?**
EfficientNet-B0 was pre-trained on ImageNet using these exact mean/std values. The model's internal feature maps expect input normalized this way. Without it, activations don't match what the model learned — resulting in random/wrong predictions (always "No Tumor").

**Why augmentation?**
With only 5,600 training images, the model overfits quickly without augmentation. Random flips, rotations, and color changes force the model to learn rotation/color-invariant features rather than memorizing specific images.

---

### Step 3: Model Architecture

```python
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=4)
```

This does three things:
1. Creates EfficientNet-B0 architecture
2. Loads weights pre-trained on 1.2M ImageNet images
3. Replaces the classifier: `Linear(1280, 1000)` → `Linear(1280, 4)`

**Architecture breakdown:**

```
Input: [Batch, 3, 224, 224]  (RGB MRI image, ImageNet-normalized)
        │
        ▼
┌─────────────────────────┐
│  Stem Conv (3 → 32)     │  Conv2d + BatchNorm + SiLU activation
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  MBConv Blocks (×16)    │  Mobile Inverted Bottleneck blocks
│  Stages 1-7             │  with Squeeze-and-Excitation attention
│                         │
│  Stage 1: 32 → 16      │  ← Low-level features (edges, corners)
│  Stage 2: 16 → 24      │
│  Stage 3: 24 → 40      │  ← Mid-level features (textures, shapes)
│  Stage 4: 40 → 80      │
│  Stage 5: 80 → 112     │  ← High-level features (brain structures)
│  Stage 6: 112 → 192    │
│  Stage 7: 192 → 320    │  ← Tumor-specific features
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Head Conv (320 → 1280) │  Conv2d + BatchNorm + SiLU
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Global Average Pooling │  → [Batch, 1280]
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Classifier             │  Linear(1280, 4) ← MODIFIED for 4 classes
└───────────┬─────────────┘
            │
            ▼
Output: [Batch, 4] → Softmax → {Glioma, Meningioma, No Tumor, Pituitary}
```

**Key components:**
- **MBConv blocks:** Efficient depthwise-separable convolutions that reduce computation by 8-9x vs standard convolutions
- **Squeeze-and-Excitation:** Channel attention mechanism — learns which feature channels are important for each input
- **SiLU activation:** Smooth non-linearity (x × sigmoid(x)) — better gradient flow than ReLU
- **Compound scaling:** Width, depth, and resolution scaled together using a coefficient (B0 = base model)

---

### Step 4: Training Configuration

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
EPOCHS = 10
BATCH_SIZE = 32
```

| Setting | Value | Why |
|---------|-------|-----|
| **Loss: CrossEntropyLoss** | Standard | Combines LogSoftmax + NLLLoss. Perfect for multi-class classification — penalizes wrong predictions proportionally to confidence |
| **Optimizer: AdamW** | lr=0.0001 | Adam with decoupled weight decay. Low LR because model is pre-trained — we fine-tune, not train from scratch |
| **Epochs: 10** | Short run | Pre-trained model converges fast. More epochs → overfitting (train 99% vs val 95% gap was already appearing) |
| **Batch Size: 32** | Standard | Fits in T4 GPU memory. Provides stable gradients (not too noisy) |
| **Device: CUDA** | Tesla T4 | 16GB VRAM GPU on Google Colab. Training took ~15 minutes total |

---

### Step 5: Training Loop

Each epoch:

**Training phase (`model.train()`):**
```
For each batch of 32 images:
  1. Zero gradients (reset from previous step)
  2. Forward pass: images → model → 4 logits per image
  3. Compute loss: CrossEntropy(logits, true_labels)
  4. Backward pass: compute gradients for all 5.3M parameters
  5. Optimizer step: update weights (lr=0.0001)
  6. Track loss and accuracy
```

**Validation phase (`model.eval()`):**
```
For each batch of 32 test images:
  1. Forward pass (no gradient computation — faster)
  2. Compute loss and accuracy
  3. No weight updates
```

**Model saving:** If validation accuracy improves → save `model.state_dict()` as `best_model.pth`. This keeps the best version even if later epochs overfit.

---

### Step 6: Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Saved? |
|-------|-----------|-----------|----------|---------|--------|
| 1 | 0.5928 | 83.82% | 0.5894 | 88.56% | ✅ |
| 2 | 0.1572 | 94.36% | 0.5160 | 91.88% | ✅ |
| 3 | 0.0700 | 97.41% | 0.6055 | 92.56% | ✅ |
| 4 | 0.0585 | 97.93% | 0.5613 | 93.81% | ✅ |
| 5 | 0.0414 | 98.59% | 0.5871 | 93.75% | ❌ |
| 6 | 0.0343 | 98.82% | 0.6172 | 94.31% | ✅ |
| 7 | 0.0267 | 99.16% | 0.6516 | 94.25% | ❌ |
| 8 | 0.0224 | 99.29% | 0.5664 | 94.44% | ✅ |
| **9** | **0.0212** | **99.38%** | **0.5768** | **95.00%** | **✅ Best** |
| 10 | 0.0183 | 99.41% | 0.6194 | 94.62% | ❌ |

**Observations:**
- Rapid convergence: 83% → 94% in just 2 epochs (pre-trained features help enormously)
- Overfitting beginning: Train acc 99.4% vs Val acc 95% (4.4% gap)
- Val loss fluctuating (not decreasing steadily) — sign of mild overfitting
- Best checkpoint at epoch 9 (val acc 95%)

---

## What the Model Learns at Each Layer

| Layer Depth | What It Detects | Source |
|-------------|----------------|--------|
| **Early (Stages 1-2)** | Edges, gradients, basic textures | Mostly from ImageNet pre-training |
| **Mid (Stages 3-4)** | Tissue boundaries, circular structures, density variations | Mix of ImageNet + fine-tuning |
| **Late (Stages 5-6)** | Brain anatomy landmarks, tumor shapes, mass boundaries | Mostly learned during fine-tuning |
| **Final (Stage 7)** | Tumor-specific patterns: glioma infiltration, meningioma surface attachment, pituitary location | Learned during fine-tuning |
| **Classifier** | Maps 1280 abstract features → 4 class probabilities | Trained from scratch |

---

## How Inference Works (Prediction Pipeline)

```
User uploads MRI image
        │
        ▼
┌───────────────────────┐
│ 1. Load & Convert RGB │  Handle grayscale/RGBA → RGB
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 2. Resize to 224×224  │  Bilinear interpolation
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 3. Normalize          │  pixels/255 → subtract mean → divide by std
│   mean=[.485,.456,.406]│  (ImageNet normalization)
│   std=[.229,.224,.225] │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 4. To Tensor          │  Shape: [1, 3, 224, 224] float32
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 5. EfficientNet-B0    │  Forward pass (torch.no_grad)
│    Forward Pass       │  → 4 raw logits
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 6. Softmax            │  Logits → probabilities [0,1] summing to 1
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 7. argmax             │  Highest probability = predicted class
└───────────┬───────────┘
            │
            ▼
Result: "Meningioma" (confidence: 98.47%)
```

**Example:** Raw logits `[-2.1, 4.8, -1.3, -0.5]` → Softmax → `[0.001, 0.985, 0.002, 0.005]` → argmax = index 1 → "Meningioma" at 98.5%

---

## Grad-CAM Explainability

**What:** Gradient-weighted Class Activation Mapping — shows which MRI regions the model focused on for its prediction.

**How:**
1. Hook into the last convolutional layer (`model.blocks[-1]` for timm)
2. Forward pass to get prediction
3. Backpropagate gradients for the predicted class
4. Global average pool gradients → importance weights per channel
5. Weighted sum of feature maps → heatmap
6. Apply ReLU (only positive contributions matter)
7. Normalize to [0,1], resize to 224×224
8. Overlay on original MRI with JET colormap at 40% opacity

**Why it matters:** Clinicians won't trust a "black box." Grad-CAM shows that the model looks at the tumor region (not background noise), building confidence in the AI's reasoning.

---

## Known Limitations

| Issue | Cause | Impact |
|-------|-------|--------|
| **Glioma recall = 0.81** | Glioma and meningioma look similar in some cases | 19% of gliomas misclassified |
| **Overfitting** | Only 10 epochs, no LR scheduler, basic augmentation | Train/val gap of 4.4% |
| **No "unknown" class** | Model only trained on MRI images | Non-MRI input always predicts "No Tumor" |
| **Single Linear classifier** | No hidden layers for feature refinement | May limit complex decision boundaries |

---

## File References

| File | Location | Purpose |
|------|----------|---------|
| Training notebook | `notebooks/NeuraSight_BrainMRI_Training.ipynb` | Google Colab training code |
| Model weights | `models/Brain_MRI_scan.pth` | Trained state_dict |
| Inference service | `backend/fastapi/app/services/inference.py` | Loads model + runs prediction |
| Preprocessor | `backend/fastapi/app/services/preprocessor.py` | Image → tensor pipeline |
| Grad-CAM service | `backend/fastapi/app/services/gradcam.py` | Heatmap generation |
| Improvement guide | `docs/improvements.md` | How to retrain for better results |

---

**Last Updated:** July 2026  
**Training Platform:** Google Colab (Tesla T4 GPU)  
**Best Model:** Epoch 9 — Validation Accuracy 95.00%
