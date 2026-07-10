# Model Architecture - NeuraSight

## Overview

NeuraSight uses **EfficientNet-B0** as the backbone architecture for brain tumor classification. This document outlines the complete model architecture, training strategy, and implementation details.

---

## Why EfficientNet-B0?

### Advantages
1. **Research-Backed:** State-of-the-art performance on ImageNet
2. **Efficient:** Optimal balance between accuracy and computational cost
3. **Lightweight:** ~5.3M parameters (suitable for deployment)
4. **Transfer Learning:** Pre-trained on ImageNet for better feature extraction
5. **Medical Imaging:** Proven effectiveness in medical image classification

### Comparison with Alternatives

| Model | Parameters | Accuracy | Speed | Medical Imaging |
|-------|------------|----------|-------|-----------------|
| EfficientNet-B0 | 5.3M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| ResNet50 | 25.6M | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| VGG16 | 138M | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Custom CNN | Variable | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

---

## Model Architecture

### Base Model: EfficientNet-B0

```
Input: 224 x 224 x 3 (RGB MRI Image)
    ↓
EfficientNet-B0 (Pre-trained on ImageNet)
    ↓ (Feature Extraction)
Global Average Pooling 2D
    ↓
Dense Layer (256 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense Layer (128 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (4 units, Softmax)
    ↓
[Glioma, Meningioma, Pituitary, No Tumor]
```

### Layer Details

#### 1. Input Layer
- **Shape:** (224, 224, 3)
- **Preprocessing:** Normalization to [0, 1]

#### 2. EfficientNet-B0 Base
- **Weights:** ImageNet pre-trained
- **Trainable:** Initially frozen, fine-tuned later
- **Output Shape:** (7, 7, 1280)

#### 3. Global Average Pooling
- **Purpose:** Reduce spatial dimensions
- **Output Shape:** (1280,)

#### 4. Dense Layer 1
- **Units:** 256
- **Activation:** ReLU
- **Purpose:** Feature refinement

#### 5. Dropout 1
- **Rate:** 0.5
- **Purpose:** Prevent overfitting

#### 6. Dense Layer 2
- **Units:** 128
- **Activation:** ReLU
- **Purpose:** Further feature compression

#### 7. Dropout 2
- **Rate:** 0.3
- **Purpose:** Additional regularization

#### 8. Output Layer
- **Units:** 4 (number of classes)
- **Activation:** Softmax
- **Output:** Probability distribution over classes

---

## Training Strategy

### Phase 1: Transfer Learning (Frozen Base)

**Epochs:** 10-15  
**Learning Rate:** 0.001  
**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  

```python
# Freeze EfficientNet base
base_model.trainable = False

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)
```

### Phase 2: Fine-Tuning (Unfrozen Base)

**Epochs:** 10-20  
**Learning Rate:** 0.0001 (reduced)  
**Optimizer:** Adam  
**Strategy:** Unfreeze last few layers of EfficientNet  

```python
# Unfreeze top layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)
```

---

## Hyperparameters

### Model Hyperparameters
- **Input Size:** 224 x 224 x 3
- **Batch Size:** 32
- **Dense Layer 1 Units:** 256
- **Dense Layer 2 Units:** 128
- **Dropout Rate 1:** 0.5
- **Dropout Rate 2:** 0.3

### Training Hyperparameters
- **Initial Learning Rate:** 0.001
- **Fine-tuning Learning Rate:** 0.0001
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Epochs (Phase 1):** 10-15
- **Epochs (Phase 2):** 10-20

### Data Augmentation
- **Rotation Range:** ±10°
- **Horizontal Flip:** True
- **Brightness Range:** [0.9, 1.1]
- **Contrast Range:** [0.9, 1.1]
- **Zoom Range:** 0.1
- **Fill Mode:** Nearest

---

## Callbacks

### 1. ModelCheckpoint
```python
ModelCheckpoint(
    filepath='models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
```

### 2. EarlyStopping
```python
EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

### 3. ReduceLROnPlateau
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)
```

### 4. TensorBoard
```python
TensorBoard(
    log_dir='logs/',
    histogram_freq=1
)
```

---

## Evaluation Metrics

### Primary Metrics
1. **Accuracy:** Overall classification accuracy
2. **Precision:** True positives / (True positives + False positives)
3. **Recall:** True positives / (True positives + False negatives)
4. **F1 Score:** Harmonic mean of precision and recall

### Additional Metrics
5. **ROC-AUC:** Area under ROC curve (per class)
6. **Confusion Matrix:** Detailed classification breakdown
7. **Classification Report:** Per-class metrics

### Target Performance
- **Accuracy:** > 95%
- **Precision:** > 94%
- **Recall:** > 94%
- **F1 Score:** > 94%

---

## Model Variants (Future)

### Variant 1: EfficientNet-B1
- **Parameters:** 7.8M
- **Input Size:** 240 x 240
- **Expected Accuracy:** +1-2%

### Variant 2: EfficientNet-B2
- **Parameters:** 9.2M
- **Input Size:** 260 x 260
- **Expected Accuracy:** +2-3%

### Variant 3: Ensemble Model
- **Combination:** EfficientNet-B0 + EfficientNet-B1
- **Strategy:** Weighted average of predictions
- **Expected Accuracy:** +2-4%

---

## Implementation Checklist

- [ ] Load EfficientNet-B0 with ImageNet weights
- [ ] Add custom classification head
- [ ] Implement data augmentation pipeline
- [ ] Set up training callbacks
- [ ] Train Phase 1 (frozen base)
- [ ] Train Phase 2 (fine-tuning)
- [ ] Evaluate on test set
- [ ] Generate confusion matrix
- [ ] Calculate all metrics
- [ ] Save final model

---

## References

1. EfficientNet Paper: https://arxiv.org/abs/1905.11946
2. Transfer Learning Guide: https://www.tensorflow.org/tutorials/images/transfer_learning
3. Medical Image Classification: https://arxiv.org/abs/2010.14701

---

**Last Updated:** [Date]  
**Author:** [Your Name]
