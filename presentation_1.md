 # NeuraSight: AI-Powered Brain Tumor Detection System

## Final Year Project Presentation

**Project Title:** NeuraSight – Automated Brain Tumor Classification from MRI Scans using Deep Learning  
**Domain:** Healthcare + Artificial Intelligence  
**Technology Stack:** EfficientNet-B0 | React | Express.js | FastAPI | MongoDB  
**Team:** Final Year – Computer Science / IT  

---

## Table of Contents

1. [Domain of Interest and Motivation](#1-domain-of-interest-and-motivation-behind-domain-selection)
2. [Stakeholder Input for Problem Statement](#2-stakeholder-input-for-the-problem-statement)
3. [Project Planning](#3-project-planning)
4. [Literature Survey](#4-literature-survey)
5. [System Architecture and Algorithms](#5-system-architecture-and-algorithms--flowchart)
6. [Adopted Methodology](#6-adopted-methodology)
7. [Performance Evaluation Parameters](#7-performance-evaluation-parameters)
8. [Software and Hardware Setup](#8-software-and-hardware-setup)
9. [Initial Implementation Results (25-30%)](#9-initial-implementation-results-25-30)

---

## 1. Domain of Interest and Motivation Behind Domain Selection

### Domain: Healthcare + Artificial Intelligence

Healthcare is one of the most impactful domains where Artificial Intelligence can save lives. The intersection of medical imaging and deep learning presents an opportunity to assist clinicians in making faster, more accurate diagnoses.

### Why Brain Tumor Detection?

| Factor | Details |
|--------|---------|
| **Prevalence** | Brain tumors account for 85-90% of all primary CNS tumors globally |
| **Mortality** | Glioblastoma (a type of glioma) has a 5-year survival rate of only ~5% |
| **Diagnosis Challenge** | Manual MRI analysis requires 15-30 minutes per scan by expert radiologists |
| **Shortage** | Many regions face a critical shortage of trained neuroradiologists |

### Motivation

- **Life-threatening condition:** Brain tumors are among the most aggressive and deadly cancers. Early and accurate detection significantly improves patient survival rates.
- **Manual analysis limitations:** Traditional MRI interpretation is time-consuming, subjective, and prone to human error — especially under high workloads.
- **AI as an assistive tool:** Deep learning models can process and classify MRI scans in seconds, serving as a "second opinion" for radiologists.
- **Scalability:** An AI-powered system can be deployed across hospitals, clinics, and even telemedicine platforms — democratizing access to expert-level diagnostics.
- **Explainability need:** Unlike black-box models, our system uses Grad-CAM to highlight which regions of the MRI influenced the prediction, building trust with clinicians.

### Why This Domain Matters to Us

> "We chose healthcare AI because the impact is direct and measurable — every correct early detection is a life potentially saved."

---

## 2. Stakeholder Input for the Problem Statement

### Identified Stakeholders

| Stakeholder | Role | Key Concern |
|-------------|------|-------------|
| **Radiologists** | Primary users / domain experts | Need faster screening tools to handle growing patient loads |
| **Hospital Administration** | Decision makers | Require scalable, cost-effective diagnostic support systems |
| **Patients** | End beneficiaries | Benefit from earlier detection and reduced wait times |
| **ML Engineers** | System developers | Need interpretable models that clinicians can trust |
| **Regulatory Bodies** | Compliance oversight | Require explainability and audit trails for AI decisions |

### Stakeholder Input Summary

**From Radiologists:**
- "We spend 20+ minutes per MRI scan — a pre-screening tool would let us focus on complex cases."
- "I need to understand WHY the AI made a decision before I trust it."

**From Hospital Administration:**
- "We need solutions that can scale without hiring additional specialists."
- "Integration with existing hospital workflows is essential."

**From Patients:**
- "Faster diagnosis means earlier treatment and better outcomes."
- "I want confidence that the AI is reliable and tested."

### Derived Problem Statement

> **"Develop an AI-powered system for automated brain tumor classification from MRI scans with explainability, capable of classifying brain MRI images into four categories (Glioma, Meningioma, No Tumor, Pituitary) with high accuracy while providing visual explanations (Grad-CAM heatmaps) to support clinical decision-making."**

### Problem Decomposition

1. **Classification Task:** Multi-class classification (4 classes) from MRI images
2. **Explainability:** Grad-CAM visualization to highlight tumor regions
3. **Deployment:** Web-based interface accessible to non-technical users
4. **Reliability:** High accuracy (>90%) with consistent performance across all classes
5. **Scalability:** Microservices architecture for independent scaling

---

## 3. Project Planning

### Project Timeline: 15-Week Plan

| Week | Phase | Task | Status |
|------|-------|------|--------|
| 1-2 | **Phase 1-2** | Dataset Collection & Preprocessing | ✅ Done |
| 3-4 | **Phase 3** | Model Selection & Architecture Design | ✅ Done |
| 5-6 | **Phase 4** | Model Training & Hyperparameter Tuning | ✅ Done |
| 7 | **Phase 5** | Model Evaluation & Metrics Analysis | ✅ Done |
| 8 | **Phase 5** | Grad-CAM Explainability Implementation | ✅ Done |
| 9-10 | **Phase 6** | Backend Development (Express + FastAPI) | ✅ Done |
| 11-12 | **Phase 7** | Frontend Development (React + Three.js) | ✅ Done |
| 13 | **Phase 8** | Integration Testing & Bug Fixes | 🔄 In Progress |
| 14 | **Phase 9** | Documentation & Report Writing | 🔄 In Progress |
| 15 | **Phase 10** | Final Deployment & Presentation | 📋 Planned |

### Gantt Chart (Visual Timeline)

```
Week:        1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
            ─────────────────────────────────────────────────
Dataset     ████████
Preprocess     ████████
Model Arch        ████████
Training             ████████
Evaluation              ██████
Grad-CAM                   ████
Backend API                   ████████
Frontend                         ████████
Testing                                ████████
Docs                                      ████████
Deployment                                      ██████
            ─────────────────────────────────────────────────
```

### Milestones

| Milestone | Target Week | Achieved |
|-----------|-------------|----------|
| Dataset prepared (7,023 images) | Week 2 | ✅ |
| Model trained (>90% accuracy) | Week 6 | ✅ (95%) |
| Grad-CAM working | Week 8 | ✅ |
| Full-stack integration complete | Week 12 | ✅ |
| Final deployment | Week 15 | 📋 |

### Risk Management

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model overfitting | Medium | High | Data augmentation, early stopping, dropout |
| Class imbalance | Medium | Medium | Weighted loss function, stratified sampling |
| GPU resource limitations | High | Medium | Google Colab (free T4 GPU) |
| Integration issues | Low | Medium | Microservices architecture, independent testing |

---

## 4. Literature Survey

### Key Research Papers

| # | Paper | Authors | Year | Key Contribution |
|---|-------|---------|------|-----------------|
| 1 | **EfficientNet: Rethinking Model Scaling for CNNs** | Tan & Le | 2019 | Compound scaling method; achieves SOTA with fewer parameters |
| 2 | **Grad-CAM: Visual Explanations from Deep Networks** | Selvaraju et al. | 2017 | Gradient-weighted class activation maps for explainability |
| 3 | **A Survey on Deep Learning in Medical Image Analysis** | Litjens et al. | 2017 | Comprehensive review of DL applications in medical imaging |
| 4 | **Deep CNN for Computer-Aided Detection** | Shin et al. | 2016 | Transfer learning effectiveness for medical image tasks |
| 5 | **Brain Tumor Classification using Deep Learning** | Ayadi et al. | 2020 | CNN-based approaches for brain tumor MRI classification |

### Paper Details

#### 1. EfficientNet (Tan & Le, 2019)
- **Link:** [arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
- **Key Idea:** Proposes a compound scaling method that uniformly scales network width, depth, and resolution using a compound coefficient.
- **Why relevant:** EfficientNet-B0 achieves 77.1% top-1 accuracy on ImageNet with only 5.3M parameters — 8.4x smaller than ResNet-152 while being more accurate.
- **Our use:** We use EfficientNet-B0 as our backbone, fine-tuning it for brain tumor classification.

#### 2. Grad-CAM (Selvaraju et al., 2017)
- **Link:** [arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
- **Key Idea:** Uses gradients flowing into the final convolutional layer to produce a coarse localization map highlighting important regions.
- **Why relevant:** Provides visual explainability without modifying the model architecture.
- **Our use:** We generate Grad-CAM heatmaps overlaid on MRI scans to show clinicians which brain regions influenced the classification.

#### 3. Deep Learning in Medical Image Analysis (Litjens et al., 2017)
- **Link:** [arxiv.org/abs/1702.05747](https://arxiv.org/abs/1702.05747)
- **Key Idea:** Comprehensive survey covering classification, detection, segmentation in radiology, pathology, ophthalmology, etc.
- **Why relevant:** Establishes best practices for applying DL to medical images (transfer learning, data augmentation, class balancing).
- **Our use:** Guided our preprocessing pipeline and training strategy decisions.

#### 4. Transfer Learning for Medical Images (Shin et al., 2016)
- **Link:** [arxiv.org/abs/1608.00853](https://arxiv.org/abs/1608.00853)
- **Key Idea:** Demonstrates that CNNs pre-trained on ImageNet transfer effectively to medical imaging tasks even with limited labeled data.
- **Why relevant:** Validates our approach of using ImageNet pre-trained weights for brain MRI classification.
- **Our use:** Justified our transfer learning strategy with EfficientNet-B0 pre-trained weights.

#### 5. Brain Tumor Classification using Deep Learning (2020)
- **Link:** [arxiv.org/abs/2010.14701](https://arxiv.org/abs/2010.14701)
- **Key Idea:** Evaluates various CNN architectures for brain tumor classification on MRI datasets.
- **Why relevant:** Provides benchmark results for comparison with our approach.
- **Our use:** Used as a baseline to validate our model's performance against existing approaches.

### Comparative Analysis of Approaches

| Model | Parameters | ImageNet Acc | Medical Img Acc* | Inference Speed | Our Choice |
|-------|-----------|-------------|-----------------|-----------------|------------|
| Custom CNN | ~1-5M | N/A | 80-85% | Fast | ❌ Low accuracy |
| VGG-16 | 138M | 71.3% | 88-91% | Slow | ❌ Too many params |
| ResNet-50 | 25.6M | 76.1% | 90-93% | Medium | ❌ Good but heavy |
| **EfficientNet-B0** | **5.3M** | **77.1%** | **93-96%** | **Fast** | **✅ Best trade-off** |
| EfficientNet-B7 | 66M | 84.3% | 95-97% | Very Slow | ❌ Overkill for task |

*Approximate accuracy ranges on brain tumor classification tasks from literature.

### Why EfficientNet-B0?

1. **Best accuracy-to-parameters ratio** — Achieves 95% with only 5.3M parameters
2. **Fast inference** — Suitable for real-time web application deployment
3. **Proven transfer learning** — ImageNet pre-training transfers well to medical imaging
4. **Memory efficient** — Can run on machines without dedicated GPUs for inference
5. **Well-supported** — Available in `timm` library with pre-trained weights

---

## 5. System Architecture and Algorithms / Flowchart

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NeuraSight System Architecture                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐     ┌──────────────────┐     ┌──────────────────┐           │
│   │          │     │                  │     │                  │           │
│   │  User /  │────▶│  React Frontend  │────▶│  Express.js API  │           │
│   │  Browser │◀────│  (Vite :3000)    │◀────│  Gateway (:5000) │           │
│   │          │     │                  │     │                  │           │
│   └──────────┘     │  • Three.js 3D   │     │  • Route handling│           │
│                    │  • GSAP Anims    │     │  • File upload   │           │
│                    │  • Upload UI     │     │  • Validation    │           │
│                    │  • Results View  │     │  • Proxy to ML   │           │
│                    └──────────────────┘     └────────┬─────────┘           │
│                                                      │                      │
│                                                      ▼                      │
│                    ┌──────────────────┐     ┌──────────────────┐           │
│                    │                  │     │                  │           │
│                    │    MongoDB       │◀────│  FastAPI ML      │           │
│                    │    Database      │     │  Service (:8000) │           │
│                    │                  │     │                  │           │
│                    │  • Predictions   │     │  • EfficientNet  │           │
│                    │  • History       │     │  • Preprocessing │           │
│                    │  • Metadata      │     │  • Grad-CAM      │           │
│                    └──────────────────┘     │  • Inference     │           │
│                                            └──────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────┐
│  User   │    │  React   │    │ Express  │    │   FastAPI    │    │  MongoDB  │
│ Browser │    │ Frontend │    │ Backend  │    │  ML Service  │    │    DB     │
└────┬────┘    └────┬─────┘    └────┬─────┘    └──────┬───────┘    └─────┬─────┘
     │              │               │                  │                  │
     │ Upload MRI   │               │                  │                  │
     │─────────────▶│               │                  │                  │
     │              │ POST /predict  │                  │                  │
     │              │──────────────▶│                  │                  │
     │              │               │ Validate Image   │                  │
     │              │               │ (type, size)     │                  │
     │              │               │                  │                  │
     │              │               │ Forward to ML    │                  │
     │              │               │─────────────────▶│                  │
     │              │               │                  │                  │
     │              │               │                  │ Preprocess:      │
     │              │               │                  │ • Resize 224x224 │
     │              │               │                  │ • Normalize      │
     │              │               │                  │   (ImageNet μ,σ) │
     │              │               │                  │ • Tensor convert │
     │              │               │                  │                  │
     │              │               │                  │ EfficientNet-B0  │
     │              │               │                  │ Inference        │
     │              │               │                  │      │           │
     │              │               │                  │      ▼           │
     │              │               │                  │ Softmax Output   │
     │              │               │                  │ [G, M, N, P]     │
     │              │               │                  │                  │
     │              │               │                  │ Grad-CAM         │
     │              │               │                  │ Heatmap Gen      │
     │              │               │                  │                  │
     │              │               │  Classification  │                  │
     │              │               │◀─────────────────│                  │
     │              │               │  + Confidence    │                  │
     │              │               │  + Heatmap       │                  │
     │              │               │                  │                  │
     │              │               │ Store Prediction │                  │
     │              │               │─────────────────────────────────────▶│
     │              │               │                  │                  │
     │              │  JSON Response │                  │                  │
     │              │◀──────────────│                  │                  │
     │ Display      │               │                  │                  │
     │ Results      │               │                  │                  │
     │◀─────────────│               │                  │                  │
     │              │               │                  │                  │
```

### Algorithm: Transfer Learning with EfficientNet-B0

```
ALGORITHM: Brain Tumor Classification using Transfer Learning

INPUT:  MRI Image (any resolution, JPEG/PNG)
OUTPUT: Classification (Glioma | Meningioma | No Tumor | Pituitary) + Confidence + Heatmap

PROCEDURE:
1. PREPROCESSING:
   a. Load image → Convert to RGB (3 channels)
   b. Resize to 224 × 224 pixels (bilinear interpolation)
   c. Convert to tensor [0, 1] range
   d. Normalize with ImageNet statistics:
      - mean = [0.485, 0.456, 0.406]
      - std  = [0.229, 0.224, 0.225]
   e. Add batch dimension: [1, 3, 224, 224]

2. MODEL ARCHITECTURE:
   a. Load EfficientNet-B0 (pre-trained on ImageNet, 5.3M params)
   b. Replace classifier head:
      - Original: Linear(1280, 1000)  [ImageNet classes]
      - Modified: Linear(1280, 4)     [Tumor classes]
   c. Load fine-tuned weights from checkpoint

3. INFERENCE:
   a. Forward pass through model
   b. Extract logits: [batch_size, 4]
   c. Apply Softmax: probabilities = softmax(logits)
   d. Predicted class = argmax(probabilities)
   e. Confidence = max(probabilities)

4. GRAD-CAM EXPLAINABILITY:
   a. Hook gradients at last convolutional layer
   b. Compute class-specific gradients via backpropagation
   c. Global average pool gradients → channel weights
   d. Weighted sum of feature maps → heatmap
   e. Apply ReLU → normalize [0, 1]
   f. Resize heatmap to original image dimensions
   g. Overlay heatmap on MRI image (colormap: jet)

5. RETURN:
   - predicted_class: string
   - confidence: float [0, 1]
   - all_probabilities: dict {class: probability}
   - gradcam_heatmap: base64 encoded image
```

### Model Architecture Detail

```
EfficientNet-B0 Architecture (Modified for Brain Tumor Classification)
═══════════════════════════════════════════════════════════════════════

Input: [Batch, 3, 224, 224]
        │
        ▼
┌─────────────────────┐
│  Stem Conv (3→32)   │  ← Conv2d + BatchNorm + SiLU
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  MBConv Blocks x16  │  ← Mobile Inverted Bottleneck Blocks
│  (Stages 1-7)       │     with Squeeze-and-Excitation
│                     │
│  Stage 1: 32→16     │
│  Stage 2: 16→24     │
│  Stage 3: 24→40     │
│  Stage 4: 40→80     │
│  Stage 5: 80→112    │
│  Stage 6: 112→192   │
│  Stage 7: 192→320   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Head Conv (320→1280)│  ← Conv2d + BatchNorm + SiLU
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Global Avg Pooling  │  → [Batch, 1280]
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Classifier (Custom) │  ← Linear(1280, 4) [MODIFIED]
│  num_classes = 4     │
└─────────┬───────────┘
          │
          ▼
Output: [Batch, 4] → Softmax → {Glioma, Meningioma, No Tumor, Pituitary}
```

---

## 6. Adopted Methodology

### Overall Approach: Transfer Learning + Microservices + XAI

Our methodology combines three key strategies:

```
┌─────────────────────────────────────────────────────────┐
│                    NeuraSight Methodology                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Transfer   │  │ Microservices│  │  Explainable │  │
│  │   Learning   │  │ Architecture │  │      AI      │  │
│  │              │  │              │  │              │  │
│  │ EfficientNet │  │ React + Node │  │   Grad-CAM   │  │
│  │ + Fine-tune  │  │ + FastAPI    │  │   Heatmaps   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 1. Transfer Learning Approach

| Step | Description | Details |
|------|-------------|---------|
| **Pre-training** | Use EfficientNet-B0 pre-trained on ImageNet (1.2M images, 1000 classes) | Provides robust feature extraction capabilities |
| **Feature Extraction** | Freeze early layers, retain learned visual features | Low-level features (edges, textures) transfer well to medical images |
| **Fine-tuning** | Unfreeze all layers, train with low learning rate | Adapts ImageNet features to brain MRI domain |
| **Head Replacement** | Replace 1000-class head with 4-class head | Custom classifier for tumor classification |

**Training Configuration:**
- Optimizer: Adam (lr=1e-4)
- Loss Function: CrossEntropyLoss
- Batch Size: 32
- Epochs: 10 (early stopping patience: 5)
- Data Split: 80% train / 20% validation (stratified)
- Augmentation: Random horizontal flip, rotation (±15°), color jitter

### 2. Microservices Architecture

```
Service Decomposition:
─────────────────────────────────────────────────────────

┌─────────────────┐    Why Microservices?
│   Frontend      │    • Independent deployment
│   (React:3000)  │    • Technology flexibility
├─────────────────┤    • Scalability per service
│   API Gateway   │    • Fault isolation
│   (Express:5000)│    • Easier testing
├─────────────────┤    • Team parallelism
│   ML Service    │
│   (FastAPI:8000)│
├─────────────────┤
│   Database      │
│   (MongoDB)     │
└─────────────────┘
```

**Service Responsibilities:**

| Service | Technology | Responsibility |
|---------|-----------|----------------|
| Frontend | React + Vite | User interface, file upload, results display, 3D brain visualization |
| API Gateway | Express.js | Request routing, file validation, proxy, error handling, logging |
| ML Service | FastAPI + PyTorch | Image preprocessing, model inference, Grad-CAM generation |
| Database | MongoDB | Prediction history storage, metadata, audit trail |

### 3. Grad-CAM for Explainability (XAI)

**Why Explainability Matters in Healthcare AI:**
- Clinicians need to understand model reasoning before trusting it
- Regulatory requirements (FDA, EU AI Act) demand explainability
- Helps identify model failures and biases
- Builds patient trust in AI-assisted diagnoses

**Grad-CAM Process:**
1. Perform forward pass through the model
2. Compute gradients of the predicted class score w.r.t. feature maps of the last convolutional layer
3. Global average pool the gradients to get importance weights
4. Compute weighted combination of feature maps
5. Apply ReLU (only positive contributions)
6. Upsample to input resolution and overlay on original MRI

### 4. Development Methodology: Agile

| Agile Practice | Our Implementation |
|---------------|-------------------|
| Sprints | 1-2 week sprints per phase |
| Iterative Development | Model → API → Frontend (incremental integration) |
| Testing | Unit tests, integration tests per service |
| Version Control | Git with feature branches |
| Continuous Integration | Automated testing on push |
| Documentation | Living documentation (updated each sprint) |

---

## 7. Performance Evaluation Parameters

### Overall Model Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | **95%** | Proportion of correct predictions across all classes |
| **Precision** | **0.95** (macro avg) | Proportion of true positives among predicted positives |
| **Recall** | **0.95** (macro avg) | Proportion of true positives among actual positives |
| **F1-Score** | **0.95** (macro avg) | Harmonic mean of precision and recall |

### Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score | Support | Analysis |
|-------|-----------|--------|----------|---------|----------|
| **Glioma** | 1.00 | 0.81 | 0.89 | ~300 | High precision but some missed cases (FN) |
| **Meningioma** | 0.90 | 0.99 | 0.95 | ~306 | Excellent recall, few false negatives |
| **No Tumor** | 0.92 | 1.00 | 0.96 | ~405 | Perfect recall, very reliable for ruling out tumors |
| **Pituitary** | 0.99 | 1.00 | 1.00 | ~300 | Near-perfect performance across all metrics |

### Classification Performance Visualization

```
Precision by Class:
─────────────────────────────────────────────────────
Glioma      |████████████████████████████████████████| 1.00
Meningioma  |████████████████████████████████████    | 0.90
No Tumor    |█████████████████████████████████████   | 0.92
Pituitary   |████████████████████████████████████████| 0.99

Recall by Class:
─────────────────────────────────────────────────────
Glioma      |████████████████████████████████        | 0.81
Meningioma  |████████████████████████████████████████| 0.99
No Tumor    |████████████████████████████████████████| 1.00
Pituitary   |████████████████████████████████████████| 1.00

F1-Score by Class:
─────────────────────────────────────────────────────
Glioma      |████████████████████████████████████    | 0.89
Meningioma  |██████████████████████████████████████  | 0.95
No Tumor    |██████████████████████████████████████  | 0.96
Pituitary   |████████████████████████████████████████| 1.00
```

### Training History

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Notes |
|-------|-----------|-----------|----------|---------|-------|
| 1 | 0.82 | 72% | 0.45 | 84% | Initial learning |
| 2 | 0.38 | 86% | 0.28 | 89% | Rapid improvement |
| 3 | 0.24 | 91% | 0.21 | 91% | Converging |
| 4 | 0.18 | 93% | 0.18 | 92% | Steady progress |
| 5 | 0.14 | 94% | 0.16 | 93% | |
| 6 | 0.11 | 95% | 0.15 | 93% | |
| 7 | 0.09 | 96% | 0.14 | 94% | |
| 8 | 0.07 | 97% | 0.13 | 94% | |
| **9** | **0.06** | **97%** | **0.12** | **95%** | **Best model saved** |
| 10 | 0.05 | 98% | 0.13 | 95% | Slight overfitting begins |

### System Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Model Inference Time | ~1.2s | <2s | ✅ Met |
| Grad-CAM Generation | ~0.5s | <1s | ✅ Met |
| Total API Response Time | ~1.8s | <3s | ✅ Met |
| Frontend Load Time | ~1.5s | <3s | ✅ Met |
| Concurrent Users Supported | 10+ | 5+ | ✅ Met |

### Confusion Matrix Analysis

```
                    Predicted
              Glioma  Menin.  NoTumor  Pituit.
         ┌─────────┬────────┬────────┬────────┐
Glioma   │   243   │   45   │   10   │    2   │  Actual: 300
         ├─────────┼────────┼────────┼────────┤
Menin.   │    0    │  303   │    3   │    0   │  Actual: 306
         ├─────────┼────────┼────────┼────────┤
NoTumor  │    0    │    0   │  405   │    0   │  Actual: 405
         ├─────────┼────────┼────────┼────────┤
Pituit.  │    0    │    0   │    0   │  300   │  Actual: 300
         └─────────┴────────┴────────┴────────┘

Key Observation: Glioma samples occasionally misclassified as Meningioma
→ Both are intra-axial tumors with similar MRI appearance in some cases
→ Improvement planned: additional glioma training data + augmentation
```

---

## 8. Software and Hardware Setup

### Software Stack

#### Machine Learning / AI

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Python | 3.11 | ML service development |
| DL Framework | PyTorch | 2.2 | Model training and inference |
| Model Library | timm | Latest | Pre-trained EfficientNet-B0 |
| ML API | FastAPI | Latest | ML service REST endpoints |
| Image Processing | Pillow, OpenCV | Latest | MRI preprocessing |
| Data Science | NumPy, Matplotlib | Latest | Data manipulation & visualization |
| ASGI Server | Uvicorn | Latest | Serving FastAPI application |

#### Backend (API Gateway)

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Runtime | Node.js | 18.x | Server-side JavaScript |
| Framework | Express.js | 4.18 | HTTP server and routing |
| File Upload | Multer | Latest | Multipart form data handling |
| HTTP Client | Axios | Latest | Communication with FastAPI |
| Database Driver | Mongoose | Latest | MongoDB ODM |
| CORS | cors | Latest | Cross-origin resource sharing |

#### Frontend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Library | React | 18 | UI component framework |
| Build Tool | Vite | Latest | Fast development server and bundler |
| 3D Graphics | Three.js | Latest | Interactive 3D brain visualization |
| Animation | GSAP | Latest | Smooth UI animations and transitions |
| HTTP Client | Axios | Latest | API communication |
| Styling | Tailwind CSS / CSS Modules | Latest | Responsive design |

#### Database & Infrastructure

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Database | MongoDB | 6.x | NoSQL document storage for predictions |
| Version Control | Git + GitHub | Latest | Source code management |
| Package Manager | npm / pip | Latest | Dependency management |

### Hardware Setup

#### Training Environment (Google Colab)

| Specification | Details |
|--------------|---------|
| **GPU** | NVIDIA Tesla T4 |
| **VRAM** | 16 GB GDDR6 |
| **RAM** | 12 GB |
| **Storage** | ~100 GB (Colab disk) |
| **CUDA Version** | 12.x |
| **Training Time** | ~15 minutes (10 epochs) |
| **Cost** | Free (Colab free tier) |

#### Development Environment

| Specification | Details |
|--------------|---------|
| **OS** | Windows 10/11 |
| **Processor** | Intel / AMD (multi-core) |
| **RAM** | 8-16 GB |
| **Storage** | SSD (256 GB+) |
| **GPU (Inference)** | CPU-only (GPU optional) |
| **IDE** | VS Code with Python & JS extensions |

#### Minimum Deployment Requirements

| Specification | Minimum | Recommended |
|--------------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **CPU** | 4 cores | 8 cores |
| **Storage** | 10 GB | 20 GB |
| **GPU** | Not required | NVIDIA GPU (faster inference) |
| **OS** | Any (Linux/Windows/macOS) | Linux (Ubuntu 22.04) |
| **Network** | Required | Low-latency connection |

### Development Tools

| Tool | Purpose |
|------|---------|
| VS Code | Primary IDE |
| Postman | API testing |
| MongoDB Compass | Database management |
| Google Colab | Model training |
| Git / GitHub | Version control |
| Chrome DevTools | Frontend debugging |
| PyTorch Profiler | Model performance analysis |

---

## 9. Initial Implementation Results (25-30%)

### Completed Implementation Summary

| Component | Status | Completion |
|-----------|--------|------------|
| Dataset Collection & Preprocessing | ✅ Complete | 100% |
| Model Training & Evaluation | ✅ Complete | 100% |
| Grad-CAM Explainability | ✅ Complete | 100% |
| FastAPI ML Service | ✅ Complete | 100% |
| Express.js API Gateway | ✅ Complete | 100% |
| React Frontend | ✅ Complete | 95% |
| MongoDB Integration | ✅ Complete | 100% |
| Testing & Documentation | 🔄 In Progress | 60% |
| Deployment | 📋 Planned | 10% |

### Key Results Achieved

#### 1. Model Performance ✅
- **95% overall accuracy** on test set (1,311 images)
- Trained in **~15 minutes** on Google Colab Tesla T4
- Model size: **~20 MB** (efficient for deployment)
- Best validation accuracy achieved at **epoch 9**

#### 2. Backend API ✅
- Express.js gateway handling file uploads (up to 10MB)
- Image validation (JPEG, PNG, DICOM support planned)
- FastAPI ML service with async inference
- Health check endpoints for monitoring
- Request logging and error handling middleware

#### 3. Frontend Application ✅
- Interactive dashboard with file upload (drag & drop)
- Real-time prediction results display
- Confidence scores with visual indicators
- Grad-CAM heatmap overlay visualization
- 3D brain model visualization (Three.js)
- Smooth animations and transitions (GSAP)
- Responsive design (mobile + desktop)
- Prediction history view

#### 4. Grad-CAM Explainability ✅
- Heatmap generation from last convolutional layer
- Color-coded overlay (blue = low attention, red = high attention)
- Highlights tumor region for clinician verification
- Generated in <0.5 seconds per image

#### 5. Database Integration ✅
- MongoDB stores all predictions with metadata
- Prediction history with timestamps
- Searchable prediction records
- Stores: filename, class, confidence, probabilities, timestamp

### Sample Prediction Output

```json
{
  "prediction": {
    "class": "Meningioma",
    "confidence": 0.9847,
    "probabilities": {
      "glioma": 0.0023,
      "meningioma": 0.9847,
      "no_tumor": 0.0089,
      "pituitary": 0.0041
    },
    "inference_time_ms": 1247,
    "gradcam_available": true
  },
  "metadata": {
    "filename": "brain_mri_scan_001.jpg",
    "timestamp": "2024-12-15T10:30:00Z",
    "model_version": "efficientnet_b0_v1"
  }
}
```

### Project Structure

```
NeuraSight/
├── backend/
│   ├── express/              # API Gateway (Node.js)
│   │   ├── src/
│   │   │   ├── config/       # Environment configuration
│   │   │   ├── middleware/   # Error handling, validation, logging
│   │   │   ├── models/       # MongoDB schemas (Prediction)
│   │   │   ├── routes/       # API endpoints (predict, gradcam, health)
│   │   │   ├── services/     # FastAPI client service
│   │   │   └── server.js     # Entry point
│   │   └── package.json
│   │
│   └── fastapi/              # ML Service (Python)
│       ├── app/
│       │   ├── models/       # Pydantic schemas
│       │   ├── routers/      # API routes (predict, gradcam, health)
│       │   ├── services/     # Inference, preprocessing, Grad-CAM
│       │   ├── config.py     # Configuration
│       │   └── main.py       # FastAPI entry point
│       ├── requirements.txt
│       └── tests/
│
├── frontend/                 # React Application
│   ├── src/
│   │   ├── components/       # UI components
│   │   ├── pages/            # Page views
│   │   ├── assets/           # Static assets
│   │   └── App.jsx           # Root component
│   └── package.json
│
├── data/
│   └── brainMRI/             # Dataset (7,023 images)
│       ├── Training/         # Train split
│       │   ├── glioma/
│       │   ├── meningioma/
│       │   ├── notumor/
│       │   └── pituitary/
│       └── Testing/          # Test split
│           ├── glioma/
│           ├── meningioma/
│           ├── notumor/
│           └── pituitary/
│
├── models/                   # Saved model weights
├── notebooks/                # Colab training notebooks
└── docs/                     # Documentation
```

### Known Limitations & Planned Improvements

| Limitation | Impact | Planned Solution |
|-----------|--------|-----------------|
| Glioma recall at 81% | Some glioma cases missed | Additional training data, focal loss |
| CPU-only inference | Slower than GPU | Optional GPU acceleration support |
| Single model | No ensemble benefit | Ensemble with ResNet for borderline cases |
| No DICOM support | Limited hospital integration | DICOM parser integration planned |
| No user authentication | Security gap | JWT authentication in next sprint |

### Next Steps (Remaining 70-75%)

1. **Improve Glioma recall** — Additional augmentation, class-weighted sampling
2. **Add user authentication** — JWT-based login system
3. **DICOM support** — Parse standard medical imaging format
4. **Docker containerization** — Consistent deployment across environments
5. **Comprehensive testing** — Unit, integration, and end-to-end tests
6. **Performance optimization** — Model quantization for faster inference
7. **Report generation** — PDF reports for clinicians
8. **Final deployment** — Cloud hosting (AWS/GCP) with CI/CD

---

## Summary

| Aspect | Details |
|--------|---------|
| **Project** | NeuraSight – AI Brain Tumor Detection |
| **Model** | EfficientNet-B0 (Transfer Learning) |
| **Accuracy** | 95% (4-class classification) |
| **Architecture** | React → Express → FastAPI → MongoDB |
| **Explainability** | Grad-CAM heatmaps |
| **Training** | Google Colab, Tesla T4 GPU, 10 epochs |
| **Status** | Core functionality complete, testing & deployment in progress |

---

## References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*. [arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
2. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*. [arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
3. Litjens, G., et al. (2017). A Survey on Deep Learning in Medical Image Analysis. *Medical Image Analysis*. [arxiv.org/abs/1702.05747](https://arxiv.org/abs/1702.05747)
4. Shin, H. C., et al. (2016). Deep Convolutional Neural Networks for Computer-Aided Detection. *IEEE TMI*. [arxiv.org/abs/1608.00853](https://arxiv.org/abs/1608.00853)
5. Ayadi, W., et al. (2020). Brain Tumor Classification Based on Deep Learning. [arxiv.org/abs/2010.14701](https://arxiv.org/abs/2010.14701)

---

*Document prepared for Final Year Project Presentation | NeuraSight Team*  
*Last Updated: 2024*
