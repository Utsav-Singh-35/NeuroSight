# NeuraSight: AI-Powered Brain Tumor Detection System

## Final Year Project Presentation

**Project Title:** NeuraSight – Automated Brain Tumor Classification from MRI Scans using a Deep Learning Stacking Ensemble
**Domain:** Healthcare + Artificial Intelligence
**Tech Stack:** EfficientNet-B0 · ResNet-50 · DenseNet-121 · VGG-16 · Logistic Regression · React · Express.js · FastAPI · MongoDB
**Team:** Final Year – Computer Science / IT

---

## Table of Contents

1. [Domain & Motivation](#1-domain--motivation)
2. [Problem Statement](#2-problem-statement)
3. [Project Planning](#3-project-planning)
4. [Literature Survey](#4-literature-survey)
5. [System Architecture & Flowchart](#5-system-architecture--flowchart)
6. [Adopted Methodology](#6-adopted-methodology)
7. [Performance Evaluation Parameters](#7-performance-evaluation-parameters)
8. [Tools & Tech Stack](#8-tools--tech-stack)
9. [Implementation Status](#9-implementation-status)

---

## 1. Domain & Motivation

### Domain: Healthcare + Artificial Intelligence

Healthcare is one of the most impactful domains for AI. The intersection of medical imaging and deep learning helps clinicians make faster, more accurate diagnoses.

### Why Brain Tumor Detection?

| Factor | Details |
|--------|---------|
| **Prevalence** | Brain tumors account for 85–90% of all primary CNS tumors globally |
| **Mortality** | Glioblastoma has a 5-year survival rate of only ~5% |
| **Diagnosis Challenge** | Manual MRI analysis takes 15–30 minutes per scan by expert radiologists |
| **Shortage** | Many regions face a critical shortage of trained neuroradiologists |

### Motivation

- **Early detection saves lives** — accurate, early classification improves survival rates.
- **Manual analysis is slow and subjective** — AI provides a fast, consistent "second opinion."
- **Explainability builds trust** — Grad-CAM shows which MRI regions drove the prediction.
- **A stronger research story** — instead of one CNN, we train and compare four architectures and combine them with a stacking ensemble.

---

## 2. Problem Statement

> **"Develop an AI-powered system for automated brain tumor classification from MRI scans with explainability — classifying brain MRI images into four categories (Glioma, Meningioma, No Tumor, Pituitary) with high accuracy, using a stacking ensemble of multiple deep-learning models, while providing visual explanations (Grad-CAM) to support clinical decision-making."**

### Problem Decomposition

1. **Classification** — 4-class classification from MRI images
2. **Model comparison** — evaluate multiple CNN architectures quantitatively
3. **Ensemble** — combine model strengths via a stacking meta-learner
4. **Explainability** — Grad-CAM heatmaps to highlight tumor regions
5. **Deployment** — web interface accessible to non-technical users

---

## 3. Project Planning

### Revised Roadmap (Stacking Ensemble Approach)

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Dataset, EDA, preprocessing, EfficientNet-B0 (95.06%), Grad-CAM, evaluation | ✅ Complete |
| **Phase 2** | Train multiple base models (ResNet-50, DenseNet-121, VGG-16) with hyperparameter tuning | ✅ Complete |
| **Phase 3** | Quantitative comparison of all base models | ✅ Complete |
| **Phase 4** | Build stacking ensemble (Logistic Regression meta-learner) — **96.75%** | ✅ Complete |
| **Phase 5** | Backend integration + new modules (auth, DICOM, PDF) | ⏳ Planned |
| **Phase 6** | Testing, Docker, cloud deployment | ⏳ Planned |

### Research Workflow

```
Brain MRI Dataset
        │
   ┌────┴─────┬───────────┬───────────┐
   ▼          ▼           ▼           ▼
EfficientNet ResNet50  DenseNet121  VGG16
   │          │           │           │
   ▼          ▼           ▼           ▼
Hyperparam  Hyperparam  Hyperparam  Hyperparam
 Tuning      Tuning      Tuning      Tuning
   │          │           │           │
   └──────────┴─────┬─────┴───────────┘
                    ▼
          Quantitative Comparison
                    ▼
          Stacking Ensemble
      (Logistic Regression Meta-Learner)
                    ▼
             Final System
```

### Milestones

| Milestone | Status |
|-----------|--------|
| Dataset prepared (7,023 images) | ✅ |
| EfficientNet-B0 trained (95.06%) | ✅ |
| Grad-CAM working | ✅ |
| Full-stack integration | ✅ |
| ResNet-50 / DenseNet-121 / VGG-16 trained | ✅ |
| Stacking ensemble built (96.75%) | ✅ |
| Deployment | ⏳ |

---

## 4. Literature Survey

| # | Paper | Authors | Year | Key Contribution |
|---|-------|---------|------|-----------------|
| 1 | EfficientNet: Rethinking Model Scaling for CNNs | Tan & Le | 2019 | Compound scaling; SOTA with fewer parameters |
| 2 | Deep Residual Learning (ResNet) | He et al. | 2015 | Residual connections enable very deep networks |
| 3 | Densely Connected Networks (DenseNet) | Huang et al. | 2017 | Dense connectivity; feature reuse, fewer params |
| 4 | Very Deep CNNs (VGG) | Simonyan & Zisserman | 2014 | Simple, deep stacked 3×3 conv architecture |
| 5 | Grad-CAM: Visual Explanations | Selvaraju et al. | 2017 | Gradient-weighted class activation maps |
| 6 | Stacked Generalization | Wolpert | 1992 | Foundational stacking ensemble method |

### Comparative Analysis of Architectures

| Model | Parameters | ImageNet Acc | Role in NeuraSight |
|-------|-----------|-------------|--------------------|
| EfficientNet-B0 | 5.3M | 77.1% | Base learner (primary) |
| ResNet-50 | 25.6M | 76.1% | Base learner (comparison) |
| DenseNet-121 | 8.0M | 74.4% | Base learner (comparison) |
| VGG-16 | 138M | 71.3% | Base learner (comparison) |
| Logistic Regression | — | — | Meta-learner (stacking) |

**Why a stacking ensemble?** Different architectures make different errors. Combining their probability outputs via a simple, explainable meta-learner (Logistic Regression) typically improves accuracy and robustness over any single model.

---

## 5. System Architecture & Flowchart

### 5.1 High-Level System Architecture

```
┌──────────┐   HTTP  ┌─────────────────────┐  HTTP  ┌──────────────────────┐
│  React   │────────▶│  Express.js Gateway │───────▶│  FastAPI ML Service   │
│ Frontend │◀────────│  (:5000)            │◀───────│  (:8000)              │
│ (:3000)  │         │  • Auth (JWT)       │        │  • Preprocessing      │
└──────────┘         │  • Image validation │        │  • Stacking Ensemble  │
                     │  • Request routing  │        │    - EfficientNet-B0  │
                     │  • PDF generation   │        │    - ResNet-50        │
                     └─────────┬───────────┘        │    - DenseNet-121     │
                               │                    │    - VGG-16           │
                               ▼                    │    - LR Meta-Learner  │
                     ┌─────────────────────┐        │  • Grad-CAM           │
                     │      MongoDB        │        │  • AI Report          │
                     │  • predictions      │        └──────────────────────┘
                     │  • users            │
                     └─────────────────────┘
```

### 5.2 Stacking Ensemble Architecture

```
                         Brain MRI Image
                               │
                               ▼
                        Preprocessing
              (RGB, resize 224×224, ImageNet norm)
                               │
        ┌──────────────┬───────┴───────┬──────────────┐
        ▼              ▼               ▼              ▼
  EfficientNet-B0  ResNet-50     DenseNet-121      VGG-16
        │              │               │              │
        ▼              ▼               ▼              ▼
   Softmax [4]     Softmax [4]     Softmax [4]    Softmax [4]
        │              │               │              │
        └──────────────┴───────┬───────┴──────────────┘
                               ▼
                Feature Concatenation → [16 features]
                               ▼
              Logistic Regression Meta-Learner
                               ▼
                    Final Prediction (4 classes)
                               ▼
                Grad-CAM (from agreeing base model)
                               ▼
                        AI Clinical Report
```

### 5.3 Prediction Flow

```
User → React → Express (validate) → FastAPI
                                       │
                                       ▼
                           Preprocess image
                                       ▼
                     Run 4 base models → 4 probability vectors
                                       ▼
                     Concatenate → Logistic Regression meta-learner
                                       ▼
                           Final prediction + confidence
                                       ▼
             Grad-CAM on the base model matching the final class
                                       ▼
Express → store in MongoDB → return {prediction, confidence, heatmap, report}
                                       ▼
                             React displays results
```

### 5.4 Grad-CAM Strategy for the Ensemble

- **Default view:** heatmap from whichever base model's prediction agreed with the ensemble's final class (most faithful single explanation).
- **Detailed view (optional):** all base-model heatmaps side by side for full transparency.
- The frontend calls the same API — it does not need to know there are four models behind the "stacking engine."

---

## 6. Adopted Methodology

### Three Pillars

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Transfer    │  │  Stacking    │  │ Explainable  │
│  Learning    │  │  Ensemble    │  │      AI      │
│              │  │              │  │              │
│ 4 pretrained │  │ LR meta-     │  │  Grad-CAM    │
│ CNNs         │  │ learner      │  │  heatmaps    │
└──────────────┘  └──────────────┘  └──────────────┘
```

### 1. Transfer Learning

Each base model is a `timm` architecture pre-trained on ImageNet, fine-tuned on 5,600 brain MRI images with a 4-class head. Shared, fair pipeline (same data, splits, augmentation) so the comparison is valid.

### 2. Hyperparameter Tuning

Tune only the important parameters per model: **Learning Rate, Batch Size, Epochs, Optimizer, Weight Decay**. CosineAnnealing LR scheduler applied across training.

### 3. Stacking Ensemble

- Each base model outputs a 4-class softmax probability vector.
- Vectors are concatenated into a 16-dimensional feature vector.
- A **Logistic Regression** meta-learner is trained on these features.
- **Leakage-safe:** the meta-learner trains on one half of the test-set probabilities and is evaluated on the untouched other half.

**Why Logistic Regression as meta-learner?** Simple, fast, highly explainable, and excellent for stacking probability inputs — a standard research choice over a second CNN.

### 4. Explainability (Grad-CAM)

Gradient-weighted class activation maps target the last convolutional layer of the base model that agreed with the ensemble's final prediction. JET colormap overlaid at 40% opacity.

### 5. Development Methodology: Agile

1–2 week sprints per phase, iterative integration (model → API → frontend), Git version control, living documentation.

---

## 7. Performance Evaluation Parameters

### 7.1 Current Result — EfficientNet-B0 (Phase 1, Completed)

| Metric | Score |
|--------|-------|
| **Accuracy** | **95.06%** |
| **Precision** (macro) | 95.52 |
| **Recall** (macro) | 95.06 |
| **F1-Score** (macro) | 94.92 |

**Per-class (EfficientNet-B0, 1600 test images):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Glioma | 1.00 | 0.81 | 0.89 | 400 |
| Meningioma | 0.90 | 0.99 | 0.95 | 400 |
| No Tumor | 0.92 | 1.00 | 0.96 | 400 |
| Pituitary | 0.99 | 1.00 | 1.00 | 400 |

**Per-class (Stacking Ensemble, 800-image meta-test split):**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 0.98 | 0.90 | 0.93 |
| Meningioma | 0.95 | 0.98 | 0.97 |
| No Tumor | 0.95 | 1.00 | 0.97 |
| Pituitary | 0.99 | 0.99 | 0.99 |

**Ensemble overall:** Accuracy 96.75% | Precision 96.81% | Recall 96.75% | F1 96.71%

**Fair comparison on same 800-image meta-test split:**

| Model | Accuracy |
|-------|----------|
| EfficientNet-B0 | 95.12% |
| ResNet-50 | 95.12% |
| DenseNet-121 | 95.88% |
| VGG-16 | 94.75% |
| **Stacking Ensemble** | **96.75%** |

### 7.2 Model Comparison Table (Full 1600-image Test Set)

| Model | Accuracy | Precision | Recall | F1 | Best Val Acc |
|-------|----------|-----------|--------|-----|-------------|
| DenseNet-121 | 96.06% | 96.32 | 96.06 | 95.97 | 96.06% |
| EfficientNet-B0 | 95.06% | 95.52 | 95.06 | 94.92 | 95.06% |
| ResNet-50 | 94.88% | 95.29 | 94.88 | 94.74 | 94.88% |
| VGG-16 | 94.56% | 95.12 | 94.56 | 94.41 | 94.56% |
| **Stacking Ensemble** | **96.75%** | **96.81** | **96.75** | **96.71** | — (meta-test) |

> DenseNet-121 is the best single model (96.06%), but the stacking ensemble (96.75%) beats all individual models on the held-out 800-image meta-test split.

### 7.3 Evaluation Methodology

- **Base models** evaluated on the 1,600-image test set (400 per class).
- **Ensemble** evaluated on a held-out half of the test-set probabilities (leakage-safe stratified split) — its accuracy is compared against each base model on that same split for a fair comparison.
- Metrics: Accuracy, macro Precision/Recall/F1, per-class breakdown, confusion matrix, inference time.

### 7.4 System Performance Targets

| Metric | Target | EfficientNet-B0 (actual) |
|--------|--------|--------------------------|
| Single-model inference | < 2s | ~1.2s ✅ |
| Ensemble inference (4 models, CPU) | < 4s | TBD |
| Grad-CAM generation | < 1s | ~0.5s ✅ |
| Total API response | < 5s | ~1.8s (single) ✅ |

### 7.5 Research Story (Report Highlight)

Instead of "trained one CNN → 95%", the report shows:

```
EfficientNet-B0   → 95.06%
ResNet-50         → 94.88%
DenseNet-121      → 96.06%  ← best single model
VGG-16            → 94.56%
Stacking Ensemble → 96.75%  ← best overall
```

Demonstrating: multiple architectures evaluated, each tuned, combined via stacking, final system selected on quantitative evidence. The ensemble improved on the best single-model baseline (96.06% → 96.75%).

---

## 8. Tools & Tech Stack

### Machine Learning & AI

| Tool | Purpose |
|------|---------|
| Python 3.11 | ML development language |
| PyTorch 2.2 | Deep learning framework |
| timm | Pretrained models (EfficientNet, ResNet, DenseNet, VGG) |
| scikit-learn | Logistic Regression meta-learner, metrics |
| Pillow, OpenCV | Image processing, Grad-CAM overlay |
| NumPy, pandas | Array ops, comparison tables |
| pytorch-grad-cam | Explainability heatmaps |

### Backend

| Tool | Purpose |
|------|---------|
| FastAPI + Uvicorn | ML inference service (:8000) |
| Express.js + Node.js 18 | API gateway (:5000) |
| Multer, Axios | File upload, service-to-service HTTP |
| Mongoose + MongoDB | Prediction & user storage |
| JWT, bcrypt | Authentication (planned) |
| pydicom | DICOM support (planned) |

### Frontend

| Tool | Purpose |
|------|---------|
| React 18 + Vite | UI + build tooling |
| Three.js | 3D brain visualization |
| GSAP + Lenis | Animations, smooth scroll |

### DevOps & Training

| Tool | Purpose |
|------|---------|
| Google Colab (Tesla T4) | Model training |
| Docker + Docker Compose | Containerization (planned) |
| GitHub Actions | CI/CD (planned) |
| AWS / GCP | Cloud hosting (planned) |
| Git / GitHub | Version control |

### Tech Stack Layers

```
CLIENT      React · Vite · Three.js · GSAP
   │
GATEWAY     Express.js · Multer · JWT · Mongoose
   │
ML SERVICE  FastAPI · PyTorch · timm · scikit-learn · pytorch-grad-cam
   │
DATA        MongoDB · model weights (.pth) · meta_model.pkl
   │
INFRA       Docker · nginx · GitHub Actions · AWS/GCP  (planned)
```

---

## 9. Implementation Status

| Component | Status |
|-----------|--------|
| Dataset, preprocessing, EDA | ✅ Complete |
| EfficientNet-B0 (95.06%) | ✅ Complete |
| Grad-CAM explainability | ✅ Complete |
| FastAPI ML service | ✅ Complete |
| Express.js gateway + MongoDB | ✅ Complete |
| React frontend (landing + dashboard) | ✅ Complete |
| AI report generator | ✅ Complete |
| ResNet-50 / DenseNet-121 / VGG-16 training | ✅ Complete |
| Model comparison | ✅ Complete |
| Stacking ensemble (meta-learner, 96.75%) | ✅ Complete |
| Backend `ai/` ensemble integration | ⏳ Planned |
| JWT auth, DICOM, PDF reports | ⏳ Planned |
| Docker + cloud deployment | ⏳ Planned |

### Future Expansion (Multi-Modal Platform)

CT scan analysis · Chest X-ray diagnostics · Retinal imaging · Skin lesion detection · Multi-modal fusion.

---

## Summary

| Aspect | Details |
|--------|---------|
| **Project** | NeuraSight – AI Brain Tumor Detection with Stacking Ensemble |
| **Architecture** | React → Express → FastAPI → MongoDB (microservices) |
| **Base Models** | EfficientNet-B0, ResNet-50, DenseNet-121, VGG-16 |
| **Meta-Learner** | Logistic Regression (stacking) |
| **Explainability** | Grad-CAM (from agreeing base model) |
| **Accuracy** | 95.06% (EfficientNet-B0) → **96.75% (stacking ensemble)** |
| **Training** | Google Colab, Tesla T4 GPU |

---

*Prepared for Final Year Project Presentation | NeuraSight Team*
