# NeuraSight 🧠

**AI-Powered Medical Imaging Platform for Tumor Detection & Clinical Decision Support**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933.svg)](https://nodejs.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

NeuraSight is a medical imaging platform that classifies brain tumors from MRI scans using deep learning. It is designed as an extensible clinical decision support system that will expand to cover multiple imaging modalities including CT scans, X-rays, retinal imaging, and skin lesion detection.

### Current Capabilities (Brain MRI)
- **95% classification accuracy** across 4 tumor types using EfficientNet-B0
- **Grad-CAM heatmaps** showing which MRI regions the model focuses on
- **AI clinical reports** with risk assessment and recommendations
- **REST API** for integration with any frontend
- **Prediction history** stored in MongoDB

### Planned Capabilities (Future Modules)
- **CT Scan Analysis** — Lung nodules, abdominal abnormalities
- **Chest X-Ray Diagnostics** — Pneumonia, COVID-19, tuberculosis screening
- **Retinal Imaging** — Diabetic retinopathy, macular degeneration
- **Skin Lesion Detection** — Melanoma and skin cancer from dermoscopic images
- **Multi-Modal Fusion** — Combine multiple imaging types for comprehensive diagnosis

---

## Architecture

```
┌──────────┐      ┌─────────────────────┐      ┌──────────────────┐
│  React   │ HTTP │   Express Backend    │ HTTP │  FastAPI Service  │
│ Frontend │─────▶│   (Node.js :5000)   │─────▶│  (Python :8000)   │
│ :3000    │◀─────│                     │◀─────│                   │
└──────────┘      │ • Authentication    │      │ • Preprocessing   │
                  │ • Image validation  │      │ • Model inference │
                  │ • Request routing   │      │ • Grad-CAM        │
                  │ • Prediction history│      │ • AI Reports      │
                  │ • PDF generation    │      │ • DICOM parsing   │
                  └─────────┬───────────┘      │ • Model comparison│
                            │                  └──────────────────┘
                            ▼
                  ┌─────────────────────┐
                  │     MongoDB :27017  │
                  │  • predictions      │
                  │  • user accounts    │
                  └─────────────────────┘
```

---

## Brain MRI Classification Results

### Tumor Classes (Current Model: EfficientNet-B0)
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 1.00 | 0.81 | 0.89 |
| Meningioma | 0.90 | 0.99 | 0.95 |
| No Tumor | 0.92 | 1.00 | 0.96 |
| Pituitary | 0.99 | 1.00 | 1.00 |
| **Overall** | **0.95** | **0.95** | **0.95** |

### Stacking Ensemble (Research Approach)

NeuraSight trains **four base CNN models**, compares them, then combines them with a **Logistic Regression stacking meta-learner** for the final prediction.

| Model | Parameters | Role | Status |
|-------|-----------|------|--------|
| **EfficientNet-B0** | 5.3M | Base learner (primary) | ✅ Trained |
| **ResNet-50** | 25.6M | Base learner | ✅ Trained |
| **DenseNet-121** | 8.0M | Base learner | ✅ Trained |
| **VGG-16** | 138M | Base learner | ✅ Trained |
| **Logistic Regression** | — | Meta-learner (stacking) | ✅ Trained |
| **Stacking Ensemble** | — | Final system | ✅ **96.75% accuracy** |

**How stacking works:** each base model outputs a 4-class softmax probability vector; the four vectors are concatenated into a 16-dimensional feature vector; a Logistic Regression meta-learner produces the final prediction. The meta-learner was trained leakage-safe on a held-out half of the test-set probabilities and reached **96.75%** accuracy — an improvement over the 95% single EfficientNet-B0 baseline.

**Trained model artifacts:** `BRAIN_MRI_EFFICIENTNET.pth`, `BRAIN_MRI_RESNET.pth`, `BRAIN_MRI_DENSENET.pth`, `BRAIN_MRI_VGG.pth`, plus `meta_model.pkl` and `ensemble_config.json`.

---

## Project Structure

```
NeuraSight/
├── backend/
│   ├── express/              # Node.js API gateway (port 5000)
│   │   ├── src/
│   │   │   ├── server.js     # Entry point
│   │   │   ├── config/       # Environment config
│   │   │   ├── middleware/   # Validation, logging, error handling
│   │   │   ├── routes/       # API endpoints
│   │   │   ├── services/     # FastAPI HTTP client
│   │   │   └── models/       # Mongoose schemas
│   │   └── package.json
│   │
│   ├── fastapi/              # Python ML service (port 8000)
│   │   ├── app/
│   │   │   ├── main.py       # Entry point
│   │   │   ├── config.py     # Settings
│   │   │   ├── routers/      # API endpoints (predict, gradcam, report, health)
│   │   │   ├── services/     # Preprocessing, inference, Grad-CAM, reports
│   │   │   └── models/       # Pydantic schemas
│   │   └── requirements.txt
│   │
│   └── README.md             # Backend setup guide
│
├── frontend/                 # React web interface
│   ├── src/                  # Landing page components
│   ├── index.html            # Landing page
│   ├── dashboard.html        # MRI upload & results dashboard
│   ├── research.html         # Research & documentation page
│   └── package.json
│
├── models/
│   ├── BRAIN_MRI_EFFICIENTNET.pth  # EfficientNet-B0 base learner
│   ├── BRAIN_MRI_RESNET.pth        # ResNet-50 base learner
│   ├── BRAIN_MRI_DENSENET.pth      # DenseNet-121 base learner
│   ├── BRAIN_MRI_VGG.pth           # VGG-16 base learner
│   ├── meta_model.pkl              # Logistic Regression meta-learner
│   └── ensemble_config.json        # Ensemble model order + config
│
├── data/
│   └── brainMRI/             # Dataset (Training + Testing)
│       ├── Training/         # 5,600 images (4 classes)
│       └── Testing/          # 1,600 images (400 per class)
│
├── docs/                     # Documentation
│   ├── system_architecture.md
│   ├── gantt_chart.md
│   ├── brain_mri_model_results.md
│   ├── model_architecture.md
│   ├── dataset_analysis.md
│   ├── improvements.md
│   └── references.md
│
├── notebooks/                # Jupyter training notebooks
│   ├── BRAIN_MRI_SCAN.ipynb      # Trains all 4 base models (one run)
│   └── BRAIN_MRI_ENSEMBLE.ipynb  # Trains stacking meta-learner
├── model_architecture.md     # Detailed model architecture doc
├── presentation.md           # Final year project presentation
└── README.md                 # ← You are here
```

---

## Development Status

### ✅ Completed

| Module | Description | Completion |
|--------|-------------|------------|
| Dataset Collection & Preprocessing | 7,023 brain MRI images, augmentation pipeline | 100% |
| EfficientNet-B0 Training | Transfer learning, 95% accuracy, 10 epochs | 100% |
| Model Evaluation | Classification report, confusion matrix, metrics | 100% |
| Grad-CAM Explainability | Heatmap generation, JET colormap overlay | 100% |
| FastAPI ML Service | Inference, Grad-CAM, report endpoints | 100% |
| Express.js API Gateway | Routing, validation, error handling, logging | 100% |
| MongoDB Integration | Prediction history storage | 100% |
| AI Report Generator | Clinical summary, risk assessment, recommendations | 100% |
| React Frontend (Landing) | Landing page, 3D brain model, animations | 100% |
| React Frontend (Dashboard) | MRI upload, results display, heatmap view | 100% |
| Research Page | Documentation, datasets, model info | 100% |

### ⏳ In Progress / Planned

| Module | Description | Target Date |
|--------|-------------|-------------|
| **ResNet-50 Training** | Base learner (25.6M params) | Jul–Sep 2026 |
| **DenseNet-121 Training** | Base learner (8.0M params) | Jul–Sep 2026 |
| **VGG-16 Training** | Base learner (138M params) | Jul–Sep 2026 |
| **Model Comparison Report** | Accuracy/speed/size comparison across all base models | Sep–Oct 2026 |
| **Stacking Ensemble** | Logistic Regression meta-learner over the 4 base models | Sep–Oct 2026 |
| **Backend Ensemble Integration** | `ai/` module loads all 4 CNNs + meta-learner; ensemble Grad-CAM | Oct 2026 |
| **User Authentication** | JWT-based login, role-based access (doctor/researcher/admin) | Jul–Aug 2026 |
| **DICOM Support** | Parse standard medical imaging format (.dcm files) | Aug–Sep 2026 |
| **PDF Report Generation** | Downloadable clinical reports with heatmaps | Sep–Oct 2026 |
| **Docker Containerization** | Containerize all services for consistent deployment | Oct 2026 |
| **Comprehensive Testing** | Unit, integration, and end-to-end test suites | Oct–Nov 2026 |
| **Model Quantization** | INT8 quantization for faster inference | Nov 2026 |
| **Cloud Deployment** | AWS/GCP hosting with CI/CD (GitHub Actions) | Nov–Dec 2026 |

### 🔮 Future Expansion (Post-Submission)

| Module | Description | Imaging Type |
|--------|-------------|--------------|
| **CT Scan Analysis** | Lung nodule detection, abdominal abnormalities | Computed Tomography |
| **Chest X-Ray Diagnostics** | Pneumonia, COVID-19, tuberculosis screening | X-Ray |
| **Retinal Imaging** | Diabetic retinopathy, macular degeneration | Fundus Photography |
| **Skin Lesion Detection** | Melanoma, skin cancer classification | Dermoscopic Images |
| **Multi-Modal Fusion** | Combine MRI + CT + X-Ray for comprehensive diagnosis | Multiple |
| **Real-Time Edge Inference** | Optimized models for hospital edge devices | All |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB 6.0+
- Trained model file: `BRAIN_MRI_EFFICIENTNET.pth` (in `models/`)

### 1. Start FastAPI (ML Service)

```bash
cd backend/fastapi
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Ensure models/BRAIN_MRI_EFFICIENTNET.pth exists
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Start Express (API Gateway)

```bash
cd backend/express
npm install

# Ensure MongoDB is running
npm start
```

### 3. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Test the API

```bash
# Health check
curl http://localhost:5000/api/health

# Predict tumor class
curl -X POST -F "image=@path/to/mri_scan.jpg" http://localhost:5000/api/predict

# Get Grad-CAM heatmap
curl -X POST -F "image=@path/to/mri_scan.jpg" http://localhost:5000/api/gradcam

# Get full AI report
curl -X POST -F "image=@path/to/mri_scan.jpg" http://localhost:5000/api/report

# View prediction history
curl http://localhost:5000/api/predictions
```

---

## API Endpoints

### Express Backend (port 5000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Upload MRI → get classification + confidence |
| `POST` | `/api/gradcam` | Upload MRI → get Grad-CAM heatmap (base64 PNG) |
| `POST` | `/api/report` | Upload MRI → get full AI clinical report |
| `GET` | `/api/predictions` | List prediction history (max 100) |
| `GET` | `/api/predictions/:id` | Get single prediction record |
| `GET` | `/api/health` | Service + FastAPI reachability status |

### Response Examples

**Prediction:**
```json
{
  "prediction": "Glioma",
  "confidence": 95.42,
  "probabilities": {
    "Glioma": 95.42,
    "Meningioma": 2.15,
    "No Tumor": 1.30,
    "Pituitary": 1.13
  }
}
```

**AI Report:**
```json
{
  "prediction": "Meningioma",
  "confidence": 98.7,
  "risk_level": "Medium",
  "description": "Meningioma is a tumor that arises from the meninges...",
  "ai_summary": "The MRI shows features consistent with a Meningioma...",
  "recommendation": "Consultation with a neurologist or neurosurgeon is recommended...",
  "disclaimer": "This AI-generated report is for clinical decision support only...",
  "probabilities": { ... }
}
```

---

## Model Details

### Current Model (EfficientNet-B0)

| Attribute | Value |
|-----------|-------|
| Architecture | EfficientNet-B0 (timm) + Linear(1280, 4) |
| Framework | PyTorch 2.2 |
| Input Size | 224 × 224 × 3 (RGB, ImageNet normalized) |
| Parameters | ~5.3M |
| Model File | `models/BRAIN_MRI_EFFICIENTNET.pth` |
| Training Platform | Google Colab (Tesla T4 GPU) |
| Epochs | 10 |
| Best Val Accuracy | 95.00% (Epoch 9) |
| Inference Time | ~1.2s (CPU) |

### Base Models & Meta-Learner (Stacking Ensemble)

| Model | Architecture | Training Plan |
|-------|-------------|---------------|
| **ResNet-50** | 50-layer residual network, 25.6M params | Transfer learning from ImageNet, same dataset |
| **DenseNet-121** | Densely connected network, 8.0M params | Transfer learning from ImageNet, same dataset |
| **VGG-16** | 16-layer network, 138M params | Transfer learning from ImageNet, lower LR (1e-5) |
| **Logistic Regression** | Meta-learner over concatenated softmax probabilities (16 features) | Trained leakage-safe on held-out test-set probabilities |

**Training notebooks:** `notebooks/BRAIN_MRI_SCAN.ipynb` trains all four base models in one run (order: EfficientNet → ResNet → DenseNet → VGG) and exports per-model probabilities + metrics. `notebooks/BRAIN_MRI_ENSEMBLE.ipynb` trains the Logistic Regression meta-learner and saves `meta_model.pkl`.

---

## Dataset

- **Source:** [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes:** Glioma, Meningioma, No Tumor, Pituitary
- **Training:** 5,600 images
- **Testing:** 1,600 images (400 per class, balanced)
- **Location:** `data/brainMRI/`

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Models** | PyTorch, timm (EfficientNet-B0, ResNet-50, DenseNet-121, VGG-16) | Base learners |
| **Meta-Learner** | scikit-learn (Logistic Regression) | Stacking ensemble |
| **ML Service** | FastAPI, Uvicorn, Pillow, pytorch-grad-cam | Inference, Grad-CAM, reports |
| **API Gateway** | Express.js, Multer, Axios, Mongoose | Routing, validation, proxy |
| **Database** | MongoDB | Prediction + user storage |
| **Frontend** | React 18, Vite, Three.js, GSAP, Lenis | UI, 3D visuals, animations |
| **Auth** | JWT, bcrypt | Authentication (planned) |
| **DICOM** | pydicom | Medical image format (planned) |
| **DevOps** | Docker, GitHub Actions | Containerization, CI/CD (planned) |
| **Cloud** | AWS / GCP | Production hosting (planned) |
| **Training** | Google Colab (Tesla T4 GPU) | Model training |

---

## Future Vision: Multi-Modal Medical AI Platform

NeuraSight is designed to evolve from a brain MRI classifier into a comprehensive medical imaging AI platform:

```
┌─────────────────────────────────────────────────────────────────┐
│                  NeuraSight Platform (Future)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Brain MRI   │  │  Chest X-Ray │  │    CT Scan           │  │
│  │  Module      │  │  Module      │  │    Module            │  │
│  │              │  │              │  │                      │  │
│  │  • Glioma    │  │  • Pneumonia │  │  • Lung nodules      │  │
│  │  • Meningioma│  │  • COVID-19  │  │  • Abdominal lesions │  │
│  │  • Pituitary │  │  • TB        │  │  • Brain hemorrhage  │  │
│  │  • No Tumor  │  │  • Normal    │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Retinal     │  │  Skin Lesion │  │  Multi-Modal         │  │
│  │  Module      │  │  Module      │  │  Fusion              │  │
│  │              │  │              │  │                      │  │
│  │  • Diabetic  │  │  • Melanoma  │  │  • Combine multiple  │  │
│  │    retinopathy│  │  • Benign    │  │    imaging types     │  │
│  │  • Macular   │  │  • Basal cell│  │  • Comprehensive     │  │
│  │    degeneration│  │  • Squamous │  │    diagnosis         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Shared Infrastructure                        │   │
│  │  • Unified API  • Grad-CAM for all  • PDF Reports       │   │
│  │  • Model Registry  • DICOM Support  • User Auth         │   │
│  │  • Prediction History  • Real-time Inference             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Google Drive Assets

```
My Drive/NeuraSight/
├── best_model.pth          # Original training checkpoint
├── neurasight_model.pt     # Exported model
└── brainMRI.zip            # Dataset backup
```

Local model files: `models/BRAIN_MRI_*.pth` + `meta_model.pkl`

---

## Documentation

| Document | Location | Description |
|----------|----------|-------------|
| System Architecture | `docs/system_architecture.md` | Full architecture, algorithms, flowcharts |
| Gantt Chart | `docs/gantt_chart.md` | Project timeline with all tasks |
| Model Architecture | `model_architecture.md` | EfficientNet-B0 details, training info |
| Model Results | `docs/brain_mri_model_results.md` | Training results, metrics |
| Improvements Guide | `docs/improvements.md` | How to retrain for better results |
| Backend Setup | `backend/README.md` | Express + FastAPI setup instructions |
| References | `docs/references.md` | Papers, datasets, tools |

---

## License

This project is licensed under the MIT License.

---

## Disclaimer

This system is a **clinical decision support tool**, not a replacement for professional medical judgment. All predictions should be validated by qualified healthcare professionals. The explainable AI component (Grad-CAM) ensures transparency in the AI's reasoning process.

---

**NeuraSight** — Final Year Project | AI-Powered Medical Imaging Platform  
Brain Tumor Classification • Grad-CAM Explainability • Clinical Decision Support
