# NeuraSight 🧠

**AI-Powered Brain MRI Tumor Detection & Clinical Decision Support Platform**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933.svg)](https://nodejs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

NeuraSight is a medical imaging platform that classifies brain tumors from MRI scans using deep learning. It provides:

- **95% classification accuracy** across 4 tumor types using EfficientNet-B0
- **Grad-CAM heatmaps** showing which MRI regions the model focuses on
- **REST API** for integration with any frontend
- **Prediction history** stored in MongoDB

### Tumor Classes
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 1.00 | 0.81 | 0.89 |
| Meningioma | 0.90 | 0.99 | 0.95 |
| No Tumor | 0.92 | 1.00 | 0.96 |
| Pituitary | 0.99 | 1.00 | 1.00 |
| **Overall** | **0.95** | **0.95** | **0.95** |

---

## Architecture

```
┌──────────┐      ┌─────────────────────┐      ┌──────────────────┐
│  React   │ HTTP │   Express Backend    │ HTTP │  FastAPI Service  │
│ Frontend │─────▶│   (Node.js :5000)   │─────▶│  (Python :8000)   │
│ :3000    │◀─────│                     │◀─────│                   │
└──────────┘      │ • Image validation  │      │ • Preprocessing   │
                  │ • Request routing   │      │ • Model inference │
                  │ • Prediction history│      │ • Grad-CAM        │
                  └─────────┬───────────┘      └──────────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │     MongoDB :27017  │
                  │  (prediction store) │
                  └─────────────────────┘
```

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
│   │   │   ├── routers/      # API endpoints
│   │   │   ├── services/     # Preprocessing, inference, Grad-CAM
│   │   │   └── models/       # Pydantic schemas
│   │   └── requirements.txt
│   │
│   └── README.md             # Backend setup guide
│
├── models/
│   └── Brain_MRI_scan.pth    # Trained EfficientNet-B0 model weights
│
├── data/
│   └── brainMRI/             # Dataset (Training + Testing)
│       ├── Training/         # 4 classes
│       └── Testing/          # 400 images per class (1600 total)
│
├── docs/                     # Documentation
│   ├── brain_mri_model_results.md
│   ├── model_architecture.md
│   ├── dataset_analysis.md
│   ├── deployment_architecture.md
│   ├── roadmap.md
│   └── references.md
│
├── frontend/                 # React web interface (TBD)
├── notebooks/                # Jupyter notebooks
├── preprocessing/            # Data preprocessing scripts
├── training/                 # Model training scripts
├── evaluation/               # Model evaluation code
├── explainability/           # Grad-CAM implementation
└── reports/                  # Generated reports
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB 6.0+
- The trained model file: `Brain_MRI_scan.pth`

### 1. Start FastAPI (ML Service)

```bash
cd backend/fastapi
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Ensure models/Brain_MRI_scan.pth exists at project root
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Start Express (API Gateway)

```bash
cd backend/express
npm install

# Ensure MongoDB is running
npm start
```

### 3. Test the API

```bash
# Health check
curl http://localhost:5000/api/health

# Predict tumor class
curl -X POST -F "image=@path/to/mri_scan.jpg" http://localhost:5000/api/predict

# Get Grad-CAM heatmap
curl -X POST -F "image=@path/to/mri_scan.jpg" http://localhost:5000/api/gradcam

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
| `GET` | `/api/predictions` | List prediction history (max 100) |
| `GET` | `/api/predictions/:id` | Get single prediction record |
| `GET` | `/api/health` | Service + FastAPI reachability status |

### Response Example

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

---

## Model Details

| Attribute | Value |
|-----------|-------|
| Architecture | EfficientNet-B0 + custom head |
| Framework | PyTorch |
| Input Size | 224 × 224 × 3 (RGB) |
| Parameters | ~5.3M |
| Model File | `models/Brain_MRI_scan.pth` |
| Training Platform | Google Colab (Tesla T4 GPU) |
| Epochs | 10 |
| Best Val Accuracy | 95.00% (Epoch 9) |
| Test Accuracy | 95% |

### Training Results

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 83.82% | 88.56% |
| 2 | 94.36% | 91.88% |
| 3 | 97.41% | 92.56% |
| 5 | 98.59% | 93.75% |
| 9 | 99.38% | **95.00%** |
| 10 | 99.41% | 94.62% |

---

## Dataset

- **Source:** [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes:** Glioma, Meningioma, No Tumor, Pituitary
- **Test Set:** 1,600 images (400 per class, balanced)
- **Location:** `data/brainMRI/`

---

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Foundation | ✅ Complete | Project structure, documentation |
| 2. Preprocessing | ✅ Complete | Data pipeline, augmentation |
| 3. Model Training | ✅ Complete | EfficientNet-B0, 95% accuracy |
| 4. Evaluation | ✅ Complete | Classification report, metrics |
| 5. Grad-CAM | ✅ Complete | Explainability via FastAPI |
| 6. Backend API | ✅ Complete | Express + FastAPI microservices |
| 7. Frontend | ⏳ Next | React web interface |
| 8. Deployment | ⏳ Planned | Cloud hosting |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | PyTorch, torchvision, EfficientNet-B0 |
| ML Service | FastAPI, Uvicorn, Pillow, pytorch-grad-cam |
| API Gateway | Express.js, Multer, Axios |
| Database | MongoDB, Mongoose |
| Frontend | React (planned) |
| Training | Google Colab (GPU) |

---

## Google Drive Assets

```
My Drive/NeuraSight/
├── best_model.pth          # Original training checkpoint
├── neurasight_model.pt     # Exported model
└── brainMRI.zip            # Dataset backup
```

Local model file: `models/Brain_MRI_scan.pth`

---

## License

This project is licensed under the MIT License.

---

## Disclaimer

This system is a **clinical decision support tool**, not a replacement for professional medical judgment. All predictions should be validated by qualified healthcare professionals.

---

**NeuraSight** — Final Year Project | Brain Tumor Classification using Deep Learning & Explainable AI
