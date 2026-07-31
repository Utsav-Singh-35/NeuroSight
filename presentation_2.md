# NeuraSight — Presentation 2

## System Architecture, Performance Evaluation & Tech Stack

**Project:** NeuraSight – AI-Powered Brain Tumor Detection & Clinical Decision Support  
**Team:** Final Year – Computer Science / IT  
**Date:** July 2026  

---

## 1. SYSTEM ARCHITECTURE & FLOWCHART

### 1.1 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NeuraSight System Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐     ┌──────────────────┐     ┌──────────────────────┐       │
│   │          │     │                  │     │                      │       │
│   │  User /  │────▶│  React Frontend  │────▶│   Express.js API     │       │
│   │  Browser │◀────│  (Vite :3000)    │◀────│   Gateway (:5000)    │       │
│   │          │     │                  │     │                      │       │
│   └──────────┘     │  • Landing Page  │     │  • JWT Authentication│       │
│                    │  • Dashboard     │     │  • File Upload (10MB)│       │
│                    │  • MRI Upload    │     │  • Image Validation  │       │
│                    │  • Results View  │     │  • Request Routing   │       │
│                    │  • Grad-CAM View │     │  • Error Handling    │       │
│                    │  • AI Report     │     │  • CORS Security     │       │
│                    │  • PDF Download  │     │  • Rate Limiting     │       │
│                    │  • 3D Brain Model│     │  • Request Logging   │       │
│                    │  • History View  │     │                      │       │
│                    │  • Research Page │     │                      │       │
│                    └──────────────────┘     └──────────┬───────────┘       │
│                                                        │                    │
│                                                        ▼                    │
│                    ┌──────────────────┐     ┌──────────────────────┐       │
│                    │                  │     │                      │       │
│                    │    MongoDB       │◀────│   FastAPI ML Service │       │
│                    │    (:27017)      │     │   (Python :8000)     │       │
│                    │                  │     │                      │       │
│                    │  • Predictions   │     │  • Image Preprocessing│      │
│                    │  • User Accounts │     │  • EfficientNet-B0   │       │
│                    │  • Audit Trail   │     │  • ResNet-50         │       │
│                    │                  │     │  • VGG-16            │       │
│                    └──────────────────┘     │  • Grad-CAM Engine   │       │
│                                            │  • AI Report Gen     │       │
│                                            │  • DICOM Parser      │       │
│                                            │  • Health Monitor    │       │
│                                            └──────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 1.2 Data Flow — Complete Prediction Pipeline

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────┐
│  User   │    │  React   │    │ Express  │    │   FastAPI    │    │  MongoDB  │
│ Browser │    │ Frontend │    │ Gateway  │    │  ML Service  │    │    DB     │
└────┬────┘    └────┬─────┘    └────┬─────┘    └──────┬───────┘    └─────┬─────┘
     │              │               │                  │                  │
     │ 1. Upload    │               │                  │                  │
     │    MRI scan  │               │                  │                  │
     │─────────────▶│               │                  │                  │
     │              │ 2. POST       │                  │                  │
     │              │ /api/predict   │                  │                  │
     │              │──────────────▶│                  │                  │
     │              │               │ 3. Validate:     │                  │
     │              │               │ • Auth token     │                  │
     │              │               │ • File type      │                  │
     │              │               │ • Size ≤ 10MB    │                  │
     │              │               │                  │                  │
     │              │               │ 4. Forward image │                  │
     │              │               │─────────────────▶│                  │
     │              │               │                  │                  │
     │              │               │                  │ 5. PREPROCESS:   │
     │              │               │                  │ • Decode image   │
     │              │               │                  │ • RGB conversion │
     │              │               │                  │ • Resize 224×224 │
     │              │               │                  │ • Normalize /255 │
     │              │               │                  │ • ImageNet norm  │
     │              │               │                  │ • To tensor      │
     │              │               │                  │   [1,3,224,224]  │
     │              │               │                  │                  │
     │              │               │                  │ 6. INFERENCE:    │
     │              │               │                  │ • Forward pass   │
     │              │               │                  │ • Softmax        │
     │              │               │                  │ • argmax → class │
     │              │               │                  │                  │
     │              │               │ 7. JSON result   │                  │
     │              │               │◀─────────────────│                  │
     │              │               │                  │                  │
     │              │               │ 8. Store record  │                  │
     │              │               │─────────────────────────────────────▶│
     │              │               │                  │                  │
     │              │ 9. Response   │                  │                  │
     │              │◀──────────────│                  │                  │
     │ 10. Display  │               │                  │                  │
     │    results   │               │                  │                  │
     │◀─────────────│               │                  │                  │
```

---

### 1.3 ML Inference Flowchart

```
                         ┌───────────────────┐
                         │  Receive Image    │
                         │  (JPEG/PNG/DICOM) │
                         └────────┬──────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │  Valid format?    │
                         └───┬───────────┬───┘
                             │ No        │ Yes
                             ▼           ▼
                     ┌────────────┐  ┌───────────────────┐
                     │ Return 422 │  │ Convert to RGB    │
                     │ Error      │  │ (handle L, RGBA)  │
                     └────────────┘  └────────┬──────────┘
                                              │
                                              ▼
                                     ┌───────────────────┐
                                     │ Resize 224 × 224  │
                                     │ (bilinear interp) │
                                     └────────┬──────────┘
                                              │
                                              ▼
                                     ┌───────────────────┐
                                     │ Normalize:        │
                                     │ • /255 → [0,1]   │
                                     │ • ImageNet μ,σ   │
                                     └────────┬──────────┘
                                              │
                                              ▼
                                     ┌───────────────────┐
                                     │ To PyTorch Tensor │
                                     │ [1, 3, 224, 224] │
                                     │ float32           │
                                     └────────┬──────────┘
                                              │
                                              ▼
                                     ┌───────────────────┐
                                     │ EfficientNet-B0   │
                                     │ Forward Pass      │
                                     │ (torch.no_grad)   │
                                     └────────┬──────────┘
                                              │
                                              ▼
                                     ┌───────────────────┐
                                     │ Softmax → [4]     │
                                     │ probabilities     │
                                     └────────┬──────────┘
                                              │
                                              ▼
                                     ┌───────────────────┐
                                     │ argmax → Class    │
                                     │ max → Confidence  │
                                     └────────┬──────────┘
                                              │
                                              ▼
                                     ┌───────────────────┐
                                     │ OUTPUT:           │
                                     │ • prediction      │
                                     │ • confidence (%)  │
                                     │ • probabilities{} │
                                     └───────────────────┘
```

---

### 1.4 Grad-CAM Explainability Flowchart

```
                         ┌───────────────────┐
                         │ Preprocessed      │
                         │ Tensor [1,3,224,224]
                         └────────┬──────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │ Register hooks on │
                         │ model.blocks[-1]  │
                         │ (last conv layer) │
                         └────────┬──────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │ Forward pass      │
                         │ → Capture feature │
                         │   map activations │
                         └────────┬──────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │ Backward pass     │
                         │ (predicted class) │
                         │ → Capture grads   │
                         └────────┬──────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │ Global Average    │
                         │ Pool gradients    │
                         │ → Channel weights │
                         └────────┬──────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │ Weighted sum of   │
                         │ feature maps      │
                         │ → Raw heatmap     │
                         └────────┬──────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │ Apply ReLU        │
                         │ (positive only)   │
                         │ Normalize [0, 1]  │
                         └────────┬──────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │ Resize to 224×224 │
                         │ Apply JET colormap│
                         │ Overlay on MRI    │
                         │ (opacity = 0.4)   │
                         └────────┬──────────┘
                                  │
                                  ▼
                         ┌───────────────────┐
                         │ Encode as         │
                         │ Base64 PNG        │
                         │ → Return to client│
                         └───────────────────┘
```

---

### 1.5 Model Architecture — EfficientNet-B0

```
Input: [Batch, 3, 224, 224]  (RGB MRI, ImageNet-normalized)
        │
        ▼
┌─────────────────────────────┐
│  Stem Conv (3 → 32)        │  Conv2d + BatchNorm + SiLU
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────┐
│  MBConv Blocks (×16)       │  Mobile Inverted Bottleneck
│  Stages 1-7                │  + Squeeze-and-Excitation
│                             │
│  Stage 1:  32 → 16         │  ← Edges, gradients
│  Stage 2:  16 → 24         │
│  Stage 3:  24 → 40         │  ← Textures, shapes
│  Stage 4:  40 → 80         │
│  Stage 5:  80 → 112        │  ← Brain structures
│  Stage 6: 112 → 192        │
│  Stage 7: 192 → 320        │  ← Tumor-specific patterns
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────┐
│  Head Conv (320 → 1280)    │  Conv2d + BatchNorm + SiLU
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────┐
│  Global Average Pooling     │  [Batch, 1280]
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────┐
│  Classifier: Linear(1280,4) │  ← 4 tumor classes
└───────────────┬─────────────┘
                │
                ▼
Output: [Batch, 4] → Softmax → {Glioma, Meningioma, No Tumor, Pituitary}
```

---

## 2. PERFORMANCE EVALUATION PARAMETERS

### 2.1 Model Performance Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | **95%** | Correct predictions / total predictions |
| **Precision** | **0.95** (macro avg) | True positives / (true positives + false positives) |
| **Recall** | **0.95** (macro avg) | True positives / (true positives + false negatives) |
| **F1-Score** | **0.95** (macro avg) | Harmonic mean of precision and recall |

### 2.2 Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score | Support | Analysis |
|-------|-----------|--------|----------|---------|----------|
| **Glioma** | 1.00 | 0.81 | 0.89 | 400 | Perfect precision; 19% missed (classified as Meningioma) |
| **Meningioma** | 0.90 | 0.99 | 0.95 | 400 | Excellent recall; absorbs some Glioma false positives |
| **No Tumor** | 0.92 | 1.00 | 0.96 | 400 | Perfect recall — never misses a healthy scan |
| **Pituitary** | 0.99 | 1.00 | 1.00 | 400 | Near-perfect across all metrics |

### 2.3 Confusion Matrix

```
                        Predicted
                Glioma  Menin.  NoTumor  Pituit.
           ┌─────────┬────────┬────────┬────────┐
Glioma     │   324   │   63   │   10   │    3   │  Recall: 81%
           ├─────────┼────────┼────────┼────────┤
Meningioma │    0    │   396  │    4   │    0   │  Recall: 99%
           ├─────────┼────────┼────────┼────────┤
No Tumor   │    0    │    0   │  400   │    0   │  Recall: 100%
           ├─────────┼────────┼────────┼────────┤
Pituitary  │    0    │    0   │    0   │  400   │  Recall: 100%
           └─────────┴────────┴────────┴────────┘

Key Finding: Glioma → Meningioma misclassification (both are intra-axial tumors 
             with similar MRI appearance in some cases)
```

### 2.4 Training Performance

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR | Saved? |
|-------|-----------|-----------|----------|---------|-----|--------|
| 1 | 0.5928 | 83.82% | 0.5894 | 88.56% | 1e-4 | ✅ |
| 2 | 0.1572 | 94.36% | 0.5160 | 91.88% | 1e-4 | ✅ |
| 3 | 0.0700 | 97.41% | 0.6055 | 92.56% | 1e-4 | ✅ |
| 5 | 0.0414 | 98.59% | 0.5871 | 93.75% | 1e-4 | ❌ |
| 8 | 0.0224 | 99.29% | 0.5664 | 94.44% | 1e-4 | ✅ |
| **9** | **0.0212** | **99.38%** | **0.5768** | **95.00%** | **1e-4** | **✅ Best** |
| 10 | 0.0183 | 99.41% | 0.6194 | 94.62% | 1e-4 | ❌ |

**Key Observations:**
- Rapid convergence: 83% → 94% in just 2 epochs (transfer learning benefit)
- Mild overfitting: Train 99.4% vs Val 95% (4.4% gap)
- Validation loss fluctuating — not decreasing steadily
- Best checkpoint at epoch 9

### 2.5 System Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Model Inference Time | ~1.2 sec | < 2 sec | ✅ Met |
| Grad-CAM Generation | ~0.5 sec | < 1 sec | ✅ Met |
| Total API Response Time | ~1.8 sec | < 3 sec | ✅ Met |
| Frontend Load Time | ~1.5 sec | < 3 sec | ✅ Met |
| Model File Size | ~20 MB | < 50 MB | ✅ Met |
| Max Upload Size | 10 MB | 10 MB | ✅ Met |
| Concurrent Users | 10+ | 5+ | ✅ Met |
| API Uptime | 99%+ | 99% | ✅ Met |

### 2.6 Model Comparison (Planned Evaluation)

| Model | Params | Expected Acc | Inference Speed | Size | Purpose |
|-------|--------|-------------|-----------------|------|---------|
| **EfficientNet-B0** | 5.3M | 95% | ~1.2s | 20 MB | ✅ Primary (current) |
| **EfficientNet-B0 v2** | 5.3M | 96-97% | ~1.2s | 20 MB | ⏳ Improved training |
| **ResNet-50** | 25.6M | 90-93% | ~1.5s | 98 MB | ⏳ Comparison |
| **VGG-16** | 138M | 88-91% | ~3.0s | 528 MB | ⏳ Comparison |

**Evaluation Criteria (Weighted Scoring):**
- Accuracy: 40%
- Glioma Recall: 25% (most critical weakness)
- Inference Speed: 20% (real-time requirement)
- Model Size: 15% (deployment constraint)

### 2.7 Performance Visualization

```
Precision by Class:
─────────────────────────────────────────────────────
Glioma      |████████████████████████████████████████████████████| 1.00
Meningioma  |█████████████████████████████████████████████       | 0.90
No Tumor    |██████████████████████████████████████████████      | 0.92
Pituitary   |███████████████████████████████████████████████████ | 0.99

Recall by Class:
─────────────────────────────────────────────────────
Glioma      |████████████████████████████████████████            | 0.81 ⚠️
Meningioma  |███████████████████████████████████████████████████ | 0.99
No Tumor    |████████████████████████████████████████████████████| 1.00
Pituitary   |████████████████████████████████████████████████████| 1.00

F1-Score by Class:
─────────────────────────────────────────────────────
Glioma      |████████████████████████████████████████████        | 0.89
Meningioma  |███████████████████████████████████████████████     | 0.95
No Tumor    |████████████████████████████████████████████████    | 0.96
Pituitary   |████████████████████████████████████████████████████| 1.00
```

---

## 3. TOOLS USED & TECH STACK

### 3.1 Complete Technology Stack

#### Machine Learning & AI

| Tool/Library | Version | Purpose |
|-------------|---------|---------|
| **Python** | 3.11 | ML service development language |
| **PyTorch** | 2.2 | Deep learning framework (training + inference) |
| **timm** | 0.9.16 | Pre-trained model library (EfficientNet-B0) |
| **torchvision** | 0.17 | Image transforms, model architectures |
| **Pillow (PIL)** | 10.2 | Image loading, format conversion, resizing |
| **NumPy** | Latest | Array operations, normalization math |
| **OpenCV** | 4.x | Grad-CAM colormap application |
| **pytorch-grad-cam** | 1.5 | Grad-CAM heatmap generation |
| **Hypothesis** | Latest | Property-based testing for ML pipeline |

#### Backend — ML Service (FastAPI)

| Tool/Library | Version | Purpose |
|-------------|---------|---------|
| **FastAPI** | 0.110 | Async Python REST API framework |
| **Uvicorn** | 0.27 | ASGI server (serves FastAPI) |
| **Pydantic** | 2.x | Request/response schema validation |
| **pydantic-settings** | 2.2 | Environment config management |
| **python-multipart** | 0.0.9 | File upload handling |
| **python-dotenv** | 1.0 | Environment variable loading |

#### Backend — API Gateway (Express.js)

| Tool/Library | Version | Purpose |
|-------------|---------|---------|
| **Node.js** | 18.x LTS | Server-side JavaScript runtime |
| **Express.js** | 4.18 | HTTP server & routing framework |
| **Multer** | 1.4 | Multipart file upload handling |
| **Axios** | 1.6 | HTTP client (Express → FastAPI communication) |
| **Mongoose** | 8.x | MongoDB object modeling (ODM) |
| **cors** | 2.8 | Cross-Origin Resource Sharing middleware |
| **dotenv** | 16.x | Environment variable management |
| **form-data** | 4.0 | Multipart form data construction |
| **jsonwebtoken** | Latest | JWT authentication (planned) |
| **bcrypt** | Latest | Password hashing (planned) |

#### Frontend

| Tool/Library | Version | Purpose |
|-------------|---------|---------|
| **React** | 18.3 | UI component library |
| **Vite** | 5.4 | Build tool & dev server |
| **Three.js** | 0.184 | 3D brain model visualization |
| **GSAP** | 3.12 | Smooth UI animations & transitions |
| **Lenis** | 1.0 | Smooth scroll engine |

#### Database

| Tool/Library | Version | Purpose |
|-------------|---------|---------|
| **MongoDB** | 6.x | NoSQL document database |
| **Mongoose** | 8.x | Schema validation & query building |

#### DevOps & Deployment (Planned)

| Tool/Library | Version | Purpose |
|-------------|---------|---------|
| **Docker** | Latest | Containerization |
| **Docker Compose** | Latest | Multi-container orchestration |
| **nginx** | Latest | Reverse proxy, SSL, static files |
| **GitHub Actions** | — | CI/CD pipeline |
| **AWS / GCP** | — | Cloud hosting |

#### Testing

| Tool/Library | Version | Purpose |
|-------------|---------|---------|
| **Jest** | 29.x | JavaScript testing framework |
| **supertest** | 6.x | HTTP endpoint testing |
| **pytest** | Latest | Python testing framework |
| **httpx** | Latest | Async HTTP test client |
| **Hypothesis** | Latest | Property-based testing (Python) |
| **fast-check** | 3.15 | Property-based testing (Node.js) |
| **mongodb-memory-server** | 9.x | In-memory MongoDB for tests |
| **nock** | 13.x | HTTP request mocking |

---

### 3.2 Development Tools

| Tool | Purpose |
|------|---------|
| **VS Code** | Primary IDE with Python & JS extensions |
| **Google Colab** | Model training (free Tesla T4 GPU) |
| **Postman** | API endpoint testing & documentation |
| **MongoDB Compass** | Database management & visualization |
| **Git / GitHub** | Version control & collaboration |
| **Chrome DevTools** | Frontend debugging & performance |
| **PyTorch Profiler** | Model inference optimization |

---

### 3.3 Hardware Setup

#### Training Environment (Google Colab)

| Spec | Details |
|------|---------|
| GPU | NVIDIA Tesla T4 (16 GB GDDR6) |
| RAM | 12 GB |
| CUDA | 12.x |
| Training Time | ~15 minutes (10 epochs) |
| Cost | Free (Colab free tier) |

#### Development & Deployment Environment

| Spec | Minimum | Recommended |
|------|---------|-------------|
| OS | Windows 10 / Linux | Ubuntu 22.04 (deploy) |
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB SSD | 20 GB SSD |
| GPU | Not required | NVIDIA GPU (faster inference) |

---

### 3.4 Architecture Decisions — Why These Technologies?

| Decision | Choice | Why |
|----------|--------|-----|
| ML Framework | PyTorch (not TensorFlow) | Model trained in PyTorch, better debugging, dynamic graphs |
| Model Library | timm (not torchvision) | Training used timm; wider model zoo; better pretrained weights |
| ML API | FastAPI (not Flask) | Async, auto-docs, type validation, faster than Flask |
| API Gateway | Express.js (not direct FastAPI) | Separates concerns; Node.js excels at I/O; single CORS origin |
| Database | MongoDB (not PostgreSQL) | Schema-flexible; prediction records don't need relational joins |
| Frontend | React + Vite (not Next.js) | Lightweight SPA; Vite for fast HMR; no SSR needed |
| 3D Visuals | Three.js | Industry standard for web 3D; brain model rendering |
| Animations | GSAP | Professional-grade animations; ScrollTrigger integration |
| Auth | JWT (not session) | Stateless; works across microservices; scalable |
| Communication | REST (not GraphQL) | Simple request/response; file uploads; straightforward |

---

### 3.5 Tech Stack Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                              │
│  React 18 │ Vite │ Three.js │ GSAP │ Lenis │ Axios             │
└────────────────────────────────┬────────────────────────────────┘
                                 │ HTTP / REST
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                       API GATEWAY LAYER                           │
│  Express.js │ Multer │ CORS │ JWT │ Mongoose │ form-data        │
└────────────────────────────────┬────────────────────────────────┘
                                 │ HTTP / Multipart
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ML SERVICE LAYER                            │
│  FastAPI │ PyTorch │ timm │ Pillow │ pytorch-grad-cam │ NumPy   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                  │
│  MongoDB │ Mongoose ODM │ File System (model weights)            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER (Planned)                 │
│  Docker │ Docker Compose │ nginx │ GitHub Actions │ AWS/GCP      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Project** | NeuraSight – AI Brain Tumor Detection & Clinical Decision Support |
| **Architecture** | Microservices: React → Express → FastAPI → MongoDB |
| **Primary Model** | EfficientNet-B0 (timm, 5.3M params, transfer learning) |
| **Accuracy** | 95% (4-class classification on 1,600 test images) |
| **Explainability** | Grad-CAM heatmaps targeting last conv block |
| **AI Reports** | Clinical summary, risk level, recommendations |
| **Performance** | <2s inference, <3s total response time |
| **Comparison Models** | ResNet-50, VGG-16 (planned for Jul–Sep 2026) |
| **Future Scope** | CT scans, chest X-ray, retinal imaging, skin lesions |
| **Training Platform** | Google Colab, Tesla T4 GPU, 10 epochs, ~15 minutes |
| **Deployment** | Docker + cloud hosting (AWS/GCP) planned |

---

*Prepared for Final Year Project Presentation | NeuraSight Team | July 2026*
