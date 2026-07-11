# NeuraSight Backend Services

## Overview

NeuraSight uses a two-service backend architecture for brain tumor classification:

```
┌──────────┐      ┌─────────────────────┐      ┌──────────────────┐
│  React   │ HTTP │   Express Backend    │ HTTP │  FastAPI Service  │
│ Frontend │─────▶│   (Node.js)         │─────▶│  (Python)         │
│ :3000    │◀─────│   :5000             │◀─────│  :8000            │
└──────────┘      └─────────┬───────────┘      └──────────────────┘
                            │
                            │ MongoDB Driver
                            ▼
                  ┌─────────────────────┐
                  │     MongoDB         │
                  │     :27017          │
                  └─────────────────────┘
```

- **Express Backend (Node.js :5000)** — API gateway that handles frontend communication, image validation, request routing, prediction history storage in MongoDB, CORS, and error handling.
- **FastAPI Service (Python :8000)** — ML microservice that loads a pre-trained EfficientNet-B0 PyTorch model, performs inference on MRI images, and generates Grad-CAM explainability heatmaps.

The flow is: React Frontend → Express Backend → FastAPI ML Service → Express Backend → React Frontend.

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Node.js | 18+ | Express backend runtime |
| Python | 3.10+ | FastAPI service runtime |
| MongoDB | 6.0+ | Prediction history storage |
| pip | Latest | Python dependency management |
| PyTorch model file | — | `best_model.pth` (EfficientNet-B0 weights) |

## Setup Instructions

### 1. FastAPI Service (ML Inference)

```bash
# Navigate to the FastAPI service directory
cd backend/fastapi

# Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file and adjust if needed
cp .env.example .env

# Place the trained model file
# Option A: Copy your trained best_model.pth to the models/ directory at project root
# Option B: Generate a dummy model for testing (no trained weights needed):
python ../../models/create_dummy_model.py

# Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The FastAPI service will be available at `http://localhost:8000`. You can view the auto-generated API docs at `http://localhost:8000/docs`.

### 2. Express Backend (API Gateway)

```bash
# Navigate to the Express service directory
cd backend/express

# Install dependencies
npm install

# Copy environment file and adjust if needed
cp .env.example .env

# Ensure MongoDB is running (default: localhost:27017)
# On Windows: net start MongoDB
# On macOS: brew services start mongodb-community
# On Linux: sudo systemctl start mongod

# Start the service
npm start

# Or for development with auto-reload:
npm run dev
```

The Express backend will be available at `http://localhost:5000`.

## Environment Variables

### Express Backend (`backend/express/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Express server port |
| `MONGODB_URI` | `mongodb://localhost:27017/neurasight` | MongoDB connection string |
| `FASTAPI_URL` | `http://localhost:8000` | FastAPI service URL |
| `FRONTEND_ORIGIN` | `http://localhost:3000` | Allowed CORS origin |
| `MAX_FILE_SIZE` | `10485760` | Max upload size in bytes (10 MB) |
| `FASTAPI_TIMEOUT` | `30000` | FastAPI request timeout in ms |
| `HEALTH_CHECK_TIMEOUT` | `3000` | Health check timeout in ms |

### FastAPI Service (`backend/fastapi/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `../../models/Brain_MRI_scan.pth` | Path to PyTorch model weights |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `CLASS_LABELS` | `Glioma,Meningioma,No Tumor,Pituitary` | Comma-separated class labels |
| `GRADCAM_OPACITY` | `0.4` | Heatmap overlay opacity |
| `GRADCAM_COLORMAP` | `jet` | Colormap for Grad-CAM heatmaps |

## API Endpoints

### Express Backend (port 5000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health check (includes FastAPI reachability) |
| `POST` | `/api/predict` | Upload MRI image for classification |
| `POST` | `/api/gradcam` | Upload MRI image for Grad-CAM heatmap |
| `GET` | `/api/predictions` | List prediction history (max 100, newest first) |
| `GET` | `/api/predictions/:id` | Get a single prediction record by ID |

### FastAPI Service (port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health and model status |
| `POST` | `/predict` | Run inference on an MRI image |
| `POST` | `/gradcam` | Generate Grad-CAM heatmap for an MRI image |

## Running Tests

### Express Backend

```bash
cd backend/express
npm test
```

### FastAPI Service

```bash
cd backend/fastapi
pytest
```

## Development Tips

1. **Start FastAPI first, then Express.** The Express service attempts to reach FastAPI for health checks and inference requests. It handles FastAPI being unavailable gracefully, but full functionality requires both services running.

2. **Use the dummy model for development.** Run `python models/create_dummy_model.py` from the project root to generate a model file with the correct architecture but random weights. This avoids needing the full trained model for local development.

3. **MongoDB must be running** before starting the Express service, as it connects on startup.

4. **Hot reload for Express** — Use `npm run dev` to run with nodemon for automatic restarts on file changes.

5. **FastAPI auto-docs** — Visit `http://localhost:8000/docs` for interactive Swagger UI documentation of the ML service endpoints.

6. **CORS is pre-configured** for `http://localhost:3000` (the React frontend). Update `FRONTEND_ORIGIN` in the Express `.env` if your frontend runs on a different port.

## Project Structure

```
backend/
├── express/                    # Express API gateway
│   ├── src/
│   │   ├── server.js           # Entry point
│   │   ├── config/index.js     # Environment config
│   │   ├── middleware/
│   │   │   ├── imageValidator.js
│   │   │   ├── errorHandler.js
│   │   │   └── requestLogger.js
│   │   ├── routes/
│   │   │   ├── health.js
│   │   │   ├── predict.js
│   │   │   ├── gradcam.js
│   │   │   └── predictions.js
│   │   ├── services/
│   │   │   └── fastapiClient.js
│   │   └── models/
│   │       └── Prediction.js
│   ├── package.json
│   ├── .env.example
│   └── .env
│
├── fastapi/                    # FastAPI ML service
│   ├── app/
│   │   ├── main.py             # Entry point
│   │   ├── config.py           # Settings
│   │   ├── routers/
│   │   │   ├── health.py
│   │   │   ├── predict.py
│   │   │   └── gradcam.py
│   │   ├── services/
│   │   │   ├── preprocessor.py
│   │   │   ├── inference.py
│   │   │   └── gradcam.py
│   │   └── models/
│   │       └── schemas.py
│   ├── requirements.txt
│   ├── .env.example
│   └── .env
│
└── README.md                   # ← You are here
```
