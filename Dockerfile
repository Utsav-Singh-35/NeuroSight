FROM python:3.11-slim

WORKDIR /app

# Install system deps for OpenCV (Grad-CAM needs it)
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/fastapi/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI application code
COPY backend/fastapi/app/ ./app/

# Copy model weights + ensemble config
COPY models/BRAIN_MRI_EFFICIENTNET.pth ./models/
COPY models/BRAIN_MRI_RESNET.pth ./models/
COPY models/BRAIN_MRI_DENSENET.pth ./models/
COPY models/BRAIN_MRI_VGG.pth ./models/
COPY models/meta_model.pkl ./models/
COPY models/ensemble_config.json ./models/

# Environment variables
ENV MODELS_DIR=./models
ENV USE_ENSEMBLE=True
ENV MODEL_PATH=./models/BRAIN_MRI_EFFICIENTNET.pth
ENV CLASS_LABELS=Glioma,Meningioma,No Tumor,Pituitary
ENV GRADCAM_OPACITY=0.4
ENV GRADCAM_COLORMAP=jet
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
