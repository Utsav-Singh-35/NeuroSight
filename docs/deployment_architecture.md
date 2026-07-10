# NeuraSight Deployment Architecture 🚀

## Overview

This document outlines the **production-ready deployment architecture** for NeuraSight using:
- **Training:** Google Colab / Kaggle (Free GPU)
- **Backend:** Django REST Framework on AWS/Railway
- **Frontend:** React on Vercel/Netlify
- **Model:** Trained `.h5` file deployed with backend

---

## 🏗️ Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING ENVIRONMENT                       │
│              (Google Colab / Kaggle Notebook)                │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  📊 Dataset (from Kaggle)                                    │
│       ↓                                                       │
│  🧠 Train EfficientNet-B0                                    │
│       ↓                                                       │
│  📈 Evaluate Model                                           │
│       ↓                                                       │
│  🔍 Implement Grad-CAM                                       │
│       ↓                                                       │
│  💾 Download: neurasight_model.h5 (5-10 MB)                 │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            ↓
                    Upload model.h5
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                    BACKEND SERVER                             │
│              (Django REST API on AWS/Railway)                │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  📦 Django Project Structure:                                │
│     ├── models/                                              │
│     │   └── neurasight_model.h5                             │
│     ├── api/                                                 │
│     │   ├── views.py (prediction logic)                     │
│     │   ├── serializers.py                                  │
│     │   └── urls.py                                         │
│     ├── utils/                                               │
│     │   ├── preprocessing.py                                │
│     │   └── gradcam.py                                      │
│     └── requirements.txt                                     │
│                                                               │
│  🔌 API Endpoints:                                           │
│     POST /api/predict/        → Get prediction              │
│     POST /api/gradcam/        → Get Grad-CAM heatmap        │
│     GET  /api/health/         → Health check                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            ↑
                    HTTP Requests (JSON)
                            ↑
┌──────────────────────────────────────────────────────────────┐
│                    FRONTEND APPLICATION                       │
│              (React on Vercel/Netlify)                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  🎨 React Components:                                        │
│     ├── Dashboard.jsx                                        │
│     ├── UploadSection.jsx                                   │
│     ├── PredictionResults.jsx                               │
│     └── GradCAMVisualization.jsx                            │
│                                                               │
│  📱 User Flow:                                               │
│     1. Upload MRI image                                      │
│     2. Send to Django API                                    │
│     3. Display prediction + confidence                       │
│     4. Show Grad-CAM heatmap                                 │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 📋 Step-by-Step Implementation

### Phase 1: Training on Google Colab/Kaggle ☁️

#### Step 1.1: Create Training Notebook

**File:** `NeuraSight_Training.ipynb`

```python
# 1. Mount Google Drive (for Colab)
from google.colab import drive
drive.mount('/content/drive')

# 2. Install dependencies
!pip install tensorflow==2.13.0 tf-keras-vis

# 3. Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# 4. Load dataset (from Kaggle or upload)
# ... (preprocessing code)

# 5. Build model
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# 6. Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[...]
)

# 8. Evaluate
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# 9. Save model
model.save('neurasight_model.h5')

# 10. Download model (Colab)
from google.colab import files
files.download('neurasight_model.h5')
```

#### Step 1.2: Implement Grad-CAM

```python
import tf_keras_vis
from tf_keras_vis.gradcam import Gradcam

def get_gradcam(model, image, class_idx):
    gradcam = Gradcam(model)
    cam = gradcam(
        score,
        image,
        penultimate_layer=-1
    )
    return cam

# Test Grad-CAM
sample_image = test_images[0]
heatmap = get_gradcam(model, sample_image, predicted_class)
plt.imshow(heatmap)
plt.show()
```

#### Step 1.3: Download Artifacts

After training, download:
- ✅ `neurasight_model.h5` (trained model)
- ✅ Training history (metrics, plots)
- ✅ Confusion matrix
- ✅ Sample predictions

---

### Phase 2: Django Backend Setup 🐍

#### Step 2.1: Create Django Project

```bash
# Create project
django-admin startproject neurasight_backend
cd neurasight_backend
python manage.py startapp api

# Install dependencies
pip install djangorestframework tensorflow pillow django-cors-headers
```

#### Step 2.2: Project Structure

```
neurasight_backend/
├── api/
│   ├── models.py
│   ├── views.py
│   ├── serializers.py
│   ├── urls.py
│   └── utils/
│       ├── preprocessing.py
│       └── gradcam.py
├── models/
│   └── neurasight_model.h5  ← PUT YOUR MODEL HERE
├── media/
│   └── uploads/
├── neurasight_backend/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── requirements.txt
└── manage.py
```

#### Step 2.3: API Views (`api/views.py`)

```python
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from .utils.preprocessing import preprocess_image
from .utils.gradcam import generate_gradcam

# Load model once at startup
MODEL_PATH = 'models/neurasight_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

@api_view(['POST'])
def predict(request):
    """
    Predict tumor class from uploaded MRI image
    """
    try:
        # Get uploaded image
        image_file = request.FILES.get('image')
        if not image_file:
            return Response(
                {'error': 'No image provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Preprocess image
        img = Image.open(image_file)
        processed_img = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Prepare response
        response_data = {
            'prediction': CLASS_NAMES[predicted_class],
            'confidence': confidence * 100,
            'probabilities': {
                CLASS_NAMES[i]: float(predictions[0][i] * 100)
                for i in range(len(CLASS_NAMES))
            }
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def gradcam_view(request):
    """
    Generate Grad-CAM heatmap for uploaded image
    """
    try:
        image_file = request.FILES.get('image')
        if not image_file:
            return Response(
                {'error': 'No image provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Generate Grad-CAM
        img = Image.open(image_file)
        heatmap_base64 = generate_gradcam(model, img)
        
        return Response(
            {'heatmap': heatmap_base64},
            status=status.HTTP_200_OK
        )
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def health_check(request):
    """
    Health check endpoint
    """
    return Response({'status': 'healthy'}, status=status.HTTP_200_OK)
```

#### Step 2.4: Preprocessing Utility (`api/utils/preprocessing.py`)

```python
import numpy as np
from PIL import Image

def preprocess_image(img):
    """
    Preprocess image for model input
    """
    # Resize to 224x224
    img = img.resize((224, 224))
    
    # Convert to RGB if grayscale
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
```

#### Step 2.5: Grad-CAM Utility (`api/utils/gradcam.py`)

```python
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import base64
import io

def generate_gradcam(model, img):
    """
    Generate Grad-CAM heatmap
    """
    # Preprocess
    img_array = preprocess_image(img)
    
    # Get last conv layer
    last_conv_layer = model.get_layer('top_conv')  # Adjust layer name
    
    # Create gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    # Get gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Generate heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert to base64
    _, buffer = cv2.imencode('.png', heatmap)
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return heatmap_base64
```

#### Step 2.6: URLs Configuration

```python
# api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('gradcam/', views.gradcam_view, name='gradcam'),
    path('health/', views.health_check, name='health'),
]

# neurasight_backend/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
]
```

#### Step 2.7: Settings Configuration

```python
# settings.py

INSTALLED_APPS = [
    # ...
    'rest_framework',
    'corsheaders',
    'api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    # ...
]

# CORS settings
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",  # React dev server
    "https://your-frontend-domain.vercel.app",  # Production
]

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```

#### Step 2.8: Requirements.txt

```txt
Django==4.2.0
djangorestframework==3.14.0
django-cors-headers==4.0.0
tensorflow==2.13.0
pillow==10.0.0
numpy==1.24.3
opencv-python==4.8.0
gunicorn==21.2.0
```

---

### Phase 3: React Frontend Integration ⚛️

#### Step 3.1: API Service (`src/services/api.js`)

```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

export const predictTumor = async (imageFile) => {
  const formData = new FormData();
  formData.append('image', imageFile);

  const response = await fetch(`${API_BASE_URL}/predict/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Prediction failed');
  }

  return await response.json();
};

export const getGradCAM = async (imageFile) => {
  const formData = new FormData();
  formData.append('image', imageFile);

  const response = await fetch(`${API_BASE_URL}/gradcam/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Grad-CAM generation failed');
  }

  return await response.json();
};
```

#### Step 3.2: Dashboard Component

```javascript
import { useState } from 'react';
import { predictTumor, getGradCAM } from '../services/api';

export default function Dashboard() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [gradcam, setGradcam] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    try {
      // Get prediction
      const predResult = await predictTumor(selectedFile);
      setPrediction(predResult);

      // Get Grad-CAM
      const gradcamResult = await getGradCAM(selectedFile);
      setGradcam(gradcamResult.heatmap);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      {/* Upload Section */}
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setSelectedFile(e.target.files[0])}
      />
      <button onClick={handleAnalyze} disabled={!selectedFile || loading}>
        {loading ? 'Analyzing...' : 'Analyze MRI'}
      </button>

      {/* Results Section */}
      {prediction && (
        <div className="results">
          <h3>Prediction: {prediction.prediction}</h3>
          <p>Confidence: {prediction.confidence.toFixed(2)}%</p>
          
          {/* Probability Distribution */}
          {Object.entries(prediction.probabilities).map(([cls, prob]) => (
            <div key={cls}>
              <span>{cls}</span>
              <span>{prob.toFixed(2)}%</span>
            </div>
          ))}
        </div>
      )}

      {/* Grad-CAM Visualization */}
      {gradcam && (
        <div className="gradcam">
          <h3>Grad-CAM Heatmap</h3>
          <img src={`data:image/png;base64,${gradcam}`} alt="Grad-CAM" />
        </div>
      )}
    </div>
  );
}
```

---

### Phase 4: Deployment 🚀

#### Option 1: AWS Elastic Beanstalk (Backend)

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.9 neurasight-backend

# Create environment
eb create neurasight-env

# Deploy
eb deploy
```

#### Option 2: Railway (Backend) - Easier!

1. Push code to GitHub
2. Go to railway.app
3. Click "New Project" → "Deploy from GitHub"
4. Select your repository
5. Railway auto-detects Django
6. Add environment variables
7. Deploy!

#### Option 3: Heroku (Backend)

```bash
# Install Heroku CLI
# Create Procfile
echo "web: gunicorn neurasight_backend.wsgi" > Procfile

# Deploy
heroku create neurasight-api
git push heroku main
```

#### Frontend Deployment (Vercel)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd frontend
vercel

# Set environment variable
# REACT_APP_API_URL=https://your-backend-url.com/api
```

---

## 📊 Advantages of This Architecture

### ✅ **Pros**

1. **Cost-Effective**
   - Free GPU training (Colab/Kaggle)
   - Cheap deployment (Railway/Heroku free tier)
   - No expensive hardware needed

2. **Scalable**
   - Backend can handle multiple requests
   - Frontend CDN distribution
   - Easy to add load balancing

3. **Professional**
   - Industry-standard architecture
   - Microservices approach
   - RESTful API design

4. **Maintainable**
   - Separate concerns (training vs deployment)
   - Easy to update model (just replace .h5 file)
   - Clear code organization

5. **Presentation-Friendly**
   - Show training notebook
   - Demo live web app
   - Explain architecture

### ⚠️ **Considerations**

1. **Model Size**
   - EfficientNet-B0: ~20MB (manageable)
   - Larger models may need optimization

2. **Cold Start**
   - First request may be slow (model loading)
   - Solution: Keep-alive pings

3. **Memory**
   - TensorFlow needs ~500MB RAM
   - Choose appropriate hosting plan

---

## 🎯 **Recommended Stack**

### For Final Year Project:

**Training:**
- ✅ Google Colab (Free GPU)
- ✅ Kaggle Notebooks (Free GPU)

**Backend:**
- ✅ Django REST Framework
- ✅ Railway.app (Free tier, easy deployment)
- Alternative: Heroku, AWS

**Frontend:**
- ✅ React (already built)
- ✅ Vercel (Free hosting)
- Alternative: Netlify

**Database (if needed):**
- ✅ PostgreSQL on Railway
- ✅ MongoDB Atlas (Free tier)

---

## 📝 **For Your Report/Viva**

### What to Show:

1. **Training Phase:**
   - Colab/Kaggle notebook
   - Training curves
   - Evaluation metrics
   - Grad-CAM examples

2. **Deployment Phase:**
   - Django API endpoints
   - React frontend
   - Live demo

3. **Architecture Diagram:**
   - Show separation of concerns
   - Explain why this approach

### Key Points to Mention:

- ✅ "Used cloud GPU for efficient training"
- ✅ "Deployed model as REST API"
- ✅ "Microservices architecture"
- ✅ "Scalable and maintainable"
- ✅ "Industry-standard practices"

---

## ✅ **Next Steps**

1. **Train model on Colab/Kaggle**
2. **Download model.h5 file**
3. **Set up Django backend**
4. **Test API locally**
5. **Deploy to Railway/Heroku**
6. **Connect React frontend**
7. **Deploy frontend to Vercel**
8. **Test end-to-end**

---

**This is the BEST approach for your project! 🚀**
