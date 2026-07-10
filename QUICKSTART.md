# NeuraSight - Quick Start Guide 🚀

## What You Need to Do TODAY

### Step 1: Download the Dataset ⬇️

1. **Go to Kaggle:**
   - Visit: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
   
2. **Download the Dataset:**
   - Click "Download" button
   - You'll get a ZIP file (~500MB)
   
3. **Extract the Dataset:**
   - Extract to: `NeuraSight/data/raw/`
   - Final structure should be:
   ```
   data/raw/
   ├── Training/
   │   ├── glioma/
   │   ├── meningioma/
   │   ├── pituitary/
   │   └── notumor/
   └── Testing/
       ├── glioma/
       ├── meningioma/
       ├── pituitary/
       └── notumor/
   ```

4. **Verify Download:**
   - Check that you have images in each folder
   - Take a screenshot of the folder structure

---

### Step 2: Set Up Python Environment 🐍

1. **Open Terminal/Command Prompt**

2. **Navigate to Project:**
   ```bash
   cd NeuraSight
   ```

3. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   ```

4. **Activate Virtual Environment:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source venv/bin/activate
     ```

5. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   ⏰ This will take 5-10 minutes

---

### Step 3: Run Dataset Analysis 📊

1. **Create Analysis Notebook:**
   - Open Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   
2. **Create New Notebook:**
   - Navigate to `notebooks/` folder
   - Create: `01_dataset_analysis.ipynb`

3. **Run This Code:**

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (15, 10)

# Dataset path
data_path = '../data/raw/Training'

# Classes
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

# Count images per class
class_counts = {}
for cls in classes:
    cls_path = os.path.join(data_path, cls)
    count = len(os.listdir(cls_path))
    class_counts[cls] = count
    print(f"{cls.capitalize()}: {count} images")

print(f"\nTotal Training Images: {sum(class_counts.values())}")

# Plot distribution
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values(), color=['#22d3ee', '#c084fc', '#f472b6', '#10b981'])
plt.title('Class Distribution - Training Set', fontsize=16, fontweight='bold')
plt.xlabel('Tumor Class', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze image sizes
print("\n" + "="*50)
print("IMAGE RESOLUTION ANALYSIS")
print("="*50)

widths = []
heights = []

for cls in classes:
    cls_path = os.path.join(data_path, cls)
    images = os.listdir(cls_path)[:100]  # Sample 100 images per class
    
    for img_name in images:
        img_path = os.path.join(cls_path, img_name)
        try:
            img = Image.open(img_path)
            widths.append(img.size[0])
            heights.append(img.size[1])
        except:
            pass

print(f"Min Width: {min(widths)}px")
print(f"Max Width: {max(widths)}px")
print(f"Avg Width: {int(np.mean(widths))}px")
print(f"\nMin Height: {min(heights)}px")
print(f"Max Height: {max(heights)}px")
print(f"Avg Height: {int(np.mean(heights))}px")

# Visualize samples
print("\n" + "="*50)
print("SAMPLE IMAGES")
print("="*50)

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle('Sample MRI Images from Each Class', fontsize=16, fontweight='bold')

for i, cls in enumerate(classes):
    cls_path = os.path.join(data_path, cls)
    images = os.listdir(cls_path)[:5]  # Get 5 samples
    
    for j, img_name in enumerate(images):
        img_path = os.path.join(cls_path, img_name)
        img = Image.open(img_path)
        
        axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')
        if j == 0:
            axes[i, j].set_title(cls.capitalize(), fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n✅ Dataset Analysis Complete!")
print("\nNext Steps:")
print("1. Review the visualizations")
print("2. Check for any data quality issues")
print("3. Proceed to preprocessing pipeline")
```

4. **Run All Cells**
   - Click "Run All" or press Shift+Enter on each cell

5. **Take Screenshots:**
   - Screenshot of class distribution chart
   - Screenshot of sample images
   - Screenshot of resolution analysis

---

### Step 4: Share Your Progress 📸

**Send me:**
1. ✅ Screenshot of folder structure showing dataset
2. ✅ Screenshot of class distribution chart
3. ✅ Screenshot of sample MRI images
4. ✅ Tell me:
   - Total number of images
   - Number of images per class
   - Any observations about the data

---

## What Happens Next? 🎯

Once you complete the above steps, I'll help you with:

### Phase 2: Preprocessing Pipeline
- Image resizing to 224x224
- Normalization
- Data augmentation
- Train/Val/Test split

### Phase 3: Model Training
- Load EfficientNet-B0
- Build classification head
- Train the model
- Monitor performance

### Phase 4: Evaluation
- Test set evaluation
- Confusion matrix
- Performance metrics
- Error analysis

### Phase 5: Grad-CAM
- Implement explainable AI
- Generate heatmaps
- Visualize attention regions

---

## Common Issues & Solutions 🔧

### Issue 1: "pip install fails"
**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Issue 2: "TensorFlow GPU not detected"
**Solution:**
- For training, you can use Google Colab (free GPU)
- Or continue with CPU (slower but works)

### Issue 3: "Dataset folder not found"
**Solution:**
- Make sure you extracted to `data/raw/`
- Check the folder structure matches exactly

### Issue 4: "Jupyter notebook won't start"
**Solution:**
```bash
pip install jupyter notebook
jupyter notebook
```

---

## Time Estimate ⏱️

- **Dataset Download:** 10-15 minutes
- **Environment Setup:** 10-15 minutes
- **Dataset Analysis:** 15-20 minutes
- **Total:** ~45 minutes

---

## Need Help? 🆘

If you encounter any issues:
1. Check the error message carefully
2. Google the error (most are common)
3. Ask me with:
   - What you were trying to do
   - The exact error message
   - Screenshot if possible

---

## Success Checklist ✅

- [ ] Dataset downloaded and extracted
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Jupyter notebook running
- [ ] Dataset analysis complete
- [ ] Screenshots taken
- [ ] Ready for next phase

---

**Once you complete this, you'll have:**
- ✅ Complete dataset ready
- ✅ Understanding of data distribution
- ✅ Visual samples of each class
- ✅ Foundation for model training

**Let's build NeuraSight! 🚀**
