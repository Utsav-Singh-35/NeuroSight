# Sample Images

This folder contains sample images for each module/scan type used in the NeuraSight dashboard for quick testing.

## Structure

```
data/samples/
├── brainMRI/
│   ├── glioma_sample.jpg        ← Sample glioma MRI
│   ├── meningioma_sample.jpg    ← Sample meningioma MRI
│   ├── notumor_sample.jpg       ← Sample no-tumor MRI
│   └── pituitary_sample.jpg     ← Sample pituitary tumor MRI
│
└── (future modules go here)
    ├── lungCT/
    ├── chestXray/
    └── retinal/
```

## Usage

These samples are copied to `frontend/samples/` for serving via the Vite dev server. They appear as "Try Sample" buttons on the dashboard so users can test the AI without needing their own MRI images.

## Adding New Modules

When adding a new scan type (e.g., Lung CT):
1. Create a subfolder: `data/samples/lungCT/`
2. Place one representative sample per class
3. Copy to `frontend/samples/` for the dashboard
4. Update the dashboard UI to include new sample buttons
