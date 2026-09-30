# NeuraSight — Technical Presentation Script

**Project:** NeuraSight — Multi-Module Medical Image Classification with Deep Stacking Ensembles,
Gradient-Based Explainability, and a Calibrated, Evidence-Grounded Decision-Support Layer
**Domain:** Medical Imaging + Deep Learning
**Stack:** PyTorch · timm · scikit-learn · sentence-transformers · FastAPI · Express.js · MongoDB · Vite
**Document type:** Technical script (formulas, protocols, measured results, failure analysis)

---

## Provenance and Honesty Statement

Every number in this document is tagged with its source so nothing is presented as
measured when it is not:

| Tag | Meaning |
|-----|---------|
| **[CONFIG]** | Read directly from a committed artifact (`ensemble_config.json`, `metadata.json`) |
| **[PICKLE]** | Introspected from the actual serialised meta-learner `.pkl` |
| **[MEASURED]** | Measured on this machine during preparation of this document |
| **[DOC]** | Carried forward from `docs/brain_mri_model_results.md` / earlier notebook runs |
| **[NOT RECORDED]** | The value does not exist anywhere in the repository and must be regenerated |

> **Critical limitation to state openly in the viva:** all seven training notebooks in
> `notebooks/` have **cleared outputs** (0 of 25/13/19/38/8/12/23 cells retain output), and no
> `model_comparison.csv`, `*_test_probs.npy`, or `ensemble_metrics.json` exists in the working
> tree. Consequently **per-model and per-class metrics for the chest X-ray modules cannot be
> reproduced from this repository.** Only the aggregate `ensemble_accuracy` persisted, because
> it was written into the ensemble config JSON. This is flagged throughout as **[NOT RECORDED]**.
>
> **Brain MRI is the exception, and deliberately so.** Its base-model probability matrices were
> regenerated from the committed weights during this work, and the published 96.75% ensemble
> accuracy **reproduces exactly (+0.00 pp)** — see §13.12. Every calibration, conformal, OOD and
> retrieval number in §13.12–§13.19 is therefore **[MEASURED]** on the 480-image final test split.

> **Reading order.** §1–§12 describe the system as originally built and audited. **§13 documents
> what was then implemented in response** — calibration, conformal prediction, OOD screening, a
> cited knowledge base, local retrieval, reviewed narratives, PDF reporting, and the integration of
> all of it. Where §12 lists a defect and §13 reports it fixed, both are kept: the audit trail is
> part of the contribution.

---

## 1. What the System Identifies

NeuraSight is not a single classifier. It is a **module registry** serving two independent
diagnostic tasks, each backed by its own stacking ensemble, its own class taxonomy, and its own
weight directory.

**What a single request returns is also not a single class.** For brain MRI the response carries a
**class label with calibrated confidence**, a **conformal prediction set with a coverage guarantee**,
an **uncertainty band** (confident / borderline / indeterminate) that can escalate the risk level, a
**Grad-CAM overlay** with a flag saying whether it faithfully explains the reported class, an
**input-validation verdict** (quality gate + novelty score), **ensemble transparency**
(which models contributed, which were skipped), **cited clinical guidance** with resolved
references, and a **provenance statement** of whether the wording came from a human-reviewed
narrative or a deterministic template. All of it is available as a 4-page PDF. Measured results for
each of those layers are in §13.12–§13.19.

### 1.1 Module inventory (live-verified)

Queried from the running service at `GET /modules` **[MEASURED]**:

```
id=brain_mri    available=True  classes=Glioma / Meningioma / No Tumor / Pituitary
id=chest_xray   available=True  classes=Normal / Pneumonia / Tuberculosis
```

### 1.2 Diseases and conditions identified — complete list

**Module A — Brain MRI Tumor Classification** (4 classes, `models/`) **[CONFIG]**

| Idx | Class | Clinical meaning | Risk tier in report engine |
|-----|-------|------------------|----------------------------|
| 0 | **Glioma** | Tumour of glial cells; includes glioblastoma, the most aggressive primary brain malignancy | High |
| 1 | **Meningioma** | Tumour arising from the meninges; usually benign, slow-growing | Medium |
| 2 | **No Tumor** | No neoplastic finding on the MRI | Low |
| 3 | **Pituitary** | Tumour of the pituitary gland; typically benign adenoma | Medium |

Internal training label order is lowercase `glioma, meningioma, notumor, pituitary`; the API
presents `Glioma, Meningioma, No Tumor, Pituitary`. **Index order is identical**, so the mapping
is safe — but it is convention-only, enforced nowhere in code.

**Module B — Chest X-ray Disease Detection** (3 classes, `chest/`) **[CONFIG]**

| Idx | Class | Clinical meaning | Risk tier |
|-----|-------|------------------|-----------|
| 0 | **Normal** | Clear lung fields, no consolidation/infiltrate/effusion | Low |
| 1 | **Pneumonia** | Pulmonary infection with consolidation | High |
| 2 | **Tuberculosis** | Mycobacterial infection; cavitation, upper-lobe infiltrates | High |

**Module B′ — Chest X-ray, archived alternative taxonomy** (3 classes, `chest V1/`) **[CONFIG]**

| Idx | Class | Clinical meaning |
|-----|-------|------------------|
| 0 | **Bacteria** | Bacterial pneumonia |
| 1 | **Normal** | No pneumonia |
| 2 | **Virus** | Viral pneumonia |

This third taxonomy is **trained, serialised, and on disk, but not wired into the application.**
Section 3.3 explains why it exists and Section 12.4 explains why it is not deployed.

**Total distinct conditions the project has trained models for: 8**
(Glioma, Meningioma, No Tumor, Pituitary, Normal-chest, Pneumonia, Tuberculosis, and the
Bacteria/Virus pneumonia-aetiology split.)

---

## 2. The Two Chest Datasets — Different Models, Different Tasks

This is the part most often misexplained, so state it precisely. **Two separate chest X-ray
training programmes were run, on two different datasets, producing two different label
taxonomies and two different sets of weights.** They are not two versions of the same run.

### 2.1 Side-by-side

| | **Chest Programme 1** (archived) | **Chest Programme 2** (deployed) |
|---|---|---|
| Weight directory | `chest V1/` | `chest/` |
| Dataset | Kermany et al. Chest X-ray | Chest X-ray Dataset (Muhammad Rehan) |
| Kaggle slug | `paultimothymooney/chest-xray-pneumonia` | `muhammadrehan00/chest-xray-dataset` |
| Classes | Bacteria / Normal / Virus | Normal / Pneumonia / Tuberculosis |
| Clinical question | *Which pathogen class* caused the pneumonia? | *Which disease* is present? |
| Corpus size | ~5,856 images | **[NOT RECORDED]** (computed at runtime in setup notebook) |
| Split protocol | Stratified 70 / 15 / 15, generated in-notebook | Pre-split `train/val/test` shipped with dataset |
| Label derivation | Parsed from filename substrings `bacteria` / `virus` inside the `PNEUMONIA/` folder | Directory names |
| Base models | EfficientNet-B0, ResNet-50, DenseNet-121 | EfficientNet-B0, ResNet-50, DenseNet-121 |
| Feature dim | 9 | 9 |
| Meta-learner | LogisticRegression | LogisticRegression |
| **Ensemble accuracy** | **85.68%** [CONFIG] | **82.00%** [CONFIG] |
| Training notebook | `Chest_Xray_SageMaker_Training.ipynb` | `Chest_Xray_Model_Training.ipynb` → `_V2.ipynb` |
| Platform | AWS SageMaker | Google Colab |
| Status | **Dead weight** — gitignored, referenced by no code | **Live** — served by `chest_xray` module |

### 2.2 Why the archived model scores higher but was still replaced

`chest V1` reports 85.68% versus `chest` at 82.00%, yet the lower-scoring model is deployed.
This is a deliberate and defensible trade-off, and it is worth arguing explicitly:

1. **The tasks are not equally hard or equally useful.** Distinguishing bacterial from viral
   pneumonia on a plain radiograph is a task where even expert radiologists perform close to
   chance; the Kermany labels derive from filename conventions, not independent microbiological
   confirmation. A high score on a weakly-labelled task is not evidence of clinical value.
2. **Tuberculosis is the higher-value target.** TB is a notifiable, screenable disease with a
   defined radiographic signature. Detecting *TB vs pneumonia vs normal* answers a question a
   clinic actually asks.
3. **Accuracy across different label spaces is not comparable.** 85.68% on
   {Bacteria, Normal, Virus} and 82.00% on {Normal, Pneumonia, Tuberculosis} are measured against
   different priors and different class difficulties. Placing them in the same ranking is a
   category error.

**Presentation line:** "We did not pick the bigger number. We picked the more meaningful task,
and we report the honest cost of that choice — a 3.7-point drop on a metric that was never
comparable in the first place."

### 2.3 Brain MRI dataset

| Property | Value |
|----------|-------|
| Dataset | Brain Tumor MRI Dataset (Masoud Nickparvar) **[DOC]** |
| Kaggle slug | `masoudnickparvar/brain-tumor-mri-dataset` |
| Total corpus | 7,023 images **[DOC]** |
| Training subset | 5,600 images **[DOC]** |
| Test set | 1,600 images, **perfectly balanced at 400 per class** **[DOC]** |
| Local copy | `data/brainMRI/Training/{glioma,meningioma,notumor,pituitary}` (present on disk) |

The balanced test set matters methodologically: with 400 images per class, a naive
majority-class baseline scores exactly 25%, and macro-averaged metrics equal micro-averaged
accuracy. Every brain metric below is therefore directly interpretable.

---

## 3. Why These Models Were Chosen

### 3.1 The four base architectures

Selection was driven by **decorrelated inductive bias**, not by leaderboard position. A stacking
ensemble only gains over its best member when members make *different* mistakes.

| Model | Params | ImageNet top-1 | Inductive bias it contributes | Why included |
|-------|--------|----------------|-------------------------------|--------------|
| **EfficientNet-B0** | 5.3 M | 77.1% | Compound depth/width/resolution scaling; inverted-residual MBConv blocks with squeeze-excitation | Best accuracy-per-FLOP; serves as the single-model fallback path |
| **ResNet-50** | 25.6 M | 76.1% | Additive residual identity mappings; bottleneck 1×1→3×3→1×1 | The field's reference baseline; residual gradient flow is well understood and reproducible |
| **DenseNet-121** | 8.0 M | 74.4% | Concatenative feature reuse — every layer sees all preceding feature maps | Strong on fine texture, which is what distinguishes tumour margins and lung infiltrates |
| **VGG-16** | 138 M | 71.3% | Plain deep stack of 3×3 convolutions, no skips | Architecturally *unlike* the other three, so its errors decorrelate — its value is diversity, not accuracy |

**Why VGG-16 despite being the weakest and 26× larger than EfficientNet-B0:** in a stacking
ensemble a weak-but-different learner can still raise the ceiling, because the meta-learner can
learn *when* to trust it. Section 6.3 shows the brain meta-learner assigns VGG-16 a 25.6%
coefficient share — the second-highest of the four — which empirically justifies keeping it.
It is dropped from both chest ensembles, where the 512 MB checkpoint was not worth its marginal
contribution on a 3-class problem.

### 3.2 Why transfer learning rather than training from scratch

Medical imaging corpora are small (thousands, not millions). Training a 138 M-parameter network
from random initialisation on 5,600 images would overfit catastrophically. ImageNet pre-training
supplies generic low-level filters — edges, textures, gradients — that transfer to greyscale
radiology; only the classifier head and upper blocks need to re-specialise. This is the standard
protocol in the literature and is why every base model is instantiated with `pretrained=True`
during training.

### 3.3 Why Logistic Regression as the meta-learner

| Criterion | Logistic Regression | A second neural network |
|-----------|--------------------|-------------------------|
| Parameters to fit | 68 (brain: 4×16 + 4) | Thousands |
| Overfitting risk on a small meta-set | Low | High |
| Interpretability | Coefficients are directly readable (see §6.3) | Opaque |
| Training cost | Milliseconds | Minutes + tuning |
| Convexity | Convex — one global optimum, reproducible | Non-convex, seed-dependent |

The meta-feature space is already highly processed — it consists of calibrated class
probabilities, not raw pixels. The decision boundary needed is close to linear, so a linear model
is the correct capacity. Wolpert's original 1992 stacked-generalization paper makes exactly this
argument: keep the combiner simple to avoid overfitting the meta-level.

**Verified meta-learner configuration [PICKLE]** — introspected from the actual `.pkl` files:

| | `models/meta_model.pkl` | `chest/meta_model_Chest_Xray.pkl` | `chest V1/meta_model_Chest_Xray.pkl` |
|---|---|---|---|
| Class | `LogisticRegression` | `LogisticRegression` | `LogisticRegression` |
| `n_features_in_` | **16** | **9** | **9** |
| `classes_` | `[0 1 2 3]` | `[0 1 2]` | `[0 1 2]` |
| `coef_` shape | `(4, 16)` | `(3, 9)` | `(3, 9)` |
| `intercept_` | `[2.254, −1.890, −0.524, 0.160]` | `[0.880, −0.751, −0.129]` | `[0.869, −1.267, 0.399]` |
| Regularisation `C` | 1.0 | 1.0 | 1.0 |
| Solver | `lbfgs` | `lbfgs` | `lbfgs` |
| `max_iter` | 1000 | 1000 | 1000 |

All three feature dimensions match their config `feature_dim` exactly (16 = 4 models × 4 classes,
9 = 3 models × 3 classes), confirming the serialised artefacts are mutually consistent with the
configs the server reads.

---

## 4. System Architecture — How Everything Actually Works

### 4.1 Three-tier topology

```
┌─────────────────┐              ┌──────────────────────┐            ┌────────────────────────────┐
│  React + Vite   │  multipart   │  Express.js Gateway  │ multipart  │   FastAPI ML Service       │
│   :3000         │─────────────▶│  :5000               │───────────▶│   :8000                    │
│                 │◀─────────────│  • multer memory     │◀───────────│  • module registry         │
│  landing +      │   JSON       │  • 10 MB cap         │   JSON     │  • preprocessing           │
│  dashboard      │              │  • MIME allow-list   │            │  • stacking ensembles      │
└─────────────────┘              │  • 30 s timeout      │            │  • Grad-CAM                │
                                 └──────────┬───────────┘            │  • calibration + conformal │
                                            │                        │  • OOD / quality gate      │
                                            │                        │  • cited KB + retrieval    │
                                            │                        │  • report + PDF            │
                                            ▼                        └────────────┬───────────────┘
                                  ┌───────────────────┐                           │
                                  │  MongoDB          │       ┌───────────────────┴──────────────┐
                                  │  predictions +    │       │  Weights + fitted artefacts      │
                                  │  uncertainty      │       │  models/ (brain, 4×.pth)         │
                                  └───────────────────┘       │  chest/  (chest, 3×.pth)         │
                                                              │  models/calibration/*.json       │
                                                              │  knowledge/ (28 chunks, index)   │
                                                              └──────────────────────────────────┘
```

**Separation rationale:** Python owns the ML runtime because PyTorch/timm/scikit-learn live
there. Node owns the gateway because file upload, persistence, and session concerns are better
served by the JS ecosystem. The boundary is a single multipart POST, so either side can be
scaled, containerised, or replaced independently.

**Everything in `models/calibration/` and `knowledge/` is a *fitted artefact*, not code** —
temperature, conformal quantile, OOD thresholds, embeddings, narratives. They are committed, so the
deployed behaviour is reproducible and auditable, and every number in §13 can be traced back to the
file it came from.

### 4.2 Endpoint contract (FastAPI)

All three inference endpoints are `POST`, `multipart/form-data`, form field **`image`**, with a
`module` query parameter defaulting to `brain_mri`.

| Endpoint | Returns | Response model |
|----------|---------|----------------|
| `POST /predict?module=…` | `{prediction, confidence, probabilities, module}` | `PredictionResponse` |
| `POST /gradcam?module=…` | `{heatmap, prediction, confidence, module}` | `GradCAMResponse` |
| `POST /report?module=…` | full report dict + `module` | **none — unvalidated raw dict** |
| `GET /health` | `{status, model_loaded, available_modules}` | `HealthResponse` (schema incomplete) |
| `GET /modules` | `{modules:[{id,name,classes,available}]}` | declared but not attached |

> **Contract as it stands now (§13.19).** `/predict`, `/gradcam` and `/report` additionally return
> `uncertainty` (band, prediction set, coverage guarantee, `calibrated`), `validation` (quality gate,
> `is_ood`, `ood_score`, `ood_reason`) and `ensemble` (`ensemble_active`, `models_used`,
> `models_skipped`, `agreeing_model`, `base_predictions`). `/report` now has a `ReportResponse`
> model, closing defect #35. `POST /report/pdf` was added and streams a rendered PDF. `/health`
> reports every layer — calibration, conformal, OOD, retrieval index, narratives — so a missing
> artefact is visible without reading logs (#34). Shared guards were extracted to
> `app/routers/_common.py` instead of being duplicated across three routers.

Shared guard sequence, in order:

1. `content_type` not in `{image/jpeg, image/png}` → **422**
2. read failure → **500**
3. zero bytes → **422**
4. unknown `module` → **400**
5. `module.is_available()` false → **503**
6. any exception inside the module → **500** with a generic message
7. success → structured log line with UTC ISO timestamp, module, filename, prediction,
   confidence, `duration_ms`

### 4.3 Request lifecycle, end to end

```
Browser selects file
   │
   ▼
Express: multer memoryStorage, fileSize ≤ 10,485,760 B, MIME ∈ {jpeg,png}
   │   413 if oversize · 400 if wrong type/empty/absent
   ▼
axios POST → FASTAPI_URL/predict   (timeout 30,000 ms)
   │   ECONNREFUSED / ETIMEDOUT → 502 "ML service is unavailable"
   ▼
FastAPI: registry.get(module) → guards → module.predict(bytes)
   │
   ▼
preprocess_image(bytes) → float32 tensor (1, 3, 224, 224) on CPU
   │
   ▼
for each base model in config["model_order"]:
      timm.create_model(name, pretrained=False, num_classes=K)
      load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
      p⁽ᵐ⁾ = softmax(model(x))        ← (K,) vector
      del model; gc.collect()          ← one model resident at a time
   │
   ▼
X = concat(p⁽¹⁾ … p⁽ᴹ⁾) → (1, M·K)
   │
   ▼
meta.predict(X) → class index ; meta.predict_proba(X) → confidence vector
   │
   ▼
Express: persist to MongoDB (filename, prediction, confidence, probabilities)
   │   save failure → 500 "Could not save prediction record"
   ▼
JSON to browser
```

**Memory strategy — load-one-free-one.** The four brain checkpoints total ~645 MB on disk
(VGG-16 alone is 512 MB). Holding all four resident would exceed a small cloud instance, so
`run_ensemble_inference` loads exactly one model, extracts its probability vector, then
`del`s it and calls `gc.collect()` before the next. This trades latency for a flat memory
ceiling — the direct cause of the multi-second response times in §9.

---

## 5. Preprocessing — Exact Specification

Implemented in `app/services/preprocessor.py`; identical for both modules.

### 5.1 Step sequence

1. **Decode and verify.** `PIL.Image.open(BytesIO(bytes))`, then `.verify()`, then **re-open**
   (verify leaves the handle unusable). Any exception → `ValueError("Image could not be processed")`.
2. **Colour normalisation.** `.convert("RGB")` — greyscale replicates to 3 channels; RGBA drops
   alpha. Necessary because ImageNet-pretrained stems expect 3 input channels, while MRI and
   X-ray are single-channel.
3. **Resize.** `.resize((224, 224), Image.BILINEAR)`.
4. **Scale to unit interval.** Divide by 255.
5. **ImageNet standardisation**, per channel.
6. **Layout transform.** HWC → CHW, then prepend batch axis.

### 5.2 Formulas

Bilinear interpolation at target coordinate $(x, y)$ with neighbours $Q_{11}, Q_{21}, Q_{12}, Q_{22}$:

$$
I(x,y) \approx (1-dx)(1-dy)\,Q_{11} + dx(1-dy)\,Q_{21} + (1-dx)dy\,Q_{12} + dx\,dy\,Q_{22}
$$

Per-channel standardisation, for channel $c \in \{R,G,B\}$:

$$
x'_c = \frac{\dfrac{x_c}{255} - \mu_c}{\sigma_c}
$$

with the ImageNet constants hard-coded in the source:

$$
\boldsymbol{\mu} = [0.485,\; 0.456,\; 0.406], \qquad
\boldsymbol{\sigma} = [0.229,\; 0.224,\; 0.225]
$$

Output tensor: shape $(1, 3, 224, 224)$, dtype `float32`, device CPU.

### 5.3 Why 224×224 and why ImageNet constants

224×224 is the native input resolution for all four ImageNet-pretrained backbones; deviating
would force spatial re-interpolation of learned positional statistics. The ImageNet mean/std must
be reused verbatim because the pretrained weights were optimised in that normalised space —
substituting dataset-specific statistics would shift every activation distribution away from what
the frozen lower layers expect.

### 5.4 Train/serve skew — a real defect to disclose

The serving path applies a **direct squash-resize to 224×224 with no aspect-ratio preservation
and no centre crop.** The V2 training transform is `transforms.Resize((224, 224))`, which is the
same squash — so **V2 is consistent**. But any model trained with the more common
`Resize(256) → CenterCrop(224)` recipe would see a systematically different field of view at
inference than during training. This is the classic silent accuracy leak in deployed vision
systems. It is consistent today; it must be re-verified whenever a model is retrained.

> **Now verified empirically, not just by reading the transforms (§13.12).** Regenerating the base
> probability matrices from the committed weights through the **serving** preprocessor reproduced all
> four published per-model accuracies to 2 dp and the ensemble accuracy at **+0.00 pp**. An exact
> match is only possible if the serving preprocessor reproduces training behaviour, so **there is no
> train/serve skew in the brain pipeline today.** That check is the regression test for this section:
> re-run `scripts/generate_test_probs.py` after any preprocessing change and the numbers must still
> land on 95.06 / 94.88 / 96.06 / 94.56 and 96.75.

---

## 6. The Stacking Ensemble — Complete Mechanics

### 6.1 Level 0 — base learners

Each base model $m \in \{1,\dots,M\}$ maps the preprocessed tensor to $K$ logits, converted to a
probability simplex by softmax:

$$
p^{(m)}_i = \frac{e^{z^{(m)}_i}}{\displaystyle\sum_{j=1}^{K} e^{z^{(m)}_j}},
\qquad \sum_{i=1}^{K} p^{(m)}_i = 1,\quad p^{(m)}_i > 0
$$

- Brain: $M = 4$, $K = 4$
- Chest: $M = 3$, $K = 3$

### 6.2 Meta-feature construction

Probability vectors are concatenated **in the exact order given by `config["model_order"]`**:

$$
\mathbf{X} = \left[\, p^{(1)}_1 \dots p^{(1)}_K \;\middle|\; p^{(2)}_1 \dots p^{(2)}_K \;\middle|\; \dots \;\middle|\; p^{(M)}_1 \dots p^{(M)}_K \,\right] \in \mathbb{R}^{M \cdot K}
$$

**Brain — 16 dimensions, concrete layout:**

```
index:  0    1    2    3  │  4    5    6    7  │  8    9   10   11  │ 12   13   14   15
model:  ──EfficientNet──  │  ────ResNet-50───  │  ──DenseNet-121──  │  ────VGG-16─────
class:  gli  men  not  pit│ gli  men  not  pit│ gli  men  not  pit│ gli  men  not  pit
```

**Chest — 9 dimensions:**

```
index:  0    1    2  │  3    4    5  │  6    7    8
model:  ─EfficientNet│  ──ResNet-50─ │ ─DenseNet-121─
class:  Nor  Pne  TB │ Nor  Pne  TB  │ Nor  Pne  TB
```

> **This ordering is the single most fragile contract in the system.** It is enforced only by
> convention — the server reads `model_order` from JSON and trusts it matches the column order the
> meta-learner was fitted on. There is no runtime assertion against `meta.n_features_in_` beyond
> sklearn's own dimension check, and no check at all on *permutation*. Reordering
> `model_order` in the config would silently produce confidently wrong predictions, because the
> feature count would still be 16.

### 6.3 Level 1 — multinomial logistic regression

$$
P(y = k \mid \mathbf{X}) = \frac{\exp(\mathbf{w}_k^{\top}\mathbf{X} + b_k)}{\displaystyle\sum_{j=1}^{K}\exp(\mathbf{w}_j^{\top}\mathbf{X} + b_j)}
$$

Prediction and reported confidence:

$$
\hat{y} = \arg\max_k P(y = k \mid \mathbf{X}), \qquad
\text{confidence} = 100 \times \max_k P(y = k \mid \mathbf{X})
$$

Fitted by L-BFGS minimising L2-penalised multinomial cross-entropy:

$$
\mathcal{L}(W) = -\sum_{n=1}^{N}\sum_{k=1}^{K} y_{nk}\log P(y=k \mid \mathbf{X}_n) \;+\; \frac{1}{2C}\lVert W\rVert_2^2, \qquad C = 1.0
$$

**Key consequence for the viva:** the confidence NeuraSight reports is the **meta-learner's own
posterior**, not an average of base-model softmaxes. It is a learned recalibration, which is why
ensemble confidence can legitimately differ from every individual base model's confidence.

### 6.4 Derived insight — which base models the meta-learner actually relies on

Computed directly from the fitted coefficient matrices **[PICKLE]**, summing $|w|$ over each
model's column block and normalising:

$$
\text{share}(m) = \frac{\sum_{k=1}^{K}\sum_{i \in \text{block}(m)} |w_{ki}|}{\sum_{k=1}^{K}\sum_{i=1}^{MK} |w_{ki}|}
$$

**Brain MRI ensemble:**

| Base model | $\sum\lvert\text{coef}\rvert$ | Share |
|------------|------------------------------|-------|
| EfficientNet-B0 | 10.553 | 23.5% |
| ResNet-50 | 11.570 | **25.8%** |
| DenseNet-121 | 11.246 | 25.1% |
| VGG-16 | 11.461 | 25.6% |

**Chest (deployed, TB taxonomy):**

| Base model | $\sum\lvert\text{coef}\rvert$ | Share |
|------------|------------------------------|-------|
| EfficientNet-B0 | 6.063 | 28.2% |
| ResNet-50 | 8.808 | **41.0%** |
| DenseNet-121 | 6.611 | 30.8% |

**Chest V1 (archived, Kermany taxonomy):**

| Base model | $\sum\lvert\text{coef}\rvert$ | Share |
|------------|------------------------------|-------|
| EfficientNet-B0 | 7.574 | **42.8%** |
| ResNet-50 | 4.556 | 25.8% |
| DenseNet-121 | 5.550 | 31.4% |

**Interpretation — this is a genuine finding, not decoration:**

- The **brain** meta-learner is almost perfectly **uniform** (23.5–25.8%). No single architecture
  dominates, which is the signature of a healthy ensemble: all four contribute complementary
  information. It also retrospectively **justifies keeping VGG-16** despite it being the weakest
  and largest member — it draws the second-highest share.
- The **chest TB** meta-learner concentrates **41% on ResNet-50**, suggesting ResNet's residual
  features are the most discriminative for cavitation and infiltrate texture in this corpus.
- The **chest V1** meta-learner inverts this, leaning **42.8% on EfficientNet-B0**. Two different
  datasets produce two different reliance profiles, which is exactly why a per-task meta-learner
  is the right design rather than one global combiner.

### 6.5 Agreement-based model selection for explainability

After the meta-learner decides class $\hat{y}$, the system selects the base model to explain:

$$
m^{*} = \arg\max_{m \,:\, \arg\max_i p^{(m)}_i \,=\, \hat{y}} p^{(m)}_{\hat{y}}
$$

In words: among base models whose own top-1 **agrees** with the ensemble, take the one most
confident in the ensemble's chosen class. Grad-CAM then runs on $m^{*}$, so the heatmap is a
faithful explanation of a model that actually believes the reported answer.

**Degenerate case (must be disclosed):** if *no* base model agrees with the meta-learner, the
code falls back to "the first model that ran". The heatmap then explains a **different class**
than the one reported — and because the library Grad-CAM path passes `targets=None` (explaining
the base model's own argmax rather than $\hat{y}$), the mismatch is guaranteed in that branch.

---

## 7. Explainability — Grad-CAM Specification

### 7.1 Formulation (Selvaraju et al., 2017)

Let $A^k \in \mathbb{R}^{u \times v}$ be the $k$-th feature map of the target convolutional layer
and $Y^c$ the pre-softmax score for class $c$.

**Step 1 — neuron importance by gradient global-average-pooling:**

$$
\alpha^{c}_{k} = \frac{1}{Z}\sum_{i}\sum_{j} \frac{\partial Y^{c}}{\partial A^{k}_{ij}}, \qquad Z = u \times v
$$

**Step 2 — weighted combination, ReLU-rectified to keep only positive evidence:**

$$
L^{c}_{\text{Grad-CAM}} = \mathrm{ReLU}\!\left(\sum_{k} \alpha^{c}_{k} A^{k}\right)
$$

ReLU is essential: negative contributions correspond to evidence *against* class $c$ and would
otherwise dilute the localisation.

**Step 3 — min–max normalisation to $[0,1]$:**

$$
\tilde{L} = \frac{L - \min(L)}{\max(L) - \min(L) + \varepsilon}
$$

**Step 4 — bilinear upsample to 224×224, then JET colour-map and alpha blend:**

$$
I_{\text{out}} = (1-\alpha)\, I_{\text{orig}} + \alpha \cdot \mathrm{JET}(\tilde{L}), \qquad \alpha = 0.4
$$

Output is encoded as a **base64 PNG string with no `data:` URI prefix** — the frontend must
prepend `data:image/png;base64,`. Measured payload length: **72,864 characters** for a
224×224 overlay **[MEASURED]**.

### 7.2 Target layer per architecture

The layer choice matters: too deep and spatial resolution collapses, too shallow and the map
shows generic edges instead of class evidence. The last convolutional stage is the standard
compromise.

| Architecture | Target layer | Rationale |
|--------------|--------------|-----------|
| EfficientNet-B0 | `model.blocks[-1]` | Final MBConv stage, 7×7 spatial |
| ResNet-50 | `model.layer4[-1]` | Final bottleneck residual block |
| DenseNet-121 | `model.features[-1]` | Terminal stage — **note: for timm this resolves to `norm5`, a BatchNorm, not a conv** |
| VGG-16 / fallback | last `nn.Conv2d` found by walking `model.modules()` | Architecture-agnostic fallback |

### 7.3 Dual implementation

A primary path uses the `pytorch-grad-cam` library; if the import fails, a hand-written fallback
registers `register_forward_hook` and `register_full_backward_hook` on the target layer, performs
`output[0, predicted_index].backward()`, and computes the formulas above directly with hooks
released in a `finally` block. The manual path honours the requested class index; the library path
does not (see §6.5).

---

## 8. Metrics — Definitions and Formulas

### 8.1 Confusion matrix basis

For class $c$, against the rest:

| | Predicted $c$ | Predicted not $c$ |
|---|---|---|
| **Actually $c$** | TP | FN |
| **Actually not $c$** | FP | TN |

### 8.2 Core metrics

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \qquad\text{(multi-class: } \tfrac{1}{N}\textstyle\sum_n \mathbb{1}[\hat{y}_n = y_n]\text{)}
$$

$$
\text{Precision}_c = \frac{TP_c}{TP_c + FP_c} \qquad
\text{Recall}_c = \frac{TP_c}{TP_c + FN_c} \qquad
\text{F1}_c = 2\cdot\frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
$$

$$
\text{Macro-}F1 = \frac{1}{K}\sum_{c=1}^{K} F1_c \qquad\qquad
\text{Weighted-}F1 = \sum_{c=1}^{K} \frac{n_c}{N} F1_c
$$

$$
\text{Specificity}_c = \frac{TN_c}{TN_c + FP_c} \qquad\qquad
\text{AUC} = \int_{0}^{1} \text{TPR}\,d(\text{FPR})
$$

### 8.3 Why each metric is reported, and which one matters clinically

| Metric | What it answers | Clinical stake in this project |
|--------|-----------------|--------------------------------|
| **Accuracy** | Overall hit rate | Only trustworthy because the brain test set is exactly balanced (400/class). On the imbalanced chest corpus, accuracy alone is misleading. |
| **Precision** | Of flagged cases, how many were real? | Governs false-alarm burden — unnecessary follow-up imaging, patient anxiety, cost |
| **Recall (Sensitivity)** | Of real cases, how many did we catch? | **The metric that matters most.** A missed glioma or missed TB is the catastrophic error. A false positive costs a second opinion; a false negative can cost a life. |
| **F1** | Harmonic balance of the two | Single comparison figure when precision/recall trade off |
| **Macro average** | Unweighted mean across classes | Prevents a large easy class from masking failure on a small hard one |
| **Specificity** | True-negative rate | Screening suitability |
| **AUC** | Threshold-independent separability | Lets a clinic re-tune the operating point without retraining |

**The asymmetry must be stated explicitly in the presentation:** in screening, recall on
pathological classes outranks accuracy. Section 10.1's glioma recall of 0.81 is therefore the
single most important weakness in the entire brain pipeline, notwithstanding a headline 95%.

### 8.4 Training objective and schedule formulas

**Cross-entropy with label smoothing** ($\epsilon = 0.1$, chest V2):

$$
y^{\text{LS}}_k = (1-\epsilon)\,y_k + \frac{\epsilon}{K}, \qquad
\mathcal{L} = -\sum_{k=1}^{K} y^{\text{LS}}_k \log \hat{p}_k
$$

Smoothing prevents the network from driving logits to saturation, which improves calibration —
directly relevant because the stacking features *are* probabilities.

**Inverse-frequency class weighting** (chest V2, for corpus imbalance):

$$
w_c \propto \frac{1}{n_c}, \qquad
\mathcal{L} = -\sum_{k} w_k\, y^{\text{LS}}_k \log \hat{p}_k
$$

paired with a `WeightedRandomSampler` drawing samples with probability $\propto 1/n_c$, so batches
are class-balanced *and* the loss is reweighted.

**Linear warmup then cosine annealing** (3 warmup epochs, $T = 30$):

$$
\eta_t =
\begin{cases}
\eta_0 \cdot \dfrac{t+1}{3}, & t < 3 \\[2ex]
\eta_0 \cdot \dfrac{1}{2}\left(1 + \cos\left(\pi \dfrac{t-3}{T-3}\right)\right), & t \geq 3
\end{cases}
$$

Warmup avoids destroying pretrained features with large early gradients; cosine decay anneals to a
flat minimum that generalises better than a step schedule.

**AdamW decoupled weight decay:**

$$
\theta_{t+1} = \theta_t - \eta_t\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon} + \lambda\theta_t\right)
$$

Decoupling $\lambda$ from the adaptive term is what distinguishes AdamW from Adam+L2 and is the
correct choice for fine-tuning.

---

## 9. Training Configuration — Verified Hyperparameters

### 9.1 Chest X-ray V2 (deployed pipeline) — read from the notebook source

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Epochs | 30 (early stopping, patience 7) | Enough for convergence; patience prevents overfit |
| Learning rate | $5\times10^{-5}$ | Deliberately lowered from V1's $1\times10^{-4}$ |
| Weight decay | $1\times10^{-3}$ | Lowered from V1's $1\times10^{-2}$ |
| Batch size | 32 | GPU-memory bound at 224×224 |
| Optimiser | AdamW | Decoupled weight decay |
| Scheduler | 3-epoch linear warmup → cosine anneal | See §8.4 |
| Label smoothing | 0.1 | Calibration |
| Class weighting | Inverse frequency + `WeightedRandomSampler` | Corpus imbalance |
| Augmentation | `RandomAffine(degrees=0, translate=(0.05,0.05))` | Translation only |
| **Deliberately excluded** | `ColorJitter` | **Radiographic intensity is diagnostic information** — perturbing brightness/contrast destroys the very signal being classified |
| Normalisation | ImageNet mean/std | Matches pretrained weights |
| Model init | `pretrained=True` | Transfer learning |

The notebook documents these as five numbered "FIXES" over V1 — class weighting, weighted
sampling, class-weighted loss with smoothing, lower LR, and warmup+cosine. This is a legitimate
ablation narrative: V1 underperformed, five specific interventions were applied, V2 was retrained.

The "no ColorJitter for medical images" decision is worth calling out in the viva as evidence of
domain reasoning rather than blind recipe-copying.

### 9.2 Brain MRI EfficientNet-B0 (Phase 1) **[DOC]**

| Hyperparameter | Value |
|----------------|-------|
| Input | 224 × 224 × 3 |
| Epochs | 10 |
| Optimiser | Adam |
| Loss | Cross-entropy |
| Best checkpoint | Epoch 9 |
| Platform | Google Colab, Tesla T4 |

Classifier head **[DOC]**:

```
EfficientNet-B0 (ImageNet pretrained)
        ↓
Global Average Pooling 2D
        ↓
Dense(256, ReLU) + Dropout(0.5)
        ↓
Dense(128, ReLU) + Dropout(0.3)
        ↓
Dense(4, Softmax)
```

> **Inconsistency to disclose:** this documented head does not match the serving path. At
> inference the code calls `timm.create_model(name, pretrained=False, num_classes=K)`, which
> builds timm's **single-layer** classifier, and then `load_state_dict` with `strict=True`. Since
> loading succeeds, the deployed checkpoints must in fact have timm's plain head — so the
> two-hidden-layer diagram above describes an earlier Phase-1 experiment, not the shipped model.
> Present the timm head as the real architecture.

### 9.3 Stacking protocol — verified from both ensemble notebooks

Identical protocol for brain and chest:

1. Each trained base model's softmax probabilities on the **test set** are saved as
   `<SAVE_NAME>_test_probs.npy`.
2. Horizontally concatenate into $\mathbf{X}$ (16-dim brain, 9-dim chest).
3. **Stratified 50/50 split**: `train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)`.
4. Fit `LogisticRegression(max_iter=1000, C=1.0)` — chest adds
   `multi_class='multinomial', solver='lbfgs'` — on the **meta-train** half.
5. Evaluate on the **untouched meta-test** half.
6. Score each base model on that *same* meta-test half by
   `X_meta_test[:, i*K:(i+1)*K].argmax(1)`, giving an apples-to-apples comparison.
7. Persist `meta_model.pkl` + `ensemble_config.json` with `ensemble_accuracy`.

**Why the 50/50 split is the correct control here:** the meta-learner never sees the rows it is
scored on, so the reported ensemble accuracy is not circular, and the base-model comparison uses
the identical subset — eliminating the usual confound where an ensemble is compared against
base models evaluated on a different set.

**The methodological caveat that must be volunteered, not hidden:** canonical stacked
generalization fits the meta-learner on **out-of-fold predictions over the training set**
(K-fold cross-validation). Here the meta-features are drawn from the **test set** instead. The
50/50 split keeps the *final* number honest, but two consequences follow:

1. The effective meta-evaluation set is only **800 images** (brain), so the confidence interval on
   96.75% is wider than the three significant figures imply. At $n = 800$, the 95% Wald interval
   is roughly $\pm 1.2$ points.
2. Base models were early-stopped on the validation split, so the pipeline as a whole has seen
   validation data — standard practice, but it means the ensemble gain is an estimate, not a
   guarantee.

Section 13 lists the fix: regenerate meta-features by K-fold out-of-fold prediction on the
training set and reserve the full test set for final evaluation only.

---

## 10. Results — Per Model, Per Disease Category

### 10.1 Brain MRI — EfficientNet-B0, full 1,600-image test set **[DOC]**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Glioma | **1.00** | **0.81** | 0.89 | 400 |
| Meningioma | 0.90 | 0.99 | 0.95 | 400 |
| No Tumor | 0.92 | **1.00** | 0.96 | 400 |
| Pituitary | 0.99 | 1.00 | **1.00** | 400 |
| **Accuracy** | | | **0.95** | 1600 |
| Macro avg | 0.95 | 0.95 | 0.95 | 1600 |
| Weighted avg | 0.95 | 0.95 | 0.95 | 1600 |

**Per-category clinical reading — this is the analytical core of the results section:**

- **Glioma — precision 1.00, recall 0.81.** Every glioma call is correct, but **19% of gliomas
  are missed**. In a balanced 400-image class that is ~76 missed malignancies. Glioma is the most
  aggressive of the four; this is the project's most serious single weakness and must be presented
  as such rather than buried under the 95% headline.
- **Meningioma — precision 0.90, recall 0.99.** Catches nearly everything but over-claims.
  The 10% false-positive rate is the **mirror image of glioma's missed cases**: the model is
  absorbing gliomas into the meningioma class. Both tumours can present with similar
  contrast-enhancement patterns on T1, so the confusion is radiologically plausible, not random.
- **No Tumor — precision 0.92, recall 1.00.** Perfect recall is the desirable direction of error:
  no diseased patient is cleared as healthy. The 8% false-positive cost is over-referral, which is
  the acceptable failure mode in screening.
- **Pituitary — precision 0.99, recall 1.00, F1 1.00.** Best class. Pituitary tumours occupy a
  stereotyped anatomical location (sella turcica), giving the CNN a strong positional prior.

### 10.2 Brain MRI — base-model comparison, full test set **[DOC]**

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| **DenseNet-121** | **96.06%** | 96.32 | 96.06 | 95.97 |
| EfficientNet-B0 | 95.06% | 95.52 | 95.06 | 94.92 |
| ResNet-50 | 94.88% | 95.29 | 94.88 | 94.74 |
| VGG-16 | 94.56% | 95.12 | 94.56 | 94.41 |

### 10.3 Brain MRI — ensemble vs bases on the identical 800-image meta-test half **[DOC]**

| Model | Accuracy on meta-test |
|-------|----------------------|
| EfficientNet-B0 | 95.12% |
| ResNet-50 | 95.12% |
| DenseNet-121 | 95.88% |
| VGG-16 | 94.75% |
| **Stacking Ensemble** | **96.75%** [CONFIG] |

**Ensemble per-class on the meta-test half [DOC]:**

| Class | Precision | Recall | F1 | Δ recall vs EfficientNet |
|-------|-----------|--------|-----|--------------------------|
| Glioma | 0.98 | **0.90** | 0.93 | **+0.09** |
| Meningioma | 0.95 | 0.98 | 0.97 | −0.01 |
| No Tumor | 0.95 | 1.00 | 0.97 | 0.00 |
| Pituitary | 0.99 | 0.99 | 0.99 | −0.01 |

Overall: Accuracy 96.75% · Precision 96.81% · Recall 96.75% · F1 96.71%.

**The single most important result in the project:** stacking lifted **glioma recall from 0.81 to
0.90**. The ensemble did not merely add 1.7 points of accuracy — it specifically repaired the
clinically dangerous failure mode, by learning to trust whichever base model distinguishes glioma
from meningioma best. That is a substantive argument for the architecture, not a marginal gain.

### 10.4 Chest X-ray — what is and is not available

| Metric | Chest (TB, deployed) | Chest V1 (Kermany, archived) |
|--------|---------------------|------------------------------|
| Ensemble accuracy | **82.00%** [CONFIG] | **85.68%** [CONFIG] |
| Per-model accuracy | **[NOT RECORDED]** | **[NOT RECORDED]** |
| Per-class precision/recall/F1 | **[NOT RECORDED]** | **[NOT RECORDED]** |
| Confusion matrix | **[NOT RECORDED]** | **[NOT RECORDED]** |
| Meta-learner reliance | ResNet-50 41.0% [PICKLE] | EfficientNet-B0 42.8% [PICKLE] |

The notebooks compute all of these (`classification_report`, `confusion_matrix`,
per-model loop, ROC/AUC) but were saved with outputs cleared, and the `*_test_probs.npy`
artefacts live on Google Drive / SageMaker rather than in the repo.

**Recovery procedure — one notebook run, no retraining needed:** re-execute
`Chest_Xray_Stacking_Ensemble.ipynb` against the saved `*_test_probs.npy` files. Steps 3–7 of
§9.3 are pure NumPy/sklearn over cached probabilities, so the full per-class table regenerates in
seconds. **Do this before the final submission** — an examiner will ask for the chest confusion
matrix.

### 10.5 Live end-to-end verification on this machine **[MEASURED]**

Run against the deployed stack, all four brain samples:

| Sample | Predicted | Confidence | Correct |
|--------|-----------|-----------|---------|
| `glioma_sample.jpg` | Glioma | 99.08% | ✅ |
| `meningioma_sample.jpg` | Meningioma | 94.86% | ✅ |
| `notumor_sample.jpg` | No Tumor | 97.98% | ✅ |
| `pituitary_sample.jpg` | Pituitary | 99.35% | ✅ |

**Brain: 4/4.**

All six chest samples (all ground-truth tuberculosis):

| Sample | Predicted | Conf. | TB prob | Normal prob | Pneu prob | Correct |
|--------|-----------|-------|---------|-------------|-----------|---------|
| `tuberculosis-1117.jpg` | Tuberculosis | 97.68% | 97.68 | 2.26 | 0.05 | ✅ |
| `tuberculosis-1122.jpg` | **Normal** | **56.51%** | **42.99** | **56.51** | 0.50 | ❌ |
| `tuberculosis-1123.jpg` | Tuberculosis | 98.06% | 98.06 | 1.89 | 0.05 | ✅ |
| `tuberculosis-1125.jpg` | Tuberculosis | 98.06% | 98.06 | 1.89 | 0.05 | ✅ |
| `tuberculosis-1127.jpg` | Tuberculosis | 97.26% | 97.26 | 2.67 | 0.07 | ✅ |
| `tuberculosis-1129.jpg` | Tuberculosis | 98.05% | 98.05 | 1.90 | 0.05 | ✅ |

**Chest: 5/6 = 83.3%, against a reported ensemble accuracy of 82.00%.** The sample is far too
small for a real estimate, but the agreement is notable.

**Two findings worth presenting from this micro-evaluation:**

1. **Confidence separates correct from incorrect predictions cleanly.** The five correct calls sit
   at 97.26–98.06%. The single error sits at 56.51% — the only sub-90% prediction in the set, and
   a near-tie (56.51% Normal vs 42.99% TB). This is empirical support for a **confidence-threshold
   referral policy** (§13.1): abstaining below ~90% would have caught this error and referred it to
   a radiologist instead of reporting it.
2. **The failure direction is the dangerous one.** `tuberculosis-1122` is a **false negative** — an
   active TB case reported as Normal. And because the rule-based report engine keys risk level off
   the predicted class, the generated report declared `risk_level: "Low"` with
   `"No immediate intervention required"` for a tuberculosis patient. Verified actual output:

   ```json
   { "prediction": "Normal", "confidence": 56.51, "risk_level": "Low",
     "ai_summary": "... classified this scan as 'Normal' with 56.5% confidence.
                    A secondary consideration of Tuberculosis (43.0%) was noted
                    but deemed less likely.",
     "recommendation": "No immediate intervention required based on AI analysis. ..." }
   ```

   The secondary-consideration sentence did fire, which is the mitigating design feature — but
   "Low risk / no intervention" as the headline for a 43%-TB scan is exactly the behaviour a
   confidence gate must prevent. **This is the most important safety finding in the document.**

---

## 11. Measured Runtime Performance **[MEASURED]**

Measured on this machine, CPU-only, cold then warm:

| Operation | Models loaded | Cold | Warm (OS page cache) |
|-----------|--------------|------|----------------------|
| `POST /predict?module=brain_mri` | 4 (incl. 512 MB VGG-16) | **14,864 ms** | **9,851 ms** |
| `POST /predict?module=chest_xray` | 3 | **4,678 ms** | **3,793 ms** |
| `POST /gradcam?module=brain_mri` | 4 + 1 re-load | **13,683 ms** | — |
| `POST /api/predict` via Express (brain) | 4 | — | 6,531 ms (from gateway log) |
| FastAPI startup (ensemble + model load) | 1 eager | ~25–30 s | — |

### 11.1 Why latency is seconds, not milliseconds — and the honest verdict

Three compounding causes:

1. **Per-request disk loading.** Nothing is cached across requests. Every call re-instantiates the
   timm graph and re-reads the `.pth` from disk. Brain reads ~645 MB per request.
2. **CPU-only execution.** Every `torch.load` uses `map_location="cpu"`; there is **no
   `torch.cuda.is_available()` check and no `.to(device)` call anywhere in the application.** The
   service cannot use a GPU even when one is present.
3. **Sequential evaluation.** Models run one at a time by design, to cap memory (§4.3).

`/gradcam` is the worst case: it runs the entire ensemble to find the agreeing model, then loads
that model **a second time** to attach hooks.

**Target vs reality — state this plainly:**

| Target (from original plan) | Measured | Verdict |
|------------------------------|----------|---------|
| Single-model inference < 2 s | ~1.2 s [DOC] | ✅ met |
| Ensemble inference < 4 s | **9.9–14.9 s brain**, 3.8–4.7 s chest | ❌ **brain misses by 2.5–3.7×**; chest meets it |
| Grad-CAM < 1 s | 13.7 s end-to-end | ❌ missed |
| Total API response < 5 s | 6.5 s via gateway | ❌ missed |

The brain ensemble **does not meet its latency target**, and the cause is architectural (load-per-request
+ CPU-only + VGG-16's 512 MB), not incidental. §13.2 gives the fix and the expected gain.

### 11.2 After the fix — measured again **[MEASURED]**

The process-wide model cache from §13.2 was implemented. Same machine, same weights, same
predictions:

| Operation | Before | After | Factor |
|-----------|--------|-------|--------|
| `POST /predict?module=brain_mri` | 9,851–14,864 ms | **650 ms** | **15–21×** |
| `POST /gradcam?module=brain_mri` | 13,683 ms | **3,196 ms** | **4.3×** |
| `POST /predict?module=chest_xray` | 3,793–4,678 ms | **490 ms** | **7.7–9.5×** |
| `POST /report/pdf` (4-page PDF, both image panels) | n/a | **2,654 ms** | — |
| Evidence retrieval, warm | n/a | **15.3 ms** | — |
| Evidence retrieval, cold | n/a | 18.7 s (one-off encoder load) | — |
| `POST /api/scan` via Express + Vite proxy (report ∥ Grad-CAM, persisted) | — | **3,492–3,996 ms** | — |

**Predictions are bit-identical before and after** — the four brain samples still return
99.08 / 94.86 / 97.98 / 99.35 %, and all six chest TB samples still return their original values
including the 56.51% misclassification. Speed came from not re-reading 645 MB of weights per
request, not from changing the maths.

**Revised verdict:**

| Target | Measured after fix | Verdict |
|--------|--------------------|---------|
| Single-model inference < 2 s | ~1.2 s [DOC] | ✅ met |
| Ensemble inference < 4 s | **0.65 s brain**, 0.49 s chest | ✅ **now met** |
| Grad-CAM < 1 s | 3.2 s | ❌ still missed (runs the full ensemble to pick the agreeing model first) |
| Total API response < 5 s | 3.5–4.0 s end-to-end including persistence | ✅ **now met** |

The one honest caveat: the **first** request after a restart still costs ~15 s while the four models
load. That is a warm-up cost, not per-request cost, and it is visible in `/health`.

---

## 12. When the System Fails — Complete Failure Register

### 12.1 Hard failures (service will not start or request rejected)

| # | Condition | Behaviour | Severity |
|---|-----------|-----------|----------|
| 1 | Brain EfficientNet weights missing **or** wrong working directory | `sys.exit(1)` in the lifespan handler. **The chest module is taken down with it even though its weights are fine.** | Critical |
| 2 | Process started from any CWD other than `backend/fastapi` | All paths are **relative** (`"../../models"`, `"../../chest"`) and resolve against the process CWD. Every weight path breaks. | Critical |
| 3 | Upload > 10 MB | Express returns **413** | Correct |
| 4 | `Content-Type` not JPEG/PNG | Express **400**, FastAPI **422** | Correct |
| 5 | Zero-byte upload | 400 / 422 | Correct |
| 6 | Unknown `module` value | **400** | Correct |
| 7 | Module's EfficientNet missing | **503** | Correct |
| 8 | FastAPI unreachable from Express | **502** "ML service is unavailable" | Correct |
| 9 | Inference exceeds 30 s gateway timeout | **502** | Plausible under load, given §11 |
| 10 | MongoDB write fails | **500** — *prediction is computed then discarded* | Medium |

### 12.2 Silent degradations — the dangerous category

These produce a **normal-looking 200 response with no indication anything went wrong**.

> **Status after §13:** #11, #12, #14, #15, #17 are **fixed** — the API now returns
> `ensemble_active`, `models_skipped`, `explanation_faithful`, and an uncertainty band, and the
> report engine no longer substitutes a reassuring "No Tumor" narrative for an unknown label. The
> pattern that fixed them was the same each time: **make the degradation visible rather than
> guessing at it.** #13, #16 and #18 remain open.

| # | Condition | What happens | Why it is dangerous |
|---|-----------|--------------|---------------------|
| 11 | A base model fails to load mid-inference | Caught (`OSError`/`MemoryError`/`RuntimeError`), a **zero vector** is substituted to preserve the 16-dim shape | The meta-learner receives input from a region it never saw in training. Note `FileNotFoundError` **is** an `OSError`, so missing weights are silently zero-filled. Output looks confident. **No flag in the API response.** |
| 12 | Chest ensemble config/pickle unreadable | `except (FileNotFoundError, Exception): return None` → **silently drops to single EfficientNet** | The response is indistinguishable from a true ensemble prediction. A catch-all that also swallows sklearn version errors. |
| 13 | `is_available()` returns true on an incomplete install | Both modules probe **only EfficientNet** | Reports "available" while ResNet/DenseNet/VGG or the meta-pickle are missing; failure is deferred to #11/#12 |
| 14 | No base model agrees with the meta-learner | `agreeing_model` degrades to "first model that ran" | **Grad-CAM explains a different class than the one reported.** Guaranteed on the library path (`targets=None`) |
| 15 | Unknown class label reaches the report engine | `TUMOR_INFO.get(prediction, TUMOR_INFO["No Tumor"])` | **Renders a reassuring "No Tumor" narrative for an unrecognised label.** Any renaming of classes silently triggers this |
| 16 | `model_order` permuted in config | Feature count still matches, so sklearn does not complain | Confidently wrong predictions with no error |
| 17 | Low-confidence prediction | Reported as fact, with risk level keyed off the predicted class only | **Demonstrated live in §10.5:** a 56.51% "Normal" on a TB scan produced `risk_level: Low`, `"No immediate intervention required"` |
| 18 | sklearn version mismatch | `InconsistentVersionWarning`: pickles written with **1.6.1**, runtime has **1.5.1** | Logged on every startup; unpickling across versions is explicitly unsupported by sklearn and can alter results |

### 12.3 Clinical and dataset limitations

> **Status after §13:** #19 is **partly addressed** — a two-layer screen (quality gate + novelty
> detector, AUROC 0.9538) now exists for brain MRI, and conformal prediction supplies the "unknown"
> answer as a multi-class set rather than a forced top-1. It is not eliminated: see §13.14 for the
> measured false-positive and false-negative rates. #20–#26 remain open and are honest scope limits.

| # | Limitation | Consequence |
|---|-----------|-------------|
| 19 | **Closed-world assumption.** Softmax always sums to 1 over the module's classes | A knee X-ray, a CT slice, or a corrupted scan still yields a confident tumour class. There was **no out-of-distribution detector and no "unknown" class** — now partly addressed, §13.14 |
| 20 | **No modality auto-detection.** `AIRouter.modality_detector` is `None` | A chest X-ray sent to `brain_mri` is classified as a brain tumour. Routing depends entirely on the caller passing the right `module` |
| 21 | Single 2D slice only | Real radiology reads full 3D volumes and multiple sequences (T1/T2/FLAIR/contrast) |
| 22 | No DICOM support — JPEG/PNG only | Clinical PACS emits DICOM with 12–16-bit depth and windowing metadata. 8-bit JPEG discards dynamic range |
| 23 | No patient context | Age, sex, symptoms, and history are all diagnostically material and unused |
| 24 | Kaggle-sourced, single-source corpora | No multi-centre or multi-scanner validation; scanner-specific artefacts may be learned as shortcuts |
| 25 | Chest V1 labels derived from **filenames** (`'bacteria'`/`'virus'` substrings) | Not microbiologically confirmed ground truth |
| 26 | Brain classifier has no tumour **grading**, segmentation, or volumetry | Reports type, not stage or extent |

### 12.4 Integration defects found during this audit

> **Status after §13:** #27 (chest unreachable), #34, #35 (`/report` had no response model), #36
> (`agreeing_model` / `base_predictions` discarded) and #40 (`run.py` crash detection) are **fixed**.
> #28–#33, #37, #38, #39 remain open; #38 and #39 are credential leaks and need **your** action —
> see §15.

| # | Defect | Evidence | Impact |
|---|--------|----------|--------|
| 27 | **The chest module is unreachable through the Express gateway.** `fastapiClient.js` never sends a `module` parameter on any of its three calls, so FastAPI always applies its `brain_mri` default | The only occurrences of `module` in `backend/express/src/**/*.js` are `module.exports` | **The entire chest X-ray capability is inaccessible from the web UI.** It works only by calling FastAPI directly — verified: `POST /predict?module=chest_xray` returns correct TB predictions |
| 28 | **No authentication on any endpoint**, and `allow_origins=["*"]` | `main.py` CORS block; no auth dependency on any route | Any origin can POST medical images to the service |
| 29 | No upload size or pixel-count limit at the FastAPI layer | Only PIL's default decompression-bomb guard | Direct FastAPI access bypasses the Express 10 MB cap |
| 30 | MIME check trusts the client-supplied `Content-Type` | Spoofable; a spoofed type falls through to PIL and surfaces as a generic **500**, not a 422 | Misleading error semantics |
| 31 | `app/engine/runner.py` is dead code that would **break chest if wired in** — `get_models_dir_for_module()` returns `settings.MODELS_DIR` for every module, but `models/` holds no `CHEST_XRAY_*` files | No importer outside its own `- Copy` twin | Latent trap |
| 32 | Two parallel discovery mechanisms: `init_registry()` is a **hardcoded list**, while `discover_modules()` **scans the filesystem** | `registry.py` | A new module added to one but not the other makes `/health`, `/modules`, and actual routing disagree |
| 33 | `GRADCAM_COLORMAP` setting is **inert** — JET is hardcoded | `_overlay_heatmap` | Config implies configurability that does not exist |
| 34 | `HealthResponse` schema omits `available_modules`, which the route actually returns via raw `JSONResponse` | `health.py` / `schemas.py` | OpenAPI docs understate the payload |
| 35 | `/report` has **no `response_model`** | `report.py` | Report shape is unvalidated |
| 36 | `PredictionResponse` **drops** `agreeing_model` and `base_predictions` that the brain ensemble computes | `schemas.py` | Per-model transparency is computed then thrown away |
| 37 | Pickle files are `pickle.load`ed | `ensemble.py`, `module.py` | Arbitrary-code-execution primitive if a weights directory is ever untrusted |
| 38 | **Hardcoded dataset API token inside a training notebook** | A literal token assigned to `os.environ` and written to a credentials file, instead of being read from the environment | **Credential leak — revoked and reissued.** Notebooks are now excluded from the repository. See §15 |
| 39 | Private key file present in the working tree | Repo root; **not** matched by `.gitignore` at the time | Key pair rotated; `*.pem` now ignored. See §15 |
| 40 | `run.py` spawns children with `shell=True`, so `poll()` watches the `cmd.exe` wrapper, not the server | Observed: monitor reported "running" while ports 8000/5000/3000 were dead | Crash detection is unreliable in both directions |

---

## 13. Recommendations, and What Was Actually Built

§13.1–§13.11 are the recommendations as originally written, each now annotated with its status.
**§13.12–§13.19 are new and report measured results** for the work that was carried out: the system
moved from *MRI → CNN → tumour class* to a calibrated, uncertainty-aware, evidence-grounded
decision-support pipeline.

### 13.0 Status of the original recommendations

| § | Recommendation | Status |
|---|----------------|--------|
| 13.1 | Confidence-gated abstention | ✅ **Superseded and improved** — replaced with conformal prediction, which carries a coverage guarantee instead of a hand-picked threshold (§13.13) |
| 13.2 | Eliminate per-request model loading | ✅ **Done** — 15–21× faster, §11.2 |
| 13.3 | Regenerate and commit the chest metrics | ⬜ **Outstanding** — still [NOT RECORDED] |
| 13.4 | Close the chest-module integration gap | ✅ **Done** — `module` threaded end to end, §13.19 |
| 13.5 | Correct stacking methodology (out-of-fold) | ⬜ **Outstanding by decision** — refitting invalidates the 96.75% headline; documented upgrade path, §13.12 |
| 13.6 | Uncertainty intervals + McNemar | ⬜ **Outstanding** — the statistical rigour gap, §13.19 |
| 13.7 | Probability calibration | ✅ **Done with measured ECE**, §13.12 |
| 13.8 | Out-of-distribution rejection | ✅ **Done, two layers, AUROC 0.9538**, §13.14 |
| 13.9 | Attack the glioma recall deficit | ⬜ **Outstanding** — needs retraining |
| 13.10 | Robustness / reproducibility hardening | ◐ **Partial** — degradation is now surfaced in the API; sklearn pin and auth still open |
| 13.11 | Clinical and regulatory maturation | ⬜ **Future work**, correctly out of scope |

### 13.1 Confidence-gated abstention — highest value, lowest cost

> ✅ **Superseded by §13.13.** A fixed $\tau$ is a heuristic with no guarantee. Conformal prediction
> delivers the same abstention behaviour with a distribution-free coverage guarantee, so the
> three-band policy below was implemented with a **calibrated quantile** rather than a chosen
> number. The original reasoning is kept because it is why the band exists at all.

The single most valuable safety change, and §10.5 supplies the empirical justification: the one
misclassification in the live chest run was also the only prediction below 90%.

Introduce a three-band decision policy instead of always asserting a class:

$$
\text{decision} =
\begin{cases}
\text{report class} & \max_k P_k \geq \tau_{\text{high}} \quad (\approx 0.90)\\
\text{report with explicit uncertainty flag} & \tau_{\text{low}} \leq \max_k P_k < \tau_{\text{high}}\\
\textbf{abstain} \rightarrow \text{refer to radiologist} & \max_k P_k < \tau_{\text{low}} \quad (\approx 0.60)
\end{cases}
$$

Select $\tau$ by maximising coverage subject to a clinical constraint on selective risk:

$$
\text{Risk}(\tau) = \frac{\sum_n \mathbb{1}[\max P(x_n) \geq \tau]\cdot\mathbb{1}[\hat{y}_n \neq y_n]}{\sum_n \mathbb{1}[\max P(x_n) \geq \tau]}
\quad\text{subject to}\quad \text{Risk}(\tau) \leq \text{Risk}_{\max}
$$

Plot the **risk–coverage curve** and report **AURC**. Crucially, couple the report engine's
`risk_level` to confidence so the §10.5 failure — `risk_level: "Low"` on a 43%-TB scan — becomes
impossible.

### 13.2 Eliminate per-request model loading — fixes the latency miss

> ✅ **Done.** Implemented as a process-wide cache in
> `backend/fastapi/app/services/model_cache.py`, with LRU eviction because the machine has 7.7 GB
> total RAM. Genuine device selection was added. Measured result: **§11.2** — 650 ms brain predict,
> predictions bit-identical. `/gradcam` now reuses the already-loaded agreeing model instead of
> loading it a second time.

Load every base model **once at startup** into a registry held on `app.state`, keyed by module and
architecture, instead of re-reading `.pth` files on each request.

- **Expected gain:** brain ensemble from ~9.9 s to well under 1 s, since only forward passes remain.
- **Memory cost:** ~645 MB resident for brain (~160 MB if VGG-16 is dropped), ~135 MB for chest.
- Add genuine device selection: `device = "cuda" if torch.cuda.is_available() else "cpu"` and
  `.to(device)` on both model and tensor. Currently **no GPU can ever be used.**
- Batch the base-model forward passes and wrap in `torch.inference_mode()`.
- For `/gradcam`, reuse the already-loaded agreeing model rather than loading it a second time.

This is the change that converts missed latency targets into met ones.

### 13.3 Regenerate and commit the chest metrics

Re-run `Chest_Xray_Stacking_Ensemble.ipynb` over the cached `*_test_probs.npy` files to restore the
per-model and per-class tables in §10.4, then **commit `ensemble_metrics.json` and
`model_comparison.csv` to the repo** so the numbers survive notebook-output clearing. Save
notebooks *with* outputs, or export an HTML report per run.

### 13.4 Close the chest-module integration gap

> ✅ **Done.** `module` is threaded frontend selector → Express route → `fastapiClient` query string
> → FastAPI, and the `Prediction` schema's brain-only label enum was widened so chest results can
> actually persist. Regression test, run through the Vite proxy on :3000:
> `POST /api/scan?module=chest_xray` returns `module: "chest_xray"` and all six TB samples save.

Thread a `module` parameter through the stack: frontend selector → Express route → `fastapiClient`
query string → FastAPI. Without it, defect #27 means the chest capability cannot be demonstrated
through the UI during the viva — only via direct API calls.

### 13.5 Correct stacking methodology

Replace test-set meta-features with **out-of-fold predictions**:

1. Partition the **training** set into $K = 5$ stratified folds.
2. For each fold $k$: train base models on the other $K-1$ folds, predict probabilities for fold $k$.
3. Concatenate the out-of-fold probability matrix → meta-features covering the whole training set.
4. Fit the meta-learner on that matrix.
5. Evaluate **once** on the fully held-out test set.

This removes the test-set dependency, uses all 1,600 test images for the final number instead of
800, and narrows the confidence interval. It also enables honest reporting of
$\text{mean} \pm \text{std}$ across folds.

### 13.6 Report uncertainty intervals, not bare point estimates

With $n = 800$, a 96.75% accuracy carries a 95% Wald interval of roughly $\pm 1.2$ points:

$$
\hat{p} \pm z_{0.975}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

Prefer the **Wilson score interval** at these proportions. Then test whether the ensemble's gain
over DenseNet-121 is statistically real, using **McNemar's test** on paired predictions:

$$
\chi^2 = \frac{(|b - c| - 1)^2}{b + c}
$$

where $b$ and $c$ count the discordant pairs. **This is the question an examiner is most likely to
ask about a 96.06% → 96.75% improvement.** Have the answer ready.

### 13.7 Probability calibration

> ✅ **Done, with a result that contradicted the expectation.** Measured ECE, fitted temperature
> scaling, reliability diagrams before and after — **§13.12**. The finding: the *base* models are
> badly over-confident ($T$ up to 3.35) but the logistic-regression meta-learner is already
> well-calibrated, so the achievable gain at the output is small. That is worth more than a large
> improvement would have been, because it explains *why*.

Stacking consumes probabilities, so their calibration directly determines meta-learner quality.
Measure **Expected Calibration Error** over $M$ bins:

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n}\,\bigl|\,\text{acc}(B_m) - \text{conf}(B_m)\,\bigr|
$$

If miscalibrated, apply **temperature scaling** — one parameter $T$ fitted on validation NLL:

$$
\hat{p}_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

Report reliability diagrams before and after. Label smoothing (already used in chest V2) helps;
temperature scaling completes it.

### 13.8 Out-of-distribution rejection

> ✅ **Done — and the "free signal" won.** All four options below were implemented and benchmarked
> on a three-tier OOD set. **Mean pairwise Jensen–Shannon divergence between base models — the
> signal that was already being computed and thrown away — beat energy score, max-softmax and
> entropy** on the hard tier. Measured results in **§13.14**.

Directly addresses limitation #19. Cheapest effective options:

- **Max-softmax baseline** with a threshold chosen on a held-out OOD set.
- **Energy score:** $E(x) = -T\log\sum_j e^{z_j/T}$ — outperforms max-softmax at negligible cost.
- **Mahalanobis distance** in penultimate feature space against class-conditional Gaussians.
- **Ensemble disagreement as a free OOD signal:** the system already computes all base-model
  probabilities. High predictive entropy or low pairwise agreement is a strong novelty indicator
  and requires **no new model** — just expose it. Given `base_predictions` is already computed and
  then discarded (defect #36), this is nearly free.

### 13.9 Attack the glioma recall deficit directly

Even post-ensemble, glioma recall (0.90) trails the other classes. Options in order of expected effect:

- **Focal loss** to concentrate learning on hard examples:
  $\mathcal{L}_{\text{focal}} = -\alpha_t(1-\hat{p}_t)^{\gamma}\log(\hat{p}_t)$, with $\gamma = 2$.
- **Class-specific augmentation** for glioma, and **explicit cost-sensitive weighting** that makes
  a missed glioma more expensive than a false meningioma — the clinically correct asymmetry.
- **Threshold adjustment** on the glioma posterior: trade precision (currently a perfect 1.00, so
  there is headroom) for recall.
- **Two-stage cascade:** tumour/no-tumour first, then subtype — lets the subtype model specialise
  on the glioma–meningioma boundary where all the error concentrates.

### 13.10 Robustness, reproducibility, and deployment hardening

- **Reproducibility:** pin seeds for `torch`, `numpy`, `random`; set
  `torch.backends.cudnn.deterministic = True`; **pin `scikit-learn==1.6.1`** to eliminate defect #18;
  record a manifest of weight-file SHA-256 hashes.
- **Replace relative paths with package-anchored absolute paths** (`Path(__file__).resolve().parents[n]`)
  to remove the CWD fragility (#2), and load config with explicit validation at startup.
- **Assert the ensemble contract at load time:** verify `meta.n_features_in_ == M × K` and that
  `model_order` matches a checksum recorded at training time — closing defects #11, #16.
- **Surface degradation in the API:** add `models_used`, `models_skipped`, and `ensemble_active`
  to the response so silent fallbacks (#11, #12) become visible to the caller.
- **Add authentication and tighten CORS** to a concrete origin allow-list (#28); enforce an upload
  size and pixel-count limit at the FastAPI layer (#29); validate image type by **magic bytes**
  rather than the client-supplied header (#30).
- **Prefer `safetensors`** over `pickle` for weights to remove the code-execution primitive (#37).
- Add **structured audit logging** with a request ID, and **model versioning** so every stored
  prediction records which weights produced it.
- **Test-time augmentation** (horizontal flip + small rotations, averaged) typically adds 0.5–1
  point for ~2× compute — worth it once §13.2 makes inference cheap.
- **Knowledge distillation:** distil the 4-model ensemble into a single student network to retain
  most of the accuracy at a fraction of the latency, the natural production endgame.

### 13.11 Clinical and regulatory maturation

- **Multi-centre external validation** on data from a different scanner and hospital — the only
  real test of generalisation and the standard reviewers demand.
- **DICOM ingestion** via `pydicom`, honouring window centre/width and preserving 12–16-bit depth.
- **Radiologist reader study**: AI-alone vs radiologist-alone vs radiologist-with-AI, reporting
  inter-rater agreement (Cohen's $\kappa$) — the accepted way to demonstrate clinical utility.
- **Prospective silent trial** before any deployment decision.
- **Segmentation and volumetry** (U-Net / nnU-Net) to move from classification to measurement.
- **3D volumetric and multi-sequence input** (T1/T2/FLAIR/contrast) instead of a single slice.
- **Foundation models:** evaluate MedSAM, RadImageNet, or BiomedCLIP pretraining, which transfer
  better to radiology than ImageNet.
- **Fairness audit:** stratify every metric by age, sex, and scanner/source to surface subgroup
  underperformance.
- **Drift monitoring** in production on input statistics and confidence distributions.

---

### 13.12 Calibration — measured, with an unexpected result **[MEASURED]**

**Artefact:** `models/calibration/calibration.json` · script `backend/fastapi/scripts/fit_calibration.py`
· plot `models/calibration/reliability.png`

**Split protocol.** The 1,600-image brain test set is split by the documented
`train_test_split(test_size=0.5, stratify=y, random_state=42)`. Half A (800) is the meta-learner's
training half. Half B (800) is split 40/60 into **320 calibration** and **480 final test** images.
Nothing is calibrated and evaluated on the same rows.

**Reproduction gate, passed first.** Before fitting anything, the base-model probability matrices
were regenerated from the committed `.pth` weights and the published numbers re-derived:

| Model | Published [DOC] | Reproduced [MEASURED] | Δ |
|-------|-----------------|-----------------------|---|
| EfficientNet-B0 | 95.06% | 95.06% | 0.00 |
| ResNet-50 | 94.88% | 94.88% | 0.00 |
| DenseNet-121 | 96.06% | 96.06% | 0.00 |
| VGG-16 | 94.56% | 94.56% | 0.00 |
| **Stacking ensemble** | **96.75%** | **96.75%** | **+0.00 pp** |

This matters twice: it independently validates the headline number, and **an exact match proves
there is no train/serve preprocessing skew** — the serving preprocessor reproduces training
behaviour bit-for-bit. (Contrast §5.4, which flagged skew as a risk.)

**Two designs were fitted, and the better one was not deployed.**

| | Design A — scale base logits, refit meta | Design B — scale meta output (**deployed**) |
|---|---|---|
| Accuracy | 97.50% | **97.71%** |
| ECE (15 equal-width bins) | **0.020344** | 0.025949 |
| Adaptive ECE (equal-mass) | **0.011783** | 0.013298 |
| Brier | 0.032710 | 0.032739 |
| NLL | 0.069706 | 0.071985 |
| AURC | **0.001459** | 0.001845 |
| Selective risk @ 80% coverage | **0.2604%** | 0.5208% |

Uncalibrated baseline: accuracy 97.7083%, ECE **0.026201**, adaptive ECE **0.013439**, mean
confidence 96.5451%.

**Why B ships despite A being better on five of seven metrics.** Design A scales the *base* models,
which changes the meta-learner's input distribution and therefore requires refitting and
re-exporting `meta_model.pkl`. That would invalidate the 96.75% headline and every baseline in §10.
Design B scales the shipped meta-learner's own output — one parameter, $T = 0.996425$ — and is
deployable as-is. A is reported as a documented upgrade path, not hidden.

**The finding worth presenting.** The per-model temperatures fitted in Design A are the interesting
number:

| Base model | Fitted $T$ | Reading |
|------------|-----------|---------|
| EfficientNet-B0 | **3.3542** | severely over-confident |
| VGG-16 | **3.0329** | severely over-confident |
| DenseNet-121 | 1.9523 | over-confident |
| ResNet-50 | 1.7743 | over-confident |
| Meta-learner (Design B) | **0.996425** | already essentially calibrated |

$T > 1$ means softening is needed. The base CNNs need it badly; the logistic-regression meta-learner
needs almost none. **The stacking layer is doing calibration work as a side effect** — that is a
substantive claim about why stacking helps here, beyond raw accuracy, and it is measured rather
than asserted.

**A reporting honesty point.** Conventional 15-**equal-width**-bin ECE inflates to 0.0262 because
the top bin holds 454 of 480 samples while several bins hold 1–4 samples each, so tiny bins
dominate the average. Equal-mass (**adaptive**) ECE, which gives every bin the same weight of
evidence, gives 0.0134. Both are reported; quoting only the flattering one would be misleading, and
quoting only the conventional one would overstate the miscalibration.

### 13.13 Conformal prediction — a referral mechanism with a guarantee **[MEASURED]**

**Artefact:** `models/calibration/conformal.json` · **split conformal** with the LAC score
$s = 1 - \hat{p}_{y_{\text{true}}}$.

Fit the empirical quantile on the 320 calibration images:

$$
\hat{q} = \text{Quantile}\left(\{s_i\}_{i=1}^{n};\ \frac{\lceil (n+1)(1-\alpha)\rceil}{n}\right),
\qquad
C(x) = \{\,k : 1 - \hat{p}_k \leq \hat{q}\,\}
$$

The guarantee is distribution-free and finite-sample: $\mathbb{P}(y \in C(x)) \geq 1-\alpha$ under
exchangeability. **No threshold is chosen by hand.**

**Deployed operating point:** $\alpha = 0.01$, $\hat{q} = 0.981281$ (probability threshold
$0.018719$).

| Metric | Value |
|--------|-------|
| Empirical coverage on 480 held-out images | **1.0000** (target 0.99) |
| Mean set size | **1.3313** |
| Set size 1 / 2 / 3 / 4 | 334 / 137 / 5 / 4 |
| Coverage by class | 1.0000 for all four |

**Band policy:** size 1 → `confident`; size 2 → `borderline`; size ≥ 3 → `indeterminate`.
Distribution: **334 confident · 137 borderline · 9 indeterminate.**

**Why $\alpha = 0.01$ and not the textbook 0.10.** Conformal sets are all singletons whenever
$1-\alpha <$ model accuracy — coverage then merely equals accuracy and the layer carries **zero
information**. With 97.71% accuracy, $\alpha$ must be below **0.0229** to say anything. The full
sweep, all measured:

| $\alpha$ | $\hat{q}$ | Empirical coverage | Mean set size | Status |
|---|---|---|---|---|
| 0.20 | 0.0511 | 0.9771 | 1.000 | vacuous — all singletons |
| 0.10 | 0.0615 | 0.9771 | 1.000 | vacuous — all singletons |
| 0.05 | 0.5188 | 0.9771 | 1.000 | vacuous — all singletons |
| 0.025 | 0.9205 | 0.9938 | 1.058 | barely informative |
| 0.02 | 0.9328 | 0.9938 | 1.067 | barely informative |
| **0.01** | **0.9813** | **1.0000** | **1.331** | **deployed** |
| 0.005 | 1.0000 | 1.0000 | 4.000 | degenerate — admits every class |

There is also a **finite-sample floor**: with $n_{\text{cal}} = 320$, the quantile level
$\lceil (n+1)(1-\alpha)\rceil / n$ reaches 1.0 for $\alpha < 0.003115$, forcing $\hat{q} = 1$. So
the usable window is $0.0031 < \alpha < 0.0229$ — narrow, and narrow *because the model is good*.
Both bounds are recorded in the artefact.

**Risk is now coupled to uncertainty.** A `borderline` or `indeterminate` band escalates
`risk_level` to **`Indeterminate`**, overriding the class's default tier. This directly closes
failure #17. Verified: `meningioma_sample.jpg` at 94.86% previously reported `risk: Medium,
escalated: false` despite a two-class set; it now reports **`risk: Indeterminate,
risk_escalated: true`** with set $\{$Glioma, Meningioma$\}$.

### 13.14 Two-layer input validation and OOD detection **[MEASURED]**

**Artefact:** `models/calibration/ood.json` · scripts `build_ood_set.py`, `fit_ood.py`

**Three-tier evaluation set**, because a single "OOD" bucket would let synthetic noise flatter the
detector:

| Tier | n | Content | Purpose |
|------|---|---------|---------|
| `near_ood` | 6 | Real chest X-rays | **Hardest and most realistic — the headline tier** |
| `corrupted` | 40 | Degraded real MRIs (blur, noise, crops) | Tests the quality gate, not novelty |
| `far_ood` | 60 | Synthetic patterns | Sanity floor only; inflates AUROC |

In-distribution reference: all **480** final-test images, never used for calibration or conformal.

**Detector comparison on `near_ood`** — thresholds fitted at 95% target TPR:

| Detector | AUROC | FPR @ 95% TPR | Threshold |
|----------|-------|---------------|-----------|
| **Mean pairwise JS divergence** | **0.9538** | **0.0729** | **0.026446** (deployed) |
| Energy score | 0.9420 | 0.0938 | −4.8236 |
| Predictive entropy | 0.8778 | 0.2833 | 0.1783 |
| Max-softmax (MSP) | 0.8753 | 0.2833 | 0.0399 |
| Ensemble disagreement (argmax votes) | 0.6403 | 1.0000 | 0.0 |

$$
\text{JS}_{\text{mean}}(x) = \frac{2}{M(M-1)}\sum_{i<j} \text{JSD}\!\left(P_i(x)\,\|\,P_j(x)\right)
$$

**The selection is the point.** On `far_ood` alone, MSP and entropy look best (AUROC 0.978). Had the
detector been chosen on that tier, the deployed system would use the detector that is **worst on
real chest X-rays** (0.875 vs 0.954). The tier a detector is selected on is a design decision with
consequences, and it was made explicitly (`selection_tier: near_ood` is recorded in the artefact).
Note also that *graded* disagreement (JS divergence) beats *binary* disagreement (argmax votes,
AUROC 0.64) by 0.31 AUROC — the information is in the probability geometry, not the votes.

**Layer 1 — pre-inference quality gate.** Thresholds fitted at the 1st percentile of the 480 ID
images: minimum dimension 64 px, maximum aspect ratio 3.0, Laplacian variance ≥ 22.2668 (blur),
pixel std ≥ 30.2187 (contrast).

| Tier | Rejected by the gate | Rate |
|------|---------------------|------|
| In-distribution (480) | 10 | **2.08%** (the cost of the gate) |
| `corrupted` (40) | 25 | 62.5% |
| `far_ood` (60) | 23 | 38.3% |
| `near_ood` (6) | **0** | **0%** |

**The two layers are complementary, and the numbers prove it.** The quality gate catches **none** of
the real chest X-rays — they are perfectly good images, just of the wrong organ. The novelty
detector catches them (AUROC 0.954) but cannot see blur. Neither layer alone is sufficient; that is
the argument for having both, and it is measured rather than assumed.

**Conformal set size also behaves as a novelty signal**, for free:

| Tier | Mean conformal set size |
|------|------------------------|
| In-distribution | **1.3313** |
| `near_ood` | 2.1667 |
| `far_ood` | 2.4333 |
| `corrupted` | 2.4500 |

**Honest limitation, stated in the UI.** At the deployed threshold the detector also fires on
**7.29%** of valid in-distribution MRIs. `frontend/samples/glioma_sample.jpg` is one: flagged at
`js_divergence = 0.2606` while being classified correctly at 99.08%. The user-facing message
therefore quotes the score, the threshold **and the false-positive rate** — read from `ood.json`, so
the quoted figure cannot drift from the measured one — and frames the flag as a prompt to check the
input, not as proof the input is invalid. The gate is also **report-only by default**
(`ENFORCE_QUALITY_GATE = False`): enforcement would block ~2% of valid scans, so blocking is opt-in
while assessment is always reported.

### 13.15 Cited clinical knowledge base **[MEASURED]**

**Artefacts:** `knowledge/sources.json` (**12 sources**), `knowledge/brain_mri/*.md` (4 files,
**28 chunks**)

| Source type | Count |
|-------------|-------|
| Government clinical summaries (NCI, NINDS, NCBI Bookshelf) | 7 |
| Peer-reviewed reference articles / consensus reviews (StatPearls, *Neuro-Oncology* 2024) | 3 |
| Classification standard (WHO CNS5, 2021) | 1 |
| `neurasight-system` self-reference | 1 |

**Every chunk carries at least one citation id**, enforced by a build-time check. Four chunks
initially failed it — they described *this system's* behaviour (e.g. "the model cannot grade a
tumour", "the heatmap is not a boundary"), which no medical source can support. Attaching a medical
citation to a software statement would have been a false citation, so a `neurasight-system`
self-reference source was added instead. The distinction between "clinically cited" and
"self-reported about our own system" is preserved rather than blurred.

**Verification status is recorded, including what could not be done.** All 12 URLs were confirmed to
exist and be topically correct. Full page text was **not** programmatically extracted — cancer.gov
and ninds.nih.gov block automated fetching, and NCBI Bookshelf yielded no text — so the entries
state well-established clinical fundamentals and point to the sources for verification. They are
**not verbatim extracts**, and `clinical_review_required: true` is set in the artefact.

**A report-engine bug fixed here.** `TUMOR_INFO.get(prediction, TUMOR_INFO["No Tumor"])` (failure
#15) rendered a reassuring "No Tumor" narrative for any unrecognised label. The fallback is now an
explicit unknown-label error path, not the most reassuring class in the dictionary.

### 13.16 Local evidence retrieval — RAG without the stack **[MEASURED]**

**Artefacts:** `knowledge/index.npz` (**40,091 bytes**), `knowledge/index_meta.json`,
`knowledge/retrieval_eval.json` (30 hand-labelled queries), `knowledge/retrieval_results.json`

**Encoder:** `sentence-transformers/all-MiniLM-L6-v2`, 384-dim. The entire corpus is a
$(28, 384)$ float array. **No vector database, no LangChain** — at 28 chunks / 40 KB, exact
brute-force cosine similarity over a NumPy array is both faster and simpler than any index, and
introducing Pinecone or Chroma here would be unjustifiable complexity. Retrieval is warm in
**15.3 ms**; the 18.7 s cold cost is the one-off encoder load.

**Evaluation against a lexical baseline**, 30 hand-labelled queries, $k = 5$, **19 of 30
deliberately paraphrased away from corpus wording**:

| Metric | TF-IDF | **Embedding** | Δ |
|--------|--------|---------------|---|
| Hit@1 | 0.5000 | **0.5333** | +0.033 |
| Hit@3 | 0.8000 | **0.9000** | +0.100 |
| Recall@5 | 0.7472 | **0.8194** | +0.072 |
| MRR | 0.6389 | **0.7139** | +0.075 |
| Precision@5 | 0.2800 | 0.3067 | +0.027 |
| **Complete misses (0 relevant in top-5)** | **6 / 30** | **2 / 30** | **−4** |

Split by query type, which is where the mechanism shows:

| Subset | Metric | TF-IDF | Embedding |
|--------|--------|--------|-----------|
| Paraphrased (n = 19) | Hit@3 | 0.7368 | **0.8947** |
| Paraphrased (n = 19) | Recall@5 | 0.6535 | **0.7939** |
| Literal (n = 11) | Hit@1 | 0.5455 | **0.7273** |

The gain concentrates on paraphrased queries — exactly the behaviour semantic embeddings are
supposed to provide, confirmed rather than assumed. **Complete misses is the metric that matters
clinically**: a retriever that returns nothing relevant leaves a narrative ungrounded. Embedding
cuts those by two-thirds.

`precision@5` is reported for completeness only and is **capped below 1.0 by construction** —
relevant sets contain 1–4 chunks while $k = 5$. Reporting it as a headline would be misleading, so
it is labelled as such in the artefact.

### 13.17 LLM narratives — offline authoring under a review gate **[MEASURED]**

**Artefacts:** `knowledge/narratives/*.json` (12 files), `knowledge/narratives/authoring_summary.json`
· `backend/fastapi/scripts/author_narratives.py`

**The architectural decision.** The output space is finite and small: **4 classes × 3 uncertainty
bands = 12 narratives**. So the LLM is used **once, offline, as an authoring aid** — not called at
request time. This eliminates per-request latency, rate limits, cost, and the possibility of
serving unvetted clinical text. Runtime behaviour stays deterministic.

**Provider:** Groq, `openai/gpt-oss-120b`. Retrieval-augmented with $k = 5$ chunks per prompt.

| Metric | Value |
|--------|-------|
| Narratives written | **12 / 12** |
| Grounded (every claim traceable to a retrieved chunk) | **12 / 12** |
| Grounded **on the first attempt** | **12 / 12** (0 retries needed of 3 allowed) |
| Fabricated citations | **0** |
| Invented percentages / statistics | **0** |
| Non-ASCII characters after normalisation | **0** |
| Total tokens consumed | **27,539** |
| Word count range | 131–207 |

**The review gate is real, not decorative.** Every file carries `reviewed: false`, and the serving
path **refuses to serve an unreviewed narrative** — it falls back to the deterministic template and
reports `narrative_source: "template"`. Verified live: the running system currently serves
`template`, never LLM prose. A qualified reviewer must set `reviewed: true` per file before any
generated text reaches a user. An LLM that writes clinical text without human sign-off is not a
feature.

**Four bugs the grounding check caught**, each of which would have been invisible without it:

1. `llama-3.3-70b-versatile` returns **404** on this account. Enumerating available models was the
   only way to find out; the seam existed because the client validates its model list.
2. gpt-oss models emit **reasoning tokens that consume the completion budget**, so responses
   truncated mid-sentence until `DEFAULT_MAX_TOKENS` was raised to 3,000.
3. The model sometimes emitted **full-width CJK brackets** 【id】 instead of `[id]`, which the
   citation regex missed — producing a *false* "ungrounded" verdict. Normalisation was added.
4. A non-breaking hyphen (U+2011) crashed both the cp1252 console and reportlab. ASCII
   normalisation was added upstream, with XML-escaping as a second net in the PDF writer.

### 13.18 Professional PDF report **[MEASURED]**

**Artefact:** `backend/fastapi/app/services/pdf_report.py` · samples in
`models/calibration/sample_reports/` · endpoint `POST /report/pdf`

Built on reportlab **platypus flowables** so content paginates rather than being placed at fixed
coordinates. **4 pages, 121–139 KB** with both the original scan and the Grad-CAM overlay embedded.

Sections in order: branded header with scan metadata · primary finding with calibrated confidence,
risk level, conformal prediction set and coverage guarantee · **caveats box** · probability table
with inline bars and an "in set" column · Grad-CAM beside the original · clinical interpretation ·
condition overview · limitations for this finding · further evaluation · clinical considerations ·
follow-up · **warning signs** · treatment information · recommended next step · evidence and
references · AI limitations disclaimer · per-page footer.

**The PDF deliberately carries more caveats than the JSON**, because it is the artefact most likely
to be read detached from the UI. The caveats box fires on any of: risk escalated,
`explanation_faithful` false, `models_skipped` non-empty, ensemble inactive, quality gate failed,
or OOD flagged.

**A silent failure caught in verification.** reportlab's `Image` flowable accepts a path or a
file-like object; passing an `ImageReader` raises `expected str, bytes or os.PathLike object`. The
exception was caught and logged, so the PDF **still built — at 9.8 KB with no images at all.** Fixed
by passing `BytesIO` directly. A "successful" build is not evidence of a correct artefact; the file
size was.

### 13.19 Integration, and the rigour that is still missing **[MEASURED]**

All of the above is reachable from the browser. Verified through the **Vite proxy on :3000** — the
path a real user takes — not by calling services directly:

| Check | Result |
|-------|--------|
| `glioma_sample.jpg` | Glioma **99.08%** — §10.5 baseline unchanged |
| Band / set / coverage | `confident` / $\{$Glioma$\}$ / 0.99, `calibrated: true` |
| `meningioma_sample.jpg` | Meningioma **94.86%**, `borderline`, $\{$Glioma, Meningioma$\}$ |
| Escalated case persists to MongoDB | `riskLevel: Indeterminate`, `riskEscalated: true`, `saved: true` |
| All 6 chest TB samples | 97.68 / **56.51** / 98.06 / 98.06 / 97.26 / 98.05 — unchanged, and the 56.51% case now escalates to `Indeterminate` |
| `GET /api/predictions?band=borderline` | 3 records, all 2-class sets, all escalated |
| `GET /api/predictions/stats` | `byBand` and `escalatedCount` populated |
| `POST /api/report/pdf` via proxy | **139,354 B**, `application/pdf`, `Cache-Control: no-store` |
| `pytest tests/` | **11 passed, 0 failed** |

The dashboard now leads with the **conformal band**, not a confidence number: a band pill, the
prediction set with its coverage guarantee, a single caveats box gathering every condition that
weakens the result, the clinical sections, the resolved citation list, and a provenance line stating
whether the wording came from a reviewed narrative or the template and whether the probabilities
were calibrated. History filters by band and by escalation.

**Two integration bugs worth naming**, both of the same species — *a thing that appeared to work*:

- `riskLevel` was a Mongoose enum of `Low | Medium | High`. Once risk escalation started emitting
  `Indeterminate`, **every escalated scan silently failed to save** — precisely the clinically
  important ones. It was invisible because the error handler was `req.log?.warn?.()` and no
  middleware attaches `req.log`, so the optional-chaining swallowed it. Fixed both: the enum, and
  the logging that hid it.
- `POST /api/report/pdf` worked **by accident**. It was mounted after `/api/report`, and Express
  `app.use` matches by prefix; the broader mount was consulted first and only fell through because
  its router happened to have no matching path. Adding any catch-all would have broken PDF
  downloads silently. Mounts are now ordered most-specific-first.

**What is still missing, stated plainly.** §13.6 is the real remaining gap, and an examiner will
find it:

- **No confidence intervals on any accuracy figure.** 96.75% on $n = 800$ carries a 95% Wilson
  interval of roughly ±1.2 points. The 96.06% → 96.75% ensemble gain **is inside it**.
- **No McNemar's test** on paired ensemble-vs-DenseNet predictions, which is the correct test for
  that comparison. The defensible claim remains the **glioma recall gain, 0.81 → 0.90**.
- **Conformal coverage is a point estimate.** 1.0000 on 480 images is consistent with the 0.99
  guarantee but is not itself a bound; the honest statement is "no coverage violation observed in
  480 trials".
- **OOD `near_ood` has n = 6.** AUROC 0.9538 on six positives is indicative, not established. This
  is stated wherever the number appears.
- **Chest metrics remain [NOT RECORDED]** (§13.3), and chest has **no calibration or conformal
  layer** — the artefacts were fitted on the brain ensemble's 16-dimensional features and 4 classes,
  so applying them to 3-class chest output would be meaningless. Chest reports
  `calibrated: false` and falls back to a confidence threshold. Saying "calibrated" about chest
  would be the easy lie; the API says otherwise.

---

## 14. Standards and Compliance

### 14.1 Reporting and methodology standards

| Standard | Scope | Applicability here |
|----------|-------|--------------------|
| **CLAIM** (Checklist for AI in Medical Imaging) | Reporting AI imaging studies | Primary checklist for the final report |
| **TRIPOD+AI** | Prediction-model reporting | Governs how §10 results should be presented |
| **STARD 2015** | Diagnostic accuracy studies | Requires sensitivity/specificity with CIs — see §13.6 |
| **CONSORT-AI / SPIRIT-AI** | AI clinical trials | Applies only if a prospective trial is run |
| **QUADAS-2** | Bias assessment in diagnostic studies | Self-assess: single-source data is a known bias risk |
| **PROBAST** | Risk-of-bias for prediction models | Flags the §9.3 test-set-derived meta-features |
| **CRISP-DM** | Data-mining process model | Matches the phase structure of the project |
| **FAIR** | Findable, Accessible, Interoperable, Reusable | Motivates §13.3 committing metrics |

### 14.2 Regulatory framing (position the project honestly)

| Framework | Relevance |
|-----------|-----------|
| **FDA SaMD** / IMDRF risk categorisation | A diagnostic aid of this class would be Class II, requiring 510(k) |
| **EU MDR 2017/745** | Diagnostic software is a medical device in the EU |
| **EU AI Act** | Medical AI is **high-risk** — mandates risk management, data governance, logging, human oversight |
| **IEC 62304** | Medical device software lifecycle |
| **ISO 14971** | Risk management for medical devices |
| **ISO/IEC 23894** | AI risk management guidance |
| **GDPR / HIPAA** | Medical images are protected health data; the current service has **no authentication** (#28) |
| **DICOM (ISO 12052)** | The clinical imaging interchange standard — not yet supported (#22) |
| **HL7 FHIR / ImagingStudy** | Target for EHR interoperability |

**The compliance statement to make on stage:** NeuraSight is a **research prototype and clinical
decision-support demonstrator**, not a certified diagnostic device. The system already emits a
non-removable disclaimer on every report:

> *"This AI-generated report is for clinical decision support only. It must not replace
> professional medical diagnosis. All findings should be validated by a qualified radiologist or
> specialist physician."*

### 14.3 Engineering standards observed

| Area | Standard applied |
|------|-----------------|
| Python style | PEP 8, PEP 257 docstrings, PEP 484 type hints |
| API design | REST over HTTP, OpenAPI 3.1 auto-generated by FastAPI |
| Data validation | Pydantic v2 models with field constraints (`ge=0, le=100`) |
| Config | `pydantic-settings`, 12-Factor environment configuration |
| Error handling | Generic client messages, full detail server-side only — **no stack traces leaked** |
| Image encoding | RFC 4648 base64, PNG (ISO/IEC 15948) |
| Transport | RFC 7578 `multipart/form-data` |
| Versioning | Git |
| Accessibility | WCAG 2.1 AA as the target for the UI |

> Full WCAG conformance cannot be asserted from code inspection alone — it requires manual testing
> with assistive technologies and expert accessibility review.

---

## 15. Security Actions Required Before Submission

A repository audit found **three** credentials that had been committed to a public history. The
published history has been replaced so the artefacts no longer appear in the repository, and the
patterns that let them in are now blocked.

🔴 **Rotation of all three is still outstanding and is the remaining action.** Replacing history
reduces further exposure; it does not undo the exposure that already happened.

| # | Credential class | How it got in | Status |
|---|-----------------|---------------|--------|
| 1 | Cloud / SSH private key | key file committed before `*.pem` was gitignored | Removed from the tree, `*.pem` now ignored — **key pair still to be rotated** |
| 2 | Dataset API token | hardcoded literal inside a training notebook | Notebooks excluded from the repository — **token still to be revoked and reissued**, and the literal replaced by an environment lookup |
| 3 | Database user password | placed in a committed `.env.example` instead of the ignored `.env` | `.env.example` now carries only placeholders — **password still to be rotated** and network access restricted |

**The lesson, which is the part worth presenting:** `.gitignore` prevents *future* commits, it does
not remove *past* ones. Rewriting history reduces further exposure but cannot undo it — anything
pushed to a public remote must be treated as disclosed from that moment, so **rotation is the only
real remediation** and it has to happen regardless of what the history looks like afterwards.

Controls now in place to stop a recurrence:

- All three `.env.example` files contain placeholders only and are the documented source of truth for
  which variables exist; real values live in gitignored `.env` files.
- `.gitignore` blocks `*.pem`, `*.key`, `id_rsa*`, `.env`, `kaggle.json`, `*token*.json` and
  credential-shaped filenames, and excludes the notebook directories (including a stray
  space-named duplicate that previously slipped past the pattern).
- A pre-publication scan over every committable file for private-key headers, provider key prefixes
  and connection strings. It currently returns no findings.

The Groq API key was never exposed — it lives only in the gitignored repo-root `.env`, and the
committed example carries an empty value.

**Also outstanding, non-credential:** the 12 files in `knowledge/narratives/` all carry
`reviewed: false`. The system correctly refuses to serve them and falls back to the deterministic
template (§13.17), so nothing unsafe is being shown — but a qualified reviewer must read each one
and set `reviewed: true` before any generated clinical text reaches a user.

---

## 16. Appendix A — Extra / Redundant Files Inventory

Audited across the working tree, excluding `node_modules` and `.git`.

### 16.1 `- Copy` duplicates — 52 files, safe to delete

Every one was hash-compared against its original. **38 are byte-identical; the other 14 differ
only in line endings** (the copies are LF, the originals CRLF — verified content-identical after
normalisation). **No copy contains unique work.**

| Location | Count | Files |
|----------|-------|-------|
| `backend/fastapi/app/services/` | 6 | `ensemble`, `gradcam`, `inference`, `preprocessor`, `report`, `__init__` |
| `backend/fastapi/app/routers/` | 5 | `predict`, `gradcam`, `health`, `report`, `__init__` |
| `backend/fastapi/app/engine/` | 5 | `base_module`, `registry`, `router`, `runner`, `__init__` |
| `backend/fastapi/app/models/` | 2 | `schemas`, `__init__` |
| `backend/fastapi/app/modules/` | 1 | `__init__` |
| `backend/express/src/routes/` | 6 | `predict`, `gradcam`, `health`, `predictions`, `report`, `.gitkeep` |
| `backend/express/src/middleware/` | 5 | `errorHandler`, `errorHandler.test`, `imageValidator`, `requestLogger`, `.gitkeep` |
| `backend/express/src/services/` | 3 | `fastapiClient`, `fastapiClient.test`, `.gitkeep` |
| `backend/express/src/models/` | 2 | `Prediction`, `.gitkeep` |
| `backend/express/src/config/` | 1 | `index` |
| `frontend/src/components/` | 16 | 8 components × (`.jsx` + `.module.css`) |
| `backend/fastapi/app/__pycache__/` | 3 | stale `.pyc` copies |

**Why these matter beyond tidiness:** they sit *inside* the `app/` package, so they are importable
and will silently drift from the originals. `engine/runner - Copy.py` is the only "importer" of the
dead `runner.py`, which is why that latent chest-breaking bug (#31) went unnoticed.

Removal (after committing current work):

```powershell
Get-ChildItem -Recurse -File -Filter "*- Copy*" |
  Where-Object { $_.FullName -notmatch 'node_modules|\.git\\' } |
  Remove-Item -Force
```

### 16.2 Large redundant artefacts

| File / directory | Size | Assessment |
|------------------|------|------------|
| `models.zip` | **598.3 MB** | Archive of the model weights. Gitignored (`*.zip`). The weights already exist unpacked in `models/`. **Largest single reclaimable item.** |
| `chest V1/` | **136.9 MB** | Archived Kermany chest ensemble (3 `.pth` + `.pkl` + config). Gitignored, referenced by **no code**. Keep only as the §2 provenance record for the two-dataset story; it is not needed to run the system. |
| `frontend/dist/` | 11 files | Vite build output, gitignored, regenerated by `npm run build` |
| `.pytest_cache/`, `backend/fastapi/.pytest_cache/` | 8 files | Test caches, gitignored |
| `backend/fastapi/app/__pycache__/` | 6 files | Bytecode cache, gitignored |

Note `models/BRAIN_MRI_VGG.pth` alone is **512.2 MB** — 79% of the brain model footprint, for the
weakest base learner. Section 13.2 discusses dropping it.

### 16.3 Duplicate content

| Item | Detail |
|------|--------|
| `frontend/samples/*.jpg` | All 4 brain samples are **byte-identical** to `data/samples/brainMRI/*.jpg`. Frontend copies exist to be web-servable; consolidate via a static route or build-time copy. |
| `docs/system_architecture_and_ml_pipeline.md` **and** `.txt` | Two formats of the same document, **not identical** in content — they have diverged. Pick one as canonical. |
| `docs/system_architecture.md` | A third architecture document (0.1 MB) overlapping the above two. |

### 16.4 Scratch, scaffolding, and sensitive files

| File | Assessment |
|------|------------|
| `notebooks/_gen_sage2.py` | A 1 KB scratch generator for the SageMaker notebook. Excluded along with the rest of the notebook directory. |
| `models/create_dummy_model.py` | Development scaffolding for generating placeholder weights; obsolete now that real weights exist. |
| Private key file at the repo root | **Rotated and removed — see §15.** `*.pem` is now gitignored. |
| `backend/express/.env`, `backend/fastapi/.env` | Gitignored. Verified not present in the published tree. |
| `notebooks copy/` | Stray duplicate of the notebook directory; the space in the name dodged the `notebooks/` ignore rule. Now matched explicitly. |

### 16.5 Notebook redundancy

| Notebook | Role | Note |
|----------|------|------|
| `00_Dataset_Setup_Chest_Xray.ipynb` | Downloads/organises the TB dataset | Keep |
| `Chest_Xray_Model_Training.ipynb` | 38 cells, TB taxonomy, V1 hyperparameters | **Superseded by V2** |
| `Chest_Xray_Model_Training_V2.ipynb` | 8 cells, the 5 fixes, deployed pipeline | Keep — **contains the leaked token (§15)** |
| `Chest_Xray_SageMaker_Training.ipynb` | Kermany Bacteria/Normal/Virus on SageMaker | Keep as provenance for `chest V1/` |
| `Chest_Xray_Stacking_Ensemble.ipynb` | Chest meta-learner | Keep — **re-run to restore §10.4 metrics** |
| `BRAIN_MRI_SCAN.ipynb` | Brain base-model training | Keep |
| `BRAIN_MRI_ENSEMBLE.ipynb` | Brain meta-learner | Keep |

**All seven have zero retained cell outputs** — the root cause of every `[NOT RECORDED]` in §10.4.

### 16.6 Summary

| Category | Reclaimable |
|----------|-------------|
| `models.zip` | 598.3 MB |
| `chest V1/` (if provenance kept in docs only) | 136.9 MB |
| 52 `- Copy` files | ~0.2 MB, but removes real drift/confusion risk |
| Caches and build output | small |
| **Total disk** | **~735 MB** |

---

## 17. Appendix B — Presentation Script (Technical Talk Track)

Timings assume a 20-minute technical slot. Each beat gives the claim to make and the evidence
backing it.

### Beat 1 — Framing (1 min)

> "NeuraSight is a two-module medical imaging platform. It classifies brain MRI into four
> tumour categories and chest X-rays into three disease categories — eight conditions in total
> across three trained taxonomies. Each module is an independent stacking ensemble of ImageNet-pretrained
> CNNs with a logistic-regression combiner, served behind a three-tier architecture, with
> Grad-CAM explainability on every prediction."

### Beat 2 — The datasets, and why there are two chest models (2 min)

State plainly that two separate chest programmes were run on two different datasets. Show the §2.1
table. Then make the argument in §2.2:

> "The archived Kermany model scores higher — 85.68% against 82.00%. We deployed the lower number
> deliberately. Those two figures are measured over different label spaces, so ranking them is a
> category error. Bacteria-versus-virus on a plain radiograph is a task where the labels come from
> filename conventions rather than microbiology, and where expert performance is near chance.
> Normal-versus-pneumonia-versus-tuberculosis answers a question a clinic actually asks."

### Beat 3 — Model selection (2 min)

Walk the §3.1 table. The point to land:

> "We did not pick four architectures by leaderboard rank. We picked them for decorrelated
> inductive bias, because stacking only beats its best member when members fail differently.
> VGG-16 is the weakest and 26 times larger than EfficientNet-B0 — we kept it because it is
> architecturally unlike the others."

Then pay it off with the derived evidence from §6.4:

> "That choice is measurable. Reading the fitted coefficients out of the pickle, the brain
> meta-learner distributes reliance almost uniformly — 23.5, 25.8, 25.1, 25.6 percent. VGG-16 draws
> the second-highest share. The ensemble genuinely uses all four."

### Beat 4 — Pipeline and formulas (4 min)

Preprocessing (§5.2), softmax (§6.1), the 16-dimensional concatenation with the explicit index
layout (§6.2), multinomial logistic regression (§6.3).

> "Each base model emits a probability simplex over K classes. We concatenate them in the order
> the config declares — four models by four classes gives sixteen features — and a multinomial
> logistic regression maps that to the final posterior. The confidence we report is the
> meta-learner's own posterior, not an average of the base models."

Flag the fragility honestly:

> "That column ordering is the most fragile contract in the system. It is convention, not
> assertion. Permute `model_order` in the config and the feature count still matches, so nothing
> errors — you just get confidently wrong answers."

### Beat 5 — Explainability (2 min)

Grad-CAM formulas (§7.1), then the agreement rule (§6.5):

> "We do not explain an arbitrary model. We find the base model whose own top-1 agrees with the
> ensemble and is most confident in that class, and we explain that one — so the heatmap is
> faithful to a model that actually believes the reported answer. When no model agrees, we fall
> back, and in that branch the heatmap can explain a different class than the one reported. That
> is a known limitation, not a hidden one."

### Beat 6 — Results (4 min)

Lead with the honest headline, then the real finding:

> "Brain: 95.06% for EfficientNet-B0, 96.06% for DenseNet-121 as the best single model, 96.75% for
> the stacking ensemble on a held-out meta-test half. But the accuracy is not the interesting part.
> Glioma recall was 0.81 — we were missing nineteen percent of the most aggressive tumour class,
> while meningioma recall sat at 0.99. The model was absorbing gliomas into meningioma, which is
> radiologically plausible since both can enhance similarly on T1. Stacking lifted glioma recall
> from 0.81 to 0.90. The ensemble did not just add 1.7 points of accuracy — it repaired the
> clinically dangerous failure mode."

Then the chest honesty:

> "For chest we have the ensemble accuracies — 82.00% deployed, 85.68% archived — but the
> per-class tables are not reproducible from this repository, because the notebooks were saved with
> outputs cleared. I am not going to quote numbers I cannot show you. The notebook regenerates them
> from cached probability arrays without retraining, and that is a pre-submission task."

Then §10.5:

> "I ran the deployed stack live. Brain: four for four, all above 94%. Chest: five of six on
> tuberculosis samples — 83.3% against a reported 82%. And the single failure is the instructive
> one. It was the only prediction below 90% — 56.5% Normal against 43.0% tuberculosis. An active TB
> case reported as normal, and because the report engine keys risk off the predicted class alone, it
> generated 'Low risk, no immediate intervention required.' That is the strongest argument in the
> project for confidence-gated abstention."

### Beat 7 — What was built in response (4 min) — §13.12–§13.19

This is the strongest part of the talk, because every claim has a number behind it.

> "That failure drove the rest of the work. Four things, each measured.
>
> **One, calibration.** I regenerated the probability matrices from the committed weights first, and
> the published 96.75% came back at plus zero point zero zero — which also proves there is no
> train/serve preprocessing skew. Then I measured Expected Calibration Error and fitted temperature
> scaling. The interesting result is not the ECE, it's the temperatures: the base CNNs need T up to
> three point three five, and the logistic-regression meta-learner needs zero point nine nine six.
> The base models are badly over-confident; the stacking layer is already calibrating them as a side
> effect. I also report adaptive, equal-mass ECE alongside the conventional number, because with 454
> of 480 samples in the top bin the equal-width version is dominated by bins holding three samples.
>
> **Two, conformal prediction** — instead of the confidence threshold I'd recommended. A threshold
> is a number I pick; conformal gives a distribution-free coverage guarantee. Alpha of one percent,
> quantile fitted on 320 held-out images, and on the 480 final-test images I measure one hundred
> percent coverage with a mean set size of one point three three. Three hundred thirty-four
> confident, one hundred thirty-seven borderline, nine indeterminate. And I want to flag why alpha
> is one percent and not the textbook ten: conformal sets are all singletons whenever one-minus-alpha
> is below model accuracy. At 97.7% accuracy, anything above alpha 0.023 is *vacuous* — it carries
> no information. The usable window is narrow precisely because the model is good, and both bounds
> are recorded.
>
> **Three, out-of-distribution detection.** Two layers: a quality gate on blur and contrast, and a
> novelty detector. I benchmarked four detectors on three tiers, and the winner was the signal the
> system was already computing and throwing away — mean pairwise Jensen-Shannon divergence between
> the base models. AUROC nought point nine five four on real chest X-rays sent to the brain module,
> beating energy score and max-softmax. And the tier selection mattered: on synthetic noise alone,
> max-softmax looks best, so choosing there would have shipped the detector that is *worst* on the
> realistic case. The two layers are complementary and I can show it — the quality gate catches
> zero of six chest X-rays, because they are perfectly good images of the wrong organ.
>
> **Four, evidence grounding.** Twelve cited clinical sources, twenty-eight chunks, local embedding
> retrieval that beats TF-IDF on every metric and cuts complete misses from six in thirty to two.
> The LLM is used *offline*, once, to author twelve narratives — four classes times three
> uncertainty bands — and every one is checked for citation grounding. Twelve of twelve grounded
> first attempt, zero fabricated citations. They all sit behind a review flag that is currently
> false, so the running system serves the deterministic template. An LLM writing clinical text
> without human sign-off is not a feature."

Then the demo: upload `meningioma_sample.jpg`, show the borderline band, the two-class set, the
escalation to Indeterminate, and download the PDF.

### Beat 8 — What is still missing (2 min)

Say this before you are asked. It is the difference between a defensible project and a defended one.

> "Three gaps I have not closed. First and most important: **no confidence intervals and no
> McNemar's test.** 96.75% on 800 images carries a Wilson interval of about plus or minus one point
> two, and the 96.06 to 96.75 ensemble gain sits inside it. The claim I will defend is the glioma
> recall gain, 0.81 to 0.90. Second, my near-OOD tier has six positives — AUROC 0.954 there is
> indicative, not established. Third, chest has no calibration or conformal layer at all, because
> the artefacts were fitted on the brain ensemble's sixteen features and four classes; applying them
> to three-class chest output would be meaningless, so the API reports `calibrated: false` rather
> than pretending. Stacking is also still fitted on test-set meta-features rather than out-of-fold,
> which I left deliberately: fixing it means refitting the meta-learner and invalidating the 96.75%
> headline, so it is documented as an upgrade path."

### Beat 9 — Close (1 min)

> "NeuraSight is a research prototype and decision-support demonstrator, not a certified device.
> Every report carries a non-removable disclaimer. The contribution is not a single accuracy
> number — it is a reproducible comparison across four architectures, a stacking combiner whose
> learned reliance we can read directly out of its coefficients, faithful per-prediction
> explainability, a calibrated uncertainty layer that abstains with a coverage guarantee rather than
> a guessed threshold, clinical text that is cited and human-gated rather than generated on demand,
> and an honest account of where all of it still fails."

### Anticipated questions

| Question | Answer |
|----------|--------|
| Is 96.06% → 96.75% statistically significant? | Not established, and I say so unprompted. With $n = 800$ the 95% interval is about ±1.2 points, so the gain is within it. The correct test is McNemar's on paired predictions (§13.6, still outstanding — §13.19). The defensible claim is the **glioma recall gain, 0.81 → 0.90**, which is large and clinically meaningful. |
| How do I know the 96.75% is real and not a stale number in a doc? | It was **reproduced from the committed `.pth` weights at +0.00 pp**, along with all four per-model accuracies to 2 dp (§13.12). The same check doubles as proof there is no train/serve preprocessing skew. |
| Why conformal prediction instead of a confidence threshold? | A threshold is a number I choose; conformal gives a **distribution-free finite-sample coverage guarantee** under exchangeability. Measured: 1.0000 coverage at a 0.99 target, mean set size 1.3313 (§13.13). |
| Why $\alpha = 0.01$ and not 0.10? | Because at 97.71% accuracy any $\alpha > 0.0229$ makes every set a singleton — coverage just equals accuracy and the layer is **vacuous**. There is also a finite-sample floor at $\alpha = 0.0031$ from $n_{\text{cal}} = 320$. The whole sweep is in `conformal.json` (§13.13). |
| Did calibration actually improve anything? | Marginally at the output (ECE 0.0262 → 0.0259), **and that is the result.** The per-model temperatures show why: bases need $T$ up to 3.35, the meta-learner needs 0.996. The stacking layer is already doing the calibration (§13.12). |
| Is your ECE flattered by the binning? | The opposite — conventional equal-width ECE is *inflated* here because 454 of 480 samples land in one bin. Both equal-width (0.0259) and equal-mass adaptive (0.0133) are reported (§13.12). |
| Does the LLM write the medical text users see? | **No.** Twelve narratives were authored offline and every one is behind a `reviewed: false` gate, so the system serves a deterministic template and reports `narrative_source: "template"`. Verified live (§13.17). |
| How do you know the narratives aren't hallucinated? | A build-time grounding check requires every claim to trace to a retrieved chunk and every citation id to resolve. 12/12 grounded on the first attempt, 0 fabricated citations, 0 invented percentages (§13.17). |
| Why no vector database for the RAG? | 28 chunks is a $(28, 384)$ array — 40 KB. Exact brute-force cosine is faster and simpler than any index; a vector DB here would be unjustifiable complexity. Retrieval is 15.3 ms warm (§13.16). |
| Why Logistic Regression and not a neural meta-learner? | 68 parameters versus thousands on a small meta-set; convex so reproducible; and the coefficients are directly interpretable — §6.4 is only possible because the combiner is linear. |
| Why is the deployed chest model the less accurate one? | Different label spaces, so the numbers were never comparable. See §2.2. |
| Why is VGG-16 there if it's the weakest? | Diversity. The meta-learner assigns it a 25.6% reliance share — second highest of the four. |
| What happens on a non-medical image? | It is now **screened**, not silently classified. A quality gate plus a JS-divergence novelty detector (AUROC 0.9538 on real chest X-rays sent to the brain module) flags it, and conformal prediction returns a multi-class set instead of a forced top-1. Not eliminated: the detector also fires on 7.29% of valid MRIs, and that rate is stated in the UI (§13.14). |
| Isn't ensemble disagreement a weak OOD signal? | *Binary* disagreement is — argmax-vote disagreement scores AUROC 0.6403. **Graded** disagreement, mean pairwise JS divergence over the probability distributions, scores 0.9538. The information is in the probability geometry, not the votes (§13.14). |
| Is the ensemble accuracy leakage-free? | The meta-learner never sees the rows it is scored on — stratified 50/50 split, and calibration/conformal use a further disjoint 320/480 split of half B. But meta-features still come from the test set rather than out-of-fold training predictions, a deviation from canonical stacking. §13.5 is the fix, deliberately deferred because it invalidates the headline (§13.19). |
| Can it run on GPU? | **Yes now.** Device selection was added with the model cache (§13.2). This machine is CPU-only, so every number quoted is a CPU number — a GPU would improve them. |
| Why is it still ~3 s per scan if predict is 650 ms? | The scan endpoint also runs Grad-CAM, which re-runs the ensemble to find the agreeing base model (3.2 s), plus persistence. Report and heatmap run in parallel, so wall-clock matches the slower one, not the sum (§11.2). |
| Are the two chest taxonomies interchangeable? | No. Different datasets, different classes, different meta-learners, and different reliance profiles (ResNet 41% vs EfficientNet 42.8%). |

---

## 18. Summary

| Aspect | Detail |
|--------|--------|
| **Modules** | 2 live (`brain_mri`, `chest_xray`) + 1 archived taxonomy (`chest V1`) |
| **Conditions identified** | 8 across 3 taxonomies |
| **Brain** | 4 classes · 4 base models · 16-dim meta-features · **96.75%** [CONFIG] |
| **Chest (deployed)** | 3 classes · 3 base models · 9-dim · **82.00%** [CONFIG] |
| **Chest (archived)** | 3 classes · 3 base models · 9-dim · **85.68%** [CONFIG] |
| **Meta-learner** | Multinomial LogisticRegression, C=1.0, lbfgs, max_iter=1000 [PICKLE] |
| **Key scientific result** | Stacking raised glioma recall 0.81 → 0.90 |
| **Explainability** | Grad-CAM on the agreeing base model, JET at α=0.4, base64 PNG |
| **Reproduction check** | 96.75% ensemble reproduced from raw weights at **+0.00 pp** — also proves no train/serve skew [MEASURED] |
| **Calibration** | ECE 0.0262 → 0.0259 (adaptive 0.0134 → 0.0133), $T = 0.9964$; bases need $T$ up to **3.35**, the meta-learner needs none [MEASURED] |
| **Uncertainty** | Split conformal, $\alpha = 0.01$, coverage **1.0000**, mean set size **1.3313**; 334 confident / 137 borderline / 9 indeterminate [MEASURED] |
| **OOD** | Quality gate + JS-divergence novelty detector, **AUROC 0.9538**, FPR 7.29% on the hard tier [MEASURED] |
| **Evidence** | 12 cited sources, 28 chunks, embedding retrieval MRR **0.7139** vs TF-IDF 0.6389; complete misses 6/30 → **2/30** [MEASURED] |
| **Narratives** | 12/12 grounded first attempt, 0 fabricated citations, all `reviewed: false` so the **template is served** [MEASURED] |
| **Reporting** | 4-page PDF, 121–139 KB, caveats box, cited references [MEASURED] |
| **Measured latency** | brain **650 ms** · chest **490 ms** · Grad-CAM 3.2 s · full scan 3.5–4.0 s — was 9.9–14.9 s, CPU-only throughout [MEASURED] |
| **Top safety gap (closed)** | Confidence gating replaced by conformal bands with a coverage guarantee; the 56.5% "Normal" TB scan now reports **Indeterminate** |
| **Top integration gap (closed)** | Chest module reachable end to end; escalated scans persist |
| **Top remaining gap** | **No confidence intervals and no McNemar's test** — the 96.06% → 96.75% gain is inside the ±1.2 pt interval (§13.19) |
| **Outstanding your-action items** | 🔴 Rotate 3 credentials (§15) — history replaced but rotation still pending; clinician sign-off on 12 narratives |
| **Reclaimable disk** | ~735 MB |

---

*NeuraSight — Technical Presentation Script. All metrics tagged by provenance;
`[NOT RECORDED]` items require one notebook re-run to restore. §1–§12 describe the system as built
and audited; §13 documents what was implemented in response, with measured results.*
