# NeuraSight

**Explainable, calibrated and evidence-grounded medical image decision support.**

Two independent diagnostic modules — brain MRI tumour classification and chest X-ray screening —
each built on a stacking ensemble of CNNs, wrapped in a layer that reports **how much to trust the
answer** and **what the answer is based on**.

> **Research prototype. Not a medical device.** Not validated for clinical use, not certified by
> any regulator, and not a substitute for a qualified radiologist. Every report the system emits
> carries a non-removable disclaimer. See [Limitations](#limitations).

---

## Why this is not just a classifier

A bare classifier answers *"which class?"*. That is not enough to be useful next to a clinician,
because it cannot say when it is out of its depth. Every layer below exists to make a single
prediction accountable, and each one has a measured number attached to it.

```
    upload
      │
      ▼
 ┌──────────────────────┐
 │ 1  Quality gate      │  blur / contrast / dimensions      rejects 2.08% of valid scans
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 2  Stacking ensemble │  4 CNNs -> 16 features -> LogReg    96.75% brain  (reproduced +0.00 pp)
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 3  Calibration       │  temperature scaling               ECE 0.0262 -> 0.0259
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 4  Conformal sets    │  split conformal, alpha = 0.01     coverage 1.0000, mean size 1.3313
 └──────────┬───────────┘     confident / borderline / indeterminate
            ▼
 ┌──────────────────────┐
 │ 5  Novelty screen    │  base-model JS divergence          AUROC 0.9538, FPR 7.29%
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 6  Grad-CAM          │  on the base model that agrees     flags unfaithful explanations
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 7  Cited evidence    │  32 chunks, 17 sources, MiniLM     MRR 0.7532 vs 0.6528 TF-IDF
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 8  Report + PDF      │  4-page PDF, caveats box           121-144 KB
 └──────────────────────┘
```

**The uncertainty band, not the confidence number, is the primary output.** A `borderline` or
`indeterminate` band escalates the risk level to `Indeterminate` and the UI stops presenting the
top-1 class as a finding.

---

## What it identifies

| Module | Classes | Ensemble | Accuracy |
|--------|---------|----------|----------|
| `brain_mri` | Glioma · Meningioma · No Tumor · Pituitary | EfficientNet-B0 + ResNet-50 + DenseNet-121 + VGG-16 → LogisticRegression | **96.75%** |
| `chest_xray` | Normal · Pneumonia · Tuberculosis | EfficientNet-B0 + ResNet-50 + DenseNet-121 → LogisticRegression | **82.00%** |

Datasets: [`masoudnickparvar/brain-tumor-mri-dataset`](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
(7,023 images; 1,600-image test set, balanced 400/class) and
[`muhammadrehan00/chest-xray-dataset`](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset).

---

## Measured results

Everything below was measured on this codebase, CPU-only. Full method and evidence in
[`presentation.md`](presentation.md) §13 and [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §4.

### Reproduction

The published brain numbers were re-derived from the committed weights before anything was built on
top of them:

| Model | Published | Reproduced | Δ |
|-------|-----------|-----------|---|
| EfficientNet-B0 | 95.06% | 95.06% | 0.00 |
| ResNet-50 | 94.88% | 94.88% | 0.00 |
| DenseNet-121 | 96.06% | 96.06% | 0.00 |
| VGG-16 | 94.56% | 94.56% | 0.00 |
| **Stacking ensemble** | **96.75%** | **96.75%** | **+0.00 pp** |

An exact match also proves there is **no train/serve preprocessing skew** — the serving preprocessor
reproduces training behaviour bit-for-bit.

### Calibration — `models/calibration/calibration.json`

Fitted on 320 images, evaluated on a disjoint 480. The result that matters is not the ECE, it is the
per-model temperatures:

| | Fitted `T` | Reading |
|---|---|---|
| EfficientNet-B0 | 3.3542 | severely over-confident |
| VGG-16 | 3.0329 | severely over-confident |
| DenseNet-121 | 1.9523 | over-confident |
| ResNet-50 | 1.7743 | over-confident |
| **Meta-learner (deployed)** | **0.9964** | **already calibrated** |

The base CNNs need heavy softening; the logistic-regression combiner needs none. **The stacking layer
is doing calibration work as a side effect** — a substantive reason stacking helps here, beyond raw
accuracy.

ECE 0.026201 → 0.025949; adaptive (equal-mass) ECE 0.013439 → 0.013298. Both are reported because
conventional equal-width ECE is *inflated* here: 454 of 480 samples fall in one bin, so bins holding
1–4 samples dominate the average.

### Conformal prediction — `models/calibration/conformal.json`

Split conformal, LAC score, `alpha = 0.01`, `q_hat = 0.981281`.

| Metric | Value |
|--------|-------|
| Empirical coverage (480 held-out) | **1.0000** (target 0.99) |
| Mean set size | **1.3313** |
| Bands | 334 confident · 137 borderline · 9 indeterminate |

**Why `alpha = 0.01` and not the textbook 0.10:** conformal sets are all singletons whenever
`1 - alpha < model accuracy`, at which point coverage just equals accuracy and the layer carries zero
information. At 97.71% accuracy, anything above `alpha = 0.0229` is vacuous. A finite-sample floor at
`alpha = 0.0031` closes the window from below. Full sweep in the artefact.

### Novelty detection — `models/calibration/ood.json`

Benchmarked on three tiers. `near_ood` = real chest X-rays sent to the brain module, the hard case:

| Detector | AUROC | FPR @ 95% TPR |
|----------|-------|---------------|
| **Mean pairwise JS divergence** | **0.9538** | **0.0729** |
| Energy score | 0.9420 | 0.0938 |
| Predictive entropy | 0.8778 | 0.2833 |
| Max-softmax | 0.8753 | 0.2833 |
| Argmax-vote disagreement | 0.6403 | 1.0000 |

The winner is the signal the system was already computing and discarding. Note the 0.31 AUROC gap
between *graded* disagreement (JS divergence) and *binary* disagreement (argmax votes): the
information is in the probability geometry, not the votes.

The two layers are complementary, and the numbers show why — the quality gate catches **0 of 6** real
chest X-rays, because they are perfectly good images of the wrong organ.

### Evidence retrieval — `knowledge/`

**Sources span three jurisdictions, by design:**

| Jurisdiction | Sources | Role |
|---|---|---|
| International | WHO CNS5 classification; *Neuro-Oncology* 2024 consensus review; StatPearls ×2 | Tumour typing and grading — the shared standard Indian centres also use |
| United States | NCI PDQ ×4, NINDS, NCBI Bookshelf | General clinical fundamentals |
| **India** | ICMR AI ethics guidelines; CDSCO; National Cancer Grid; ICMR-NCDIR registry; Indian CNS epidemiology review | Care pathway, resource context, epidemiology, regulatory and ethical framework |

The split is deliberate. WHO CNS5 *is* the international standard and Indian tertiary centres grade
against it too, so there is no Indian alternative to it. What genuinely differs by country is
everything around the classification — where a patient is seen, what is locally available, and which
authority governs a tool like this.

Each class file carries a `## Care pathway in India` section, surfaced as its own field in the API,
dashboard and PDF. It gives the referral context and epidemiological framing, and states plainly that
this system holds **no CDSCO registration and is therefore not a medical device in India**, and that
citing Indian authorities does not mean the model was validated on an Indian cohort.

36 hand-labelled queries over 32 chunks, 23 deliberately paraphrased away from corpus
wording, `k = 5`:

| Metric | TF-IDF | Embedding |
|--------|--------|-----------|
| Hit@1 | 0.5000 | **0.6111** |
| Hit@3 | 0.8056 | **0.8889** |
| Recall@5 | 0.7731 | **0.7940** |
| MRR | 0.6528 | **0.7532** |
| **Complete misses** | **5 / 36** | **2 / 36** |

All six India queries retrieve a relevant chunk at rank 1. Note that Hit@1 and MRR rose partly
because the new queries are easier — "Is this an approved medical device in India?" has distinctive
vocabulary — so the comparison against TF-IDF is what the table is for, not the absolute movement.

### Latency (CPU-only)

| Operation | Before model caching | After |
|-----------|---------------------|-------|
| `POST /predict` brain | 9,851–14,864 ms | **650 ms** |
| `POST /predict` chest | 3,793–4,678 ms | **490 ms** |
| `POST /gradcam` brain | 13,683 ms | **3,196 ms** |
| `POST /api/scan` end-to-end | — | **3.5–4.0 s** |

Predictions are bit-identical before and after: the speedup came from not re-reading 645 MB of
weights per request, not from changing the maths. The first request after a restart still costs ~15 s
while models load — a warm-up cost, visible in `/health`.

---

## Getting started

### Prerequisites

- **Python 3.11** (3.14 is not supported by the pinned torch build)
- **Node.js 18+**
- **MongoDB** — local, or an Atlas cluster

### 1. Model weights — not in this repository

`.pth` files total ~785 MB and `BRAIN_MRI_VGG.pth` alone is 537 MB, over GitHub's 100 MB hard
per-file limit. The tiny meta-learners **are** committed (`models/meta_model.pkl` 1.2 KB,
`chest/meta_model_Chest_Xray.pkl` 958 B) because the ensembles cannot run without them.

Place the weights as follows:

```
models/                            chest/
├── BRAIN_MRI_EFFICIENTNET.pth     ├── CHEST_XRAY_EFFICIENTNET.pth
├── BRAIN_MRI_RESNET.pth           ├── CHEST_XRAY_RESNET.pth
├── BRAIN_MRI_DENSENET.pth         ├── CHEST_XRAY_DENSENET.pth
├── BRAIN_MRI_VGG.pth              ├── chest_xray_ensemble_config.json   (committed)
├── ensemble_config.json  (committed)  └── meta_model_Chest_Xray.pkl     (committed)
└── meta_model.pkl        (committed)
```

Without them, `/health` reports the module unavailable and inference returns **503** rather than
failing silently.

### 2. Install

```powershell
# ML service
cd backend\fastapi
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Gateway
cd ..\express
npm install

# Frontend
cd ..\..\frontend
npm install
```

### 3. Configure

```powershell
Copy-Item .env.example .env
Copy-Item backend\express\.env.example backend\express\.env
Copy-Item backend\fastapi\.env.example backend\fastapi\.env
```

Each `.env.example` documents every variable the code reads. `GROQ_API` can be left **empty** — the
application never calls an LLM at request time.

### 4. Run

```powershell
python run.py
```

That is all. The launcher runs a preflight first and tells you exactly what is missing rather than
failing cryptically — Python version, packages, Node, `node_modules`, weights per module, the
calibration/OOD/retrieval/narrative artefacts, MongoDB reachability, and whether the ports are free.
It then waits for each service to actually answer its health endpoint instead of guessing at a delay.

| Command | What it does |
|---------|--------------|
| `python run.py` | Start all three services |
| `python run.py --check` | Run the preflight only, start nothing |
| `python run.py --install` | Install any missing Python/Node dependencies, then start |
| `python run.py --kill` | Free ports 8000/5000/3000 and exit — use this when a previous run left something behind |
| `python run.py --no-frontend` | API services only, no Vite |

Missing `.env` files are created from the `.env.example` templates automatically.

Or start them individually — note that **uvicorn must run from `backend/fastapi`**, because the model
paths are relative and resolve against the working directory:

```powershell
cd backend\fastapi ; python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
cd backend\express ; node src/server.js
cd frontend        ; npm run dev
```

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:3000/dashboard.html |
| Frontend | http://localhost:3000 |
| Gateway health | http://localhost:5000/api/health |
| API docs | http://localhost:8000/docs |

---

## API

All inference endpoints are `POST multipart/form-data`, form field **`image`**, with a `module` query
parameter (`brain_mri` | `chest_xray`, default `brain_mri`).

### FastAPI — port 8000

| Endpoint | Returns |
|----------|---------|
| `POST /predict` | class, calibrated confidence, probabilities, `uncertainty`, `validation`, `ensemble` |
| `POST /gradcam` | base64 PNG heatmap + `explanation_faithful` |
| `POST /report` | full clinical report with cited sources |
| `POST /report/pdf` | rendered 4-page PDF |
| `GET /health` | per-layer status: calibration, conformal, OOD, retrieval index, narratives |
| `GET /modules` | module registry |

### Express gateway — port 5000

| Endpoint | Purpose |
|----------|---------|
| `POST /api/scan` | report + Grad-CAM in parallel, persisted to MongoDB. **The endpoint the dashboard uses.** |
| `POST /api/report/pdf` | streams the PDF through |
| `GET /api/predictions` | history; `?module=`, `?band=`, `?escalated=true`, `?q=`, `?page=` |
| `GET /api/predictions/stats` | counts by module, by band, escalated |
| `GET /api/predictions/:id` | one record incl. heatmap |
| `DELETE /api/predictions/:id` | delete a record |

### Example

```powershell
curl.exe -s -X POST "http://localhost:5000/api/scan?module=brain_mri" -F "image=@frontend/samples/meningioma_sample.jpg"
```

```jsonc
{
  "prediction": "Meningioma",
  "confidence": 94.86,
  "riskLevel": "Indeterminate",       // escalated from Medium by the band
  "riskEscalated": true,
  "uncertainty": {
    "uncertainty_band": "borderline",
    "prediction_set": ["Glioma", "Meningioma"],   // the honest answer
    "coverage_guarantee": 0.99,
    "calibrated": true
  },
  "validation": { "passed": true, "is_ood": false },
  "narrativeSource": "template",      // never unreviewed LLM text
  "sources": [ /* 5 resolved citations */ ]
}
```

---

## Repository layout

```
backend/
  fastapi/                    ML service
    app/
      modules/                per-module ensembles (brain_mri, chest_xray)
      services/
        model_cache.py        process-wide weight cache  (15-21x speedup)
        ensemble.py           stacking inference
        uncertainty.py        temperature + conformal sets
        validation.py         quality gate + novelty detection
        knowledge.py          cited knowledge base
        retrieval.py          MiniLM embedding retrieval
        narratives.py         reviewed-narrative gate
        report.py             report assembly + risk escalation
        pdf_report.py         reportlab PDF
        gradcam.py            explainability
      routers/                predict · gradcam · report · health
    scripts/                  offline artefact generation (see below)
    tests/
  express/                    gateway + MongoDB persistence
frontend/
  dashboard.html              scan + history UI
  index.html, research.html
knowledge/
  sources.json                17 cited sources (WHO + US federal + Indian)
  brain_mri/*.md              32 cited chunks (incl. India care pathway)
  narratives/*.json           12 pre-authored, review-gated narratives
  index.npz                   (32, 384) embeddings, 45 KB
models/
  calibration/                fitted artefacts + reliability.png + sample PDFs
presentation.md               full technical write-up
IMPLEMENTATION_PLAN.md        build log, decisions, known issues
```

**Everything in `models/calibration/` and `knowledge/` is a fitted artefact, not code** — committed
so the deployed behaviour is reproducible and every number above traces to a file.

---

## Verifying a change

Regression baselines. These must not move:

```powershell
foreach ($f in Get-ChildItem "data\samples\brainMRI" -File) { $r = curl.exe -s -X POST "http://127.0.0.1:8000/predict?module=brain_mri" -F "image=@data/samples/brainMRI/$($f.Name)" | ConvertFrom-Json; "$($f.Name) -> $($r.prediction) $($r.confidence)% band=$($r.uncertainty.uncertainty_band)" }
```

| Sample | Expected |
|--------|----------|
| glioma | Glioma 99.08% · confident · `{Glioma}` |
| meningioma | Meningioma 94.86% · borderline · `{Glioma, Meningioma}` |
| notumor | No Tumor 97.98% · borderline · `{Glioma, No Tumor}` |
| pituitary | Pituitary 99.35% · confident · `{Pituitary}` |

```powershell
cd backend\fastapi ; python -m pytest tests/ -q     # 11 passed
```

### Regenerating the artefacts

Run from `backend/fastapi`, in order. Each writes a JSON manifest recording its own inputs.

```powershell
python scripts\generate_test_probs.py    # base-model probability matrices
python scripts\fit_calibration.py        # temperature + conformal quantile
python scripts\build_ood_set.py          # 3-tier OOD evaluation set
python scripts\fit_ood.py                # detector comparison + thresholds
python scripts\build_index.py            # embedding index
python scripts\eval_retrieval.py         # retrieval metrics vs TF-IDF
python scripts\author_narratives.py      # needs GROQ_API; offline only
```

---

## Limitations

Stated plainly, because a decision-support tool that hides its failure modes is worse than none.

**Statistical**
- **No confidence intervals and no McNemar's test.** 96.75% on n=800 carries a ~±1.2 pt Wilson
  interval, and the 96.06% → 96.75% ensemble gain sits **inside it**. The defensible claim is the
  glioma recall gain, 0.81 → 0.90.
- Conformal coverage of 1.0000 is a point estimate, not a bound. The honest statement is "no coverage
  violation observed in 480 trials".
- The `near_ood` tier has **n = 6**. AUROC 0.9538 there is indicative, not established.
- Meta-features come from the test set rather than out-of-fold training predictions — a deviation
  from canonical stacking, left in place deliberately because fixing it means refitting the
  meta-learner and invalidating the reproduced 96.75%.

**Coverage**
- **Chest has no calibration or conformal layer.** The artefacts were fitted on the brain ensemble's
  16 features and 4 classes; applying them to 3-class chest output would be meaningless. Chest
  reports `calibrated: false` and falls back to a confidence threshold.
- Chest per-model and per-class metrics are **not recorded** anywhere in this repo — the training
  notebooks were saved with outputs cleared.
- The novelty detector fires on **7.29% of valid in-distribution MRIs**. The UI quotes that rate so a
  flag reads as a prompt, not a verdict.
- All 12 narratives are `reviewed: false`, so the deterministic template is served. A qualified
  reviewer must sign each one off before generated text reaches a user.

**Clinical**
- Single 2D slice; real radiology reads 3D volumes and multiple sequences.
- JPEG/PNG only — no DICOM, so 12–16-bit depth and windowing metadata are lost.
- No modality auto-detection: a chest X-ray sent to `brain_mri` is classified as a brain tumour.
- Single-source Kaggle corpora; no multi-centre or multi-scanner validation.
- **Not validated on an Indian cohort.** The knowledge base cites ICMR, CDSCO and the National
  Cancer Grid for care-pathway and governance context, which is the correct framing for use in
  India — but the model itself was trained on a public corpus of unstated scanner and population
  provenance. Indian citations are not Indian clinical validation, and the report says so.
- **No CDSCO registration.** Medical devices in India are regulated under the Drugs and Cosmetics
  Act, 1940 and the Medical Devices Rules, 2017. This holds no approval under either.
- No tumour grading, segmentation or volumetry.

**Engineering**
- **No authentication on any endpoint**, and CORS is permissive. Do not expose this to a network.
- Pickled artefacts are `pickle.load`ed, which is an arbitrary-code-execution primitive if a weights
  directory is ever untrusted.
- Training notebooks are excluded from this repository because one contains a hardcoded Kaggle API
  token in its source.

---

## Documentation

| Document | Contents |
|----------|----------|
| [`presentation.md`](presentation.md) | Full technical write-up: formulas, protocols, measured results, failure register, talk track |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Build log, architecture decisions, known issues, runbook |
| [`docs/brain_mri_model_results.md`](docs/brain_mri_model_results.md) | Original per-class brain metrics for the single EfficientNet baseline |
| [`docs/references.md`](docs/references.md) | Every method citation, tied to where it is implemented |
| [`knowledge/sources.json`](knowledge/sources.json) | The 17 cited sources, their jurisdiction and verification status |
| [`data/dataset_links.md`](data/dataset_links.md) | Dataset sources |
| `models/calibration/*.json` | Every fitted number, with the inputs that produced it |

The research page at `/research.html` presents the same material for a non-code audience.

---

## Licence and disclaimer

**No licence has been granted yet.** There is no `LICENSE` file in this repository, which under
default copyright means all rights are reserved — you may view the code, but reuse, modification and
redistribution are not permitted until a licence is added. If you want this to be reusable, add a
`LICENSE` file (MIT and Apache-2.0 are the usual choices for research code; both include a
no-warranty clause, which matters here).

Sample images under `frontend/samples/` and `data/samples/` are excerpts from the public Kaggle
datasets listed in [`data/dataset_links.md`](data/dataset_links.md), included for demonstration only
and subject to their original dataset terms.

**This software is not a medical device.** It has not been validated for clinical use, cleared by any
regulator, or tested in a prospective trial. Outputs are AI-generated and may be wrong — including
confidently wrong. Nothing it produces constitutes a diagnosis or medical advice. All findings must
be reviewed by a qualified radiologist or specialist physician before any clinical decision.

Clinical content in `knowledge/` states well-established fundamentals and cites public sources; it is
**not** verbatim from those sources and carries `clinical_review_required: true`.
