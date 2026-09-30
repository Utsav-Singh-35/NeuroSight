# NeuraSight — Brain MRI Clinical Intelligence: Implementation Plan & Handoff

**Last updated:** 29 September 2026
**Purpose:** Single source of truth for the Brain MRI module upgrade. Written so a fresh session
(or a different account) can resume with zero prior context.

> **How to use this document.** Section 2 tells you where the work stands. Section 4 gives the
> numbers you must be able to reproduce before trusting any change. Section 7 contains full
> specifications — formulas, file layouts, metrics — for everything not yet built, so no design
> decision needs re-deriving. Section 8 lists traps that have already cost time. Section 9 is the
> runbook.

---

## 1. Goal

Transform the project from **"Brain Tumor Classification using CNN"** into:

> **An Explainable, Calibrated and Evidence-Grounded Brain MRI Clinical Decision-Support Framework**

Target pipeline:

```
MRI → Domain/Quality Validation → Ensemble Diagnosis → Calibration → Conformal Uncertainty
    → Grad-CAM Explainability → Clinical Intelligence → Evidence Retrieval (RAG)
    → Vetted Narrative → Professional PDF + Dashboard
```

### The three research contributions (this is what gets graded)

Sixteen scattered features do not make a thesis. They collapse into three examinable claims:

1. **A trustworthy-confidence pipeline** — stacking + temperature calibration + conformal
   prediction sets + OOD rejection, evaluated with ECE, AURC, empirical coverage.
   *This is the primary contribution.*
2. **Faithful explainability** — Grad-CAM on the agreeing base model, validated with
   deletion/insertion curves and a randomisation sanity check, explicitly **not** claimed as
   segmentation.
3. **Evidence-grounded reporting** — retrieval with span-level citations and a measured
   groundedness rate, so every clinical statement traces to a source.

---

## 2. Status

| # | Task | Status |
|---|------|--------|
| 1 | Audit local runnability (data, deps, Groq key) | Done |
| 2 | Load models once + device selection | Done, verified |
| 3 | Grad-CAM target-class correctness | Done |
| 4 | Generate brain test-set probability matrices | Done — reproduces docs exactly |
| 5 | Probability calibration with measured ECE | Done, see §4.1 |
| 6 | Conformal prediction for uncertainty/referral | Done, see §4.2 |
| 7 | MRI domain/quality validation (OOD) | Done, see §4.3 |
| 8 | Cited brain tumour knowledge base | Done, see §4.4 |
| 9 | Local RAG retrieval | Done, see §4.5 |
| 10 | Groq offline narrative authoring | Done, see §4.6 |
| 11 | Professional PDF report | Done, see §4.7 |
| 12 | Wire into FastAPI / Express / dashboard | Done, see §4.8 |

**Headline results so far:**

1. Brain `/predict` went from **9,851–14,864 ms to 650 ms** with predictions bit-identical.
2. The published **96.75% ensemble accuracy reproduces exactly** (+0.00 pp) from raw weights, and
   all four per-model accuracies match the docs to 2 dp. This independently validates the numbers
   *and* proves there is no train/serve preprocessing skew.
3. Conformal prediction now yields a working referral mechanism: **334 confident / 137 borderline /
   9 indeterminate** out of 480, at **100% empirical coverage**.
4. All of it is reachable from the browser. A borderline scan now reaches the user as
   `Indeterminate` with a two-class prediction set and a 99% coverage guarantee, and persists to
   history — see §4.8.

---

## 3. Environment (verified 29 Sep 2026)

| Item | Value |
|------|-------|
| OS / shell | Windows, PowerShell |
| Python | 3.11.8 |
| torch | 2.7.1+cu118 — **but `torch.cuda.is_available() == False`** |
| CPU | 12 cores, torch using 10 threads |
| timm | 1.0.24 |
| scikit-learn | **1.5.1** — pickles were written with **1.6.1** (see §8.3) |
| numpy / PIL / cv2 | 1.26.4 / 10.4.0 / 4.13.0 |
| reportlab | 4.0.9 — **present**, use for PDF (no need for fpdf) |
| scipy / matplotlib | 1.15.3 / 3.9.2 |
| **Missing** | `pytorch_grad_cam`, `sentence_transformers`, `groq`, `fpdf` |

Install when the relevant task starts:

```powershell
pip install groq sentence-transformers
# optional: pip install grad-cam   (see §8.2 before doing this)
```

### Data (all present locally)

| Path | Contents |
|------|----------|
| `data/brainMRI/Testing/` | 400 images × 4 classes = **1,600** |
| `data/brainMRI/Training/` | 1,400 images × 4 classes = **5,600** |
| `data/chest_xray/` | **Absent** — chest training data is remote only |
| `data/samples/brainMRI/` | 4 demo images |
| `data/samples/ChestXray/` | 6 demo images (all tuberculosis) |

Class directory order is `glioma, meningioma, notumor, pituitary`, which matches
`ensemble_config.json → class_names`. The probability-generation script **asserts** this rather
than assuming it.

### Groq API key

- Stored in **repo-root `.env`** as `GROQ_API` (56 chars, `gsk_` prefix).
- `.env.example` has `GROQ_API=` **empty** — correct pattern, keep it that way.
- Verified **not** in git history (`git grep gsk_` across all revs is clean). Root `.env` is
  untracked and gitignored.
- `app/config.py` now reads **two** env files so the key is visible to FastAPI:
  repo-root `.env` first, then `backend/fastapi/.env` (which wins on conflict).
- Config field: `settings.GROQ_API`; model: `settings.GROQ_MODEL` (default
  `llama-3.3-70b-versatile`).

---

## 4. Verified baselines — regression reference

**Any change to preprocessing, ensemble, or weights must reproduce these exactly.** If they move,
something broke.

### Sample predictions

| Module | Image | Expected |
|--------|-------|----------|
| brain | `glioma_sample.jpg` | Glioma **99.08%** |
| brain | `meningioma_sample.jpg` | Meningioma **94.86%** |
| brain | `notumor_sample.jpg` | No Tumor **97.98%** |
| brain | `pituitary_sample.jpg` | Pituitary **99.35%** |
| chest | `tuberculosis-1117.jpg` | Tuberculosis **97.68%** |
| chest | `tuberculosis-1122.jpg` | **Normal 56.51%** ← known misclassification, keep it |
| chest | `tuberculosis-1123.jpg` | Tuberculosis **98.06%** |
| chest | `tuberculosis-1125.jpg` | Tuberculosis **98.06%** |
| chest | `tuberculosis-1127.jpg` | Tuberculosis **97.26%** |
| chest | `tuberculosis-1129.jpg` | Tuberculosis **98.05%** |

`tuberculosis-1122` is a deliberate keeper: it is an active TB case reported as Normal at 56.51%,
and it is the empirical justification for the whole uncertainty/abstention contribution.

### Latency (CPU, after task 2)

| Operation | Before | After |
|-----------|--------|-------|
| brain `/predict` | 9,851–14,864 ms | **1,395 ms cold / 650–699 ms warm** |
| brain `/gradcam` | 13,683 ms | **3,196 ms** |
| chest `/predict` | 3,793–4,678 ms | **490–504 ms** |

### Committed artefact facts

| Artefact | Value |
|----------|-------|
| brain ensemble accuracy | 96.75% (`models/ensemble_config.json`) |
| chest (TB) ensemble accuracy | 82.00% (`chest/chest_xray_ensemble_config.json`) |
| chest V1 (Kermany) accuracy | 85.68% (`chest V1/chest_xray_ensemble_config.json`) |
| brain meta-learner | LogisticRegression, `n_features_in_=16`, `coef_ (4,16)`, C=1.0, lbfgs, max_iter=1000 |
| chest meta-learner | LogisticRegression, `n_features_in_=9`, `coef_ (3,9)` |
| brain meta reliance | efficientnet 23.5% / resnet 25.8% / densenet 25.1% / vgg 25.6% |
| chest TB meta reliance | efficientnet 28.2% / resnet **41.0%** / densenet 30.8% |
| chest V1 meta reliance | efficientnet **42.8%** / resnet 25.8% / densenet 31.4% |

Brain per-class (EfficientNet-B0, 1,600 images, from `docs/brain_mri_model_results.md`):
Glioma P1.00/R0.81, Meningioma P0.90/R0.99, No Tumor P0.92/R1.00, Pituitary P0.99/R1.00.
**Stacking lifted glioma recall 0.81 → 0.90 — this is the key scientific result, not the accuracy.**

### 4.1 Calibration results (task 5)

Artefacts: `models/calibration/calibration.json`, `reliability.png`.

**Split used** — the shipped `meta_model.pkl` was fitted on half A, so half A is spent. Half B is
subdivided; every reported number comes from 480 images the meta-learner and the temperature have
never seen.

```
1,600 → half A (800, meta-train, spent)
      → half B (800, untouched) → calibration 320 → final test 480
```

| Design | T | Accuracy | ECE | Adaptive ECE | Brier | NLL | AURC | Sel. risk @80% |
|--------|---|----------|-----|--------------|-------|-----|------|----------------|
| Uncalibrated | — | 97.71% | 0.026201 | 0.013439 | 0.032756 | 0.072168 | 0.001845 | 0.5208% |
| **B — scale meta output** (deployed) | 0.9964 | 97.71% | 0.025949 | 0.013298 | 0.032739 | 0.071985 | 0.001845 | 0.5208% |
| **A — scale bases + refit meta** (best) | per-model | 97.50% | **0.020344** | **0.011783** | 0.032710 | 0.069706 | **0.001459** | **0.2604%** |

Design A per-model temperatures: efficientnet **3.3542**, resnet **1.7743**, densenet **1.9523**,
vgg **3.0329**.

**Findings worth writing up:**

- **The base models are substantially over-confident** (all temperatures ≫ 1, up to 3.35), yet the
  meta-learner's output needs almost no correction (T = 0.9964). This empirically confirms that
  **the logistic-regression meta-learner is already acting as a calibrator** — a real result, and
  the reason Design B has so little to do.
- Design A is better on every calibration and selective-prediction metric (ECE −22.4%, AURC
  −21%, selective risk at 80% coverage halved from 0.52% to 0.26%) at a cost of 0.21 pp accuracy.
  **It is not deployed**, because it requires refitting and re-exporting `meta_model.pkl`, which
  would invalidate the published 96.75% and every regression baseline. That is a human decision —
  see `upgrade_note` in `calibration.json`.
- **Report adaptive (equal-mass) ECE alongside conventional ECE.** With a 97.7%-accurate model,
  ~460 of 480 test images land in the top equal-width confidence bin, leaving mid bins with ~3
  samples each whose observed accuracy swings 0.0→1.0 on noise alone. That inflates conventional
  ECE to 0.0262 and makes the reliability diagram look alarming. Equal-mass binning (~32
  samples/bin) gives **0.0134** and a curve that hugs the diagonal. `reliability.png` shows both,
  with sample-count bars, so the artefact is self-explaining.
- Temperature scaling never changes argmax, so **accuracy is mathematically unchanged** by Design B.

### 4.2 Conformal prediction results (task 6)

Artefact: `models/calibration/conformal.json`. Method: split conformal, LAC score
$s = 1 - p_{\text{true}}$, quantile level $\lceil (n+1)(1-\alpha)\rceil / n$.

**Critical finding — conformal is vacuous at the textbook α.** Sets can only widen once the target
coverage $1-\alpha$ exceeds the model's accuracy. At 97.71% accuracy that means **α must fall below
0.0229**; above it every set is the singleton top-1 and empirical coverage simply equals accuracy.
The default was therefore changed from α = 0.10 to **α = 0.01**.

| α | q̂ | Target cov. | Empirical cov. | Mean size | Size distribution |
|---|----|-------------|----------------|-----------|-------------------|
| 0.20 | 0.0511 | 0.800 | 0.9771 | 1.000 | all singletons — vacuous |
| 0.10 | 0.0615 | 0.900 | 0.9771 | 1.000 | all singletons — vacuous |
| 0.05 | 0.5188 | 0.950 | 0.9771 | 1.000 | all singletons — vacuous |
| 0.025 | 0.9205 | 0.975 | 0.9938 | 1.058 | 457 / 18 / 5 |
| 0.02 | 0.9328 | 0.980 | 0.9938 | 1.067 | 454 / 20 / 6 |
| **0.01** | **0.9813** | **0.990** | **1.0000** | **1.331** | **334 / 137 / 5 / 4** |
| 0.005 | 1.0000 | 0.995 | 1.0000 | 4.000 | degenerate — all classes |

**Deployed operating point (α = 0.01):** q̂ = 0.981281, threshold = 0.018719, empirical coverage
**100%**, per-class coverage 100% on all four classes.

| Band | Rule | Count (of 480) | Share |
|------|------|----------------|-------|
| `confident` | set size 1 | 334 | 69.6% |
| `borderline` | set size 2 | 137 | 28.5% |
| `indeterminate` | set size ≥ 3 | 9 | 1.9% |

So roughly **30% of cases are routed to human review** while retaining 100% empirical coverage —
a genuine, statistically-backed referral mechanism rather than a hand-picked cutoff.

**Two limits to state explicitly:**
- **Finite-sample floor:** with $n_{\text{cal}} = 320$, the quantile level reaches 1.0 for
  α < 0.003115, forcing q̂ = 1 and admitting every class. α = 0.005 already degenerates. A larger
  calibration split (i.e. out-of-fold stacking) is needed to certify tighter α.
- **Non-empty fallback:** when the threshold excludes every class, serving falls back to the top-1
  and sets `conformal_fallback: true`. This slightly breaks the formal guarantee in that rare
  branch, and is reported rather than hidden.

**Behavioural spot-check** — the conformal layer independently rediscovers the known weakness. The
real `meningioma_sample.jpg` (94.86% meningioma, 5.04% glioma) is classified **`borderline` with
set `{Glioma, Meningioma}`**, which is exactly the glioma↔meningioma confusion that drove
EfficientNet's glioma recall down to 0.81. Nothing was hand-tuned to produce that.

### 4.3 Input validation / OOD results (task 7)

Artefacts: `models/calibration/ood.json`, `data/ood/` (built by
`scripts/build_ood_set.py`, seeded and reproducible).

**Three OOD tiers, reported separately** — pooling them would let trivially-separable synthetic
images inflate the headline:

| Tier | n | What it is |
|------|---|------------|
| `near_ood` | 6 | **Real chest X-rays.** Right modality family, wrong anatomy. Hardest and most realistic — this is a user picking the wrong scan type. **Headline tier.** |
| `corrupted` | 40 | Real MRIs degraded (heavy/extreme blur, noise, 14×14 downsample, near-blank). Tests the quality gate. |
| `far_ood` | 60 | Synthetic patterns (noise, gradients, checkerboards, shapes, text-like). Sanity floor only. |

**Detector comparison** (all oriented so higher = more OOD; ID = the 480-image final test split,
which was never used for calibration or conformal fitting):

| Detector | near-OOD AUROC | FPR@95 | corrupted AUROC | far-OOD AUROC |
|----------|---------------|--------|-----------------|---------------|
| **js_divergence** (deployed) | **0.9538** | **0.0729** | 0.9583 | 0.9523 |
| energy | 0.9420 | 0.0938 | 0.9518 | 0.9740 |
| entropy | 0.8778 | 0.2833 | 0.9588 | 0.9773 |
| msp | 0.8753 | 0.2833 | 0.9573 | 0.9778 |
| disagreement (vote count) | 0.6403 | 1.0000 | 0.8507 | 0.7628 |

Deployed: `js_divergence`, threshold **0.026446**, selected on the near-OOD tier.

**Findings worth writing up:**

- **The best detector is the signal the API was throwing away.** Mean pairwise Jensen–Shannon
  divergence between base models wins on near-OOD (0.9538) and needs **no new model** — the base
  probabilities are already computed on every ensemble request and then dropped by
  `PredictionResponse`. Exposing them is free.
- **Continuous disagreement beats discrete voting by a wide margin**: JS divergence 0.9538 vs
  argmax vote-counting 0.6403. With only four base models, vote counting is too coarse — it cannot
  distinguish "all four agree weakly" from "all four agree strongly". Its FPR@95 of 1.0 means it is
  useless as a gate. Same underlying idea, very different utility.
- **Energy beats max-softmax on the hard tier** (0.9420 vs 0.8753), reproducing the standard result
  from the OOD literature. Note energy *cannot* be computed from stored softmax probabilities —
  they sum to 1, so log-sum-exp is constant. It uses `LogisticRegression.decision_function`.
- **On far-OOD, MSP and entropy look best (≈0.978)** — which is precisely why far-OOD must not be
  the selection tier. Optimising for detecting random noise would have picked the wrong detector for
  the failure that actually matters.

**Quality gate**, fitted at the 1st percentile of the ID population
(`min_laplacian_variance = 22.27`, `min_pixel_std = 30.22`):

| Set | Rejected | Rate |
|-----|----------|------|
| ID (final test split) | 10 / 480 | 2.1% (≈1% by construction) |
| corrupted | 25 / 40 | 62.5% |
| far_ood | 23 / 60 | 38.3% |
| **near_ood (chest X-rays)** | **0 / 6** | **0%** |

**This table is the argument for the two-layer design.** Chest X-rays are *high-quality* images —
sharp, good contrast — so the quality gate correctly passes all of them. They are only detectable
semantically, by the OOD scorer. Conversely the quality gate catches 62.5% of degraded MRIs that
the OOD scorer would have to guess at. Neither layer subsumes the other.

**Conformal set size doubles as a novelty signal** (α = 0.01):

| Set | Mean set size |
|-----|---------------|
| in-distribution | 1.3313 |
| near_ood | 2.1667 |
| corrupted | 2.4500 |
| far_ood | 2.4333 |

**Limitations to state plainly:**
- **Only 6 near-OOD images.** The 0.9538 AUROC has a wide confidence interval and the threshold is
  not publication-grade. `data/chest_xray/` is absent locally; pulling the full chest dataset would
  fix this and is the single highest-value improvement to this section.
- Synthetic far-OOD overstates detector performance by construction. That is why it is a floor, not
  a result.
- The operating point costs **7.3% of valid brain MRIs rejected** to catch 95% of chest X-rays. That
  trade-off is a clinical decision, not a technical one.

### 4.4 Cited knowledge base + report safety (task 8)

Files: `knowledge/sources.json`, `knowledge/brain_mri/{glioma,meningioma,pituitary,no_tumor}.md`,
`app/services/knowledge.py`, rewritten `app/services/report.py`.

**Loaded state (verified):** 4 entries · 28 retrieval chunks · 11 registered sources ·
10 citations used · **0 unresolved citations** · all 4 entries flagged
`requires_clinical_review: true`. Chunk length 37–222 words, mean 78 — already the right size for
retrieval, so section boundaries are used directly as chunk boundaries and no sliding window is
needed.

Each entry has 7 sections: `overview`, `ai_limitations` (or `interpretation` for No Tumor),
`investigations`, `clinical_considerations`, `follow_up`, `warning_signs`, `treatment_information`.

**Sources registered** (all URLs verified to exist and be topically correct on 2026-09-29): NCI
PDQ Adult CNS Tumors (patient + health-professional + Bookshelf), NCI Meningioma, StatPearls
Meningioma (NBK560538), ICOM Meningioma consensus (Neuro-Oncology 2024), NCI Pituitary Tumors PDQ
(NBK66024.6), NCI Pituitary diagnosis, StatPearls Pituitary Adenoma (NBK554451), NINDS Brain and
Spinal Cord Tumors, WHO CNS5 classification.

> **Honesty constraint recorded in `sources.json`.** Page text was **not** programmatically
> extracted — cancer.gov and ninds.nih.gov refuse automated fetching (NINDS returns HTTP 403) and the
> NCBI Bookshelf pages yielded no text to the fetcher. The entries therefore state well-established
> general clinical fundamentals and cite these sources for verification; they are **not verbatim
> extracts**. `verification_status.content_extracted` is `false` and every entry carries
> `requires_clinical_review: true`. **A supervisor or clinician must check the content against the
> primary sources before any clinical framing is relied upon.**

**Two safety defects fixed, both verified live through the API:**

1. **Unknown labels no longer produce reassurance.** The old code did
   `TUMOR_INFO.get(prediction, TUMOR_INFO["No Tumor"])`, so any unrecognised label rendered a
   "no abnormality detected" narrative. Verified: label `Astrocytoma` now returns
   `risk_level: Indeterminate`, `content_source: none`, a referral recommendation, and **no borrowed
   clinical text**.
2. **Risk level is now coupled to uncertainty.** Risk was a static per-class attribute. It now
   escalates to `Indeterminate` whenever the conformal band is `borderline`/`indeterminate`, or —
   when no conformal set is supplied — whenever confidence is below 90%.

**The headline safety result.** The `tuberculosis-1122.jpg` case that previously produced
`risk_level: "Low"` and *"No immediate intervention required"* for an active TB scan at 56.51%
confidence now returns, live through `POST /report?module=chest_xray`:

```
prediction        : Normal 56.51%
risk_level        : Indeterminate (escalated=True)
recommendation    : The AI result is inconclusive and should not be acted upon as a finding.
                    Refer this scan for review by a qualified radiologist or specialist physician...
```

Also verified: brain `/report` returns `evidence_grounded: true`, `content_source: knowledge_base`,
4 resolved sources, populated `warning_signs` and `treatment_information`, and
`clinical_review_required: true`. The real meningioma sample at 94.86% escalates to `Indeterminate`
with prediction set `{Glioma, Meningioma}`. All four brain predictions remain bit-identical to the
§4 baseline.

**Editorial rules applied throughout:** no drug names, no dosages; investigations phrased as
*"may consider discussing with a healthcare professional"* rather than directives; treatment
information kept categorical (observation / surgical / radiation-based / specialist-directed); every
clinical claim carries a source ID.

**Known gap:** `chest_xray` has **no** cited knowledge base yet and still uses the legacy inline
dictionary. This is reported honestly per-response via `evidence_grounded: false` and
`content_source: "legacy_inline"` rather than left ambiguous.

### 4.5 Evidence retrieval / RAG results (task 9)

Files: `app/services/retrieval.py`, `scripts/build_index.py`, `scripts/eval_retrieval.py`,
`knowledge/index.npz` (39 KB), `knowledge/index_meta.json`, `knowledge/retrieval_eval.json`
(30 hand-labelled queries), `knowledge/retrieval_results.json`.

**No vector database, no LangChain.** The corpus is 28 chunks, so the whole index is a `(28, 384)`
float32 array and search is one matrix–vector product. Vectors are L2-normalised at index time, so
cosine similarity reduces to a dot product.

**Two backends implemented so the choice is measured, not asserted:**

| Backend | Hit@1 | Hit@3 | Recall@5 | MRR | P@5 | Complete misses |
|---------|-------|-------|----------|-----|-----|-----------------|
| tfidf (sklearn baseline) | 0.5000 | 0.8000 | 0.7472 | 0.6389 | 0.2800 | 6 / 30 |
| **embedding (MiniLM-L6-v2)** | **0.5333** | **0.9000** | **0.8194** | **0.7139** | 0.3067 | **2 / 30** |

Breakdown by query type (19 deliberately paraphrased away from corpus wording, 11 literal):

| Subset | Backend | Hit@1 | Hit@3 | Recall@5 | MRR |
|--------|---------|-------|-------|----------|-----|
| paraphrased | tfidf | 0.4737 | 0.7368 | 0.6535 | 0.5965 |
| paraphrased | embedding | 0.4211 | **0.8947** | **0.7939** | **0.6623** |
| literal | tfidf | 0.5455 | 0.9091 | **0.9091** | 0.7121 |
| literal | embedding | **0.7273** | 0.9091 | 0.8636 | **0.8030** |

**Findings:**

- The embedding backend wins overall, and the clearest signal is **complete misses: 2/30 vs 6/30**.
  For grounding a narrative, "retrieved nothing relevant at all" is the failure that matters.
- **The semantic advantage is in Hit@3 and Recall, not Hit@1.** On paraphrased queries TF-IDF
  actually edges Hit@1 (0.4737 vs 0.4211) by occasional lexical luck, while embedding lifts Hit@3
  from 0.7368 to 0.8947. The honest claim is "semantic retrieval reliably gets relevant evidence
  into the top 3", not "semantic retrieval always ranks it first".
- **Precision@5 is capped by construction** — relevant sets hold 1–4 chunks against k=5 — so the
  ~0.30 figure is a ceiling artefact, not poor retrieval. Recall@5 and MRR are the honest metrics.
  This is recorded in `retrieval_results.json → metric_notes`.
- **Latency: 15.3 ms warm** (min 13.0, max 18.8) per query, versus 18.7 s cold because the first
  query loads the 80 MB encoder. Index construction from the saved `.npz` is 11 ms. This is what
  justifies the "LLM as offline authoring aid" decision — serving never needs the encoder if
  narratives are pre-generated.
- **Self-retrieval: 28/28 exact**, 0 misaligned. The sanity check distinguishes genuine
  misalignment from cross-class near-duplication (the `investigations` sections read similarly
  across classes), because only the former is a bug.

**Citation completeness audit.** An initial audit found **4 uncited chunks**, all in `No Tumor`.
Rather than bolt a medical reference onto statements about our own software, a `neurasight-system`
self-reference source was registered and applied, so **every one of the 28 chunks now has declared
provenance** — either an external medical source or an explicit self-reference. Result: **0 uncited
chunks.** Keeping the two kinds of provenance distinct is what makes the audit meaningful.

**Class-filtered retrieval** (what report generation uses) verified working for all four classes,
with citations attached to every hit.

**`.gitignore` exceptions added** so the runtime artefacts survive a fresh clone:
`!knowledge/index.npz` (39 KB — avoids needing the model download) and `!models/calibration/*.png`
(evaluation figures are results, not data). The calibration/conformal/OOD JSON artefacts were
already committable; only `*.npy` probability matrices remain ignored, and those are regenerable.

### 4.6 Groq narrative authoring results (task 10)

Files: `app/services/llm.py` (provider seam), `app/services/narratives.py` (serving + review gate),
`scripts/author_narratives.py`, `knowledge/narratives/*.json` (12 items + `authoring_summary.json`).

**The model name in the plan was wrong and the seam caught it.** `llama-3.3-70b-versatile` returns
**404 — model not found** for this key. Groq's lineup has changed. Models actually available:
`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b`, `qwen/qwen3.8-27b`,
`allam-2-7b`, plus whisper/prompt-guard/orpheus (not text generation). **Deployed:
`openai/gpt-oss-120b`.** Check with `python scripts/author_narratives.py --list-models` before
assuming any model name is valid.

**Three real defects found and fixed during this task:**

1. **Reasoning models consume the completion budget before emitting content.** A 16-token cap
   returned an empty message with `completion_tokens=16` — the internal reasoning had used the whole
   allowance. `DEFAULT_MAX_TOKENS` is now 3000, and an empty response with
   `completion_tokens >= max_tokens` now reports that specific cause instead of a generic error.
2. **Full-width CJK citation brackets.** The model emitted `【ncbi-meningioma-statpearls】` instead of
   `[...]`, so the citation regex matched nothing and flagged correct output as ungrounded — a false
   negative. Brackets are now normalised (`【】〔〕` → `[]`) before validation, in both the authoring
   script and `knowledge.py`, and the prompt demands ASCII explicitly.
3. **Non-ASCII typography.** Output contained U+2011 (non-breaking hyphen), which crashed console
   output under Windows cp1252 and would render as black boxes in the default reportlab fonts.
   Smart quotes, en/em dashes, ellipses and non-breaking spaces are now normalised to ASCII, and
   `non_ascii_characters` is validated and reported per item.

**Authoring results** (4 classes × 3 bands = 12, `openai/gpt-oss-120b`, embedding retriever, k=5):

| Metric | Result |
|--------|--------|
| Written | **12 / 12** |
| Grounded on first attempt | **12 / 12** |
| Grounded final | **12 / 12** |
| Fabricated citations | **0** |
| Percentages (rule breach) | **0** |
| Non-ASCII after normalisation | **0** |
| Total tokens | 27,539 |
| Length | 131–197 words |

A **retry-on-ungrounded** loop (up to 3 attempts, default) was added after an intermediate run
produced one narrative with no citations at all — roughly 1 in 12 first-pass outputs omitted them
before the prompt was tightened. Both first-attempt and final rates are recorded in
`authoring_summary.json`, so the retry improves the artefact without hiding the LLM's raw
reliability.

**The review gate, verified.** All 12 are `reviewed: false`, and all 12 are **withheld** from
serving. The report falls back to deterministic template text with
`narrative_source: "template"`. The failure mode of the LLM layer is therefore "plainer prose",
never "unvetted clinical claims reach a user".

```
narratives on disk       : 12
servable (reviewed=true) : 0
withheld (unreviewed)    : 12
```

> **Action required before the demo:** a human must read each of the 12 files in
> `knowledge/narratives/`, confirm no medicine/dose/diagnosis is stated and that the uncertainty
> framing matches the band, then set `reviewed: true` with `reviewed_by` and `reviewed_at`. Until
> then the system serves template text. Each file carries `review_instructions` inline.

**Sample output quality** (meningioma / borderline, 193 words, 4 citations): correctly refuses to
assert the class, explains that a single slice cannot establish growth rate, location or grade, and
routes to professional review — the hedged framing the band requires, produced without hand-editing.

**Privacy property maintained:** no image bytes and no patient identifiers are sent to Groq. The
prompt contains only the class label, the uncertainty band, and passages retrieved from the local
knowledge base. Total cost across the whole project is ~27.5 k tokens, orders of magnitude inside
the free tier.

### 4.7 PDF report results (task 11)

File: `app/services/pdf_report.py`. Built on reportlab platypus flowables so content paginates
rather than being placed at fixed coordinates. Samples in
`models/calibration/sample_reports/`.

`build_report_pdf(report, heatmap_b64=None, original_image_bytes=None, patient=None) -> bytes`
returns PDF bytes, ready to stream from an endpoint.

**Sections, in order:** branded header with scan metadata · primary finding with calibrated
confidence, risk level, conformal prediction set and coverage guarantee · **caveats box** ·
probability distribution table with inline bars and an "in set" column · Grad-CAM beside the
original scan · clinical interpretation (reviewed narrative or template) · condition overview ·
limitations for this finding · potential further evaluation · clinical considerations · follow-up ·
**warning signs** (highlighted box) · treatment information · recommended next step · evidence and
references · AI limitations disclaimer · per-page footer.

**The PDF carries more caveats than the JSON, deliberately** — it is the artefact most likely to be
read detached from the UI. The caveats box fires on any of: risk escalated, `explanation_faithful`
false, `models_skipped` non-empty, ensemble inactive, validation failed, or OOD flagged.

**Two bugs found and fixed during verification:**

1. **Images silently failed to embed.** reportlab's `Image` flowable takes a path or a file-like
   object; passing an `ImageReader` raises `expected str, bytes or os.PathLike object`. The error was
   caught and logged, so the PDF still built — just with no images, at 9.8 KB. Fixed by passing
   `BytesIO` directly; PDFs are now 121–139 KB with both panels present.
2. **Risk escalation was not wired for brain MRI.** `BrainMRIModule.report()` called
   `generate_report` *without* the uncertainty payload, so a borderline case was reported with its
   class's default risk tier — `Meningioma` at 94.86% came out as `risk: Medium, escalated: False`
   despite a two-class prediction set. This is precisely the failure the pipeline exists to prevent,
   and leaving escalation to the caller was the wrong design. Uncertainty is now computed **inside**
   the module (`describe_uncertainty()`, lazily loading `CalibrationBundle`) and passed in. Verified:
   the same case now returns `risk: Indeterminate, escalated: True`.

**Per-module calibration correctness.** Chest deliberately passes **no** uncertainty payload: the
artefacts in `models/calibration/` were fitted on the brain ensemble's 16-dimensional features and
4 classes, so applying that temperature or conformal quantile to 3-class chest output would be
meaningless. Chest therefore reports `calibrated: false` and relies on the confidence-threshold
fallback for escalation, which was already verified working on the TB case.

**Verified output:**

| Case | Prediction | Band / set | Risk | PDF |
|------|-----------|-----------|------|-----|
| `glioma_sample.jpg` | Glioma 99.08% | confident / `{Glioma}` | High, not escalated | 138,967 B, 4 pages |
| `meningioma_sample.jpg` | Meningioma 94.86% | borderline / `{Glioma, Meningioma}` | **Indeterminate, escalated** | 120,689 B, 4 pages |

Both carry a valid `%PDF-` header, 4 resolved sources (5 for meningioma), populated warning signs
and treatment information, and `narrative_source: template` because no narrative is approved yet.

**Text safety:** all strings are XML-escaped before reaching `Paragraph` (platypus interprets a
markup dialect, so a stray `&` or `<` would raise) and coerced to Latin-1-safe characters, because
the built-in Type 1 fonts do not cover arbitrary Unicode. This is the safety net behind the
upstream ASCII normalisation from §4.6.

> **Not recorded anywhere:** chest per-model and per-class metrics. All 7 notebooks were saved with
> outputs cleared and no `*_test_probs.npy` / `model_comparison.csv` exists for chest. Re-run
> `notebooks/Chest_Xray_Stacking_Ensemble.ipynb` to restore them, and **commit the outputs**.

---

### 4.8 Integration results (task 12)

Everything built in tasks 5–11 is now reachable from the browser. Verified through the **Vite proxy
on :3000**, i.e. the exact path a user takes, not by calling the services directly.

**FastAPI.** `/predict`, `/gradcam` and `/report` return `uncertainty`, `validation` and `ensemble`
objects; `/report` gained a response model (`ReportResponse`) where it previously had none, so its
shape is now validated and documented in the OpenAPI schema. `POST /report/pdf` streams a rendered
PDF. `/health` reports every layer (calibration, conformal, OOD, retrieval index, narratives) so a
missing artefact is visible without reading logs. Shared per-request guards live in
`app/routers/_common.py` rather than being duplicated across three routers.

**Express.** `Prediction` stores the full decision-support record: `uncertaintyBand`,
`predictionSet`, `coverageGuarantee`, `calibrated`, `riskEscalated`, `validationPassed`, `isOod`,
`oodScore`, `ensembleActive`, `modelsSkipped`, `explanationFaithful`, `evidenceGrounded`,
`narrativeSource`. `GET /api/predictions` gained `?band=` and `?escalated=true`; `/stats` gained
`byBand` and `escalatedCount`. `POST /api/report/pdf` proxies the PDF with `Cache-Control: no-store`.

**Dashboard.** The conformal band is now the primary status, not the confidence threshold: a band
pill in the header, a prediction-set box quoting the coverage guarantee, and a single caveats box
that gathers risk escalation, unfaithful explanations, skipped models, a degraded ensemble, failed
quality checks and OOD flags. Clinical sections (warning signs, considerations, investigations,
follow-up, treatment information, AI limitations) render when present, followed by the resolved
citation list and a provenance line stating whether the wording came from a reviewed narrative or
the template and whether the probabilities were calibrated. History rows show the band and set size,
filter by band or escalation, and the summary cards count confident / borderline / indeterminate.
A "Download PDF Report" button posts the in-memory image to `/api/report/pdf`; it is deliberately
absent in the history detail view, because the original upload is not stored.

**Verified end-to-end (through :3000):**

| Check | Result |
|-------|--------|
| `glioma_sample.jpg` | Glioma **99.08%** — baseline unchanged |
| Band / set / coverage | `confident` / `{Glioma}` / 0.99, `calibrated: true` |
| `meningioma_sample.jpg` | Meningioma **94.86%**, `borderline`, `{Glioma, Meningioma}` |
| Escalated case persists | `riskLevel: Indeterminate`, `riskEscalated: true`, `saved: true` |
| `?band=borderline` | 3 records, all with 2-class sets and escalated risk |
| `?band=bogus` | ignored rather than erroring (16 records, unfiltered) |
| `/stats` | `byBand: {unbanded: 11, confident: 2, borderline: 3}`, `escalatedCount: 3` |
| PDF via proxy | **139,354 B**, `application/pdf`, `Cache-Control: no-store`, 4.8 s |
| Warm scan latency | 3.5–4.0 s end-to-end (report + Grad-CAM in parallel); 14.9 s on the first scan after a restart while models load |

Render markup was verified by loading the inline `<script>` into a stub DOM and rendering real
`/api/scan` payloads — all 13 expected blocks emitted for both the confident and the borderline
case. The harness was temporary and has been deleted; the dashboard has no build step or test
runner, so there is nothing permanent to keep.

**Two things found while integrating:**

1. **The OOD reason overstated its own evidence.** `glioma_sample.jpg` is flagged
   (`js_divergence = 0.2606` vs threshold `0.026446`) even though it is in-distribution and is
   classified correctly at 99.08%. That is not a bug — it is the measured 7.29% false-positive rate
   at this operating point, and it is informative: the meta-learner is confident while the base
   models disagree unusually strongly. The wording asserted "this image does not resemble the brain
   MRI distribution", which a 7.29% FPR does not support. It now reports the score, the threshold,
   *and the false-positive rate read from `ood.json`* (`OODDetector.false_positive_rate`, derived
   from `selection_tier`, so the quoted number cannot drift from the measured one) and frames the
   flag as a prompt to check the input rather than proof it is invalid.
2. **`/api/report/pdf` was working by accident.** It was mounted *after* `/api/report`, and
   `app.use` matches by prefix, so the broader mount was consulted first and only fell through
   because its router happened to have no matching path. Adding any catch-all to the report router
   would have silently broken PDF downloads. The mounts are now ordered most-specific-first.

---

## 5. What changed (tasks 1–4)

### New files

| Path | Purpose |
|------|---------|
| `backend/fastapi/app/services/model_cache.py` | Process-wide model cache + device selection |
| `backend/fastapi/scripts/generate_test_probs.py` | Regenerates base-model probability matrices |

### Modified files

**`app/services/model_cache.py`** (new)
- `get_device()` — resolves `torch.device` once, honours `settings.FORCE_CPU`, logs the reason.
- `ModelCache` — thread-safe (RLock, double-checked locking) cache keyed `"<module_id>:<arch>"`.
  `get(module_id, arch, timm_name, weights_path, num_classes)`. Build = `timm.create_model` →
  `load_state_dict` → `eval()` → `.to(device)` → freeze params.
- `preload(specs)` — tolerates per-model failure, returns `{"loaded": [...], "failed": {...}}`.
- `gradcam_ready(model)` — re-enables parameter grads, since cached models are frozen and Grad-CAM
  needs a backward pass.
- Singleton `model_cache`.

**`app/config.py`**
- `env_file` is now a **tuple**: repo-root `.env`, then `backend/fastapi/.env`.
- Added `FORCE_CPU`, `PRELOAD_MODULES` (default `"brain_mri"`) + `preload_modules` computed field,
  `CALIBRATION_DIR` (`../../models/calibration`), `CONFORMAL_ALPHA` (`0.10`), `GROQ_API`,
  `GROQ_MODEL`, `extra="ignore"`.

**`app/services/ensemble.py`** (rewritten)
- `load_ensemble(models_dir, config_filename, meta_filename)` is now generic (chest uses it too)
  and **validates `meta.n_features_in_ == len(model_order) × num_classes`**, raising `ValueError`
  on mismatch. This closes the silent feature-order hazard.
- `get_base_model(...)` routes through the cache. `_load_single_model` kept as a back-compat shim.
- `run_ensemble_inference(..., module_id, timm_names)` uses the cache, `torch.inference_mode()`
  and the selected device. New return keys: `raw_meta_proba`, `base_probabilities`, `models_used`,
  `models_skipped`, `ensemble_active`.
- `_pick_agreeing_model` returns **`None`** when no base model agrees, instead of silently
  substituting one.

**`app/services/gradcam.py`**
- Imports `ClassifierOutputTarget`; library path now passes
  `targets=[ClassifierOutputTarget(idx)]` instead of `None`.
- `generate_gradcam(..., target_index=None)` — an explicit target overrides the individual model's
  argmax, so the heatmap explains the *ensemble's* class.
- Calls `gradcam_ready(model)` and moves the tensor to the inference device.

**`app/services/inference.py`** — `torch.inference_mode()`, tensor moved to device, `load_model`
does `.to(device)`.

**`app/modules/brain_mri/module.py`** and **`app/modules/chest_xray/module.py`** (both rewritten)
- Now **share** one ensemble code path. Chest previously duplicated ~80 lines because the shared
  helper hardcoded `num_classes=4`.
- Added `timm_names`, `base_models`, `weight_paths()`, `preload_specs()`.
- `gradcam()` reuses the **cached** agreeing model (no second load) and returns
  `explanation_model`, `explanation_faithful`, `models_used`, `models_skipped`.
- Chest's `except (FileNotFoundError, Exception)` catch-all is gone; only genuine absence and
  feature-width mismatch fall back.

**`app/engine/base_module.py`** — added non-abstract `preload_specs()` returning `[]`.

**`app/main.py`**
- Lifespan calls `get_device()`, preloads per `settings.preload_modules`.
- **No longer `sys.exit(1)`** when the brain fallback model fails — chest can still serve.

---

## 6. Architecture decisions (do not re-litigate)

| Decision | Rationale |
|----------|-----------|
| **Models cached for process lifetime** | Predictable latency beats low idle memory for a service. ~645 MB brain + ~135 MB chest. Controlled by `PRELOAD_MODULES`. |
| **LLM is an authoring aid, not a runtime oracle** | Output space is only 4 classes × 3 uncertainty bands = **12 narratives**. Pre-generate offline, human-review, commit as vetted text. Zero runtime compute, zero hallucination risk at serve time, and every clinical sentence is human-reviewed. |
| **No LangChain / LlamaIndex / vector DB** | 25–40 documents ≈ 500 chunks ≈ a 500×384 matrix ≈ 750 KB. Retrieval is one matrix multiply. 40 lines of numpy is more defensible in a viva than a framework you cannot debug. |
| **Never send images or patient identifiers to Groq** | Only structured JSON + retrieved passages leave the machine. Privacy argument for the thesis, and keeps payloads tiny. |
| **Conformal prediction instead of a hand-picked threshold** | Gives a distribution-free coverage guarantee and naturally produces `{Glioma, Meningioma}` "indeterminate" sets. Far stronger than "we chose 90%". |
| **Deploy the 82% chest model over the 85.68% one** | Different label spaces; ranking them is a category error. Normal/Pneumonia/TB answers a real clinical question; bacteria-vs-virus labels come from filename conventions. |
| **Keep VGG-16 despite being weakest and 26× larger** | Meta-learner assigns it a 25.6% reliance share — second highest. Diversity, not accuracy, is its job. |
| **Calibrate through the serving preprocessing path** | Using a different transform to generate calibration probabilities would calibrate against a distribution never seen in production. |

---

## 7. Specifications for remaining work

### 7.1 Task 4 — probability matrices (running)

Script: `backend/fastapi/scripts/generate_test_probs.py`. Run from `backend/fastapi`:

```powershell
python scripts/generate_test_probs.py --batch-size 16          # full 1,600
python scripts/generate_test_probs.py --limit 8 --batch-size 8 # smoke test
```

Outputs into `models/calibration/`:

| File | Shape |
|------|-------|
| `BRAIN_MRI_EFFICIENTNET_test_probs.npy` | (1600, 4) float32 |
| `BRAIN_MRI_RESNET_test_probs.npy` | (1600, 4) |
| `BRAIN_MRI_DENSENET_test_probs.npy` | (1600, 4) |
| `BRAIN_MRI_VGG_test_probs.npy` | (1600, 4) |
| `test_labels.npy` | (1600,) int64 |
| `test_files.json` | ordered relative paths |
| `probs_manifest.json` | split, class order, model order, per-model top-1 accuracy, timings |

Observed throughput: efficientnet ~30 img/s, resnet ~25, densenet ~2.8–3.9, vgg ~8.2.
Full run ≈ 12 minutes.

### 7.2 Task 5 — calibration

**The data-split problem to fix.** Current stacking trains the meta-learner on **test-set**
probabilities via a 50/50 split, which burns the test set. Every new component (temperature,
abstention threshold, conformal quantile, OOD threshold) needs its own held-out data.

Replace the 50/50 split with a **three-way stratified split** of the 1,600 test rows:

```
meta-fit / calibration  (40%, 640)  → temperature T, conformal quantile q̂
final test              (60%, 960)  → reported numbers, touched once
```

The rigorous version is **out-of-fold stacking on the 5,600 training images** (5 folds), which
frees the entire 1,600-image test set. That requires retraining base models per fold, so it is a
**Colab/GPU task** — infeasible on this CPU box. Write the notebook; run it on Colab.

**Temperature scaling.** Probabilities, not logits, are stored. Since softmax is invariant to
additive constants, `z = log(p)` is a valid logit surrogate:

$$\hat{p}_i(T) = \frac{\exp(\log p_i / T)}{\sum_j \exp(\log p_j / T)}$$

Fit a single scalar $T$ by minimising NLL on the calibration split (`scipy.optimize.minimize_scalar`,
bounds ~(0.05, 10)).

**Two designs to compare — the comparison is itself a result:**

- **A:** temperature-scale each base model → refit the meta-learner on calibrated features
- **B:** leave bases raw → temperature-scale the meta-learner's output

Note the meta-learner is *already* a learned recalibrator, so B may well win on effort-to-benefit.

**Metrics.**

$$\text{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{n}\bigl|\text{acc}(B_m) - \text{conf}(B_m)\bigr| \quad (M=15)$$

$$\text{Brier} = \frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K}(\hat{p}_{ik} - y_{ik})^2$$

Also report NLL and a reliability diagram (before/after) via matplotlib.

**Artefact:** `models/calibration/calibration.json` —
`{"design": "A"|"B", "temperature": float, "per_model_temperature": {...}, "ece_before": float,
"ece_after": float, "brier_before": ..., "brier_after": ..., "nll_before": ..., "nll_after": ...,
"split": {...}, "generated": iso8601}`

### 7.3 Task 6 — conformal prediction

Split conformal with the LAC / "1 − true-class probability" nonconformity score:

1. Calibration scores: $s_i = 1 - \hat{p}_{y_i}(x_i)$
2. Quantile: $\hat{q} = \text{Quantile}\!\left(\{s_i\},\ \frac{\lceil (n+1)(1-\alpha)\rceil}{n}\right)$
3. Prediction set: $\mathcal{C}(x) = \{k : \hat{p}_k(x) \geq 1 - \hat{q}\}$

Guarantee: $P(y \in \mathcal{C}(X)) \geq 1 - \alpha$. Use `settings.CONFORMAL_ALPHA` (0.10 → 90%).

**Uncertainty bands** driven by set size (this replaces the arbitrary 90% cutoff):

| Set size | Band | UI behaviour |
|----------|------|--------------|
| 1 | `confident` | Report the finding |
| 2 | `borderline` | Report with explicit "indeterminate between X and Y" |
| ≥3 | `indeterminate` | Abstain, recommend professional review |

**Metrics:** empirical coverage on the final test split (should land near $1-\alpha$), mean set
size, per-class coverage, and a **risk–coverage curve with AURC**.

**Artefact:** `models/calibration/conformal.json` —
`{"alpha": 0.10, "q_hat": float, "threshold": 1 - q_hat, "empirical_coverage": float,
"mean_set_size": float, "coverage_by_class": {...}, "n_calibration": int, "n_test": int}`

**Serving:** load both artefacts at startup into `app.state`; add a `app/services/uncertainty.py`
exposing `apply_calibration(probs)` and `prediction_set(probs)`. If the artefacts are absent,
degrade gracefully to raw probabilities and report `calibrated: false` — never silently pretend.

### 7.4 Task 7 — MRI domain/quality validation (OOD)

Two independent layers; keep them separate because they fail differently.

**Quality checks (cheap, deterministic, run first):**
- Minimum resolution (reject < 64×64 before the 224×224 resize)
- Blur: variance of Laplacian below a threshold
- Near-empty / constant image: standard deviation below a threshold
- Extreme aspect ratio (the pipeline squash-resizes, so this distorts badly)

**Domain/OOD detection (statistical):**

Energy score over logits — outperforms max-softmax at negligible cost:

$$E(x) = -T\log\sum_{j}\exp(z_j/T)$$

Plus two signals that are **already computed and currently thrown away**:
- **Ensemble disagreement** — `base_predictions` / `base_probabilities` are returned by
  `run_ensemble_inference` but dropped by `PredictionResponse`. Exposing them is free.
- **Predictive entropy** $H = -\sum_k p_k \log p_k$
- **Conformal set size** — an all-classes set is a strong novelty signal

**Evaluation:** build a small OOD set (chest X-rays from `data/samples/ChestXray`, plus ~50
non-medical photos). Report **AUROC** and **FPR@95TPR**. This is the metric that makes the
"MRI validation" claim defensible rather than decorative.

**Artefact:** `models/calibration/ood.json` with thresholds and the evaluation numbers.

### 7.5 Task 8 — cited knowledge base

`app/services/report.py → TUMOR_INFO` is already a proto knowledge base. **Extend it, do not
start over.** Add a `sources` list of IDs to every entry.

Structure per class (Glioma, Meningioma, Pituitary, No Tumor):

```
overview · relevant_investigations · clinical_considerations
follow_up · warning_signs · treatment_information · sources[]
```

Proposed layout:

```
knowledge/
├── sources.json                 # id → {title, publisher, url, accessed, licence}
├── brain_mri/
│   ├── glioma.md
│   ├── meningioma.md
│   ├── pituitary.md
│   └── no_tumor.md
```

Curate **25–40 documents you have actually read** (WHO, NCI, NIH, radiology guidelines). Small and
vetted beats large and unvetted — it is the only way retrieval can be honestly evaluated.

**Wording rules (non-negotiable):**
- "Potential investigations to **discuss with a healthcare professional**" — never "you must undergo"
- **No** drug names or dosages. Treatment information stays categorical (observation, surgical,
  radiation-based, specialist-directed).
- Every clinical statement carries a source ID.

**Also fix:** `TUMOR_INFO.get(prediction, TUMOR_INFO["No Tumor"])` currently renders a reassuring
"No Tumor" narrative for *any* unrecognised label. Make unknown labels raise or return an explicit
"unknown finding" entry.

### 7.6 Task 9 — local RAG

```powershell
pip install sentence-transformers
```

1. **Chunk** ~300 words with ~50-word overlap, preserving `(source_id, section)` metadata
2. **Embed** with `sentence-transformers/all-MiniLM-L6-v2` (80 MB, 384-dim, ~20–50 ms/query CPU)
3. **Store** `knowledge/index.npz` — `vectors (N,384) float32`, `chunk_texts`, `chunk_meta`
4. **Retrieve** top-k by cosine similarity:

```python
sims = (vectors @ q) / (norms * np.linalg.norm(q))
top_k = np.argsort(-sims)[:k]
```

**Evaluation:** hand-label ~30 query→relevant-chunk pairs (an afternoon's work; this labelled set
*is* the contribution). Report **precision@5** and **recall@5**.

**Artefacts:** `knowledge/index.npz`, `knowledge/retrieval_eval.json`.

### 7.7 Task 10 — Groq narrative authoring (offline)

```powershell
pip install groq
```

Key: `settings.GROQ_API`. Model: `settings.GROQ_MODEL` (`llama-3.3-70b-versatile`).
Free tier is roughly 30 req/min with generous daily ceilings — the ~100–300 calls this project
needs are far inside it.

**Put the provider behind one seam** so it is swappable in one line:

```python
def generate(prompt: str, *, model: str | None = None) -> str:
    """Single seam for the LLM provider."""
```

**Script:** `backend/fastapi/scripts/author_narratives.py`
- Iterate the 12 combinations (4 classes × {confident, borderline, indeterminate})
- For each: retrieve top-k evidence → prompt constrained to that evidence → save draft
- Output `knowledge/narratives/<class>_<band>.json` with `text`, `citations[]`,
  `model`, `generated`, `reviewed: false`

**Prompt constraints:** answer only from supplied passages; cite the source ID for every claim;
if the evidence does not cover something, say so; no diagnosis, no prescriptions, no dosages.

**Then a human reviews each of the 12 and flips `reviewed: true`.** Runtime never calls Groq —
it selects the vetted narrative and fills numeric slots. That is the safety argument.

**Fallback that must keep working:** if narratives are missing or unreviewed, fall back to the
existing template report. Never block a prediction on the narrative layer.

### 7.8 Task 11 — PDF report

Use **reportlab** (already installed). `app/services/pdf_report.py`, sections in order:

```
NEURASIGHT — Brain MRI AI Analysis Report
Patient ID · Modality · Analysis date
AI Finding + calibrated confidence + uncertainty band
Prediction set (conformal, with the coverage guarantee stated)
Probability distribution table
Explainability: original + Grad-CAM, with the attention-not-segmentation caveat
Clinical interpretation (vetted narrative)
Potential further evaluation · Precautions · Follow-up · Warning signs
Treatment information (categorical, "not a prescription")
AI limitations disclaimer
Evidence / references (actual retrieved source IDs)
```

Must print the `explanation_faithful=False` caveat when no base model agreed, and the
`models_skipped` note when the ensemble ran degraded.

### 7.9 Task 12 — integration - **done — results in §4.8**

Original spec, kept for the record:

- **FastAPI:** extend `PredictionResponse` with `calibrated_probabilities`, `prediction_set`,
  `uncertainty_band`, `validation`, `models_used`, `models_skipped`, `ensemble_active`,
  `explanation_faithful`. Add `POST /report/pdf`. **Currently `PredictionResponse` silently drops
  `agreeing_model` and `base_predictions` — that is the per-model transparency being computed and
  thrown away.**
- **Express:** `Prediction` schema already supports both modules (Map probabilities + `module`).
  Add `uncertaintyBand`, `predictionSet`, `calibrated`, `validationPassed`.
- **Dashboard:** show the prediction set and band; render abstention prominently rather than as a
  footnote; add a PDF download button.

---

## 8. Known issues and traps

### 8.1 Three credentials were committed to a public history — remediated

An audit found three credentials in the repository's published history: a cloud/SSH private key
committed before `*.pem` was ignored, a dataset API token hardcoded inside a training notebook, and a
database password placed in a committed `.env.example` instead of the ignored `.env`.

**The published history was replaced. Rotation of all three is still outstanding** and is the
remaining action — see §15 of `presentation.md`. The specific commits and the shape of the secrets are
deliberately not recorded here: this file is published, and documenting where to look for a
recoverable secret is itself a disclosure.

`.gitignore` prevents future commits; it does not remove past ones. Rewriting history reduces further
exposure but cannot undo it, so **rotation is the actual fix** and is required regardless of what the
history looks like afterwards.

Controls now in place:

- All three `.env.example` files carry placeholders only; real values live in gitignored `.env` files.
- `.gitignore` blocks `*.pem`, `*.key`, `id_rsa*`, `.env`, `kaggle.json`, `*token*.json` and
  credential-shaped filenames, and excludes the notebook directories — including a stray
  space-named duplicate (`notebooks copy/`) that previously slipped past the `notebooks/` pattern
  and held a second copy of the token.
- A pre-publication scan over every committable file for private-key headers, provider key prefixes
  and connection strings. Currently returns no findings.

The Groq key was never exposed: it lives only in the gitignored repo-root `.env`, and the committed
example carries an empty value. Keep it that way.

### 8.2 `pytorch_grad_cam` is not installed

`GRADCAM_LIB_AVAILABLE = False`, so the **manual** Grad-CAM path runs. The manual path honours the
requested class index, so the `targets=None` bug is **dormant, not active** — it would only bite if
someone installs `grad-cam`. Task 3 hardened the library path preventatively. If you install the
library, re-verify heatmaps against the §4 baselines.

### 8.3 sklearn version mismatch

Pickles written with **1.6.1**, runtime has **1.5.1**. `InconsistentVersionWarning` fires on every
load. Unpickling across versions is explicitly unsupported and can alter probabilities — which
matters enormously now that calibration is being measured. **Pin `scikit-learn==1.6.1` before
trusting any calibration number.**

### 8.4 Two pre-existing test failures (not caused by this work) — **fixed**

`tests/test_preprocessor.py::test_output_values_in_range` and `::test_normalization_correctness`
assert a white image maps to all ones — i.e. they were written for a preprocessor **without**
ImageNet normalisation. Real output is $(1.0-0.485)/0.229 = 2.2489$ for R. **The tests were stale,
the code is correct.** Confirmed via `git status`: `preprocessor.py` and `tests/` were untouched.

**Fixed** — the two assertions now expect the per-channel normalised range
$[-\mu/\sigma, (1-\mu)/\sigma]$, derived from `IMAGENET_MEAN`/`IMAGENET_STD` imported from the
module rather than hard-coded. Making the tests pass by changing the preprocessor would have
introduced exactly the train/serve skew that §2 headline 2 proves is absent.
Current state: **11 passed, 0 failed.**

### 8.5 Git repository damage

Two stray refs (`origin/HEAD - Copy`, `origin/main - Copy`) were breaking `git log --all` with
`fatal: bad object`; both were exact duplicates and have been removed. **Still outstanding:**

```
broken link from tree 8f6bb96 to blob 7f2fde6  →  models/Brain_MRI_scan.pth (commit 394a61b)
```

A blob is missing locally. Day-to-day work is fine, but `git gc --aggressive` or a local clone will
fail. GitHub should still have the object — re-fetch or re-clone into a fresh directory to repair.

### 8.6 Chest module is unreachable from the dashboard — **fixed, verify it stays fixed**

`fastapiClient.js` never forwarded a `module` parameter, so every gateway request defaulted to
`brain_mri`. Fixed. Regression test: `POST /api/scan?module=chest_xray` must return
`module: "chest_xray"`.

### 8.7 Relative model paths resolve against CWD

`MODELS_DIR = "../../models"` only works when the process starts in `backend/fastapi`. Starting
uvicorn from the repo root breaks every weight path. Prefer package-anchored absolute paths
(`Path(__file__).resolve().parents[n]`) when touching this.

### 8.8 App logs are invisible — **fixed**

`logger.info(...)` calls in `main.py` and `model_cache.py` did not appear in uvicorn output because
nothing configured the root logger, so preload success was only observable via latency.
`logging.basicConfig(level=logging.INFO)` is now called at app startup (task 12).

### 8.9 Report risk level ignores confidence — **fixed**

`risk_level` was a static per-class attribute. Demonstrated failure: `tuberculosis-1122` produced
`risk_level: "Low"`, `"No immediate intervention required"` for an active TB case at 56.51%
confidence. Risk is now coupled to the uncertainty band (task 8) and escalation is computed inside
the module rather than left to the caller (task 11, §4.7). Regression test: `meningioma_sample.jpg`
must return `risk_level: Indeterminate`, `risk_escalated: true`.

### 8.11 Express route mounts are prefix-matched — **fixed, keep the order**

`app.use('/api/report', …)` matches `/api/report/pdf` too. The PDF route worked only because the
report router had no path that matched. `/api/report/pdf` is now mounted **before** `/api/report`;
adding a catch-all to the report router without that ordering would silently break PDF downloads.

### 8.12 The OOD detector fires on ~7% of valid brain MRI

Measured, not a defect: `fpr_at_95_tpr = 0.072917` for `js_divergence` on the near-OOD tier.
`frontend/samples/glioma_sample.jpg` is one such case — flagged at `0.2606` while being classified
correctly at 99.08%. Do not "fix" this by raising the threshold without re-reading §4.3: the
threshold was chosen on the hard tier on purpose. The user-facing wording quotes the false-positive
rate so the flag reads as a prompt, not a verdict.

### 8.10 PowerShell gotchas that have already wasted time

- Multi-line `function`/`foreach` blocks in `execute_pwsh` get mangled — keep commands single-line.
- Piping NUL-delimited git output fails; pass file lists via `--pathspec-from-file`.
- `git rm --cached` with space-containing filenames (`" - Copy"`) silently no-ops when splatted;
  use a pathspec file.
- `git check-ignore` skips tracked files unless you pass `--no-index`.

---

## 9. Runbook

### Start everything

```powershell
python run.py          # all three services
```

Or individually:

```powershell
# FastAPI  (MUST be from backend/fastapi — relative model paths)
cd backend\fastapi; python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Express
cd backend\express; node src/server.js

# Frontend
cd frontend; npm run dev
```

URLs: frontend `:3000`, dashboard `/dashboard.html`, Express `:5000/api/health`,
FastAPI docs `:8000/docs`.

### Port already in use

```powershell
$c = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
foreach ($x in $c) { Stop-Process -Id $x.OwningProcess -Force }
```

### Regression check (run after any inference change)

```powershell
foreach ($f in Get-ChildItem "data\samples\brainMRI" -File) { $r = curl.exe -s -X POST "http://127.0.0.1:8000/predict?module=brain_mri" -F "image=@data/samples/brainMRI/$($f.Name)" | ConvertFrom-Json; "$($f.Name) -> $($r.prediction) $($r.confidence)% band=$($r.uncertainty.uncertainty_band) set=[$($r.uncertainty.prediction_set -join ',')] ood=$($r.validation.is_ood)" }
```

Expected (also the task-12 baseline): `glioma` Glioma 99.08 confident `{Glioma}` ood=True ·
`meningioma` Meningioma 94.86 borderline `{Glioma,Meningioma}` · `notumor` No Tumor 97.98 borderline
`{Glioma,No Tumor}` · `pituitary` Pituitary 99.35 confident `{Pituitary}`.

Then the integration path, which is what the browser actually uses:

```powershell
curl.exe -s -X POST "http://localhost:3000/api/scan?module=brain_mri" -F "image=@frontend/samples/meningioma_sample.jpg" | ConvertFrom-Json | Select-Object saved, prediction, riskLevel, riskEscalated
curl.exe -s -X POST "http://localhost:3000/api/report/pdf?module=brain_mri" -F "image=@frontend/samples/glioma_sample.jpg" -o rep.pdf -w "http=%{http_code} bytes=%{size_download}"
```

`saved` must be `True` and `riskLevel` `Indeterminate`; the PDF must exceed ~100 KB (a ~10 KB PDF
means the images failed to embed, see §4.7). Then:

```powershell
cd backend\fastapi; python -m pytest tests/ -q     # expect 11 passed, 0 failed
```

### Regenerate probability matrices

```powershell
cd backend\fastapi; python scripts/generate_test_probs.py --batch-size 16
```

---

## 10. Build order with gates

Each gate is a number you must have before moving on. Do not skip 1–3 to reach the LLM faster —
that is how projects end up with impressive output and no defensible results.

| Step | Gate |
|------|------|
| 1. Load-once + device | Passed - brain predict < 1 s (achieved: 650 ms) |
| 2. Grad-CAM target class | Passed - heatmap class always matches reported class |
| 3. Pin sklearn 1.6.1 | no `InconsistentVersionWarning` |
| 4. Probability matrices | 4 `.npy` files + manifest committed |
| 5. Three-way split + calibration | ECE before/after + reliability diagrams |
| 6. Conformal | empirical coverage ≈ 1−α, mean set size reported |
| 7. OOD / validation | AUROC + FPR@95TPR on a real OOD set |
| 8. Grad-CAM faithfulness | deletion/insertion curves + randomisation sanity check |
| 9. Knowledge base | every statement carries a source ID |
| 10. RAG | precision@5 on the hand-labelled set |
| 11. Narratives | 12/12 human-reviewed, zero unsourced clinical claims |
| 12. PDF + integration | end-to-end scan produces a cited PDF |

### Out of scope (say "future work")

- Longitudinal / growth analysis — needs image registration and segmentation
- Previous-report upload / document AI — large scope, does not strengthen the core claim
- **Patient-context fusion** — the dataset has **no** age/sex/symptom labels, so a context-aware
  *model* cannot be trained on it. Patient context may only be reported as clinician-facing
  narrative. Claiming fusion that was not built is a factual error an examiner will catch.

### Statistical rigour to add before submission

- Wilson score intervals on all accuracies (at $n=800$, 96.75% carries roughly ±1.2 points)
- **McNemar's test** on paired predictions for ensemble vs best single model:
  $\chi^2 = (|b-c|-1)^2/(b+c)$. This is the question most likely to be asked about 96.06% → 96.75%.
- Mean ± std across folds once out-of-fold stacking is in place

---

*NeuraSight implementation plan. Keep §2, §4 and §8 current — they are what a fresh session reads
first.*
