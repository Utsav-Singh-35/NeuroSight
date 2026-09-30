# References

Every entry below is something this project **actually uses**, with a note saying where. Nothing is
listed aspirationally.

> **Why this file was rewritten.** The previous version described a TensorFlow/Keras project — this
> one is PyTorch — and contained incorrect citations, including an arXiv ID that points to an
> unrelated paper. A references file with wrong identifiers is worse than none, because it looks
> authoritative. Every arXiv ID below was verified against arxiv.org before being written here.

---

## 1. Datasets

### Used to train and evaluate the deployed models

| Dataset | Used for | Notes |
|---------|----------|-------|
| [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — Masoud Nickparvar, Kaggle | The `brain_mri` module | 7,023 images. 5,600 train / 1,600 test, **balanced at 400 per class**. Classes: glioma, meningioma, no tumor, pituitary. |
| [Chest X-ray Dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset) — muhammadrehan00, Kaggle | The deployed `chest_xray` module | Classes: Normal, Pneumonia, Tuberculosis. |
| [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kermany et al., Kaggle | An **archived** chest programme, not deployed | Different label space (bacteria / normal / virus), so its 85.68% is not comparable to the deployed 82.00%. |

Dataset URLs are also kept in [`data/dataset_links.md`](../data/dataset_links.md).

### Referenced only — did not contribute to any reported number

| Dataset | Why it is listed |
|---------|------------------|
| [Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) — Sartaj Bhuvaji | Candidate for the external validation described as future work. **Not** used for cross-validation, despite an earlier claim in this file to the contrary. |
| [BraTS 2020 Training Data](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) | A **segmentation** benchmark. This system performs classification only, so BraTS is relevant to future work, not to current results. |

Licence terms are those of the original Kaggle datasets. The 14 sample images committed under
`frontend/samples/` and `data/samples/` are excerpts included for demonstration and regression
testing only.

---

## 2. Methods, and where each one is implemented

This is the section that matters: each method below is in the codebase, and the citation is the source
of the technique.

### Ensembling

| Reference | Used in |
|-----------|---------|
| Wolpert, D. H. (1992). *Stacked Generalization.* **Neural Networks** 5(2), 241–259. | The core architecture: base-model probabilities become meta-features for a second-level learner. `app/services/ensemble.py` |
| Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474) | Rationale for ensemble disagreement carrying uncertainty information — the basis of the novelty detector that won the benchmark. |

### Probability calibration

| Reference | Used in |
|-----------|---------|
| Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML, PMLR 70. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) | Temperature scaling, and the Expected Calibration Error definition. `scripts/fit_calibration.py`, `app/services/uncertainty.py` |
| Nixon, J., Dusenberry, M., Zhang, L., Jerfel, G., Tran, D. (2019). *Measuring Calibration in Deep Learning.* CVPR Workshops. [arXiv:1904.01685](https://arxiv.org/abs/1904.01685) | Adaptive Calibration Error with equal-mass bins. This is why the project reports **both** conventional and adaptive ECE: with 454 of 480 samples in one equal-width bin, the conventional estimate is dominated by bins holding a handful of samples. |

### Conformal prediction

| Reference | Used in |
|-----------|---------|
| Sadinle, M., Lei, J., Wasserman, L. (2019). *Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.* **JASA** 114(525), 223–234. [arXiv:1609.00451](https://arxiv.org/abs/1609.00451) | The **LAC** conformal score the project uses, `s = 1 − p_true`. It minimises expected set size at a given coverage level. `app/services/uncertainty.py` |
| Angelopoulos, A. N., Bates, S. (2023). *Conformal Prediction: A Gentle Introduction.* **Foundations and Trends in ML** 16(4). [arXiv:2107.07511](https://arxiv.org/abs/2107.07511) | Split-conformal procedure and the finite-sample quantile level `⌈(n+1)(1−α)⌉ / n`, which is what produces the α floor of 0.0031 at n = 320. |
| Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer. | The exchangeability assumption underlying the distribution-free coverage guarantee. |

### Out-of-distribution detection

| Reference | Used in |
|-----------|---------|
| Hendrycks, D., Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR. [arXiv:1610.02136](https://arxiv.org/abs/1610.02136) | The maximum-softmax-probability baseline, one of the five detectors benchmarked in `scripts/fit_ood.py`. |
| Liu, W., Wang, X., Owens, J., Li, Y. (2020). *Energy-based Out-of-distribution Detection.* NeurIPS. [arXiv:2010.03759](https://arxiv.org/abs/2010.03759) | The energy score, `E(x) = −T·log Σ exp(z_j / T)`. Second-best detector in the benchmark (AUROC 0.9420). |

The deployed detector — mean pairwise Jensen–Shannon divergence between base-model distributions —
is a direct application of Jensen–Shannon divergence to the deep-ensemble disagreement idea above. It
beat both published baselines on the hard tier; see `models/calibration/ood.json` for the numbers.

### Explainability

| Reference | Used in |
|-----------|---------|
| Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391) | `app/services/gradcam.py`. Implemented directly rather than via a library, so the target class can be forced to the class the ensemble actually reported. |

### Evidence retrieval

| Reference | Used in |
|-----------|---------|
| Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) | `all-MiniLM-L6-v2` embeddings over the knowledge base. `app/services/retrieval.py`, `scripts/build_index.py` |

---

## 3. Model architectures

All four brain base learners and all three chest base learners come from
[`timm`](https://github.com/huggingface/pytorch-image-models) (Ross Wightman), pre-trained on
ImageNet and fine-tuned.

| Architecture | Reference |
|--------------|-----------|
| EfficientNet-B0 | Tan, M., Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946) |
| ResNet-50 | He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| DenseNet-121 | Huang, G., Liu, Z., van der Maaten, L., Weinberger, K. Q. (2017). *Densely Connected Convolutional Networks.* CVPR. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993) |
| VGG-16 | Simonyan, K., Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556) |

Background on transfer learning in this domain: Litjens, G. et al. (2017). *A Survey on Deep Learning
in Medical Image Analysis.* **Medical Image Analysis** 42, 60–88.
[arXiv:1702.05747](https://arxiv.org/abs/1702.05747)

---

## 4. Clinical and jurisdictional sources

The 17 sources behind the report text are **not** individually listed here, because they are recorded
machine-readably with their verification status in
[`knowledge/sources.json`](../knowledge/sources.json). That file is the single source of truth and is
what the build-time citation check validates against.

### 4.1 Clinical backbone — international and US

7 government clinical summaries (NCI, NINDS, NCBI Bookshelf), 3 peer-reviewed reference articles and
consensus reviews (StatPearls; *Neuro-Oncology* 2024 consensus review on meningioma), and 1
classification standard — the WHO Classification of Tumours of the Central Nervous System, 5th
edition, 2021. One self-reference is used for statements about this system's own behaviour.

### 4.2 Indian sources — care pathway, epidemiology and governance

Added because the clinical backbone above is jurisdiction-neutral for *classification* but
US-flavoured for everything else. WHO CNS5 remains the typing and grading authority; these supply
what genuinely differs by country.

| Source | Role in this project |
|--------|---------------------|
| [ICMR — Ethical Guidelines for Application of Artificial Intelligence in Biomedical Research and Healthcare](https://icmr.gov.in/ethical-guidelines-for-application-of-artificial-intelligence-in-biomedical-research-and-healthcare) (2023) | India's first ethical framework for AI in health. Sets patient-centric principles, the ethics review process, governance and consent, and requires clinical and field validation before patient application. **The framework this project is answerable to.** |
| [CDSCO — Medical Device and Diagnostics regulation](https://cdsco.gov.in/opencms/opencms/en/Medical-Device-Diagnostics/Medical-Device-Diagnostics/) | Devices in India are regulated under the Drugs and Cosmetics Act, 1940 and the Medical Devices Rules, 2017. Cited to state what this project is **not**: it holds no registration and is not a medical device in India. |
| [National Cancer Grid](https://www.ncgindia.org/) | Network of major Indian cancer centres with a mandate of uniform care standards; publishes resource-stratified guidelines so evidence-based management is implementable across differing infrastructure. Supplies the referral context. |
| [ICMR-NCDIR — National Cancer Registry Programme](https://www.ncdirindia.org/) | Indian cancer incidence by age, sex and site (ICD-10). Cited as *where to obtain* figures; per-class numbers are deliberately not restated, as they vary by registry and reporting year. |
| [Indian data on central nervous system tumours: a summary of published work](https://pmc.ncbi.nlm.nih.gov/articles/PMC4991137/) | Peer-reviewed review. CNS tumours are approximately 2% of all malignancies, with burden weighted towards younger and middle-aged patients. |

**Verification.** All five Indian URLs were confirmed to exist and be topically correct via search on
2026-09-30, on the same basis as the original set. Citing them does **not** imply the model was
validated on an Indian cohort — the `## Care pathway in India` sections say the opposite explicitly.

**Verification status, recorded honestly in that file:** all 12 URLs were confirmed to exist and be
topically correct. Full page text was **not** programmatically extracted — cancer.gov and
ninds.nih.gov block automated fetching — so the knowledge-base entries state well-established
clinical fundamentals and cite these sources for verification. They are **not verbatim extracts**, and
the artefact carries `clinical_review_required: true`.

---

## 5. Software

| Library | Version | Role |
|---------|---------|------|
| PyTorch | 2.7.1 | Deep-learning runtime |
| torchvision | 0.22.1 | Image transforms |
| timm | 1.0.24 | Pre-trained CNN architectures |
| scikit-learn | 1.6.1 | Logistic-regression meta-learners |
| sentence-transformers | 6.1.0 | MiniLM retrieval embeddings |
| OpenCV | 4.13 | Grad-CAM overlays, Laplacian blur metric |
| reportlab | 4.0.9 | Paginated PDF reports |
| matplotlib | 3.9.2 | Reliability diagrams |
| FastAPI / uvicorn | 0.109 / 0.27 | ML service |
| Express.js | 4.x | API gateway |
| MongoDB / mongoose | — | Scan history |
| React / Vite | 18 / 5 | Interface |
| Three.js / GSAP | — | Landing-page visuals |

Exact pins: [`backend/fastapi/requirements.txt`](../backend/fastapi/requirements.txt) and the two
`package.json` files.

---

## 6. Reporting and governance standards

| Standard | Relevance |
|----------|-----------|
| **CLAIM** — Checklist for Artificial Intelligence in Medical Imaging | The primary reporting checklist for a study of this kind. |
| **STARD 2015** — Standards for Reporting Diagnostic Accuracy | Requires sensitivity and specificity **with confidence intervals**. This project does not yet report intervals, which is its main outstanding methodological gap. |
| **TRIPOD+AI** | Reporting of prediction-model studies including AI. |
| [FDA — AI/ML-enabled medical devices](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices) | Positions this work correctly: a research prototype, **not** a cleared device. |
| [WHO — Ethics and Governance of AI for Health](https://www.who.int/publications/i/item/9789240029200) | Ethical framing for clinical decision support. |
| **[ICMR — Ethical Guidelines for AI in Biomedical Research and Healthcare](https://icmr.gov.in/ethical-guidelines-for-application-of-artificial-intelligence-in-biomedical-research-and-healthcare)** (2023) | **The governing framework in India**, and the one that actually applies here. Requires clinical and field validation before patient use — which this project has not done, and says so. |
| **[CDSCO — Medical Devices Rules, 2017](https://cdsco.gov.in/opencms/opencms/en/Medical-Device-Diagnostics/Medical-Device-Diagnostics/)** | The Indian regulatory instrument for medical devices. No registration held. |

---

## 7. Citing this project

```bibtex
@misc{neurasight2026,
  title  = {NeuraSight: Calibrated, Evidence-Grounded Medical Image Decision Support},
  year   = {2026},
  note   = {Research prototype. Brain MRI and chest X-ray classification with
            stacking ensembles, temperature calibration, split conformal
            prediction, out-of-distribution screening and cited evidence
            retrieval.},
  howpublished = {\url{https://github.com/Utsav-Singh-35/NeuroSight}}
}
```

Primary dataset:

```bibtex
@misc{braintumormri,
  title     = {Brain Tumor MRI Dataset},
  author    = {Nickparvar, Masoud},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset}}
}
```

The two methods most worth citing alongside this work, because they are what make the system more
than a classifier:

```bibtex
@inproceedings{guo2017calibration,
  title     = {On Calibration of Modern Neural Networks},
  author    = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  booktitle = {International Conference on Machine Learning},
  volume    = {70},
  pages     = {1321--1330},
  year      = {2017},
  publisher = {PMLR}
}

@article{sadinle2019least,
  title   = {Least Ambiguous Set-Valued Classifiers with Bounded Error Levels},
  author  = {Sadinle, Mauricio and Lei, Jing and Wasserman, Larry},
  journal = {Journal of the American Statistical Association},
  volume  = {114},
  number  = {525},
  pages   = {223--234},
  year    = {2019}
}
```

---

## Where the measured numbers live

This file cites *methods*. For *results*, and for the provenance of every figure:

| Document | Contents |
|----------|----------|
| [`presentation.md`](../presentation.md) | Full technical write-up: formulas, protocols, measured results, failure register |
| [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) | Build log, architecture decisions, known issues, runbook |
| [`brain_mri_model_results.md`](brain_mri_model_results.md) | Original per-class brain metrics |
| `models/calibration/*.json` | Every fitted number, with the inputs that produced it |
| `knowledge/retrieval_results.json` | Retrieval evaluation, embeddings vs TF-IDF |
