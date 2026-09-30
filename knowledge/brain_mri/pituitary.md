---
class: Pituitary
module: brain_mri
risk_tier: Medium
requires_clinical_review: true
sources: [nci-pituitary-pdq, nci-pituitary-diagnosis, ncbi-pituitary-adenoma-statpearls, ninds-brain-tumors, who-cns5, ncg-india, icmr-ai-ethics, cdsco-md-rules, icmr-ncdir-ncrp, indian-cns-data, neurasight-system]
---

## Overview

Pituitary tumours arise in the pituitary gland at the base of the brain, within the sellar region.
The majority are adenomas. Two features make this class clinically distinct from the other tumour
classes in this module: the gland is endocrine, so tumours may disturb hormone secretion, and it
sits immediately below the optic chiasm, so an enlarging lesion can compress the visual pathway.

The most characteristic presenting features of pituitary adenomas are inappropriate pituitary
hormone secretion and visual field deficits. [nci-pituitary-pdq]

Diagnosis rests on pituitary-protocol MRI together with comprehensive endocrine testing, rather than
on imaging alone. [ncbi-pituitary-adenoma-statpearls]

## AI limitations for this class

Pituitary tumours occupy a stereotyped anatomical location, which gives a CNN a strong positional
cue. This was the best-performing class in this project's evaluation (precision 0.99, recall 1.00 at
the single-model baseline).

That strength is narrow, though. The model classifies *appearance*; it says nothing about the two
things that actually drive pituitary management — **hormonal activity** and **visual pathway
involvement**. Both require tests the model does not perform. A confident pituitary output is
therefore the *start* of an endocrine and ophthalmological assessment, not a substitute for one.
[nci-pituitary-pdq] [ncbi-pituitary-adenoma-statpearls]

## Relevant investigations

Investigations a clinician may consider discussing:

- Clinical examination including assessment for endocrine symptoms
- Comprehensive endocrine / pituitary hormone evaluation, since hormonal status is central to
  characterising these tumours [ncbi-pituitary-adenoma-statpearls]
- Formal visual field assessment. Visual field testing measures both central vision (looking
  straight ahead) and peripheral vision, with each eye tested separately while the other is
  covered. [nci-pituitary-diagnosis]
- Dedicated pituitary-protocol MRI, which differs from a general brain MRI
- Specialist review by endocrinology, neurosurgery and where relevant ophthalmology or
  neuro-ophthalmology

## Clinical considerations

- Function matters as much as size: a small hormone-secreting tumour can be more clinically
  significant than a larger non-secreting one.
- Visual field deficits reflect the anatomical relationship between the lesion and the optic chiasm.
- Visual fields alone give an incomplete picture and can mislead if read without clinical context;
  visual acuity may be preserved despite meaningful visual pathway dysfunction. [nci-pituitary-pdq]
- Incidental sellar findings occur and require clinical correlation.
- Management is genuinely multidisciplinary across endocrinology, neurosurgery and ophthalmology.

## Follow-up considerations

- Specialist review to determine appropriate monitoring
- Endocrine follow-up where hormonal abnormality is identified
- Repeat visual field assessment where the visual pathway is implicated
- Interval imaging as clinically determined
- Comparison with prior imaging where available

[nci-pituitary-pdq] [ncbi-pituitary-adenoma-statpearls]

## Warning signs

Symptoms that generally warrant prompt medical assessment:

- Sudden severe headache
- Sudden visual loss or rapidly worsening vision
- New double vision
- Sudden onset of severe headache with visual change and altered consciousness, which can indicate
  an acute pituitary event
- Confusion or marked change in mental state
- Symptoms suggesting acute adrenal insufficiency, such as severe weakness, vomiting, dehydration
  or collapse

[nci-pituitary-pdq] [ninds-brain-tumors]

## Treatment information

For clinician review. **Not a prescription, and no medicines or doses are specified.**

Management is individualised according to tumour type and functional status, and broad categories
described in the literature include medical therapy, transsphenoidal surgery and radiotherapy.
[ncbi-pituitary-adenoma-statpearls]

Which category applies depends strongly on whether the tumour secretes hormones and which hormone is
involved — some pituitary tumour types respond to medical therapy in a way that other CNS tumours do
not. Selection is a specialist decision made after endocrine and ophthalmological evaluation, not
from imaging. [nci-pituitary-pdq]

## Care pathway in India

This section is jurisdictional, not medical. Typing and grading above follow the WHO CNS
classification, which Indian tertiary centres use as well. [who-cns5] What differs by country is
where a patient is seen, what is locally available, and which authority governs a tool like this
one.

**Where specialist review happens.** India's National Cancer Grid is a network of major cancer
centres, research institutes and patient groups with the mandate of establishing uniform standards
for prevention, diagnosis and treatment, and it publishes resource-stratified guidelines so
evidence-based management stays implementable at centres with differing infrastructure. [ncg-india]

**This class needs more than imaging.** A pituitary finding is characteristically an endocrine as
well as a radiological matter: diagnosis relies on a pituitary-protocol MRI together with
comprehensive endocrine testing, and management is individualised depending on tumour type and
functional status. [ncbi-pituitary-adenoma-statpearls] Visual field assessment is also relevant,
covering central and peripheral vision with each eye tested separately. [nci-pituitary-diagnosis]
Pathway implication: specialist input is typically endocrinology and ophthalmology alongside
neurosurgery, so a referral limited to one of those may be incomplete. Hormone assays and formal
perimetry are widely available in Indian tertiary and many secondary centres, so this is usually a
question of arranging the right tests rather than of access.

**What this model cannot contribute here.** The classifier reports a class from a single image slice.
It does not measure hormone levels, assess visual fields, or distinguish functioning from
non-functioning lesions, and those are the findings that drive management for this class.
[neurasight-system]

**Epidemiological context.** Central nervous system tumours make up roughly 2% of all malignancies,
with the burden weighted towards younger and middle-aged patients. [indian-cns-data] Indian
incidence by age, sex and site is published through the National Cancer Registry Programme; specific
figures are deliberately not restated here, because they vary by registry and reporting year.
[icmr-ncdir-ncrp]

**Regulatory status of this tool.** Medical devices in India are regulated by CDSCO under the Drugs
and Cosmetics Act, 1940 and the Medical Devices Rules, 2017. [cdsco-md-rules] This system holds no
CDSCO registration or approval and is therefore **not** a medical device for use in India.
[neurasight-system]

**Ethical framework.** ICMR's ethical guidelines for AI in biomedical research and healthcare
require that AI health technologies undergo clinical and field validation before being applied to
patients, place ethics review with an ethics committee, and address accountability when a system
errs. [icmr-ai-ethics]

**What citing Indian authorities does not mean.** This model was trained and evaluated on a public
image corpus of unstated scanner and population provenance. It has **not** been validated on an
Indian patient cohort, and no multi-centre or multi-scanner validation has been performed. The
Indian sources above supply care-pathway and governance context only; they must not be read as
evidence of Indian clinical validation. [neurasight-system]
