---
class: Pituitary
module: brain_mri
risk_tier: Medium
requires_clinical_review: true
sources: [nci-pituitary-pdq, nci-pituitary-diagnosis, ncbi-pituitary-adenoma-statpearls, ninds-brain-tumors]
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
