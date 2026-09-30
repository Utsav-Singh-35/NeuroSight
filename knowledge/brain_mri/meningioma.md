---
class: Meningioma
module: brain_mri
risk_tier: Medium
requires_clinical_review: true
sources: [nci-meningioma, ncbi-meningioma-statpearls, icom-meningioma-consensus, who-cns5, ninds-brain-tumors]
---

## Overview

Meningioma arises from the meninges, the membrane layers covering the brain and spinal cord. It is
a primary central nervous system tumour, meaning it begins in the CNS rather than spreading there
from elsewhere. Meningioma is described as the most common primary brain tumour overall, while
higher-grade meningiomas are rare. [nci-meningioma]

Presentation varies widely: meningiomas may be found incidentally, or may produce neurological
deficits related to tumour location, size, growth rate and histological grade.
[ncbi-meningioma-statpearls]

Typing and grading follow the WHO CNS classification, 5th edition. [who-cns5]

## AI limitations for this class

Meningioma is typically extra-axial and dural-based, which gives it a characteristic appearance —
but the model is not performing localisation or segmentation, and a Grad-CAM heatmap indicates
where the network's attention fell, not where a tumour boundary lies.

This project's evaluation showed meningioma with high recall (0.99) but lower precision (0.90) at
the single-model baseline, meaning the class **over-claims**: some gliomas were being absorbed into
it. That is the mirror image of glioma's recall shortfall. A meningioma output therefore carries a
meaningful probability of being a misassigned glioma, which is exactly why the conformal prediction
layer frequently returns the two-class set {Glioma, Meningioma} rather than a single label.
[nci-meningioma]

## Relevant investigations

Investigations a clinician may consider discussing:

- Clinical and neurological examination
- Dedicated contrast-enhanced MRI using a brain-tumour protocol
- Neurosurgical or neurological specialist review where appropriate
- Histological grading where tissue is obtained, since grade influences management and follow-up
- Interval imaging to establish growth rate, which is frequently more informative than a single
  study

[nci-meningioma] [ncbi-meningioma-statpearls]

## Clinical considerations

- Many meningiomas are slow-growing, and incidental discovery is common.
- Because presentation depends on location, size, growth rate and grade, two meningiomas of similar
  appearance can warrant very different management. [ncbi-meningioma-statpearls]
- Growth rate over time is a key input and cannot be assessed from one image.
- A single-slice AI classification cannot distinguish grade.

## Follow-up considerations

- Specialist review to decide whether observation or intervention is appropriate
- Serial imaging where monitoring is the chosen approach, at intervals determined clinically
- Comparison with prior imaging where available
- Reassessment if new or progressive neurological symptoms develop

[nci-meningioma] [icom-meningioma-consensus]

## Warning signs

Symptoms that generally warrant prompt medical assessment:

- Sudden severe headache, especially if unlike previous headaches
- New seizure activity
- Loss of consciousness
- Sudden weakness, numbness or speech difficulty
- Significant or rapidly progressing visual change
- Confusion or marked change in mental state

General neurological red flags, applicable regardless of AI output. [ninds-brain-tumors]

## Treatment information

For clinician review. **Not a prescription, and no medicines or doses are specified.**

Management depends on tumour location, size, symptoms, growth characteristics, histological grade
and the patient's overall clinical condition. Broad categories described in the literature include:

- Observation with serial imaging, which is a recognised approach for selected asymptomatic or
  slow-growing lesions
- Surgical management
- Radiation-based approaches, including stereotactic techniques
- Symptom-directed and supportive care

Selection among these is a specialist decision made after diagnostic confirmation and complete
clinical evaluation. [nci-meningioma] [icom-meningioma-consensus]
