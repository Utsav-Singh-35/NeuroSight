---
class: No Tumor
module: brain_mri
risk_tier: Low
requires_clinical_review: true
sources: [nci-cns-pdq-patient, ninds-brain-tumors]
---

## Overview

A "No Tumor" output means the model did not find features it associates with the three tumour
classes it was trained on: glioma, meningioma and pituitary tumour.

That is a narrower statement than "this scan is normal", and the distinction is the most important
thing on this page. [neurasight-system]

## Interpretation — read this before reassuring anyone

This result is **not** a clean bill of health. Four specific limits apply:

1. **The model only knows four categories.** It was trained on glioma, meningioma, pituitary tumour
   and no-tumour images. Any other pathology — infarct, haemorrhage, demyelination, infection,
   abscess, metastasis, developmental abnormality — falls outside everything it has ever seen. Such a
   finding would most likely be pushed into the "No Tumor" class, because softmax must distribute
   probability across the four available options and has no way to express "something else".
2. **Absence of a detected tumour is not absence of disease.** The two are not the same claim.
3. **One 2D slice is not a study.** A lesion outside the imaged slice cannot be detected.
4. **Clinical correlation is required.** A normal-appearing scan in a symptomatic patient still
   requires investigation, and the scan result does not close that question.

In this project's evaluation, the No Tumor class achieved recall 1.00 with precision 0.92 at the
single-model baseline. Perfect recall is the desirable direction of error here — no diseased case in
the test set was cleared as healthy — but the 8% false-positive rate means some tumour-bearing scans
were being labelled No Tumor's complement, i.e. over-referral rather than missed disease. That
balance is the right one for a screening aid, and it only holds for the four trained categories.
[nci-cns-pdq-patient]

## Relevant investigations

If symptoms are present despite this result, matters a clinician may consider discussing:

- Clinical and neurological examination, which carries more weight than an AI screening output
- Formal radiological reporting of the full imaging study by a qualified radiologist
- Further or repeat imaging with appropriate sequences and full anatomical coverage
- Investigation of non-tumour causes of the presenting symptoms
- Specialist referral where symptoms persist or progress

[nci-cns-pdq-patient] [ninds-brain-tumors]

## Clinical considerations

- A negative AI result should never override a positive clinical finding.
- The system has no "unknown" class; out-of-distribution inputs are handled by the separate input
  validation layer, not by this label. [neurasight-system]
- If the input validation layer flagged the image, the classification should be treated as
  unreliable regardless of which class was returned. [neurasight-system]
- Persistent symptoms warrant investigation independent of this output. [ninds-brain-tumors]

## Follow-up considerations

- Clinical follow-up driven by symptoms rather than by this result
- Repeat assessment if symptoms are new, persistent or progressive
- Comparison with prior imaging where available
- Formal radiological review where any clinical concern remains

[nci-cns-pdq-patient] [neurasight-system]

## Warning signs

Symptoms that warrant prompt medical assessment **even with a No Tumor result**:

- Sudden severe headache, particularly if unlike previous headaches
- New seizure activity
- Loss of consciousness
- Sudden weakness, numbness or difficulty speaking
- Significant or rapidly progressing visual change
- Confusion or marked change in mental state
- Persistent vomiting with headache

A No Tumor classification does not reduce the urgency of any of these. [ninds-brain-tumors]

## Treatment information

Not applicable — no tumour was detected by the model, and no treatment information should be
inferred from this result. [neurasight-system]

If symptoms are present, the appropriate next step is clinical evaluation, not reassurance based on
an AI screening output. [nci-cns-pdq-patient]
