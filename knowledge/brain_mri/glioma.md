---
class: Glioma
module: brain_mri
risk_tier: High
requires_clinical_review: true
sources: [nci-cns-pdq-hp, nci-cns-pdq-patient, who-cns5, ninds-brain-tumors, ncg-india, icmr-ai-ethics, cdsco-md-rules, icmr-ncdir-ncrp, indian-cns-data, neurasight-system]
---

## Overview

Glioma is a broad term for tumours arising from glial cells — the supporting cells of the central
nervous system — rather than from neurons themselves. The group spans a wide range of biological
behaviour, from slow-growing lesions to aggressive high-grade tumours. Because behaviour varies so
much within the group, "glioma" alone is not a complete diagnosis: type and grade determine
prognosis and management. [nci-cns-pdq-hp] [who-cns5]

Current typing and grading follow the WHO Classification of Tumours of the Central Nervous System,
5th edition, which integrates histological appearance with molecular markers. Grading terminology
changed from the 4th edition, so older literature is not directly comparable. [who-cns5]

## AI limitations for this class

An image-only classifier assigns a broad tumour category. It does **not** establish grade, molecular
subtype, margins, or volume, and none of those can be inferred reliably from a single 2D slice. In
this project's own evaluation, glioma was the weakest class for recall — the single-model baseline
recovered 0.81 of true gliomas, with the missed cases tending to be absorbed into the meningioma
class. Both tumour types can enhance similarly on T1, so that confusion is radiologically
plausible rather than random. The stacking ensemble improved glioma recall to 0.90, but the class
remains the least reliable of the four. [nci-cns-pdq-hp]

Consequently a glioma output from this system should be read as "features that the model associates
with glioma are present", not as a confirmed tumour type.

## Relevant investigations

Investigations that a clinician may consider discussing, depending on presentation:

- Clinical and neurological examination to correlate imaging with symptoms and deficits
- Dedicated contrast-enhanced MRI using a brain-tumour protocol, which provides information a
  single unenhanced slice cannot
- Specialist review by neurology, neurosurgery or neuro-oncology
- Histopathological and molecular characterisation where tissue is obtained, since typing and
  grading under the current WHO classification depend on molecular markers
- Additional imaging or functional studies where surgical planning is being considered

[nci-cns-pdq-hp] [who-cns5]

## Clinical considerations

- Symptoms depend heavily on tumour location, size and growth rate rather than on tumour type alone.
- An imaging appearance suggestive of glioma does not establish grade, and grade is the dominant
  driver of prognosis and treatment intensity.
- Management is multidisciplinary; decisions are not made from imaging in isolation.
- Incidental findings occur. Correlation with the clinical picture is essential.

[nci-cns-pdq-hp] [ninds-brain-tumors]

## Follow-up considerations

- Specialist review to determine whether and when further imaging is appropriate
- Comparison against any prior imaging, where available, since change over time carries more
  information than a single study
- Reassessment if new or progressive neurological symptoms develop

[nci-cns-pdq-patient]

## Warning signs

Symptoms that generally warrant prompt medical assessment rather than waiting for a routine
appointment:

- Sudden severe headache, particularly if different in character from previous headaches
- New seizure activity
- Loss of consciousness
- Sudden weakness, numbness or difficulty speaking
- Significant or rapidly progressing visual change
- Confusion or a marked change in mental state
- Persistent vomiting with headache

These are general neurological red flags, not glioma-specific, and apply regardless of what any AI
system reports. [ninds-brain-tumors]

## Treatment information

For clinician review. **Not a prescription, and no medicines or doses are specified.**

Management depends on factors including tumour type and grade, location, size, growth
characteristics, the patient's symptoms, age and overall clinical condition. Broad categories
described in the literature include:

- Observation with interval imaging in selected situations
- Surgical management, including biopsy for diagnosis and resection where feasible
- Radiation-based approaches
- Systemic therapy
- Supportive and symptom-directed care

Which categories apply, in what sequence, and with what intent is a specialist decision made after
diagnostic confirmation and full clinical evaluation. [nci-cns-pdq-hp] [nci-cns-pdq-patient]

## Care pathway in India

This section is jurisdictional, not medical. Typing and grading above follow the WHO CNS
classification, which Indian tertiary centres use as well. [who-cns5] What differs by country is
where a patient is seen, what is locally available, and which authority governs a tool like this
one.

**Where specialist review happens.** India's National Cancer Grid is a network of major cancer
centres, research institutes and patient groups with the mandate of establishing uniform standards
for prevention, diagnosis and treatment, so comparable care is reachable without every patient
travelling to one institution. It publishes resource-stratified guidelines so that evidence-based
management stays implementable at centres with differing infrastructure. [ncg-india] A suspected
glioma is a neuro-oncology multidisciplinary matter — neurosurgery, neuro-radiology, pathology and
radiation oncology — and molecular characterisation influences both classification and management.
[who-cns5] [nci-cns-pdq-hp]

**Why the timeline matters here.** Glioma is the class this model is weakest on: recall was 0.81 at
the single-model baseline and 0.90 after stacking, so roughly one in ten gliomas in evaluation was
still assigned elsewhere. [neurasight-system] Central nervous system tumours make up roughly 2% of
all malignancies but fall disproportionately on younger and middle-aged patients, so a delay carries
a high cost in life-years. [indian-cns-data] A borderline or indeterminate output on a possible
glioma should accelerate specialist referral, not defer it.

**Epidemiological context.** Indian incidence by age, sex and site is published through the National
Cancer Registry Programme. This knowledge base deliberately does not restate specific figures,
because they vary by registry and reporting year and should be read from the source.
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
