"""AI Report Generator for brain tumor classification results.

Generates clinical-style summary reports with prediction details,
risk assessment, AI reasoning summary, and recommendations.
"""

# Tumor information database
TUMOR_INFO = {
    "Glioma": {
        "risk": "High",
        "description": "Glioma is a type of tumor that originates in the glial cells of the brain. "
                       "It is the most common type of primary malignant brain tumor.",
        "characteristics": "The MRI demonstrates imaging features consistent with a Glioma tumor, "
                          "typically appearing as an irregularly shaped mass with infiltrative borders.",
        "recommendation": "Immediate consultation with a neuro-oncologist or neurosurgeon is strongly advised. "
                         "Further diagnostic workup including contrast-enhanced MRI and possibly biopsy "
                         "may be warranted for grading and treatment planning.",
    },
    "Meningioma": {
        "risk": "Medium",
        "description": "Meningioma is a tumor that arises from the meninges — the membranes that surround "
                       "the brain and spinal cord. Most meningiomas are benign (non-cancerous).",
        "characteristics": "The MRI shows features consistent with a Meningioma, typically appearing as a "
                          "well-circumscribed, extra-axial mass attached to the dural surface.",
        "recommendation": "Consultation with a neurologist or neurosurgeon is recommended. Many meningiomas "
                         "are benign and may be monitored with periodic imaging. Treatment options depend on "
                         "size, location, and growth rate.",
    },
    "Pituitary": {
        "risk": "Medium",
        "description": "Pituitary tumors (adenomas) develop in the pituitary gland at the base of the brain. "
                       "Most are benign and treatable.",
        "characteristics": "The MRI reveals features consistent with a Pituitary tumor, typically located in "
                          "the sellar region with possible suprasellar extension.",
        "recommendation": "Referral to an endocrinologist and neurosurgeon is recommended. Hormonal evaluation "
                         "should be performed. Treatment options include medication, surgery, or radiation "
                         "depending on tumor type and hormone activity.",
    },
    "No Tumor": {
        "risk": "Low",
        "description": "No evidence of tumor pathology detected in the MRI scan.",
        "characteristics": "The MRI appears within normal limits. No mass lesions, abnormal enhancement, "
                          "or space-occupying lesions were identified by the AI model.",
        "recommendation": "No immediate intervention required based on AI analysis. If clinical symptoms "
                         "persist, consult a neurologist for further evaluation. This AI result does not "
                         "replace a formal radiological report.",
    },
}


def generate_report(prediction: str, confidence: float, probabilities: dict) -> dict:
    """Generate a clinical-style AI report for a brain tumor prediction.

    Args:
        prediction: The predicted class label.
        confidence: Confidence percentage (0-100).
        probabilities: Per-class probability distribution.

    Returns:
        Dictionary containing:
            - prediction: class label
            - confidence: percentage
            - risk_level: High/Medium/Low
            - description: tumor type description
            - ai_summary: clinical-style explanation
            - recommendation: next steps
            - disclaimer: legal note
    """
    info = TUMOR_INFO.get(prediction, TUMOR_INFO["No Tumor"])

    # Build AI summary
    ai_summary = (
        f"{info['characteristics']} "
        f"The model classified this scan as '{prediction}' with {confidence:.1f}% confidence."
    )

    # Add secondary findings if confidence is not 100%
    if confidence < 99.0:
        # Find second-highest class
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_probs) > 1 and sorted_probs[1][1] > 1.0:
            second_class = sorted_probs[1][0]
            second_prob = sorted_probs[1][1]
            ai_summary += (
                f" A secondary consideration of {second_class} ({second_prob:.1f}%) "
                f"was noted but deemed less likely."
            )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "risk_level": info["risk"],
        "description": info["description"],
        "ai_summary": ai_summary,
        "recommendation": info["recommendation"],
        "disclaimer": (
            "This AI-generated report is for clinical decision support only. "
            "It must not replace professional medical diagnosis. All findings should be "
            "validated by a qualified radiologist or neurologist."
        ),
        "probabilities": probabilities,
    }
