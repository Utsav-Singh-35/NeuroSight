const express = require('express');
const imageValidator = require('../middleware/imageValidator');
const fastapiClient = require('../services/fastapiClient');
const Prediction = require('../models/Prediction');

const router = express.Router();

/**
 * POST /?module=brain_mri|chest_xray
 *
 * Runs a complete scan in one round trip: the report and the Grad-CAM heatmap
 * are requested from the ML service in parallel, the result is persisted, and
 * the combined payload is returned.
 *
 * Doing both here rather than in the browser means every analysis is recorded,
 * which is what makes the dashboard history possible. The two ML calls run
 * concurrently so wall-clock latency matches the slower of the two rather than
 * their sum.
 */
router.post('/', imageValidator, async (req, res, next) => {
  const moduleId = fastapiClient.resolveModule(req.query.module);

  try {
    // Grad-CAM is treated as best-effort: a heatmap failure must not lose a
    // valid classification, so it is allowed to settle as a rejection.
    const [reportOutcome, gradcamOutcome] = await Promise.allSettled([
      fastapiClient.sendForReport(
        req.file.buffer,
        req.file.originalname,
        moduleId
      ),
      fastapiClient.sendForGradcam(
        req.file.buffer,
        req.file.originalname,
        moduleId
      ),
    ]);

    if (reportOutcome.status === 'rejected') {
      const err = reportOutcome.reason;
      if (err.message === 'ML service is unavailable') {
        return res.status(502).json({ error: 'ML service is unavailable' });
      }
      return res
        .status(err.status && err.status < 500 ? err.status : 500)
        .json({ error: err.message || 'Analysis failed' });
    }

    const report = reportOutcome.value;
    const heatmap =
      gradcamOutcome.status === 'fulfilled'
        ? gradcamOutcome.value?.heatmap || null
        : null;

    const threshold = Prediction.LOW_CONFIDENCE_THRESHOLD;

    // Prefer the ML service's own uncertainty band over a local confidence
    // threshold. The band comes from conformal prediction and carries a coverage
    // guarantee; a raw confidence cutoff is a weaker proxy that only exists for
    // modules without calibration artefacts.
    const uncertainty = report.uncertainty || {};
    const validation = report.validation || {};
    const ensemble = report.ensemble || {};
    const band = uncertainty.uncertainty_band || null;

    const lowConfidence = band
      ? band !== 'confident'
      : report.confidence < threshold;

    // Persist. A storage failure should not discard a completed analysis, so
    // the result is still returned with a flag telling the client it is
    // not in history.
    let saved = null;
    let savedOk = true;
    try {
      const doc = new Prediction({
        filename: req.file.originalname,
        module: moduleId,
        prediction: report.prediction,
        confidence: report.confidence,
        probabilities: report.probabilities,
        riskLevel: report.risk_level,
        description: report.description,
        aiSummary: report.ai_summary,
        recommendation: report.recommendation,
        heatmap,
        lowConfidence,

        uncertaintyBand: band,
        predictionSet: uncertainty.prediction_set || undefined,
        coverageGuarantee: uncertainty.coverage_guarantee,
        calibrated: Boolean(uncertainty.calibrated),
        riskEscalated: Boolean(report.risk_escalated),

        validationPassed: validation.passed !== false,
        isOod: validation.is_ood === true,
        oodScore: validation.ood_score,

        ensembleActive: ensemble.ensemble_active !== false,
        modelsSkipped: (ensemble.models_skipped || []).length
          ? ensemble.models_skipped
          : undefined,
        explanationFaithful:
          gradcamOutcome.status === 'fulfilled'
            ? gradcamOutcome.value?.explanation_faithful
            : undefined,

        evidenceGrounded: Boolean(report.evidence_grounded),
        narrativeSource: report.narrative_source,
      });
      saved = await doc.save();
    } catch (storageError) {
      savedOk = false;
      // Log to the console, not req.log: no middleware attaches a logger to the
      // request, so the previous `req.log?.warn?.()` silently discarded every
      // storage failure and made `saved: false` impossible to diagnose.
      console.error(
        '[scan] Failed to persist scan:',
        storageError.name,
        storageError.message
      );
      if (storageError.errors) {
        for (const [field, detail] of Object.entries(storageError.errors)) {
          console.error(`[scan]   field "${field}": ${detail.message}`);
        }
      }
    }

    return res.status(200).json({
      id: saved ? saved._id : null,
      saved: savedOk,
      module: moduleId,
      filename: req.file.originalname,
      prediction: report.prediction,
      confidence: report.confidence,
      probabilities: report.probabilities,
      riskLevel: report.risk_level,
      description: report.description,
      aiSummary: report.ai_summary,
      recommendation: report.recommendation,
      disclaimer: report.disclaimer,
      heatmap,
      heatmapAvailable: Boolean(heatmap),
      lowConfidence,
      lowConfidenceThreshold: threshold,
      createdAt: saved ? saved.createdAt : new Date(),

      // Pass the decision-support detail through untouched so the dashboard can
      // render the prediction set, the abstention state and the caveats.
      riskEscalated: Boolean(report.risk_escalated),
      uncertainty,
      validation,
      ensemble,
      evidenceGrounded: Boolean(report.evidence_grounded),
      narrativeSource: report.narrative_source,
      clinicalNarrative: report.clinical_narrative || null,
      clinicalConsiderations: report.clinical_considerations || null,
      warningSigns: report.warning_signs || null,
      investigations: report.investigations || null,
      followUp: report.follow_up || null,
      treatmentInformation: report.treatment_information || null,
      indiaCarePathway: report.india_care_pathway || null,
      aiLimitations: report.ai_limitations || null,
      sources: report.sources || [],
      clinicalReviewRequired: report.clinical_review_required !== false,
      explanationFaithful:
        gradcamOutcome.status === 'fulfilled'
          ? gradcamOutcome.value?.explanation_faithful
          : null,
    });
  } catch (error) {
    return next(error);
  }
});

module.exports = router;
