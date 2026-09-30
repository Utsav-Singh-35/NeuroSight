const mongoose = require('mongoose');

// All labels either module can emit. Kept as a flat list rather than a
// per-module enum so a single collection can hold both scan types.
const ALL_LABELS = [
  // brain_mri
  'Glioma',
  'Meningioma',
  'No Tumor',
  'Pituitary',
  // chest_xray
  'Normal',
  'Pneumonia',
  'Tuberculosis',
];

const predictionSchema = new mongoose.Schema(
  {
    filename: {
      type: String,
      required: true,
      maxlength: 255,
      trim: true,
    },

    // Which AI module produced this result. Previously absent, which meant
    // stored scans could not be attributed to brain vs chest.
    module: {
      type: String,
      required: true,
      enum: ['brain_mri', 'chest_xray'],
      default: 'brain_mri',
      index: true,
    },

    prediction: {
      type: String,
      required: true,
      enum: ALL_LABELS,
    },

    confidence: {
      type: Number,
      required: true,
      min: 0,
      max: 100,
    },

    // A Map keeps this module-agnostic: 4 keys for brain, 3 for chest,
    // without the schema having to know which.
    probabilities: {
      type: Map,
      of: Number,
      required: true,
    },

    // Report fields. Optional so a bare /api/predict call still persists.
    riskLevel: {
      type: String,
      // 'Indeterminate' is required: the report generator escalates to it
      // whenever the conformal band is borderline/indeterminate. Omitting it
      // meant every uncertain scan - precisely the ones worth reviewing -
      // failed validation and was silently dropped from history.
      enum: ['Low', 'Medium', 'High', 'Indeterminate'],
    },
    description: { type: String, maxlength: 2000 },
    aiSummary: { type: String, maxlength: 4000 },
    recommendation: { type: String, maxlength: 4000 },

    // Base64 PNG Grad-CAM overlay (~70 KB each). Stored so the history view
    // can show what the model looked at without re-running inference.
    // `select: false` keeps it out of list queries by default.
    heatmap: {
      type: String,
      select: false,
    },

    // Convenience flag so the UI can highlight uncertain results without
    // duplicating the threshold in the client.
    lowConfidence: {
      type: Boolean,
      default: false,
      index: true,
    },

    // --- Uncertainty quantification -------------------------------------
    // The conformal prediction set is the honest answer: a set larger than one
    // means the model did not discriminate, and the UI must not present the
    // top-1 as a finding. Stored so history reflects what was actually shown.
    uncertaintyBand: {
      type: String,
      enum: ['confident', 'borderline', 'indeterminate'],
      index: true,
    },
    predictionSet: {
      type: [String],
      default: undefined,
    },
    coverageGuarantee: { type: Number, min: 0, max: 1 },
    calibrated: { type: Boolean, default: false },

    // True when uncertainty overrode the class's default risk tier. Worth
    // querying on: these are the cases a reviewer should look at first.
    riskEscalated: { type: Boolean, default: false, index: true },

    // --- Input validation ------------------------------------------------
    validationPassed: { type: Boolean, default: true },
    isOod: { type: Boolean, default: false, index: true },
    oodScore: { type: Number },

    // --- Ensemble transparency -------------------------------------------
    ensembleActive: { type: Boolean, default: true },
    modelsSkipped: { type: [String], default: undefined },
    explanationFaithful: { type: Boolean },

    // --- Evidence provenance ---------------------------------------------
    evidenceGrounded: { type: Boolean, default: false },
    narrativeSource: { type: String },
  },
  {
    timestamps: { createdAt: true, updatedAt: false },
  }
);

// Confidence below this is treated as "needs human review". Matches the
// abstention threshold recommended in the project's analysis.
predictionSchema.statics.LOW_CONFIDENCE_THRESHOLD = 90;

// Compound index for the dashboard's default query (filter by module,
// newest first) and a plain one for the unfiltered feed.
predictionSchema.index({ module: 1, createdAt: -1 });
predictionSchema.index({ createdAt: -1 });

/**
 * Shape a document for API responses. Converts the probabilities Map into a
 * plain object so it serialises as JSON rather than as a Map.
 * @param {object} doc - A lean() document or hydrated doc
 * @param {boolean} [includeHeatmap=false] - Whether to include the base64 PNG
 * @returns {object} Serialisable record
 */
predictionSchema.statics.toApi = function toApi(doc, includeHeatmap = false) {
  if (!doc) return null;

  // lean() yields a plain object whose Map field is already an object;
  // a hydrated doc yields a real Map. Handle both.
  let probabilities = {};
  if (doc.probabilities instanceof Map) {
    probabilities = Object.fromEntries(doc.probabilities);
  } else if (doc.probabilities && typeof doc.probabilities === 'object') {
    probabilities = { ...doc.probabilities };
  }

  const out = {
    id: doc._id,
    filename: doc.filename,
    // Records written before the module field existed are brain MRI scans.
    module: doc.module || 'brain_mri',
    prediction: doc.prediction,
    confidence: doc.confidence,
    probabilities,
    riskLevel: doc.riskLevel,
    description: doc.description,
    aiSummary: doc.aiSummary,
    recommendation: doc.recommendation,
    lowConfidence: doc.lowConfidence,
    createdAt: doc.createdAt,

    uncertaintyBand: doc.uncertaintyBand,
    predictionSet: doc.predictionSet,
    coverageGuarantee: doc.coverageGuarantee,
    calibrated: doc.calibrated,
    riskEscalated: doc.riskEscalated,

    validationPassed: doc.validationPassed,
    isOod: doc.isOod,
    oodScore: doc.oodScore,

    ensembleActive: doc.ensembleActive,
    modelsSkipped: doc.modelsSkipped,
    explanationFaithful: doc.explanationFaithful,

    evidenceGrounded: doc.evidenceGrounded,
    narrativeSource: doc.narrativeSource,
  };

  if (includeHeatmap && doc.heatmap) {
    out.heatmap = doc.heatmap;
  }

  return out;
};

module.exports = mongoose.model('Prediction', predictionSchema);
module.exports.ALL_LABELS = ALL_LABELS;
