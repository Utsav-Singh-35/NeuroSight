const express = require('express');
const mongoose = require('mongoose');
const Prediction = require('../models/Prediction');

const router = express.Router();

const MAX_LIMIT = 100;
const DEFAULT_LIMIT = 20;

/**
 * Parse and clamp pagination parameters.
 * @param {object} query - req.query
 * @returns {{limit: number, skip: number, page: number}}
 */
function parsePaging(query) {
  const rawLimit = parseInt(query.limit, 10);
  const limit =
    Number.isFinite(rawLimit) && rawLimit > 0
      ? Math.min(rawLimit, MAX_LIMIT)
      : DEFAULT_LIMIT;

  const rawPage = parseInt(query.page, 10);
  const page = Number.isFinite(rawPage) && rawPage > 0 ? rawPage : 1;

  return { limit, skip: (page - 1) * limit, page };
}

/**
 * Build a Mongo filter from query parameters.
 * @param {object} query - req.query
 * @returns {object} Mongo filter document
 */
function buildFilter(query) {
  const filter = {};

  if (query.module && ['brain_mri', 'chest_xray'].includes(query.module)) {
    filter.module = query.module;
  }

  if (query.prediction) {
    filter.prediction = query.prediction;
  }

  if (query.lowConfidence === 'true') {
    filter.lowConfidence = true;
  }

  // Filter on the conformal uncertainty band. This is the clinically useful
  // axis: 'indeterminate' isolates the scans the model itself declined to
  // call, which is a stronger signal than a raw confidence cutoff.
  if (
    query.band &&
    ['confident', 'borderline', 'indeterminate'].includes(query.band)
  ) {
    filter.uncertaintyBand = query.band;
  }

  if (query.escalated === 'true') {
    filter.riskEscalated = true;
  }

  // Free-text match on filename. Escaped so user input cannot inject
  // regex metacharacters.
  if (query.q) {
    const escaped = String(query.q).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    filter.filename = { $regex: escaped, $options: 'i' };
  }

  return filter;
}

/**
 * GET /stats — aggregate counts for the dashboard summary cards.
 * Declared before /:id so "stats" is not parsed as an id.
 */
router.get('/stats', async (req, res, next) => {
  try {
    const [
      total,
      byModule,
      byPrediction,
      lowConfidenceCount,
      latest,
      byBand,
      escalatedCount,
    ] = await Promise.all([
        Prediction.countDocuments(),
        // $ifNull so records written before the module field existed are
        // counted as brain_mri rather than grouping under null.
        Prediction.aggregate([
          {
            $group: {
              _id: { $ifNull: ['$module', 'brain_mri'] },
              count: { $sum: 1 },
            },
          },
        ]),
        Prediction.aggregate([
          {
            $group: {
              _id: {
                module: { $ifNull: ['$module', 'brain_mri'] },
                prediction: '$prediction',
              },
              count: { $sum: 1 },
              avgConfidence: { $avg: '$confidence' },
            },
          },
          { $sort: { count: -1 } },
        ]),
        Prediction.countDocuments({ lowConfidence: true }),
        Prediction.findOne().sort({ createdAt: -1 }).select('createdAt').lean(),
        // Records with no band predate the calibration layer, or come from a
        // module without conformal artefacts. They group under 'unbanded'
        // rather than null so the dashboard can label them honestly.
        Prediction.aggregate([
          {
            $group: {
              _id: { $ifNull: ['$uncertaintyBand', 'unbanded'] },
              count: { $sum: 1 },
            },
          },
        ]),
        Prediction.countDocuments({ riskEscalated: true }),
      ]);

    res.status(200).json({
      total,
      lowConfidenceCount,
      lowConfidenceThreshold: Prediction.LOW_CONFIDENCE_THRESHOLD,
      lastScanAt: latest ? latest.createdAt : null,
      escalatedCount,
      byBand: byBand.reduce((acc, row) => {
        acc[row._id] = row.count;
        return acc;
      }, {}),
      byModule: byModule.reduce((acc, row) => {
        acc[row._id] = row.count;
        return acc;
      }, {}),
      byPrediction: byPrediction.map((row) => ({
        module: row._id.module,
        prediction: row._id.prediction,
        count: row.count,
        avgConfidence: Math.round(row.avgConfidence * 100) / 100,
      })),
    });
  } catch (err) {
    next(err);
  }
});

/**
 * GET / — paginated scan history, newest first.
 * Supports ?module=, ?prediction=, ?lowConfidence=true, ?band=, ?escalated=true,
 * ?q=, ?page=, ?limit=
 * Heatmaps are excluded here to keep the payload small.
 */
router.get('/', async (req, res, next) => {
  try {
    const { limit, skip, page } = parsePaging(req.query);
    const filter = buildFilter(req.query);

    const [docs, total] = await Promise.all([
      Prediction.find(filter)
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit)
        .lean(),
      Prediction.countDocuments(filter),
    ]);

    res.status(200).json({
      items: docs.map((d) => Prediction.toApi(d, false)),
      page,
      limit,
      total,
      totalPages: Math.max(1, Math.ceil(total / limit)),
      hasMore: skip + docs.length < total,
    });
  } catch (err) {
    next(err);
  }
});

/**
 * GET /:id — one record, including its Grad-CAM heatmap.
 */
router.get('/:id', async (req, res, next) => {
  try {
    const { id } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(404).json({ error: 'Prediction record not found' });
    }

    // heatmap is select:false on the schema, so ask for it explicitly.
    const doc = await Prediction.findById(id).select('+heatmap').lean();

    if (!doc) {
      return res.status(404).json({ error: 'Prediction record not found' });
    }

    res.status(200).json(Prediction.toApi(doc, true));
  } catch (err) {
    next(err);
  }
});

/**
 * DELETE /:id — remove a record from history.
 */
router.delete('/:id', async (req, res, next) => {
  try {
    const { id } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(404).json({ error: 'Prediction record not found' });
    }

    const doc = await Prediction.findByIdAndDelete(id).lean();

    if (!doc) {
      return res.status(404).json({ error: 'Prediction record not found' });
    }

    res.status(200).json({ deleted: true, id });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
