const express = require('express');
const mongoose = require('mongoose');
const Prediction = require('../models/Prediction');

const router = express.Router();

// GET / — return up to 100 prediction records sorted by createdAt descending
router.get('/', async (req, res, next) => {
  try {
    const predictions = await Prediction.find()
      .sort({ createdAt: -1 })
      .limit(100)
      .lean();

    const result = predictions.map((doc) => ({
      id: doc._id,
      filename: doc.filename,
      prediction: doc.prediction,
      confidence: doc.confidence,
      probabilities: doc.probabilities,
      createdAt: doc.createdAt
    }));

    res.status(200).json(result);
  } catch (err) {
    next(err);
  }
});

// GET /:id — return a single prediction record or 404
router.get('/:id', async (req, res, next) => {
  try {
    const { id } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(404).json({ error: 'Prediction record not found' });
    }

    const doc = await Prediction.findById(id).lean();

    if (!doc) {
      return res.status(404).json({ error: 'Prediction record not found' });
    }

    res.status(200).json({
      id: doc._id,
      filename: doc.filename,
      prediction: doc.prediction,
      confidence: doc.confidence,
      probabilities: doc.probabilities,
      createdAt: doc.createdAt
    });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
