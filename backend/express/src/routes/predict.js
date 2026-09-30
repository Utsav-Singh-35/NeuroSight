const express = require('express');
const imageValidator = require('../middleware/imageValidator');
const fastapiClient = require('../services/fastapiClient');
const Prediction = require('../models/Prediction');

const router = express.Router();

// POST /?module=… — classify an image, store the result, return it
router.post('/', imageValidator, async (req, res, next) => {
  const moduleId = fastapiClient.resolveModule(req.query.module);

  try {
    let result;
    try {
      result = await fastapiClient.sendForPrediction(
        req.file.buffer,
        req.file.originalname,
        moduleId
      );
    } catch (error) {
      if (error.message === 'ML service is unavailable') {
        return res.status(502).json({ error: 'ML service is unavailable' });
      }
      throw error;
    }

    const lowConfidence =
      result.confidence < Prediction.LOW_CONFIDENCE_THRESHOLD;

    try {
      const prediction = new Prediction({
        filename: req.file.originalname,
        module: moduleId,
        prediction: result.prediction,
        confidence: result.confidence,
        probabilities: result.probabilities,
        lowConfidence,
      });
      await prediction.save();
    } catch (error) {
      return res.status(500).json({ error: 'Could not save prediction record' });
    }

    return res.status(200).json({
      prediction: result.prediction,
      confidence: result.confidence,
      probabilities: result.probabilities,
      module: moduleId,
      lowConfidence,
    });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
