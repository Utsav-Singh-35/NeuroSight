const express = require('express');
const imageValidator = require('../middleware/imageValidator');
const fastapiClient = require('../services/fastapiClient');
const Prediction = require('../models/Prediction');

const router = express.Router();

// POST / — forward image to FastAPI for prediction, store result in MongoDB
router.post('/', imageValidator, async (req, res, next) => {
  try {
    // Forward image to FastAPI for classification
    let result;
    try {
      result = await fastapiClient.sendForPrediction(req.file.buffer, req.file.originalname);
    } catch (error) {
      if (error.message === 'ML service is unavailable') {
        return res.status(502).json({ error: 'ML service is unavailable' });
      }
      throw error;
    }

    // Store prediction in MongoDB
    try {
      const prediction = new Prediction({
        filename: req.file.originalname,
        prediction: result.prediction,
        confidence: result.confidence,
        probabilities: result.probabilities,
      });
      await prediction.save();
    } catch (error) {
      return res.status(500).json({ error: 'Could not save prediction record' });
    }

    // Return the classification result
    return res.status(200).json({
      prediction: result.prediction,
      confidence: result.confidence,
      probabilities: result.probabilities,
    });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
