const express = require('express');
const imageValidator = require('../middleware/imageValidator');
const fastapiClient = require('../services/fastapiClient');

const router = express.Router();

/**
 * POST /
 * Grad-CAM endpoint (mounted at /api/gradcam).
 * Validates the uploaded MRI image, forwards it to the FastAPI service
 * for Grad-CAM heatmap generation, and returns the result.
 */
router.post('/', imageValidator, async (req, res, next) => {
  try {
    const result = await fastapiClient.sendForGradcam(
      req.file.buffer,
      req.file.originalname
    );

    res.status(200).json(result);
  } catch (error) {
    if (error.message === 'ML service is unavailable') {
      return res.status(502).json({ error: 'ML service is unavailable' });
    }
    next(error);
  }
});

module.exports = router;
