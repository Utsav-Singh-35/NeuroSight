const express = require('express');
const imageValidator = require('../middleware/imageValidator');
const fastapiClient = require('../services/fastapiClient');

const router = express.Router();

// POST / — Generate full AI diagnostic report
router.post('/', imageValidator, async (req, res, next) => {
  try {
    const result = await fastapiClient.sendForReport(
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
