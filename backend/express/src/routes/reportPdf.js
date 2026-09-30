const express = require('express');
const imageValidator = require('../middleware/imageValidator');
const fastapiClient = require('../services/fastapiClient');

const router = express.Router();

/**
 * POST /?module=…&patient_id=…&age=…&sex=…
 *
 * Streams a rendered PDF report straight through from the ML service.
 *
 * Nothing is persisted here: the PDF is a rendering of a result the /api/scan
 * route already stored, so writing it again would duplicate ~120 KB per download
 * for no benefit.
 */
router.post('/', imageValidator, async (req, res, next) => {
  const moduleId = fastapiClient.resolveModule(req.query.module);

  try {
    const { buffer, filename } = await fastapiClient.sendForPdf(
      req.file.buffer,
      req.file.originalname,
      moduleId,
      {
        patient_id: req.query.patient_id,
        age: req.query.age,
        sex: req.query.sex,
      }
    );

    res.status(200);
    res.set({
      'Content-Type': 'application/pdf',
      'Content-Disposition': `attachment; filename="${filename}"`,
      'Content-Length': String(buffer.length),
      // The report embeds patient-adjacent content; never let a proxy cache it.
      'Cache-Control': 'no-store',
    });
    return res.send(buffer);
  } catch (error) {
    if (error.message === 'ML service is unavailable') {
      return res.status(502).json({ error: 'ML service is unavailable' });
    }
    if (error.status && error.status < 500) {
      return res.status(error.status).json({ error: error.message });
    }
    return next(error);
  }
});

module.exports = router;
