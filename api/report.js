const { sendForReport } = require('./_lib/fastapiClient');
const imageValidator = require('./_lib/imageValidator');

async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  imageValidator(req, res, async () => {
    try {
      const buffer = Buffer.isBuffer(req.file.buffer) ? req.file.buffer : Buffer.from(req.file.buffer);
      const result = await sendForReport(buffer, req.file.originalname || 'image.jpg');
      res.status(200).json(result);
    } catch (error) {
      if (error.message === 'ML service is unavailable') {
        return res.status(502).json({ error: 'ML service is unavailable' });
      }
      res.status(500).json({ error: error.message || 'An internal error occurred' });
    }
  });
}

module.exports = handler;
module.exports.config = { api: { bodyParser: false } };
