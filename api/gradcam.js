const { sendForGradcam } = require('./_lib/fastapiClient');
const imageValidator = require('./_lib/imageValidator');

// Disable Vercel's body parser so multer can handle multipart
module.exports.config = { api: { bodyParser: false } };

module.exports = (req, res) => {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  imageValidator(req, res, async () => {
    try {
      const result = await sendForGradcam(req.file.buffer, req.file.originalname);
      res.status(200).json(result);
    } catch (error) {
      if (error.message === 'ML service is unavailable') {
        return res.status(502).json({ error: 'ML service is unavailable' });
      }
      res.status(500).json({ error: 'An internal error occurred' });
    }
  });
};
