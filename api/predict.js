const connectDB = require('./_lib/mongodb');
const Prediction = require('./_lib/Prediction');
const { sendForPrediction } = require('./_lib/fastapiClient');
const imageValidator = require('./_lib/imageValidator');

// Disable Vercel's body parser so multer can handle multipart
module.exports.config = { api: { bodyParser: false } };

module.exports = (req, res) => {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  imageValidator(req, res, async () => {
    try {
      await connectDB();
      let result;
      try {
        result = await sendForPrediction(req.file.buffer, req.file.originalname);
      } catch (error) {
        if (error.message === 'ML service is unavailable') {
          return res.status(502).json({ error: 'ML service is unavailable' });
        }
        throw error;
      }

      try {
        await Prediction.create({
          filename: req.file.originalname,
          prediction: result.prediction,
          confidence: result.confidence,
          probabilities: result.probabilities,
        });
      } catch {
        return res.status(500).json({ error: 'Could not save prediction record' });
      }

      res.status(200).json(result);
    } catch (error) {
      res.status(500).json({ error: 'An internal error occurred' });
    }
  });
};
