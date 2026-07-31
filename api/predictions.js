const mongoose = require('mongoose');
const connectDB = require('./_lib/mongodb');
const Prediction = require('./_lib/Prediction');

module.exports = async (req, res) => {
  await connectDB();

  // GET /api/predictions/:id — handle if query param or path param
  const id = req.query.id;
  if (id) {
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(404).json({ error: 'Prediction record not found' });
    }
    const doc = await Prediction.findById(id).lean();
    if (!doc) return res.status(404).json({ error: 'Prediction record not found' });
    return res.status(200).json({
      id: doc._id, filename: doc.filename, prediction: doc.prediction,
      confidence: doc.confidence, probabilities: doc.probabilities, createdAt: doc.createdAt,
    });
  }

  // GET /api/predictions — list
  if (req.method === 'GET') {
    const predictions = await Prediction.find().sort({ createdAt: -1 }).limit(100).lean();
    const result = predictions.map((doc) => ({
      id: doc._id, filename: doc.filename, prediction: doc.prediction,
      confidence: doc.confidence, probabilities: doc.probabilities, createdAt: doc.createdAt,
    }));
    return res.status(200).json(result);
  }

  res.status(405).json({ error: 'Method not allowed' });
};
