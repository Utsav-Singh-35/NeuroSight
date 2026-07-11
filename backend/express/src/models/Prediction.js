const mongoose = require('mongoose');

const predictionSchema = new mongoose.Schema({
  filename: {
    type: String,
    required: true,
    maxlength: 255
  },
  prediction: {
    type: String,
    required: true,
    enum: ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
  },
  confidence: {
    type: Number,
    required: true,
    min: 0,
    max: 100
  },
  probabilities: {
    Glioma: { type: Number, required: true, min: 0, max: 100 },
    Meningioma: { type: Number, required: true, min: 0, max: 100 },
    'No Tumor': { type: Number, required: true, min: 0, max: 100 },
    Pituitary: { type: Number, required: true, min: 0, max: 100 }
  }
}, {
  timestamps: { createdAt: true, updatedAt: false }
});

predictionSchema.index({ createdAt: -1 });

module.exports = mongoose.model('Prediction', predictionSchema);
