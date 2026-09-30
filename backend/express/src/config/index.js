require('dotenv').config();

const config = {
  PORT: process.env.PORT || 5000,
  MONGODB_URI: process.env.MONGODB_URI || 'mongodb://localhost:27017/neurasight',
  FASTAPI_URL: process.env.FASTAPI_URL || 'http://localhost:8000',
  FRONTEND_ORIGIN: process.env.FRONTEND_ORIGIN || 'http://localhost:3000',
  MAX_FILE_SIZE: parseInt(process.env.MAX_FILE_SIZE) || 10485760,
  // The brain MRI ensemble loads 4 checkpoints (~645 MB) per request on CPU.
  // /api/scan runs report + Grad-CAM concurrently, so 30s was not enough and
  // silently cost every brain scan its heatmap.
  FASTAPI_TIMEOUT: parseInt(process.env.FASTAPI_TIMEOUT) || 120000,
  HEALTH_CHECK_TIMEOUT: parseInt(process.env.HEALTH_CHECK_TIMEOUT) || 3000,
};

module.exports = config;
