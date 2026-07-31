/**
 * NeuraSight — Local Express Server
 * Mounts the same /api/* handlers used by Vercel serverless in production.
 * Run via: node server.js (or via python run.py)
 */
const express = require('express');
const cors = require('cors');
const config = require('./api/_lib/config');
const connectDB = require('./api/_lib/mongodb');

// Import unified route handlers
const healthHandler = require('./api/health');
const predictHandler = require('./api/predict');
const gradcamHandler = require('./api/gradcam');
const reportHandler = require('./api/report');
const predictionsHandler = require('./api/predictions');

const app = express();

// CORS
app.use(cors({
  origin: config.FRONTEND_ORIGIN,
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Accept'],
  optionsSuccessStatus: 204,
}));

// Method restriction
app.use((req, res, next) => {
  if (!['GET', 'POST', 'OPTIONS'].includes(req.method)) {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  next();
});

app.use(express.json());

// Mount the serverless-compatible handlers as Express routes
app.all('/api/health', healthHandler);
app.all('/api/predict', predictHandler);
app.all('/api/gradcam', gradcamHandler);
app.all('/api/report', reportHandler);
app.all('/api/predictions', predictionsHandler);
// For /api/predictions/:id — pass as query param
app.get('/api/predictions/:id', (req, res) => {
  req.query.id = req.params.id;
  predictionsHandler(req, res);
});

// Global error handler
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err.message);
  res.status(500).json({ error: 'An internal error occurred' });
});

// Connect to MongoDB and start
connectDB()
  .then(() => {
    console.log('Connected to MongoDB');
    app.listen(config.PORT, () => {
      console.log(`Express server running on port ${config.PORT}`);
    });
  })
  .catch((err) => {
    console.error('MongoDB connection error:', err.message);
    process.exit(1);
  });

module.exports = app;
