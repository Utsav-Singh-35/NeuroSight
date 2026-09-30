const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const config = require('./config/index.js');
const requestLogger = require('./middleware/requestLogger.js');
const errorHandler = require('./middleware/errorHandler.js');
const healthRoutes = require('./routes/health.js');
const predictRoutes = require('./routes/predict.js');
const gradcamRoutes = require('./routes/gradcam.js');
const predictionsRoutes = require('./routes/predictions.js');
const reportRoutes = require('./routes/report.js');
const scanRoutes = require('./routes/scan.js');
const reportPdfRoutes = require('./routes/reportPdf.js');

const app = express();

// Configure CORS
app.use(cors({
  origin: config.FRONTEND_ORIGIN,
  methods: ['GET', 'POST', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Accept'],
  optionsSuccessStatus: 204,
}));

// HTTP method restriction middleware — reject disallowed methods before routes.
// DELETE is permitted so scan history records can be removed.
app.use((req, res, next) => {
  const allowedMethods = ['GET', 'POST', 'DELETE', 'OPTIONS'];
  if (!allowedMethods.includes(req.method)) {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  next();
});

// Parse JSON bodies
app.use(express.json());

// Request logging
app.use(requestLogger);

// Mount routes
app.use('/api/health', healthRoutes);
app.use('/api/predict', predictRoutes);
app.use('/api/gradcam', gradcamRoutes);
app.use('/api/predictions', predictionsRoutes);
// Mounted before '/api/report' on purpose: app.use matches by prefix, so the
// broader mount would otherwise be consulted first for /api/report/pdf and only
// fall through by accident (its router has no matching path).
app.use('/api/report/pdf', reportPdfRoutes);
app.use('/api/report', reportRoutes);
// Combined report + Grad-CAM in one call, persisted for the dashboard history
app.use('/api/scan', scanRoutes);

// Global error handler (must be last)
app.use(errorHandler);

// Connect to MongoDB and start server
mongoose.connect(config.MONGODB_URI)
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
