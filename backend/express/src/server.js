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

const app = express();

// Configure CORS
app.use(cors({
  origin: config.FRONTEND_ORIGIN,
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Accept'],
  optionsSuccessStatus: 204,
}));

// HTTP method restriction middleware — reject disallowed methods before routes
app.use((req, res, next) => {
  const allowedMethods = ['GET', 'POST', 'OPTIONS'];
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
app.use('/api/report', reportRoutes);

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
