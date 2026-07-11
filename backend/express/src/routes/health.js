const express = require('express');
const fastapiClient = require('../services/fastapiClient');

const router = express.Router();

/**
 * GET /
 * Health check endpoint (mounted at /api/health).
 * Checks if the FastAPI service is reachable and returns service status.
 * Always returns 200 — fastapi_reachable indicates downstream availability.
 */
router.get('/', async (req, res) => {
  const fastapiReachable = await fastapiClient.checkHealth();

  res.status(200).json({
    status: 'healthy',
    fastapi_reachable: fastapiReachable,
  });
});

module.exports = router;
