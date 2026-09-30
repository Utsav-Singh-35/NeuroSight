/**
 * Request Logger Middleware
 *
 * Logs each incoming request to stdout with:
 * - UTC timestamp (ISO 8601)
 * - HTTP method
 * - Endpoint path
 * - Response status code
 * - Response time in milliseconds
 *
 * Format: [ISO_TIMESTAMP] METHOD /path STATUS_CODE DURATIONms
 * Example: [2024-01-15T10:30:00.000Z] POST /api/predict 200 1523ms
 */

const requestLogger = (req, res, next) => {
  const startTime = Date.now();

  res.on('finish', () => {
    const duration = Date.now() - startTime;
    const timestamp = new Date().toISOString();
    const method = req.method;
    const path = req.originalUrl || req.url;
    const statusCode = res.statusCode;

    console.log(`[${timestamp}] ${method} ${path} ${statusCode} ${duration}ms`);
  });

  next();
};

module.exports = requestLogger;
