/**
 * Global error-handling middleware for Express.
 * Logs full error details and returns sanitized JSON responses.
 * Never exposes stack traces, file paths, or environment variables to clients.
 */
function errorHandler(err, req, res, next) {
  const timestamp = new Date().toISOString();
  const method = req.method;
  const path = req.path;
  const message = err.message || 'Unknown error';
  const stack = err.stack || '';

  // Log full error details to stdout
  console.error(
    `[${timestamp}] ${method} ${path} - Error: ${message}\n${stack}`
  );

  // Determine status code: use err.statusCode if set, otherwise 500
  const statusCode = err.statusCode && err.statusCode >= 400 && err.statusCode < 600
    ? err.statusCode
    : 500;

  // For 4xx errors, use the specific error message
  // For 5xx errors, ALWAYS return the generic message
  const responseMessage = statusCode < 500
    ? message
    : 'An internal error occurred';

  res.setHeader('Content-Type', 'application/json');
  res.status(statusCode).json({ error: responseMessage });
}

module.exports = errorHandler;
