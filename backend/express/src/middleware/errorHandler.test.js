const errorHandler = require('./errorHandler');

describe('errorHandler middleware', () => {
  let req, res, next, consoleSpy;

  beforeEach(() => {
    req = { method: 'POST', path: '/api/predict' };
    res = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
      setHeader: jest.fn()
    };
    next = jest.fn();
    consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    consoleSpy.mockRestore();
  });

  test('returns 500 with generic message for errors without statusCode', () => {
    const err = new Error('database connection failed');

    errorHandler(err, req, res, next);

    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalledWith({ error: 'An internal error occurred' });
  });

  test('returns 500 with generic message for 5xx errors', () => {
    const err = new Error('Something broke internally');
    err.statusCode = 503;

    errorHandler(err, req, res, next);

    expect(res.status).toHaveBeenCalledWith(503);
    expect(res.json).toHaveBeenCalledWith({ error: 'An internal error occurred' });
  });

  test('returns specific error message for 4xx errors', () => {
    const err = new Error('File must be JPEG or PNG');
    err.statusCode = 400;

    errorHandler(err, req, res, next);

    expect(res.status).toHaveBeenCalledWith(400);
    expect(res.json).toHaveBeenCalledWith({ error: 'File must be JPEG or PNG' });
  });

  test('returns specific message for 404 errors', () => {
    const err = new Error('Prediction record not found');
    err.statusCode = 404;

    errorHandler(err, req, res, next);

    expect(res.status).toHaveBeenCalledWith(404);
    expect(res.json).toHaveBeenCalledWith({ error: 'Prediction record not found' });
  });

  test('returns specific message for 413 errors', () => {
    const err = new Error('File size exceeds 10 MB limit');
    err.statusCode = 413;

    errorHandler(err, req, res, next);

    expect(res.status).toHaveBeenCalledWith(413);
    expect(res.json).toHaveBeenCalledWith({ error: 'File size exceeds 10 MB limit' });
  });

  test('sets Content-Type to application/json', () => {
    const err = new Error('test error');

    errorHandler(err, req, res, next);

    expect(res.setHeader).toHaveBeenCalledWith('Content-Type', 'application/json');
  });

  test('logs full error details including timestamp, method, path, message, and stack', () => {
    const err = new Error('something went wrong');
    err.statusCode = 500;

    errorHandler(err, req, res, next);

    expect(consoleSpy).toHaveBeenCalledTimes(1);
    const logOutput = consoleSpy.mock.calls[0][0];
    expect(logOutput).toMatch(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/); // ISO timestamp
    expect(logOutput).toContain('POST');
    expect(logOutput).toContain('/api/predict');
    expect(logOutput).toContain('something went wrong');
    expect(logOutput).toContain('Error'); // stack trace contains 'Error'
  });

  test('never exposes stack traces in 5xx response body', () => {
    const err = new Error('secret path /usr/src/app leaked');
    err.statusCode = 500;

    errorHandler(err, req, res, next);

    const responseBody = res.json.mock.calls[0][0];
    expect(responseBody.error).toBe('An internal error occurred');
    expect(responseBody.error).not.toContain('/usr/src/app');
    expect(responseBody.error).not.toContain('stack');
  });

  test('defaults to 500 for invalid statusCode', () => {
    const err = new Error('weird error');
    err.statusCode = 999;

    errorHandler(err, req, res, next);

    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalledWith({ error: 'An internal error occurred' });
  });

  test('handles error without message', () => {
    const err = new Error();
    err.message = '';

    errorHandler(err, req, res, next);

    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalledWith({ error: 'An internal error occurred' });
  });
});
