const axios = require('axios');
const FormData = require('form-data');
const config = require('../config');

// Modules the ML service exposes. Used to reject bad input before it
// reaches FastAPI, and to keep the gateway's contract explicit.
const VALID_MODULES = ['brain_mri', 'chest_xray'];
const DEFAULT_MODULE = 'brain_mri';

/**
 * Normalise a caller-supplied module name.
 * @param {string} [mod] - Requested module id
 * @returns {string} A valid module id, falling back to the default
 */
function resolveModule(mod) {
  return VALID_MODULES.includes(mod) ? mod : DEFAULT_MODULE;
}

/**
 * Translate an axios failure into a domain error with a stable message.
 * @param {Error} error - The axios error
 * @returns {Error} Normalised error
 */
function normaliseError(error) {
  if (
    error.code === 'ECONNREFUSED' ||
    error.code === 'ETIMEDOUT' ||
    error.code === 'ECONNABORTED'
  ) {
    return new Error('ML service is unavailable');
  }
  if (error.response) {
    const message =
      error.response.data?.error ||
      error.response.data?.detail ||
      'ML service error';
    const err = new Error(message);
    err.status = error.response.status;
    return err;
  }
  return new Error('ML service is unavailable');
}

/**
 * POST an image to a FastAPI inference endpoint.
 *
 * The `module` query parameter is what selects brain_mri vs chest_xray on
 * the ML service. Omitting it makes FastAPI silently fall back to its
 * brain_mri default, which previously made the chest module unreachable
 * through this gateway.
 *
 * @param {string} endpoint - FastAPI path, e.g. "/predict"
 * @param {Buffer} imageBuffer - Raw image bytes
 * @param {string} filename - Original filename
 * @param {string} [mod] - Module id
 * @returns {Promise<object>} Parsed response body
 */
async function postImage(endpoint, imageBuffer, filename, mod) {
  const form = new FormData();
  form.append('image', imageBuffer, { filename });

  try {
    const response = await axios.post(
      `${config.FASTAPI_URL}${endpoint}`,
      form,
      {
        headers: form.getHeaders(),
        params: { module: resolveModule(mod) },
        timeout: config.FASTAPI_TIMEOUT,
        // Grad-CAM returns a base64 PNG, so responses can be large.
        maxContentLength: 50 * 1024 * 1024,
        maxBodyLength: 50 * 1024 * 1024,
      }
    );
    return response.data;
  } catch (error) {
    throw normaliseError(error);
  }
}

/**
 * Classify an image.
 * @param {Buffer} imageBuffer
 * @param {string} filename
 * @param {string} [mod]
 * @returns {Promise<object>} {prediction, confidence, probabilities, module}
 */
function sendForPrediction(imageBuffer, filename, mod) {
  return postImage('/predict', imageBuffer, filename, mod);
}

/**
 * Generate a Grad-CAM heatmap.
 * @param {Buffer} imageBuffer
 * @param {string} filename
 * @param {string} [mod]
 * @returns {Promise<object>} {heatmap, prediction, confidence, module}
 */
function sendForGradcam(imageBuffer, filename, mod) {
  return postImage('/gradcam', imageBuffer, filename, mod);
}

/**
 * Generate a full rule-based clinical report.
 * @param {Buffer} imageBuffer
 * @param {string} filename
 * @param {string} [mod]
 * @returns {Promise<object>} Full report payload
 */
function sendForReport(imageBuffer, filename, mod) {
  return postImage('/report', imageBuffer, filename, mod);
}

/**
 * Request a rendered PDF report.
 *
 * Returned as a Buffer rather than parsed, because the payload is binary and
 * gets streamed straight through to the browser.
 *
 * @param {Buffer} imageBuffer
 * @param {string} filename
 * @param {string} [mod]
 * @param {object} [patient] - Optional header fields (patient_id, age, sex)
 * @returns {Promise<{buffer: Buffer, filename: string}>}
 */
async function sendForPdf(imageBuffer, filename, mod, patient = {}) {
  const form = new FormData();
  form.append('image', imageBuffer, { filename });

  const params = { module: resolveModule(mod) };
  if (patient.patient_id) params.patient_id = patient.patient_id;
  if (patient.age) params.age = patient.age;
  if (patient.sex) params.sex = patient.sex;

  try {
    const response = await axios.post(
      `${config.FASTAPI_URL}/report/pdf`,
      form,
      {
        headers: form.getHeaders(),
        params,
        timeout: config.FASTAPI_TIMEOUT,
        responseType: 'arraybuffer',
        maxContentLength: 50 * 1024 * 1024,
        maxBodyLength: 50 * 1024 * 1024,
      }
    );

    // Preserve the filename the ML service chose, if it sent one.
    const disposition = response.headers['content-disposition'] || '';
    const match = disposition.match(/filename="([^"]+)"/);

    return {
      buffer: Buffer.from(response.data),
      filename: match ? match[1] : 'neurasight-report.pdf',
    };
  } catch (error) {
    // An error body arrives as an arraybuffer too, so decode it before the
    // generic handler tries to read error.response.data as an object.
    if (error.response && error.response.data) {
      try {
        const text = Buffer.from(error.response.data).toString('utf8');
        const parsed = JSON.parse(text);
        const err = new Error(parsed.error || 'PDF generation failed');
        err.status = error.response.status;
        throw err;
      } catch (parseError) {
        if (parseError instanceof Error && parseError.message !== 'PDF generation failed') {
          // Fall through to the shared normaliser below.
        } else {
          throw parseError;
        }
      }
    }
    throw normaliseError(error);
  }
}

/**
 * Check ML service health.
 * @returns {Promise<boolean>}
 */
async function checkHealth() {
  try {
    const response = await axios.get(`${config.FASTAPI_URL}/health`, {
      timeout: config.HEALTH_CHECK_TIMEOUT,
    });
    return response.status === 200;
  } catch (error) {
    return false;
  }
}

/**
 * Fetch the ML service's module registry.
 * @returns {Promise<Array>} Module descriptors, or [] on failure
 */
async function getModules() {
  try {
    const response = await axios.get(`${config.FASTAPI_URL}/modules`, {
      timeout: config.HEALTH_CHECK_TIMEOUT,
    });
    return response.data?.modules || [];
  } catch (error) {
    return [];
  }
}

module.exports = {
  sendForPrediction,
  sendForGradcam,
  sendForReport,
  sendForPdf,
  checkHealth,
  getModules,
  resolveModule,
  VALID_MODULES,
  DEFAULT_MODULE,
};
