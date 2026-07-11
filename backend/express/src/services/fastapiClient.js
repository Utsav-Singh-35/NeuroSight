const axios = require('axios');
const FormData = require('form-data');
const config = require('../config');

/**
 * Send an image to the FastAPI /predict endpoint for classification.
 * @param {Buffer} imageBuffer - The image file buffer
 * @param {string} filename - The original filename
 * @returns {Promise<object>} The prediction response data
 */
async function sendForPrediction(imageBuffer, filename) {
  const form = new FormData();
  form.append('image', imageBuffer, { filename });

  try {
    const response = await axios.post(
      `${config.FASTAPI_URL}/predict`,
      form,
      {
        headers: form.getHeaders(),
        timeout: config.FASTAPI_TIMEOUT,
      }
    );
    return response.data;
  } catch (error) {
    if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT' || error.code === 'ECONNABORTED') {
      throw new Error('ML service is unavailable');
    }
    if (error.response) {
      const message = error.response.data?.error || error.response.data?.detail || 'ML service error';
      const err = new Error(message);
      err.status = error.response.status;
      throw err;
    }
    throw new Error('ML service is unavailable');
  }
}

/**
 * Send an image to the FastAPI /gradcam endpoint for heatmap generation.
 * @param {Buffer} imageBuffer - The image file buffer
 * @param {string} filename - The original filename
 * @returns {Promise<object>} The Grad-CAM response data
 */
async function sendForGradcam(imageBuffer, filename) {
  const form = new FormData();
  form.append('image', imageBuffer, { filename });

  try {
    const response = await axios.post(
      `${config.FASTAPI_URL}/gradcam`,
      form,
      {
        headers: form.getHeaders(),
        timeout: config.FASTAPI_TIMEOUT,
      }
    );
    return response.data;
  } catch (error) {
    if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT' || error.code === 'ECONNABORTED') {
      throw new Error('ML service is unavailable');
    }
    if (error.response) {
      const message = error.response.data?.error || error.response.data?.detail || 'ML service error';
      const err = new Error(message);
      err.status = error.response.status;
      throw err;
    }
    throw new Error('ML service is unavailable');
  }
}

/**
 * Check the health of the FastAPI service.
 * @returns {Promise<boolean>} True if the service is healthy, false otherwise
 */
async function checkHealth() {
  try {
    const response = await axios.get(
      `${config.FASTAPI_URL}/health`,
      { timeout: config.HEALTH_CHECK_TIMEOUT }
    );
    return response.status === 200;
  } catch (error) {
    return false;
  }
}

module.exports = {
  sendForPrediction,
  sendForGradcam,
  checkHealth,
};
