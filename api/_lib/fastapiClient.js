const axios = require('axios');
const FormData = require('form-data');
const config = require('./config');

async function sendForPrediction(imageBuffer, filename) {
  const form = new FormData();
  form.append('image', imageBuffer, { filename });
  try {
    const res = await axios.post(`${config.FASTAPI_URL}/predict`, form, {
      headers: form.getHeaders(), timeout: config.FASTAPI_TIMEOUT,
    });
    return res.data;
  } catch (error) {
    if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT' || error.code === 'ECONNABORTED') {
      throw new Error('ML service is unavailable');
    }
    if (error.response) {
      const msg = error.response.data?.error || error.response.data?.detail || 'ML service error';
      const err = new Error(msg); err.status = error.response.status; throw err;
    }
    throw new Error('ML service is unavailable');
  }
}

async function sendForGradcam(imageBuffer, filename) {
  const form = new FormData();
  form.append('image', imageBuffer, { filename });
  try {
    const res = await axios.post(`${config.FASTAPI_URL}/gradcam`, form, {
      headers: form.getHeaders(), timeout: config.FASTAPI_TIMEOUT,
    });
    return res.data;
  } catch (error) {
    if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT' || error.code === 'ECONNABORTED') {
      throw new Error('ML service is unavailable');
    }
    if (error.response) {
      const msg = error.response.data?.error || error.response.data?.detail || 'ML service error';
      const err = new Error(msg); err.status = error.response.status; throw err;
    }
    throw new Error('ML service is unavailable');
  }
}

async function sendForReport(imageBuffer, filename) {
  const form = new FormData();
  form.append('image', imageBuffer, { filename });
  try {
    const res = await axios.post(`${config.FASTAPI_URL}/report`, form, {
      headers: form.getHeaders(), timeout: config.FASTAPI_TIMEOUT,
    });
    return res.data;
  } catch (error) {
    if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT' || error.code === 'ECONNABORTED') {
      throw new Error('ML service is unavailable');
    }
    if (error.response) {
      const msg = error.response.data?.error || error.response.data?.detail || 'ML service error';
      const err = new Error(msg); err.status = error.response.status; throw err;
    }
    throw new Error('ML service is unavailable');
  }
}

async function checkHealth() {
  try {
    const res = await axios.get(`${config.FASTAPI_URL}/health`, { timeout: config.HEALTH_CHECK_TIMEOUT });
    return res.status === 200;
  } catch { return false; }
}

module.exports = { sendForPrediction, sendForGradcam, sendForReport, checkHealth };
