const nock = require('nock');
const { sendForPrediction, sendForGradcam, checkHealth } = require('./fastapiClient');
const config = require('../config');

describe('fastapiClient', () => {
  afterEach(() => {
    nock.cleanAll();
  });

  describe('sendForPrediction', () => {
    it('should return prediction data on success', async () => {
      const mockResponse = {
        prediction: 'Glioma',
        confidence: 95.42,
        probabilities: { Glioma: 95.42, Meningioma: 2.15, 'No Tumor': 1.30, Pituitary: 1.13 },
      };

      nock(config.FASTAPI_URL)
        .post('/predict')
        .reply(200, mockResponse);

      const imageBuffer = Buffer.from('fake-image-data');
      const result = await sendForPrediction(imageBuffer, 'test.jpg');

      expect(result).toEqual(mockResponse);
    });

    it('should throw "ML service is unavailable" on connection refused', async () => {
      nock(config.FASTAPI_URL)
        .post('/predict')
        .replyWithError({ code: 'ECONNREFUSED' });

      const imageBuffer = Buffer.from('fake-image-data');
      await expect(sendForPrediction(imageBuffer, 'test.jpg'))
        .rejects.toThrow('ML service is unavailable');
    });

    it('should throw "ML service is unavailable" on timeout', async () => {
      nock(config.FASTAPI_URL)
        .post('/predict')
        .replyWithError({ code: 'ECONNABORTED' });

      const imageBuffer = Buffer.from('fake-image-data');
      await expect(sendForPrediction(imageBuffer, 'test.jpg'))
        .rejects.toThrow('ML service is unavailable');
    });

    it('should re-throw FastAPI error message on non-2xx response', async () => {
      nock(config.FASTAPI_URL)
        .post('/predict')
        .reply(422, { error: 'Image could not be processed' });

      const imageBuffer = Buffer.from('fake-image-data');
      await expect(sendForPrediction(imageBuffer, 'test.jpg'))
        .rejects.toThrow('Image could not be processed');
    });
  });

  describe('sendForGradcam', () => {
    it('should return gradcam data on success', async () => {
      const mockResponse = {
        heatmap: 'base64encodedstring',
        prediction: 'Meningioma',
        confidence: 88.5,
      };

      nock(config.FASTAPI_URL)
        .post('/gradcam')
        .reply(200, mockResponse);

      const imageBuffer = Buffer.from('fake-image-data');
      const result = await sendForGradcam(imageBuffer, 'scan.png');

      expect(result).toEqual(mockResponse);
    });

    it('should throw "ML service is unavailable" on connection error', async () => {
      nock(config.FASTAPI_URL)
        .post('/gradcam')
        .replyWithError({ code: 'ECONNREFUSED' });

      const imageBuffer = Buffer.from('fake-image-data');
      await expect(sendForGradcam(imageBuffer, 'scan.png'))
        .rejects.toThrow('ML service is unavailable');
    });

    it('should re-throw FastAPI error on failure response', async () => {
      nock(config.FASTAPI_URL)
        .post('/gradcam')
        .reply(500, { error: 'Heatmap generation failed' });

      const imageBuffer = Buffer.from('fake-image-data');
      await expect(sendForGradcam(imageBuffer, 'scan.png'))
        .rejects.toThrow('Heatmap generation failed');
    });
  });

  describe('checkHealth', () => {
    it('should return true when FastAPI responds with 200', async () => {
      nock(config.FASTAPI_URL)
        .get('/health')
        .reply(200, { status: 'healthy', model_loaded: true });

      const result = await checkHealth();
      expect(result).toBe(true);
    });

    it('should return false when FastAPI responds with non-200', async () => {
      nock(config.FASTAPI_URL)
        .get('/health')
        .reply(503, { status: 'unhealthy', model_loaded: false });

      const result = await checkHealth();
      expect(result).toBe(false);
    });

    it('should return false when FastAPI is unreachable', async () => {
      nock(config.FASTAPI_URL)
        .get('/health')
        .replyWithError({ code: 'ECONNREFUSED' });

      const result = await checkHealth();
      expect(result).toBe(false);
    });
  });
});
