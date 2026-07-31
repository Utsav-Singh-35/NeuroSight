/**
 * Image validation middleware using multer.
 * Works in both Express server mode and Vercel serverless mode.
 */
const multer = require('multer');
const config = require('./config');

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: config.MAX_FILE_SIZE },
}).single('image');

function imageValidator(req, res, next) {
  upload(req, res, (err) => {
    if (err) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({ error: 'File size exceeds 10 MB limit' });
      }
      return res.status(400).json({ error: err.message || 'File upload error' });
    }
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }
    if (req.file.size === 0) {
      return res.status(400).json({ error: 'Uploaded file is empty' });
    }
    const allowed = ['image/jpeg', 'image/png'];
    if (!allowed.includes(req.file.mimetype)) {
      return res.status(400).json({ error: 'File must be JPEG or PNG' });
    }
    next();
  });
}

module.exports = imageValidator;
