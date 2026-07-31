/**
 * MongoDB connection singleton.
 * Reuses connections across serverless invocations (Vercel) and across
 * requests in local Express mode.
 */
const mongoose = require('mongoose');
const config = require('./config');

let cached = global._mongooseConnection;

if (!cached) {
  cached = global._mongooseConnection = { conn: null, promise: null };
}

async function connectDB() {
  if (cached.conn) return cached.conn;

  if (!cached.promise) {
    cached.promise = mongoose.connect(config.MONGODB_URI).then((m) => m);
  }

  cached.conn = await cached.promise;
  return cached.conn;
}

module.exports = connectDB;
