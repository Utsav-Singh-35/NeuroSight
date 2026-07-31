const { checkHealth } = require('./_lib/fastapiClient');

module.exports = async (req, res) => {
  if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });

  const fastapiReachable = await checkHealth();
  res.status(200).json({ status: 'healthy', fastapi_reachable: fastapiReachable });
};
