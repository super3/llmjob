const WaitlistService = require('../services/waitlistService');

// POST /api/waitlist — join the managed-mining waitlist (no auth).
//
// Whitelisted field by field, like the miner ping. `website` is a honeypot: the
// form hides it from people, so only a bot fills it in. Such a request gets the
// same success response as a real signup (nothing to learn from probing it) but
// is never stored.
async function joinWaitlist(req, res) {
  try {
    const { email, gpus, note, source, website } = req.body || {};
    if (website) return res.json({ success: true });

    const service = new WaitlistService(req.app.locals.db);
    const result = await service.join({ email, gpus, note, source });
    if (result.error) {
      return res.status(400).json({ error: result.error });
    }
    res.json(result);
  } catch (error) {
    console.error('Waitlist signup error:', error);
    res.status(500).json({ error: 'Failed to join the waitlist' });
  }
}

module.exports = { joinWaitlist };
