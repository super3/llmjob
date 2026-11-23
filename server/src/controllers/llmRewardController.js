const LlmRewardService = require('../services/llmRewardService');
const { owedCsv } = LlmRewardService;
const { handle } = require('./http');

// LLM rewards (#331). The owner's view is public and keyed by payout address,
// like a pool's address page. Listing what is owed and marking it paid are
// admin-only (requireAuth + requireAdmin in routes.js).

// GET /api/llm-rewards/:address — today, unpaid, paid so far, tokens today and
// the next payout day. The app's "LLM rewards" tile reads this.
const getSummary = handle(async (req, res) => {
  const service = new LlmRewardService(req.app.locals.db);
  res.json(await service.summary(req.params.address));
}, 'Failed to get LLM rewards');

// GET /api/admin/llm-rewards?from=YYYY-MM-DD&to=YYYY-MM-DD[&format=csv] —
// what each address is owed for the range, unpaid rows only.
const getOwed = handle(async (req, res) => {
  const service = new LlmRewardService(req.app.locals.db);
  const owed = await service.owed({ from: req.query.from, to: req.query.to });
  if (req.query.format === 'csv') {
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename="llm-rewards-${owed.from}-to-${owed.to}.csv"`);
    return res.send(owedCsv(owed));
  }
  res.json(owed);
}, 'Failed to list LLM rewards owed');

// POST /api/admin/llm-rewards/paid { from, to, tx?, address? } — record a
// payout for each payable address in the range, so it is never paid twice.
const markPaid = handle(async (req, res) => {
  const service = new LlmRewardService(req.app.locals.db);
  const { from, to, tx, address } = req.body || {};
  res.json(await service.markPaid({ from, to, tx, address }));
}, 'Failed to mark LLM rewards paid');

module.exports = { getSummary, getOwed, markPaid };
