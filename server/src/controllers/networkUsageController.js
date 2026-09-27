const ChatUsageService = require('../services/chatUsageService');
const ApiKeyService = require('../services/apiKeyService');
const OpenRouterService = require('../services/openRouterService');

// GET /api/chat/usage — the network page's "tokens served" figure, plus how much
// of the free hosted-model budget is left.
//
// The path still says "chat" because the free web chat used to live beside it.
// The chat is gone, but the network page calls this path, and that page is also
// served from GitHub Pages, which deploys separately from this server, so the
// path stays where it is. The totals it reads still hold the chat's lifetime
// tokens; new tokens come from the hosted models the /v1 API serves.
class NetworkUsageController {
  constructor(opts = {}) {
    // Only the free budget is read from the OpenRouter config, so the cap
    // reported here is the one the /v1 gateway enforces.
    this.openRouter = new OpenRouterService(opts);
    // Services are built per-request from req.app.locals.db so the controller
    // can be registered before the DB pool connects. Injectable for tests.
    this._services = opts.services || null;
  }

  get freeBudget() { return this.openRouter.freeBudget; }

  services(req) {
    if (this._services) return this._services;
    const db = req.app.locals.db;
    return { chatUsage: new ChatUsageService(db), apiKeys: new ApiKeyService(db) };
  }

  async usage(req, res) {
    const svc = this.services(req);
    const totals = await svc.chatUsage.getTotals();
    // `totals` is every token we have bought from OpenRouter, and is what the
    // free-usage cap is measured against. Node-served API traffic is
    // deliberately absent: it costs the fleet's GPU time, not our credits, and
    // folding it in would burn the free budget on somebody else's hardware.
    //
    // `network` is the headline "tokens served" figure: everything that went
    // through LLMJob, counted once. `apiTokens` (billed per key) and `totals`
    // overlap by exactly the hosted-model slice the /v1 gateway records in both
    // places, so subtract it back out.
    const apiTokens = svc.apiKeys ? await svc.apiKeys.getTotalUsage() : 0;
    const capped = this.freeBudget > 0;
    res.json({
      totals,
      network: { apiTokens, totalTokens: totals.totalTokens + apiTokens - (totals.apiTotalTokens || 0) },
      freeBudget: capped ? this.freeBudget : null,
      remaining: capped ? Math.max(0, this.freeBudget - totals.totalTokens) : null,
      exhausted: capped && totals.totalTokens >= this.freeBudget
    });
  }
}

module.exports = NetworkUsageController;
