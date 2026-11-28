const crypto = require('crypto');
const { estimateTokens, joinContent } = require('../controllers/gatewayShared');

// LLM rewards (#331): track the LLM work nodes do for the server and work out
// what each payout address is owed for it.
//
// Track. completeJob calls recordJob once per finished job. Every job comes
// from the server itself (the /v1 gateway or the speed benchmark sweep), so a
// rig can't add rows by serving its own local traffic. A request whose API key
// belongs to the same account as the node that served it is skipped: that
// owner would be paying themselves.
//
// Reward. Founder's decision of 21 Nov: priced in dollars at OpenRouter's rate
// for the same model, paid in PRL at that day's price. The dollar amount is
// fixed when the row is written. The PRL amount is worked out per address per
// UTC day from that day's price, and capped at DAILY_CAP_PRL.
//
// Pay. The founder pays by hand. owed() lists what each address is owed for a
// range of days; markPaid() records the payout and stamps its id on the rows,
// so a paid row is never counted again.

// Dollars per 1M tokens, from the issue's 21 Nov comment. No caching
// discounts. Matched against the model name the node reports, in order.
// Gemma 4 E4B isn't on OpenRouter; its smallest Gemma 4 is the ceiling.
const RATES = [
  { match: /qwen3\.8-27b/i, label: 'Qwen 3.8 27B', inUsd: 0.06, outUsd: 1.96 },
  { match: /gemma-4-e4b/i, label: 'Gemma 4 E4B', inUsd: 0.0675, outUsd: 0.225 },
];

// The guardrails from the issue while this is new.
const DAILY_CAP_PRL = 5;
const MIN_PAYOUT_PRL = 1;

// Where the PRL price comes from: prlscan first, CoinGecko second. The same
// two sources the app and the website already read.
const PRICE_URLS = {
  prlscan: 'https://api.prlscan.com/v1/market/prl',
  coingecko: 'https://api.coingecko.com/api/v3/simple/price?ids=pearl-2&vs_currencies=usd',
};

const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
const ADDRESS_RE = /^prl1p[0-9a-z]{20,80}$/;

function rateFor(model) {
  const name = String(model || '');
  return RATES.find((r) => r.match.test(name)) || null;
}

// The UTC day a timestamp falls on, as YYYY-MM-DD.
function dayOf(ms) {
  return new Date(ms).toISOString().slice(0, 10);
}

function addDays(day, n) {
  return dayOf(Date.parse(day + 'T00:00:00Z') + n * 86400000);
}

// The next payout day: the coming Monday (UTC), or today if it is Monday.
function nextPayoutDay(ms) {
  const weekday = new Date(ms).getUTCDay(); // 0 = Sunday
  return addDays(dayOf(ms), (8 - weekday) % 7);
}

function normalizeAddress(address) {
  return String(address == null ? '' : address).trim().toLowerCase();
}

function isValidAddress(address) {
  return ADDRESS_RE.test(normalizeAddress(address));
}

// A plausible price, or null. PRL is well under $1000.
function sanePrice(v) {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 && n < 1000 ? n : null;
}

function badRequest(message) {
  const error = new Error(message);
  error.status = 400;
  return error;
}

class LlmRewardService {
  constructor(db, opts = {}) {
    this.db = db;
    this.now = opts.now || Date.now;
    this.fetchFn = opts.fetchFn || globalThis.fetch;
  }

  // Record one finished job. `data` is the job record, `chunks` its stored
  // chunks. Returns the row with its model label and an estimate in PRL at the
  // latest stored price, or null when nothing was recorded.
  async recordJob({ jobId, nodeId, data, chunks, completedAt }) {
    const owner = await this.db.query('SELECT user_id FROM nodes WHERE node_id = $1', [nodeId]);
    const ownerId = owner.rows.length ? owner.rows[0].user_id : null;
    if (data.userId && ownerId && ownerId === data.userId) return null;

    const final = (chunks || []).find((c) => c && c.isFinal && c.metrics);
    const metrics = (final && final.metrics) || {};
    // The node's own count, held to the job's max_tokens so a node can't claim
    // more than it was allowed to write. Without a count, estimate from the text.
    const reported = Number(metrics.totalTokens);
    const text = (chunks || []).map((c) => (c && c.content) || '').join('')
      + ((final && final.reasoning) || '');
    let output = Number.isFinite(reported) && reported >= 0 ? Math.round(reported) : estimateTokens(text);
    if (data.maxTokens > 0) output = Math.min(output, data.maxTokens);
    const prompt = Array.isArray(data.messages) ? estimateTokens(joinContent(data.messages)) : estimateTokens(data.prompt);

    const model = metrics.model || data.model || 'unknown';
    const rate = rateFor(model);
    const usd = rate ? (prompt * rate.inUsd + output * rate.outUsd) / 1e6 : 0;

    // The PRL address this node mines to, from the board row its app last sent.
    const miner = await this.db.query(
      `SELECT address FROM miners WHERE node_id = $1 AND address LIKE 'prl1p%'
       ORDER BY last_seen DESC LIMIT 1`,
      [nodeId]
    );
    const address = miner.rows.length ? miner.rows[0].address : null;

    const assignedAt = Number(data.assignedAt);
    const seconds = assignedAt > 0 ? Math.max(0, (completedAt - assignedAt) / 1000) : null;
    // A speed benchmark, or a request pinned to one node (X-LLMJob-Node), which
    // is how the founder benchmarks a rig. Anything else is ordinary API work.
    const source = data.benchmark || data.targetNode ? 'benchmark' : 'api';

    const row = {
      id: 'llmu_' + crypto.randomBytes(8).toString('hex'),
      jobId, ts: completedAt, day: dayOf(completedAt), nodeId, address, model,
      promptTokens: prompt, outputTokens: output, seconds, source, usd,
    };
    await this.db.query(
      `INSERT INTO llm_usage (id, job_id, ts, day, node_id, address, model, prompt_tokens, output_tokens, seconds, source, usd)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)`,
      [row.id, jobId, row.ts, row.day, nodeId, address, model, prompt, output, seconds, source, usd]
    );

    const price = await this.latestPrice();
    return { ...row, label: rate ? rate.label : model, prl: price && address ? usd / price : null };
  }

  // The most recent stored price, or null when none has been stored yet.
  async latestPrice() {
    const r = await this.db.query('SELECT usd FROM prl_prices ORDER BY day DESC LIMIT 1');
    return r.rows.length ? Number(r.rows[0].usd) : null;
  }

  // Fetch today's PRL price and store it, unless today already has one. Runs
  // on a timer; returns the stored price, or null when both sources failed.
  async recordTodayPrice() {
    const today = dayOf(this.now());
    const have = await this.db.query('SELECT usd FROM prl_prices WHERE day = $1', [today]);
    if (have.rows.length) return Number(have.rows[0].usd);
    const got = await this.fetchPrice();
    if (!got) return null;
    await this.db.query(
      `INSERT INTO prl_prices (day, usd, source, fetched_at) VALUES ($1, $2, $3, $4)
       ON CONFLICT (day) DO NOTHING`,
      [today, got.usd, got.source, this.now()]
    );
    return got.usd;
  }

  // { usd, source } from prlscan, else CoinGecko, else null.
  async fetchPrice() {
    const read = async (url) => {
      try {
        const res = await this.fetchFn(url);
        return res && res.ok ? await res.json() : null;
      } catch (e) {
        return null;
      }
    };
    const scan = await read(PRICE_URLS.prlscan);
    const fromScan = sanePrice(scan && scan.price_usd);
    if (fromScan) return { usd: fromScan, source: 'prlscan' };
    const cg = await read(PRICE_URLS.coingecko);
    const fromCg = sanePrice(cg && cg['pearl-2'] && cg['pearl-2'].usd);
    return fromCg ? { usd: fromCg, source: 'coingecko' } : null;
  }

  // The price for each day: that day's, else the nearest earlier one, else the
  // earliest stored. null for every day when no price is stored at all.
  async _priceFor() {
    const r = await this.db.query('SELECT day, usd FROM prl_prices ORDER BY day ASC');
    const prices = r.rows.map((row) => ({ day: row.day, usd: Number(row.usd) }));
    return (day) => {
      let best = prices.length ? prices[0].usd : null;
      for (const p of prices) {
        if (p.day <= day) best = p.usd;
      }
      return best;
    };
  }

  // Unpaid totals per address per day, each turned into PRL and capped.
  async _unpaidDays({ address, from, to }) {
    const r = await this.db.query(
      `SELECT address, day, COUNT(*)::int AS requests, SUM(prompt_tokens) AS prompt_tokens,
              SUM(output_tokens) AS output_tokens, SUM(usd) AS usd
       FROM llm_usage
       WHERE address IS NOT NULL AND payout_id IS NULL AND day >= $1 AND day <= $2
         AND ($3::text IS NULL OR address = $3)
       GROUP BY address, day ORDER BY address, day`,
      [from, to, address || null]
    );
    const priceFor = await this._priceFor();
    return r.rows.map((row) => {
      const usd = Number(row.usd);
      const price = priceFor(row.day);
      const prl = price ? Math.min(usd / price, DAILY_CAP_PRL) : null;
      return {
        address: row.address, day: row.day, requests: Number(row.requests),
        promptTokens: Number(row.prompt_tokens), outputTokens: Number(row.output_tokens),
        usd, price, prl,
      };
    });
  }

  // Check a from/to pair. `to` defaults to yesterday, so the day still being
  // served is never in a payout; `from` defaults to the start.
  _range(from, to) {
    const range = { from: from || '1970-01-01', to: to || addDays(dayOf(this.now()), -1) };
    if (!DAY_RE.test(range.from) || !DAY_RE.test(range.to)) throw badRequest('from and to must be YYYY-MM-DD');
    if (range.from > range.to) throw badRequest('from is after to');
    return range;
  }

  // What each address is owed for a range of days, unpaid rows only. An
  // address under MIN_PAYOUT_PRL is listed but not payable; it rolls over.
  async owed({ from, to } = {}) {
    const range = this._range(from, to);
    const days = await this._unpaidDays(range);
    const byAddress = new Map();
    for (const d of days) {
      let a = byAddress.get(d.address);
      if (!a) {
        a = { address: d.address, days: 0, requests: 0, promptTokens: 0, outputTokens: 0, usd: 0, prl: 0, unpriced: 0 };
        byAddress.set(d.address, a);
      }
      a.days++;
      a.requests += d.requests;
      a.promptTokens += d.promptTokens;
      a.outputTokens += d.outputTokens;
      if (d.prl == null) {
        a.unpriced++;
      } else {
        // The dollars actually paid: the cap trims them along with the PRL.
        a.usd += d.prl * d.price;
        a.prl += d.prl;
      }
    }
    const rows = [...byAddress.values()].map((a) => ({ ...a, payable: a.unpriced === 0 && a.prl >= MIN_PAYOUT_PRL }));
    rows.sort((x, y) => y.prl - x.prl || x.address.localeCompare(y.address));
    return { ...range, dailyCapPrl: DAILY_CAP_PRL, minPayoutPrl: MIN_PAYOUT_PRL, rows };
  }

  // Record a payout for every payable address in the range (or just
  // `address`), and stamp the payout id on the rows it covers. The range must
  // end before today.
  async markPaid({ from, to, tx, address } = {}) {
    const range = this._range(from, to);
    if (range.to >= dayOf(this.now())) throw badRequest('to must be before today');
    const only = address ? normalizeAddress(address) : null;
    const { rows } = await this.owed(range);
    const paid = [];
    for (const a of rows) {
      if (!a.payable || (only && a.address !== only)) continue;
      const id = 'llmp_' + crypto.randomBytes(8).toString('hex');
      await this.db.query(
        `INSERT INTO llm_payouts (id, address, from_day, to_day, prl, usd, tx, paid_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
        [id, a.address, range.from, range.to, a.prl, a.usd, tx || null, this.now()]
      );
      await this.db.query(
        `UPDATE llm_usage SET payout_id = $1
         WHERE address = $2 AND day >= $3 AND day <= $4 AND payout_id IS NULL`,
        [id, a.address, range.from, range.to]
      );
      paid.push({ id, address: a.address, prl: a.prl, usd: a.usd, tx: tx || null });
    }
    return { ...range, paid };
  }

  // The owner's view for one address: today, unpaid, paid so far, tokens
  // served today and the next payout day.
  async summary(address) {
    const addr = normalizeAddress(address);
    if (!isValidAddress(addr)) throw badRequest('Not a PRL address');
    const now = this.now();
    const today = dayOf(now);
    const days = await this._unpaidDays({ address: addr, from: '1970-01-01', to: today });
    const t = days.find((d) => d.day === today);
    const paid = await this.db.query(
      'SELECT COALESCE(SUM(prl), 0) AS prl, COUNT(*)::int AS payouts FROM llm_payouts WHERE address = $1',
      [addr]
    );
    return {
      address: addr,
      today: {
        prl: t ? t.prl : 0,
        usd: t ? t.usd : 0,
        tokens: t ? t.promptTokens + t.outputTokens : 0,
        requests: t ? t.requests : 0,
      },
      unpaidPrl: days.reduce((sum, d) => sum + (d.prl || 0), 0),
      paidPrl: Number(paid.rows[0].prl) || 0,
      payouts: Number(paid.rows[0].payouts),
      nextPayout: nextPayoutDay(now),
      dailyCapPrl: DAILY_CAP_PRL,
      minPayoutPrl: MIN_PAYOUT_PRL,
    };
  }
}

// One CSV line per address, for the founder's payout sheet.
function owedCsv(owed) {
  const lines = ['address,days,requests,prompt_tokens,output_tokens,usd,prl,payable'];
  for (const a of owed.rows) {
    lines.push([a.address, a.days, a.requests, a.promptTokens, a.outputTokens,
      a.usd.toFixed(6), a.prl.toFixed(6), a.payable ? 'yes' : 'no'].join(','));
  }
  return lines.join('\n') + '\n';
}

module.exports = LlmRewardService;
Object.assign(module.exports, {
  RATES, DAILY_CAP_PRL, MIN_PAYOUT_PRL, PRICE_URLS,
  rateFor, dayOf, addDays, nextPayoutDay, isValidAddress, sanePrice, owedCsv,
});
