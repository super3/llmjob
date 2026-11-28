const LlmRewardService = require('../src/services/llmRewardService');
const {
  rateFor, dayOf, addDays, nextPayoutDay, isValidAddress, sanePrice, owedCsv, PRICE_URLS,
  DAILY_CAP_PRL, MIN_PAYOUT_PRL,
} = LlmRewardService;
const { createTestDb } = require('./helpers/pgmem');

const A = 'prl1p' + 'a'.repeat(30);
const B = 'prl1p' + 'b'.repeat(30);
// Wednesday 18 Nov 2026, noon UTC.
const NOW = Date.parse('2026-11-18T12:00:00Z');
const at = (day) => Date.parse(day + 'T06:00:00Z');

describe('llmRewardService helpers', () => {
  it('prices Qwen 3.8 27B and Gemma 4 E4B, and nothing else', () => {
    expect(rateFor('Qwen3.8-27B-UD-Q4_K_XL')).toMatchObject({ label: 'Qwen 3.8 27B', inUsd: 0.06, outUsd: 1.96 });
    expect(rateFor('Gemma-4-E4B-it-Q4_K_M')).toMatchObject({ label: 'Gemma 4 E4B', inUsd: 0.0675, outUsd: 0.225 });
    expect(rateFor('llama-3')).toBeNull();
    expect(rateFor(undefined)).toBeNull();
  });

  it('works in UTC days', () => {
    expect(dayOf(NOW)).toBe('2026-11-18');
    expect(addDays('2026-11-30', 1)).toBe('2026-12-01');
    expect(addDays('2026-11-01', -1)).toBe('2026-10-31');
  });

  it('puts the next payout on the coming Monday, or today on a Monday', () => {
    expect(nextPayoutDay(NOW)).toBe('2026-11-23');
    expect(nextPayoutDay(Date.parse('2026-11-22T23:00:00Z'))).toBe('2026-11-23'); // Sunday
    expect(nextPayoutDay(Date.parse('2026-11-23T01:00:00Z'))).toBe('2026-11-23'); // Monday
  });

  it('checks addresses and prices', () => {
    expect(isValidAddress(' ' + A.toUpperCase() + ' ')).toBe(true);
    expect(isValidAddress('qz123')).toBe(false);
    expect(isValidAddress(null)).toBe(false);
    expect(sanePrice('1.8')).toBe(1.8);
    expect(sanePrice(0)).toBeNull();
    expect(sanePrice(5000)).toBeNull();
    expect(sanePrice('x')).toBeNull();
  });
});

describe('LlmRewardService', () => {
  let db;
  let now;
  let fetchFn;
  let svc;

  beforeEach(async () => {
    db = await createTestDb();
    now = NOW;
    fetchFn = jest.fn();
    svc = new LlmRewardService(db, { now: () => now, fetchFn });
  });

  afterEach(async () => {
    if (db.end) await db.end();
  });

  const miner = (address, nodeId, lastSeen, id) => db.query(
    'INSERT INTO miners (id, address, node_id, last_seen) VALUES ($1, $2, $3, $4)',
    [id || address + nodeId + lastSeen, address, nodeId, lastSeen]
  );
  const price = (day, usd) => db.query('INSERT INTO prl_prices (day, usd) VALUES ($1, $2)', [day, usd]);
  // A usage row worth `usd` dollars, written straight to the table.
  let seq = 0;
  const usage = (address, day, usd, extra = {}) => db.query(
    `INSERT INTO llm_usage (id, ts, day, node_id, address, model, prompt_tokens, output_tokens, source, usd, payout_id)
     VALUES ($1, $2, $3, 'n', $4, 'm', $5, $6, 'api', $7, $8)`,
    ['u' + (seq++), at(day), day, address, extra.prompt || 10, extra.output || 20, usd, extra.payoutId || null]
  );

  describe('recordJob', () => {
    const job = (over = {}) => ({
      jobId: 'j1', nodeId: 'n1', completedAt: NOW,
      data: { messages: [{ role: 'user', content: 'x'.repeat(800) }], maxTokens: 6400, assignedAt: NOW - 30000, userId: 'caller' },
      chunks: [{ content: 'answer', isFinal: true, metrics: { totalTokens: 2000, model: 'Qwen3.8-27B-UD-Q4_K_XL' } }],
      ...over,
    });

    it('records one row with tokens, seconds, source, address and dollars', async () => {
      await miner('qz' + '1'.repeat(46), 'n1', 500, 'q'); // a Quantus card is not a PRL address
      await miner(A, 'n1', 100, 'old');
      await miner(A.replace('aaaa', 'cccc'), 'n1', 200, 'new');
      await price('2026-11-17', 1.5);
      const r = await svc.recordJob(job());
      const usd = (200 * 0.06 + 2000 * 1.96) / 1e6;
      expect(r).toMatchObject({
        jobId: 'j1', day: '2026-11-18', nodeId: 'n1', address: A.replace('aaaa', 'cccc'),
        promptTokens: 200, outputTokens: 2000, seconds: 30, source: 'api', usd, label: 'Qwen 3.8 27B',
      });
      expect(r.prl).toBeCloseTo(usd / 1.5, 12);
      const rows = (await db.query('SELECT * FROM llm_usage')).rows;
      expect(rows).toHaveLength(1);
      expect(rows[0]).toMatchObject({ job_id: 'j1', model: 'Qwen3.8-27B-UD-Q4_K_XL', source: 'api', payout_id: null });
    });

    it('tags speed benchmarks and pinned requests as benchmark', async () => {
      const bench = await svc.recordJob(job({ data: { prompt: 'abcd', benchmark: true } }));
      expect(bench.source).toBe('benchmark');
      expect(bench.promptTokens).toBe(1);
      expect(bench.seconds).toBeNull();
      const pinned = await svc.recordJob(job({ data: { prompt: 'abcd', targetNode: 'n1' } }));
      expect(pinned.source).toBe('benchmark');
    });

    it('holds the node\'s count to max_tokens, and estimates when there is none', async () => {
      const capped = await svc.recordJob(job({
        data: { prompt: 'p', maxTokens: 500 },
        chunks: [{ content: 'a', isFinal: true, metrics: { totalTokens: 99999 } }],
      }));
      expect(capped.outputTokens).toBe(500);
      // No count: 8 characters of answer and 4 of reasoning is 3 tokens.
      const est = await svc.recordJob(job({
        data: { prompt: 'p' },
        chunks: [{ content: 'abcd' }, null, { content: 'efgh', isFinal: true, reasoning: 'ijkl', metrics: {} }],
      }));
      expect(est.outputTokens).toBe(3);
      const none = await svc.recordJob(job({ data: { prompt: 'p' }, chunks: undefined }));
      expect(none.outputTokens).toBe(0);
    });

    it('records an unknown model at $0 and uses the job model when the node names none', async () => {
      const r = await svc.recordJob(job({ data: { prompt: 'p', model: 'llama-3' }, chunks: [] }));
      expect(r).toMatchObject({ model: 'llama-3', usd: 0, label: 'llama-3', prl: null, address: null });
      const u = await svc.recordJob(job({ data: { prompt: 'p' }, chunks: [] }));
      expect(u.model).toBe('unknown');
    });

    it('gives no PRL figure without an address, even with a price', async () => {
      await price('2026-11-18', 2);
      expect((await svc.recordJob(job())).prl).toBeNull();
    });

    it('skips a request the node\'s own owner sent', async () => {
      await db.query("INSERT INTO nodes (node_id, user_id) VALUES ('n1', 'caller')");
      expect(await svc.recordJob(job())).toBeNull();
      expect((await db.query('SELECT * FROM llm_usage')).rows).toHaveLength(0);
      // Someone else's node, or an unclaimed one, counts.
      await db.query("UPDATE nodes SET user_id = 'other' WHERE node_id = 'n1'");
      expect(await svc.recordJob(job())).not.toBeNull();
      await db.query("UPDATE nodes SET user_id = NULL WHERE node_id = 'n1'");
      expect(await svc.recordJob(job())).not.toBeNull();
    });
  });

  describe('prices', () => {
    const ok = (body) => ({ ok: true, json: async () => body });

    it('stores today\'s price from prlscan once a day', async () => {
      fetchFn.mockResolvedValue(ok({ price_usd: '1.84' }));
      expect(await svc.recordTodayPrice()).toBe(1.84);
      expect(fetchFn).toHaveBeenCalledWith(PRICE_URLS.prlscan);
      expect(await svc.recordTodayPrice()).toBe(1.84);
      expect(fetchFn).toHaveBeenCalledTimes(1);
      const rows = (await db.query('SELECT * FROM prl_prices')).rows;
      expect(rows).toEqual([{ day: '2026-11-18', usd: 1.84, source: 'prlscan', fetched_at: NOW }]);
      expect(await svc.latestPrice()).toBe(1.84);
    });

    it('falls back to CoinGecko, and stores nothing when both fail', async () => {
      fetchFn.mockResolvedValueOnce({ ok: false }).mockResolvedValueOnce(ok({ 'pearl-2': { usd: 1.7 } }));
      expect(await svc.fetchPrice()).toEqual({ usd: 1.7, source: 'coingecko' });
      fetchFn.mockRejectedValueOnce(new Error('down')).mockResolvedValueOnce(ok({}));
      expect(await svc.fetchPrice()).toBeNull();
      fetchFn.mockResolvedValueOnce(null).mockResolvedValueOnce(ok(null));
      expect(await svc.recordTodayPrice()).toBeNull();
      expect(await svc.latestPrice()).toBeNull();
    });

    it('uses the global fetch by default', () => {
      expect(new LlmRewardService(db).fetchFn).toBe(globalThis.fetch);
    });
  });

  describe('owed and markPaid', () => {
    it('turns each day\'s dollars into PRL at that day\'s price, capped per day', async () => {
      await price('2026-11-15', 2);
      await price('2026-11-17', 1);
      await usage(A, '2026-11-14', 1);   // before any price: the earliest price, $2
      await usage(A, '2026-11-16', 3);   // the nearest earlier price, $2 -> 1.5 PRL
      await usage(A, '2026-11-17', 4);
      await usage(A, '2026-11-17', 4);   // $8 at $1 is 8 PRL, capped at 5
      await usage(B, '2026-11-17', 0.5); // 0.5 PRL: under the minimum
      await usage(null, '2026-11-17', 9); // no address: never owed
      await usage(A, '2026-11-18', 2);   // today: outside the default range

      const owed = await svc.owed();
      expect(owed).toMatchObject({ from: '1970-01-01', to: '2026-11-17', dailyCapPrl: DAILY_CAP_PRL, minPayoutPrl: MIN_PAYOUT_PRL });
      expect(owed.rows).toEqual([
        { address: A, days: 3, requests: 4, promptTokens: 40, outputTokens: 80, usd: 1 + 3 + 5, prl: 0.5 + 1.5 + 5, unpriced: 0, payable: true },
        { address: B, days: 1, requests: 1, promptTokens: 10, outputTokens: 20, usd: 0.5, prl: 0.5, unpriced: 0, payable: false },
      ]);

      const csv = owedCsv(owed).split('\n');
      expect(csv[0]).toBe('address,days,requests,prompt_tokens,output_tokens,usd,prl,payable');
      expect(csv[1]).toBe(`${A},3,4,40,80,9.000000,7.000000,yes`);
      expect(csv[2]).toBe(`${B},1,1,10,20,0.500000,0.500000,no`);
      expect(csv[3]).toBe('');
    });

    it('lists days without any price as unpriced and not payable', async () => {
      await usage(A, '2026-11-17', 5);
      await usage(B, '2026-11-17', 5);
      const { rows } = await svc.owed({ from: '2026-11-17', to: '2026-11-17' });
      expect(rows.map((r) => [r.address, r.unpriced, r.prl, r.payable])).toEqual([[A, 1, 0, false], [B, 1, 0, false]]);
    });

    it('rejects a bad range', async () => {
      await expect(svc.owed({ from: 'yesterday' })).rejects.toMatchObject({ status: 400, message: 'from and to must be YYYY-MM-DD' });
      await expect(svc.owed({ from: '2026-11-10', to: '2026-11-01' })).rejects.toMatchObject({ status: 400, message: 'from is after to' });
    });

    it('marks payable addresses paid, once, and leaves the rest to roll over', async () => {
      await price('2026-11-01', 1);
      await usage(A, '2026-11-16', 2);
      await usage(A, '2026-11-17', 1.5);
      await usage(B, '2026-11-17', 0.5);
      await usage(A, '2026-11-18', 1);

      const res = await svc.markPaid({ from: '2026-11-01', to: '2026-11-17', tx: 'abc' });
      expect(res.paid).toEqual([{ id: expect.stringMatching(/^llmp_/), address: A, prl: 3.5, usd: 3.5, tx: 'abc' }]);
      const payouts = (await db.query('SELECT * FROM llm_payouts')).rows;
      expect(payouts).toEqual([{ id: res.paid[0].id, address: A, from_day: '2026-11-01', to_day: '2026-11-17', prl: 3.5, usd: 3.5, tx: 'abc', paid_at: NOW }]);
      // Nothing is paid twice; today's row and B's are still unpaid.
      expect((await svc.markPaid()).paid).toEqual([]);
      const left = await db.query('SELECT address, day FROM llm_usage WHERE payout_id IS NULL ORDER BY day, address');
      expect(left.rows).toEqual([{ address: B, day: '2026-11-17' }, { address: A, day: '2026-11-18' }]);
    });

    it('can pay one address, and never a range reaching today', async () => {
      await price('2026-11-01', 0.1);
      await usage(A, '2026-11-17', 0.2);
      await usage(B, '2026-11-17', 0.2);
      const res = await svc.markPaid({ to: '2026-11-17', address: B.toUpperCase() });
      expect(res.paid.map((p) => [p.address, p.tx])).toEqual([[B, null]]);
      await expect(svc.markPaid({ from: '2026-11-17', to: '2026-11-18' }))
        .rejects.toMatchObject({ status: 400, message: 'to must be before today' });
    });
  });

  describe('summary', () => {
    it('shows today, unpaid, paid so far and the next payout', async () => {
      await price('2026-11-01', 2);
      await usage(A, '2026-11-17', 4, { payoutId: 'p1' });
      await db.query("INSERT INTO llm_payouts (id, address, prl) VALUES ('p1', $1, 2)", [A]);
      await usage(A, '2026-11-17', 1);
      await usage(A, '2026-11-18', 0.5, { prompt: 100, output: 900 });
      await usage(A, '2026-11-18', 0.5, { prompt: 100, output: 900 });

      expect(await svc.summary(' ' + A.toUpperCase())).toEqual({
        address: A,
        today: { prl: 0.5, usd: 1, tokens: 2000, requests: 2 },
        unpaidPrl: 1,
        paidPrl: 2,
        payouts: 1,
        nextPayout: '2026-11-23',
        dailyCapPrl: DAILY_CAP_PRL,
        minPayoutPrl: MIN_PAYOUT_PRL,
      });
    });

    it('reads zero for an address with nothing yet, and unpriced days add nothing', async () => {
      await usage(B, '2026-11-18', 1);
      expect(await svc.summary(A)).toMatchObject({ today: { prl: 0, usd: 0, tokens: 0, requests: 0 }, unpaidPrl: 0, paidPrl: 0, payouts: 0 });
      expect(await svc.summary(B)).toMatchObject({ today: { prl: null, tokens: 30, requests: 1 }, unpaidPrl: 0 });
    });

    it('rejects something that is not a PRL address', async () => {
      await expect(svc.summary('nope')).rejects.toMatchObject({ status: 400, message: 'Not a PRL address' });
    });
  });
});
