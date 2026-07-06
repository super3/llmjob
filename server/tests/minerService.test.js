const MinerService = require('../src/services/minerService');
const { createTestDb } = require('./helpers/pgmem');

const ADDR = {
  a: 'prl1p' + 'a'.repeat(30),
  b: 'prl1p' + 'b'.repeat(30),
  c: 'prl1p' + 'c'.repeat(30),
};

describe('MinerService helpers', () => {
  test('minerFingerprint is a stable 12-char hex id', () => {
    const id = MinerService.minerFingerprint(ADDR.a, 'rig01');
    expect(id).toMatch(/^[0-9a-f]{12}$/);
    expect(MinerService.minerFingerprint(ADDR.a, 'rig01')).toBe(id);
    expect(MinerService.minerFingerprint(ADDR.a, 'rig02')).not.toBe(id);
  });

  test('isValidAddress accepts prl1p addresses and rejects others', () => {
    expect(MinerService.isValidAddress(ADDR.a)).toBe(true);
    expect(MinerService.isValidAddress('prl1pShort')).toBe(false);
    expect(MinerService.isValidAddress('bc1qwhatever')).toBe(false);
    expect(MinerService.isValidAddress(null)).toBe(false);
  });

  test('clampNum coerces to a finite, non-negative, capped number', () => {
    expect(MinerService.clampNum(12.5)).toBe(12.5);
    expect(MinerService.clampNum(-3)).toBe(0);
    expect(MinerService.clampNum('nope')).toBe(0);
    expect(MinerService.clampNum(50, 10)).toBe(10);
    expect(MinerService.clampNum(5, 10)).toBe(5);
  });

  test('formatAgo renders each tier', () => {
    expect(MinerService.formatAgo(-100)).toBe('just now');
    expect(MinerService.formatAgo(2000)).toBe('just now');
    expect(MinerService.formatAgo(12000)).toBe('12s ago');
    expect(MinerService.formatAgo(90 * 1000)).toBe('1m ago');
    expect(MinerService.formatAgo(2 * 3600 * 1000)).toBe('2h ago');
  });
});

describe('MinerService (db)', () => {
  let db;
  let service;
  beforeEach(async () => {
    db = await createTestDb();
    service = new MinerService(db);
  });
  afterEach(async () => {
    if (db.end) await db.end();
  });

  const setLastSeen = (id, ms) => db.query('UPDATE miners SET last_seen = $1 WHERE id = $2', [ms, id]);
  const count = async () => (await db.query('SELECT COUNT(*)::int AS n FROM miners', [])).rows[0].n;

  test('rejects an invalid or missing payout address', async () => {
    expect(await service.reportMiner({ address: 'nope', worker: 'rig01' })).toEqual({ error: 'Invalid payout address' });
    expect(await service.reportMiner()).toEqual({ error: 'Invalid payout address' }); // no args → default {}
    expect(await count()).toBe(0);
  });

  test('inserts a miner and defaults the worker, gpu and region', async () => {
    const r = await service.reportMiner({ address: ADDR.a, hashrate: 100 });
    expect(r.success).toBe(true);
    expect(r.id).toBe(MinerService.minerFingerprint(ADDR.a, 'rig01'));
    const row = (await db.query('SELECT * FROM miners WHERE id = $1', [r.id])).rows[0];
    expect(row.worker).toBe('rig01');
    expect(row.gpu).toBeNull();
    expect(row.region).toBeNull();
  });

  test('upserts on repeat and clamps/floors the numbers, storing VRAM', async () => {
    await service.reportMiner({ address: ADDR.a, worker: 'rig01', gpu: 'RTX 4090', region: 'us1', hashrate: 100, accepted: 5 });
    await service.reportMiner({ address: ADDR.a, worker: 'rig01', gpu: 'RTX 4090', hashrate: 5e6, accepted: 9.9, vramUsedMb: 4096, vramTotalMb: 24564 });
    expect(await count()).toBe(1);
    const row = (await db.query('SELECT * FROM miners', [])).rows[0];
    expect(Number(row.hashrate)).toBe(1e6);   // clamped to MAX_HASHRATE
    expect(Number(row.accepted)).toBe(9);      // floored
    expect(Number(row.vram_used)).toBe(4096);  // upserted
    expect(Number(row.vram_total)).toBe(24564);
  });

  test('getPublicMiners returns one row per online worker (its own GPU/VRAM/last), sorted by hashrate', async () => {
    // ADDR.a runs two different cards on one address → two rows sharing the address.
    await service.reportMiner({ address: ADDR.a, worker: 'w-6000', gpu: 'NVIDIA RTX PRO 6000 Blackwell', hashrate: 300, accepted: 12, vramUsedMb: 8000, vramTotalMb: 98304 });
    await service.reportMiner({ address: ADDR.a, worker: 'w-4090', gpu: 'NVIDIA GeForce RTX 4090', hashrate: 100, accepted: 5, vramUsedMb: 4096, vramTotalMb: 24564 });
    await service.reportMiner({ address: ADDR.b, worker: 'rig01', gpu: 'NVIDIA GeForce RTX 3090', hashrate: 200, accepted: 10 });
    await service.reportMiner({ address: ADDR.c, worker: 'rig01', hashrate: 0, accepted: 0 }); // no gpu, zero hashrate

    const out = await service.getPublicMiners();
    expect(out.totalWorkers).toBe(4);   // one row per online GPU/worker
    expect(out.totalOnline).toBe(3);    // distinct payout addresses
    expect(out.totalHashrate).toBe(600);

    // Ranked by hashrate desc: 6000(300) > b/3090(200) > a/4090(100) > c(0).
    expect(out.miners.map((m) => [m.addr, m.gpu])).toEqual([
      [ADDR.a, 'NVIDIA RTX PRO 6000 Blackwell'],
      [ADDR.b, 'NVIDIA GeForce RTX 3090'],
      [ADDR.a, 'NVIDIA GeForce RTX 4090'],
      [ADDR.c, '—'], // no worker reported a gpu → dash
    ]);

    // The two ADDR.a rows are distinct GPUs on one address, each its own VRAM.
    expect(out.miners.find((m) => m.worker === 'w-6000')).toMatchObject({
      addr: ADDR.a, hash: 300, accepted: 12, vramUsedMb: 8000, vramTotalMb: 98304, last: 'just now',
    });
    expect(out.miners.find((m) => m.addr === ADDR.c)).toMatchObject({ vramUsedMb: 0, vramTotalMb: 0 });
  });

  test('ties in hashrate are ordered deterministically by address+worker (not DB row order)', async () => {
    await service.reportMiner({ address: ADDR.b, worker: 'rig01', gpu: 'X', hashrate: 50 });
    await service.reportMiner({ address: ADDR.a, worker: 'rig02', gpu: 'Z', hashrate: 50 });
    await service.reportMiner({ address: ADDR.a, worker: 'rig01', gpu: 'Y', hashrate: 50 });

    const out = await service.getPublicMiners();
    expect(out.miners.map((m) => [m.addr, m.worker])).toEqual([
      [ADDR.a, 'rig01'], [ADDR.a, 'rig02'], [ADDR.b, 'rig01'],
    ]);
  });

  test('omits workers past the online window but keeps their rows until prune', async () => {
    const r = await service.reportMiner({ address: ADDR.a, worker: 'rig01', hashrate: 100 });
    await setLastSeen(r.id, Date.now() - 10 * 60 * 1000); // 10 min: offline, not yet pruned
    const out = await service.getPublicMiners();
    expect(out.totalOnline).toBe(0);
    expect(await count()).toBe(1);
  });

  test('an offline worker is excluded, so a renamed worker leaves just the live row', async () => {
    // One rig, renamed rig01 → rig02: the old row lingers but stopped reporting.
    const stale = await service.reportMiner({ address: ADDR.a, worker: 'rig01', gpu: 'RTX 4090', hashrate: 206 });
    await service.reportMiner({ address: ADDR.a, worker: 'rig02', gpu: 'RTX 4090', hashrate: 285 });
    await setLastSeen(stale.id, Date.now() - 10 * 60 * 1000); // rig01 quiet for 10 min

    const out = await service.getPublicMiners();
    expect(out.totalWorkers).toBe(1);                       // only the live worker is a row
    expect(out.miners.map((m) => [m.worker, m.hash])).toEqual([['rig02', 285]]);
    expect(out.totalHashrate).toBe(285);
  });

  test('prunes miners not seen within the TTL', async () => {
    const r = await service.reportMiner({ address: ADDR.a, worker: 'rig01', hashrate: 100 });
    await setLastSeen(r.id, Date.now() - 2 * 60 * 60 * 1000); // 2 h: pruned
    const out = await service.getPublicMiners();
    expect(out.totalOnline).toBe(0);
    expect(await count()).toBe(0);
  });
});
