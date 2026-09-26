const MinerService = require('../src/services/minerService');
const { createTestDb } = require('./helpers/pgmem');
const { generateKeypair, fingerprint, signRig } = require('../../earn/src/shared/node');

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

  test('clampOrNull keeps "no reading" apart from zero', () => {
    expect(MinerService.clampOrNull(61.5, 200)).toBe(61.5);
    expect(MinerService.clampOrNull('310', 5000)).toBe(310);
    expect(MinerService.clampOrNull(0, 200)).toBe(0);
    expect(MinerService.clampOrNull(900, 200)).toBe(200);
    for (const v of [undefined, null, '', 'N/A', -1, Infinity]) {
      expect(MinerService.clampOrNull(v, 200)).toBeNull();
    }
  });

  test('textOrNull trims, caps, and turns blank into null', () => {
    expect(MinerService.textOrNull('  580.82 ', 32)).toBe('580.82');
    expect(MinerService.textOrNull('x'.repeat(50), 8)).toBe('x'.repeat(8));
    expect(MinerService.textOrNull('   ', 8)).toBeNull();
    expect(MinerService.textOrNull(null, 8)).toBeNull();
  });

  test('baseWorker strips a /gpuN suffix, leaving the host name', () => {
    expect(MinerService.baseWorker('rig9/gpu0')).toBe('rig9');
    expect(MinerService.baseWorker('rig9/gpu12')).toBe('rig9');
    expect(MinerService.baseWorker('rig01')).toBe('rig01'); // bare worker unchanged
    expect(MinerService.baseWorker(null)).toBe('');
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
    expect(row.version).toBeNull();  // no version reported → stored null
  });

  test('upserts on repeat and clamps/floors the numbers, storing VRAM + version', async () => {
    await service.reportMiner({ address: ADDR.a, worker: 'rig01', gpu: 'RTX 4090', region: 'us1', hashrate: 100, accepted: 5, version: '0.1.15' });
    await service.reportMiner({ address: ADDR.a, worker: 'rig01', gpu: 'RTX 4090', hashrate: 5e6, accepted: 9.9, vramUsedMb: 4096, vramTotalMb: 24564, version: '0.1.16' });
    expect(await count()).toBe(1);
    const row = (await db.query('SELECT * FROM miners', [])).rows[0];
    expect(Number(row.hashrate)).toBe(1e6);   // clamped to MAX_HASHRATE
    expect(Number(row.accepted)).toBe(9);      // floored
    expect(Number(row.vram_used)).toBe(4096);  // upserted
    expect(Number(row.vram_total)).toBe(24564);
    expect(row.version).toBe('0.1.16');        // reported version, upserted
  });

  test('stores per-card health, clamped, with missing readings left null', async () => {
    const r = await service.reportMiner({
      address: ADDR.a, worker: 'rig01', hashrate: 250, accepted: 40,
      rejected: 3.7, tempC: 250, powerW: '310.5', powerLimitW: 450, coreClockMhz: 2520.4,
      memClockMhz: 'N/A', fanPct: 140, driver: ' 580.82 ', os: 'linux', client: 'cli',
      uptimeSec: 3600, lastShareSec: -5,
    });
    const row = (await db.query('SELECT * FROM miners WHERE id = $1', [r.id])).rows[0];
    expect(Number(row.rejected)).toBe(3);         // floored like accepted
    expect(Number(row.temp_c)).toBe(200);         // capped
    expect(Number(row.power_w)).toBe(310.5);
    expect(Number(row.power_limit_w)).toBe(450);
    expect(row.core_clock_mhz).toBe(2520);        // whole MHz
    expect(row.mem_clock_mhz).toBeNull();         // nvidia-smi could not read it
    expect(row.fan_pct).toBe(100);
    expect(row.driver).toBe('580.82');
    expect(row.os).toBe('linux');
    expect(row.client).toBe('cli');
    expect(Number(row.uptime_sec)).toBe(3600);
    expect(row.last_share_sec).toBeNull();        // negative is nonsense, not zero
    expect(row.rig_id).toBeNull();                // unsigned
  });

  test('an older client that sends no health fields stores nulls (and 0 rejected)', async () => {
    const r = await service.reportMiner({ address: ADDR.a, hashrate: 100 });
    const row = (await db.query('SELECT * FROM miners WHERE id = $1', [r.id])).rows[0];
    expect(Number(row.rejected)).toBe(0);
    for (const col of ['temp_c', 'power_w', 'power_limit_w', 'core_clock_mhz', 'mem_clock_mhz', 'fan_pct',
      'driver', 'os', 'client', 'uptime_sec', 'last_share_sec', 'rig_id']) {
      expect(row[col]).toBeNull();
    }
  });

  test('stores the rig id only while the report is signed', async () => {
    const kp = generateKeypair();
    const identity = { ...kp, nodeId: fingerprint(kp.publicKey) };
    const r = await service.reportMiner({ address: ADDR.a, hashrate: 100, ...signRig(identity, Date.now()) });
    const rigId = async () => (await db.query('SELECT rig_id FROM miners WHERE id = $1', [r.id])).rows[0].rig_id;
    expect(await rigId()).toBe(identity.nodeId);

    // The next report arrives unsigned (a downgrade, or a clock that drifted):
    // the row reflects that rather than keeping a stale claim.
    await service.reportMiner({ address: ADDR.a, hashrate: 100 });
    expect(await rigId()).toBeNull();
  });

  test('the public board never carries the health fields or the rig id', async () => {
    const kp = generateKeypair();
    const identity = { ...kp, nodeId: fingerprint(kp.publicKey) };
    await service.reportMiner({
      address: ADDR.a, hashrate: 100, tempC: 60, powerW: 300, driver: '580.82', os: 'linux', client: 'gui',
      ...signRig(identity, Date.now()),
    });
    const { miners } = await service.getPublicMiners();
    // Pinned exactly, so a field added to the public payload is a deliberate change.
    expect(Object.keys(miners[0]).sort()).toEqual(
      ['accepted', 'addr', 'cards', 'gpu', 'gpus', 'hash', 'last', 'multi', 'version', 'vramTotalMb', 'vramUsedMb', 'worker']);
    expect(Object.keys(miners[0].cards[0]).sort()).toEqual(
      ['accepted', 'gpu', 'hash', 'last', 'version', 'vramTotalMb', 'vramUsedMb', 'worker']);
    expect(JSON.stringify(miners)).not.toContain(identity.nodeId);
  });

  test('getPublicMiners returns one row per online worker (its own GPU/VRAM/last), sorted by hashrate', async () => {
    // ADDR.a runs two different cards on one address → two rows sharing the address.
    await service.reportMiner({ address: ADDR.a, worker: 'w-6000', gpu: 'NVIDIA RTX PRO 6000 Blackwell', hashrate: 300, accepted: 12, vramUsedMb: 8000, vramTotalMb: 98304, version: '0.1.16' });
    await service.reportMiner({ address: ADDR.a, worker: 'w-4090', gpu: 'NVIDIA GeForce RTX 4090', hashrate: 100, accepted: 5, vramUsedMb: 4096, vramTotalMb: 24564 });
    await service.reportMiner({ address: ADDR.b, worker: 'rig01', gpu: 'NVIDIA GeForce RTX 3090', hashrate: 200, accepted: 10 });
    await service.reportMiner({ address: ADDR.c, worker: 'rig01', hashrate: 0, accepted: 0 }); // no gpu, zero hashrate

    const out = await service.getPublicMiners();
    expect(out.totalWorkers).toBe(4);   // one row per online GPU/worker
    expect(out.totalOnline).toBe(4);    // four distinct nodes (a/w-6000, a/w-4090, b/rig01, c/rig01)
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
      addr: ADDR.a, hash: 300, accepted: 12, vramUsedMb: 8000, vramTotalMb: 98304, version: '0.1.16', last: 'just now',
    });
    // A worker that reported no version surfaces as null (not undefined).
    expect(out.miners.find((m) => m.addr === ADDR.c)).toMatchObject({ vramUsedMb: 0, vramTotalMb: 0, version: null });
  });

  test('combines a multi-GPU host (worker/gpuN) into one row that sums its cards', async () => {
    // One rig, three A4000s on ADDR.a: workers rig9/gpu0..2 → a single host row.
    await service.reportMiner({ address: ADDR.a, worker: 'rig9/gpu0', gpu: 'NVIDIA RTX A4000', hashrate: 96, accepted: 4, vramUsedMb: 5000, vramTotalMb: 16000, version: '0.2.10' });
    await service.reportMiner({ address: ADDR.a, worker: 'rig9/gpu1', gpu: 'NVIDIA RTX A4000', hashrate: 96, accepted: 4, vramUsedMb: 5000, vramTotalMb: 16000, version: '0.2.10' }); // ties gpu0's hashrate
    await service.reportMiner({ address: ADDR.a, worker: 'rig9/gpu2', gpu: 'NVIDIA RTX A4000', hashrate: 95, accepted: 3, vramUsedMb: 4900, vramTotalMb: 16000, version: '0.2.10' });
    await service.reportMiner({ address: ADDR.b, worker: 'rig01', gpu: 'NVIDIA GeForce RTX 3090', hashrate: 200, accepted: 10 });

    const out = await service.getPublicMiners();
    expect(out.totalWorkers).toBe(4); // four physical GPUs online
    expect(out.totalOnline).toBe(2);  // two nodes: the 3-card rig9 + the single 3090
    expect(out.totalHashrate).toBe(487);

    // The rig outranks the single 3090 on summed hashrate (287 > 200).
    const host = out.miners[0];
    expect(host).toMatchObject({
      addr: ADDR.a, worker: 'rig9', gpu: 'NVIDIA RTX A4000 × 3', gpus: 3, multi: true,
      hash: 287, accepted: 11, vramUsedMb: 14900, vramTotalMb: 48000, version: '0.2.10', last: 'just now',
    });
    // Cards ranked by hashrate, ties broken by worker: gpu0, gpu1 (both 96), gpu2 (95).
    expect(host.cards.map((c) => [c.worker, c.hash])).toEqual([
      ['rig9/gpu0', 96], ['rig9/gpu1', 96], ['rig9/gpu2', 95],
    ]);
    expect(host.cards[0]).toMatchObject({ gpu: 'NVIDIA RTX A4000', vramUsedMb: 5000, vramTotalMb: 16000, last: 'just now' });

    // The single-GPU host stays a plain, non-expandable row.
    expect(out.miners[1]).toMatchObject({ addr: ADDR.b, worker: 'rig01', gpu: 'NVIDIA GeForce RTX 3090', gpus: 1, multi: false });
    expect(out.miners[1].cards).toHaveLength(1);
  });

  test('drops a multi-GPU host\'s stale bare aggregate row (no phantom card, no doubled VRAM)', async () => {
    // Two real cards (rig9/gpu0..1) plus a leftover startup aggregate on the bare
    // worker "rig9": summed VRAM (32000) and zero hashrate. It must not appear as
    // a third card or double the host's VRAM.
    await service.reportMiner({ address: ADDR.a, worker: 'rig9/gpu0', gpu: 'NVIDIA RTX A4000', hashrate: 96, accepted: 4, vramUsedMb: 3000, vramTotalMb: 16000, version: '0.2.10' });
    await service.reportMiner({ address: ADDR.a, worker: 'rig9/gpu1', gpu: 'NVIDIA RTX A4000', hashrate: 95, accepted: 3, vramUsedMb: 3000, vramTotalMb: 16000, version: '0.2.10' });
    await service.reportMiner({ address: ADDR.a, worker: 'rig9', gpu: 'NVIDIA RTX A4000', hashrate: 0, accepted: 0, vramUsedMb: 100, vramTotalMb: 32000, version: '0.2.10' });

    const out = await service.getPublicMiners();
    expect(out.totalWorkers).toBe(2);   // the aggregate row is not a GPU
    const host = out.miners[0];
    expect(host).toMatchObject({ gpus: 2, gpu: 'NVIDIA RTX A4000 × 2', hash: 191, vramUsedMb: 6000, vramTotalMb: 32000 });
    expect(host.cards.map((c) => c.worker)).toEqual(['rig9/gpu0', 'rig9/gpu1']); // bare "rig9" gone

    // A genuine single-GPU host (only a bare worker) is untouched.
    await service.reportMiner({ address: ADDR.b, worker: 'rig01', gpu: 'NVIDIA GeForce RTX 3090', hashrate: 200 });
    const out2 = await service.getPublicMiners();
    expect(out2.miners.find((m) => m.addr === ADDR.b)).toMatchObject({ gpus: 1, worker: 'rig01', gpu: 'NVIDIA GeForce RTX 3090' });
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


// The server validated case-insensitively but never lowercased, and
// minerFingerprint keys the row on whatever it was handed — so one miner posting
// mixed case landed as two board rows with its hashrate split.
describe('address normalization', () => {
  it('treats case variants of one address as the same miner', async () => {
    const db = await createTestDb();
    const svc = new MinerService(db);
    const lower = 'prl1p' + 'a'.repeat(30);
    const upper = lower.toUpperCase();

    await svc.reportMiner({ address: lower, worker: 'rig1', hashrate: 10 });
    await svc.reportMiner({ address: upper, worker: 'rig1', hashrate: 20 });

    const { miners } = await svc.getPublicMiners();
    expect(miners).toHaveLength(1);            // one row, not two
    expect(miners[0].addr).toBe(lower);        // stored normalized
    expect(miners[0].hash).toBe(20);           // the second report updated the first
    if (db.end) await db.end();
  });

  it('normalizeAddress trims and lowercases', () => {
    expect(MinerService.normalizeAddress('  PRL1PAbC  ')).toBe('prl1pabc');
    expect(MinerService.normalizeAddress(null)).toBe('');
    expect(MinerService.normalizeAddress(undefined)).toBe('');
  });
});
