const crypto = require('crypto');

// A worker unseen for this long drops off the "online" list; rows unseen for
// PRUNE_TTL are deleted entirely. Clients check in every ~60s, so 5 minutes is
// ~5 missed check-ins — tight enough that a stopped/renamed/crashed worker ages
// out fast instead of lingering. This matters because an address's hashrate is
// the SUM of its online workers: a lax window would keep summing a stale worker
// row (e.g. after a worker rename) on top of the live one and inflate the total.
const OFFLINE_THRESHOLD = 5 * 60 * 1000;   // 5 minutes
const PRUNE_TTL = 90 * 60 * 1000;          // 90 minutes
const ADDRESS_RE = /^prl1p[0-9a-z]{20,80}$/i;
const MAX_HASHRATE = 1e6;                  // TH/s sanity clamp
const MAX_VRAM_MB = 1e6;                   // VRAM MB sanity clamp (~1 TB)

// Stable per-(address, worker) id.
function minerFingerprint(address, worker) {
  return crypto.createHash('sha256').update(address + '|' + worker).digest('hex').slice(0, 12);
}

function isValidAddress(address) {
  return ADDRESS_RE.test(String(address == null ? '' : address).trim());
}

// Coerce to a finite, non-negative number, optionally capped at `max`.
function clampNum(v, max) {
  const n = Number(v);
  if (!Number.isFinite(n) || n < 0) return 0;
  return (max != null && n > max) ? max : n;
}

// Compact "time since" label for the last-share column.
function formatAgo(ms) {
  const s = Math.floor((ms > 0 ? ms : 0) / 1000);
  if (s < 5) return 'just now';
  if (s < 60) return s + 's ago';
  const m = Math.floor(s / 60);
  if (m < 60) return m + 'm ago';
  return Math.floor(m / 60) + 'h ago';
}

class MinerService {
  constructor(db) {
    this.db = db;
  }

  // Upsert a mining client's live status (no auth — public leaderboard data).
  async reportMiner(input = {}) {
    const address = String(input.address == null ? '' : input.address).trim();
    if (!isValidAddress(address)) return { error: 'Invalid payout address' };

    const worker = (String(input.worker == null ? '' : input.worker).trim() || 'rig01').slice(0, 64);
    const gpu = input.gpu ? String(input.gpu).slice(0, 80) : null;
    const region = input.region ? String(input.region).slice(0, 16) : null;
    const hashrate = clampNum(input.hashrate, MAX_HASHRATE);
    const accepted = Math.floor(clampNum(input.accepted));
    const vramUsed = clampNum(input.vramUsedMb, MAX_VRAM_MB);
    const vramTotal = clampNum(input.vramTotalMb, MAX_VRAM_MB);
    const id = minerFingerprint(address, worker);
    const now = Date.now();

    await this.db.query(
      `INSERT INTO miners (id, address, worker, gpu, region, hashrate, accepted, vram_used, vram_total, first_seen, last_seen)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $10)
       ON CONFLICT (id) DO UPDATE SET
         gpu = EXCLUDED.gpu, region = EXCLUDED.region, hashrate = EXCLUDED.hashrate,
         accepted = EXCLUDED.accepted, vram_used = EXCLUDED.vram_used,
         vram_total = EXCLUDED.vram_total, last_seen = EXCLUDED.last_seen`,
      [id, address, worker, gpu, region, hashrate, accepted, vramUsed, vramTotal, now]
    );
    return { success: true, id };
  }

  // The network page's view: one row per online worker (i.e. per GPU) — a rig
  // running two cards on the same payout address shows as two rows, each with
  // its own GPU, VRAM and last-seen, sharing the address. Sorted by hashrate
  // (ties broken by address+worker so the order never depends on DB row order).
  // Prunes dead rows; offline workers are simply excluded.
  async getPublicMiners() {
    const now = Date.now();
    await this.db.query('DELETE FROM miners WHERE last_seen < $1', [now - PRUNE_TTL]);
    const r = await this.db.query('SELECT * FROM miners WHERE last_seen >= $1', [now - OFFLINE_THRESHOLD]);

    const miners = r.rows
      .map((row) => ({
        addr: row.address,
        worker: row.worker,
        gpu: row.gpu || '—',
        hash: +(Number(row.hashrate) || 0).toFixed(1),
        accepted: Number(row.accepted) || 0,
        vramUsedMb: Math.round(Number(row.vram_used) || 0),
        vramTotalMb: Math.round(Number(row.vram_total) || 0),
        last: formatAgo(now - Number(row.last_seen)),
      }))
      .sort((a, b) => (b.hash - a.hash) || (a.addr + '|' + a.worker).localeCompare(b.addr + '|' + b.worker));

    return {
      miners,
      totalOnline: new Set(miners.map((m) => m.addr)).size, // distinct payout addresses
      totalWorkers: miners.length,                           // one entry per online GPU/worker
      totalHashrate: +miners.reduce((a, m) => a + m.hash, 0).toFixed(1),
    };
  }
}

MinerService.minerFingerprint = minerFingerprint;
MinerService.isValidAddress = isValidAddress;
MinerService.clampNum = clampNum;
MinerService.formatAgo = formatAgo;

module.exports = MinerService;
