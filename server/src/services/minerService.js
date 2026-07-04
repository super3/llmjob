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
    const id = minerFingerprint(address, worker);
    const now = Date.now();

    await this.db.query(
      `INSERT INTO miners (id, address, worker, gpu, region, hashrate, accepted, first_seen, last_seen)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
       ON CONFLICT (id) DO UPDATE SET
         gpu = EXCLUDED.gpu, region = EXCLUDED.region, hashrate = EXCLUDED.hashrate,
         accepted = EXCLUDED.accepted, last_seen = EXCLUDED.last_seen`,
      [id, address, worker, gpu, region, hashrate, accepted, now]
    );
    return { success: true, id };
  }

  // The network page's view: online miners grouped by payout address (one row
  // per address; its workers summed), sorted by hashrate. Prunes dead rows.
  async getPublicMiners() {
    const now = Date.now();
    await this.db.query('DELETE FROM miners WHERE last_seen < $1', [now - PRUNE_TTL]);
    const r = await this.db.query('SELECT * FROM miners WHERE last_seen >= $1', [now - OFFLINE_THRESHOLD]);

    const byAddr = new Map();
    for (const row of r.rows) {
      let g = byAddr.get(row.address);
      if (!g) {
        g = { addr: row.address, gpu: row.gpu || '—', hash: 0, workers: 0, accepted: 0, lastSeen: 0 };
        byAddr.set(row.address, g);
      }
      g.hash += Number(row.hashrate) || 0;
      g.workers += 1;
      g.accepted += Number(row.accepted) || 0;
      if (row.gpu) g.gpu = row.gpu;
      g.lastSeen = Math.max(g.lastSeen, Number(row.last_seen));
    }

    const miners = Array.from(byAddr.values())
      .map((g) => ({
        addr: g.addr, gpu: g.gpu, hash: +g.hash.toFixed(1),
        workers: g.workers, accepted: g.accepted, last: formatAgo(now - g.lastSeen)
      }))
      .sort((a, b) => b.hash - a.hash);

    return {
      miners,
      totalOnline: miners.length,
      totalWorkers: miners.reduce((a, m) => a + m.workers, 0),
      totalHashrate: +miners.reduce((a, m) => a + m.hash, 0).toFixed(1)
    };
  }
}

MinerService.minerFingerprint = minerFingerprint;
MinerService.isValidAddress = isValidAddress;
MinerService.clampNum = clampNum;
MinerService.formatAgo = formatAgo;

module.exports = MinerService;
