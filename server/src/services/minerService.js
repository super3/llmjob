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

// A stable GPU label for an address that may run several cards. `gpus` maps each
// card name to its summed hashrate. Picks the highest-hashrate card as the
// primary (tie-broken by name, so the result never depends on DB row order) and
// appends "+N" for the rest — instead of letting one arbitrary worker's GPU win
// and flip-flop between pings on a mixed-GPU address.
function gpuLabel(gpus) {
  const names = Array.from(gpus.keys());
  if (!names.length) return '—';
  names.sort((a, b) => (gpus.get(b) - gpus.get(a)) || a.localeCompare(b));
  return names.length === 1 ? names[0] : names[0] + ' +' + (names.length - 1);
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
        g = { addr: row.address, gpus: new Map(), hash: 0, workers: 0, accepted: 0, vramUsed: 0, vramTotal: 0, lastSeen: 0 };
        byAddr.set(row.address, g);
      }
      g.hash += Number(row.hashrate) || 0;
      g.workers += 1;
      g.accepted += Number(row.accepted) || 0;
      g.vramUsed += Number(row.vram_used) || 0;
      g.vramTotal += Number(row.vram_total) || 0;
      if (row.gpu) g.gpus.set(row.gpu, (g.gpus.get(row.gpu) || 0) + (Number(row.hashrate) || 0));
      g.lastSeen = Math.max(g.lastSeen, Number(row.last_seen));
    }

    const miners = Array.from(byAddr.values())
      .map((g) => ({
        addr: g.addr, gpu: gpuLabel(g.gpus), hash: +g.hash.toFixed(1),
        workers: g.workers, accepted: g.accepted,
        vramUsedMb: Math.round(g.vramUsed), vramTotalMb: Math.round(g.vramTotal),
        last: formatAgo(now - g.lastSeen)
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
MinerService.gpuLabel = gpuLabel;

module.exports = MinerService;
