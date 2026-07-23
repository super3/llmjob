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

// The "/gpuN" suffix a multi-GPU rig appends to its worker, one per card.
const GPU_SUFFIX = /\/gpu\d+$/i;

// The host's base worker, with a trailing "/gpuN" suffix removed. A multi-GPU
// rig posts one row per card as "<worker>/gpu<index>" (see earn's minerReport),
// so rows sharing an (address, baseWorker) are the same physical host and are
// combined into one board row. A bare worker keeps its name (a host of one card).
function baseWorker(worker) {
  return String(worker == null ? '' : worker).replace(GPU_SUFFIX, '');
}

// Drop a multi-GPU host's legacy rig-level row. Before per-card stats arrive, the
// client posts one row on the BARE worker with the rig's summed VRAM and whole-rig
// hashrate; once it switches to per-card "<worker>/gpuN" rows, that bare row
// lingers in the DB until it ages out. Left in, it groups as a phantom extra card
// and double-counts the host's VRAM/hashrate. So for any host that has at least
// one per-card row, discard its bare row. A genuine single-GPU host (only a bare
// row, no per-card siblings) is untouched.
function dropHostAggregates(cards) {
  const perCardHosts = new Set();
  for (const c of cards) {
    if (GPU_SUFFIX.test(c.worker)) perCardHosts.add(c.addr + '|' + c.base);
  }
  return cards.filter((c) => GPU_SUFFIX.test(c.worker) || !perCardHosts.has(c.addr + '|' + c.base));
}

// Fold per-card rows into one entry per host (address + base worker): summed
// hashrate/VRAM/shares, the host's GPU label ("<name> × N" when it runs more
// than one card), and the individual cards nested so the board can expand a
// multi-GPU rig. Cards are ordered by hashrate (ties by worker); hosts by total
// hashrate (ties by address+worker) so order never depends on DB row order.
function groupHosts(cards, now) {
  const byHost = new Map();
  for (const c of cards) {
    const key = c.addr + '|' + c.base;
    const group = byHost.get(key);
    if (group) group.push(c); else byHost.set(key, [c]);
  }

  const hosts = [];
  for (const group of byHost.values()) {
    group.sort((a, b) => (b.hash - a.hash) || a.worker.localeCompare(b.worker));
    const top = group[0];
    const multi = group.length > 1;
    const sum = (pick) => group.reduce((a, c) => a + pick(c), 0);
    hosts.push({
      addr: top.addr,
      worker: top.base,
      gpu: multi ? top.gpu + ' × ' + group.length : top.gpu,
      gpus: group.length,
      multi,
      hash: +sum((c) => c.hash).toFixed(1),
      accepted: sum((c) => c.accepted),
      vramUsedMb: sum((c) => c.vramUsedMb),
      vramTotalMb: sum((c) => c.vramTotalMb),
      version: top.version,
      last: formatAgo(now - group.reduce((a, c) => Math.max(a, c.lastMs), 0)),
      cards: group.map((c) => ({
        worker: c.worker,
        gpu: c.gpu,
        hash: c.hash,
        accepted: c.accepted,
        vramUsedMb: c.vramUsedMb,
        vramTotalMb: c.vramTotalMb,
        version: c.version,
        last: formatAgo(now - c.lastMs),
      })),
    });
  }

  hosts.sort((a, b) => (b.hash - a.hash) || (a.addr + '|' + a.worker).localeCompare(b.addr + '|' + b.worker));
  return hosts;
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
    const version = input.version ? String(input.version).slice(0, 32) : null;
    const id = minerFingerprint(address, worker);
    const now = Date.now();

    await this.db.query(
      `INSERT INTO miners (id, address, worker, gpu, region, hashrate, accepted, vram_used, vram_total, version, first_seen, last_seen)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $11)
       ON CONFLICT (id) DO UPDATE SET
         gpu = EXCLUDED.gpu, region = EXCLUDED.region, hashrate = EXCLUDED.hashrate,
         accepted = EXCLUDED.accepted, vram_used = EXCLUDED.vram_used,
         vram_total = EXCLUDED.vram_total, version = EXCLUDED.version, last_seen = EXCLUDED.last_seen`,
      [id, address, worker, gpu, region, hashrate, accepted, vramUsed, vramTotal, version, now]
    );
    return { success: true, id };
  }

  // The network page's view: one row per online HOST — a rig running several
  // cards on one payout address (workers "<name>/gpu0", "/gpu1", …) is combined
  // into a single row that sums the cards' hashrate/VRAM and nests them for the
  // expandable per-card breakdown. `totalOnline` counts distinct payout
  // addresses and `totalWorkers` counts individual online GPUs (so the stat
  // cards read "N miners / M GPUs"). Prunes dead rows; offline workers excluded.
  async getPublicMiners() {
    const now = Date.now();
    await this.db.query('DELETE FROM miners WHERE last_seen < $1', [now - PRUNE_TTL]);
    const r = await this.db.query('SELECT * FROM miners WHERE last_seen >= $1', [now - OFFLINE_THRESHOLD]);

    // One card per online worker row, then folded into hosts by (address, base).
    // Drop each multi-GPU host's stale bare-worker aggregate row first so it
    // isn't counted as a GPU or double-summed into the host's VRAM/hashrate.
    const cards = dropHostAggregates(r.rows.map((row) => ({
      addr: row.address,
      worker: row.worker,
      base: baseWorker(row.worker),
      gpu: row.gpu || '—',
      hash: +(Number(row.hashrate) || 0).toFixed(1),
      accepted: Number(row.accepted) || 0,
      vramUsedMb: Math.round(Number(row.vram_used) || 0),
      vramTotalMb: Math.round(Number(row.vram_total) || 0),
      version: row.version || null,
      lastMs: Number(row.last_seen), // always positive: the WHERE clause filters on last_seen
    })));

    return {
      miners: groupHosts(cards, now),
      totalOnline: new Set(cards.map((c) => c.addr)).size, // distinct payout addresses
      totalWorkers: cards.length,                           // one entry per online GPU/worker
      totalHashrate: +cards.reduce((a, c) => a + c.hash, 0).toFixed(1),
    };
  }
}

MinerService.minerFingerprint = minerFingerprint;
MinerService.isValidAddress = isValidAddress;
MinerService.clampNum = clampNum;
MinerService.formatAgo = formatAgo;
MinerService.baseWorker = baseWorker;
MinerService.dropHostAggregates = dropHostAggregates;
MinerService.groupHosts = groupHosts;

module.exports = MinerService;
