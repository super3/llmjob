const crypto = require('crypto');
const NodeService = require('./nodeService');

// A worker unseen for this long drops off the "online" list; rows unseen for
// PRUNE_TTL are deleted entirely. Clients check in every ~60s, so 5 minutes is
// ~5 missed check-ins — tight enough that a stopped/renamed/crashed worker ages
// out fast instead of lingering. This matters because an address's hashrate is
// the SUM of its online workers: a lax window would keep summing a stale worker
// row (e.g. after a worker rename) on top of the live one and inflate the total.
const OFFLINE_THRESHOLD = 5 * 60 * 1000;   // 5 minutes
const PRUNE_TTL = 90 * 60 * 1000;          // 90 minutes
const ADDRESS_RE = /^prl1p[0-9a-z]{20,80}$/i;
// A node id as either width nodeService recognises — the current 16 hex
// characters, or the 6 a machine enrolled under before ids were widened.
const NODE_ID_RE = new RegExp(
  `^([0-9a-f]{${NodeService.LEGACY_NODE_ID_HEX}}|[0-9a-f]{${NodeService.NODE_ID_HEX}})$`
);
const MAX_HASHRATE = 1e6;                  // TH/s sanity clamp
const MAX_VRAM_MB = 1e6;                   // VRAM MB sanity clamp (~1 TB)
const MAX_RAM_MB = 1e7;                    // system RAM MB sanity clamp (~10 TB)
const MAX_XMR_HASHRATE = 1e7;              // Monero H/s sanity clamp (10 MH/s, far past any one machine)

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

// The host's most recently reported card that passes `has`, or null when none
// does. Ties go to the earlier card in the group, whose order groupHosts has
// already made deterministic.
function newestCard(group, has) {
  let best = null;
  for (const c of group) {
    if (has(c) && (!best || c.lastMs > best.lastMs)) best = c;
  }
  return best;
}

// Fold per-card rows into one entry per host (address + base worker): summed
// hashrate/VRAM/shares, the host's GPU label ("<name> × N" when it runs more
// than one card), the machine's RAM and Monero hashrate (taken once, never
// summed), and the individual cards nested so the board can expand a
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
    // RAM and the Monero hashrate belong to the machine, and every card row of a
    // host reports the same values, so summing them would count the machine once
    // per card. RAM comes from the newest card that has it; null when none does
    // (a client too old to send it). The Monero hashrate comes from the newest
    // card whatever it says, because there null means "not CPU mining": a card
    // that dropped out of the reports stays online for up to OFFLINE_THRESHOLD
    // with the hashrate from before, and falling back to it would show a CPU
    // miner that has stopped.
    const ramCard = newestCard(group, (c) => c.ramTotalMb != null);
    const xmrCard = newestCard(group, () => true);
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
      // The host's served model: the fleet runs one model across every serving
      // card, so any card's non-null llmModel is THE model; null if none serve.
      llmModel: (group.find((c) => c.llmModel) || {}).llmModel || null,
      // The node id is the machine's, so every card of a host reports the same one
      // — take whichever card has it. Null means this host runs the model (or just
      // mines) without serving the cluster.
      nodeId: (group.find((c) => c.nodeId) || {}).nodeId || null,
      ramUsedMb: ramCard ? ramCard.ramUsedMb : null,
      ramTotalMb: ramCard ? ramCard.ramTotalMb : null,
      xmrHashrate: xmrCard.xmrHashrate,
      last: formatAgo(now - group.reduce((a, c) => Math.max(a, c.lastMs), 0)),
      cards: group.map((c) => ({
        worker: c.worker,
        gpu: c.gpu,
        hash: c.hash,
        accepted: c.accepted,
        vramUsedMb: c.vramUsedMb,
        vramTotalMb: c.vramTotalMb,
        version: c.version,
        llmModel: c.llmModel,
        last: formatAgo(now - c.lastMs),
      })),
    });
  }

  hosts.sort((a, b) => (b.hash - a.hash) || (a.addr + '|' + a.worker).localeCompare(b.addr + '|' + b.worker));
  return hosts;
}

// Normalize at the boundary, then test — the same order earn/src/shared/address.js
// uses. The server accepted mixed case (the /i flag) but never lowercased, and
// minerFingerprint keys the row on the address it was handed, so one miner posting
// `PRL1P…` and `prl1p…` landed as TWO board rows with its hashrate split between
// them.
function normalizeAddress(address) {
  return String(address == null ? '' : address).trim().toLowerCase();
}

function isValidAddress(address) {
  return ADDRESS_RE.test(normalizeAddress(address));
}

// Coerce to a finite, non-negative number, optionally capped at `max`.
function clampNum(v, max) {
  const n = Number(v);
  if (!Number.isFinite(n) || n < 0) return 0;
  return (max != null && n > max) ? max : n;
}

// clampNum for a field where "not reported" must stay null rather than read as
// 0. Only a number, or a string holding one, counts as a reading: Number()
// would turn null, '', true or [] into 0 or 1. Anything else that isn't a
// finite, non-negative number is null too.
function clampOrNull(v, max) {
  const given = typeof v === 'number' || (typeof v === 'string' && v.trim() !== '');
  const n = given ? Number(v) : NaN;
  if (!Number.isFinite(n) || n < 0) return null;
  return n > max ? max : n;
}

// A nullable numeric column, read back: null stays null, anything else goes
// through `fn`.
function readNullable(v, fn) {
  return v == null ? null : fn(Number(v));
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
    // Normalized before it is used as fingerprint input, so case can't fork a
    // miner's row.
    const address = normalizeAddress(input.address);
    if (!isValidAddress(address)) return { error: 'Invalid payout address' };

    const worker = (String(input.worker == null ? '' : input.worker).trim() || 'rig01').slice(0, 64);
    const gpu = input.gpu ? String(input.gpu).slice(0, 80) : null;
    const region = input.region ? String(input.region).slice(0, 16) : null;
    const hashrate = clampNum(input.hashrate, MAX_HASHRATE);
    const accepted = Math.floor(clampNum(input.accepted));
    const vramUsed = clampNum(input.vramUsedMb, MAX_VRAM_MB);
    const vramTotal = clampNum(input.vramTotalMb, MAX_VRAM_MB);
    const version = input.version ? String(input.version).slice(0, 32) : null;
    // The local LLM this card is serving, if any. A card not serving (insufficient
    // VRAM) or a client too old to report it stores null → blank on the board.
    const llmModel = input.llmModel ? String(input.llmModel).slice(0, 64) : null;
    // The machine's node id, sent only while it is armed to serve cluster jobs.
    // Null (client too old, or running the model without serving) means the board
    // shows the row as advertising a model rather than serving the network.
    //
    // Shape-validated, not merely length-clamped: this ping is unauthenticated, and
    // the value is rendered into a chip on the public board. A real node id is
    // sha256(publicKey) truncated to NODE_ID_HEX hex chars — both nodeService's
    // generateNodeFingerprint and the client's node.fingerprint() produce exactly
    // that — so anything else is junk and stored as null rather than painted onto
    // the board. It remains a display hint only: nothing routes or authorizes on it.
    //
    // Both widths are accepted: ids were widened from 6 to 16 hex characters (see
    // nodeService), and a machine enrolled before that keeps its 6-character id.
    // Rejecting the old width would blank the serving chip for every existing rig.
    const rawNodeId = input.nodeId == null ? '' : String(input.nodeId).trim();
    const nodeId = NODE_ID_RE.test(rawNodeId) ? rawNodeId : null;
    // The machine's system RAM (MB) and its Monero CPU miner's hashrate (H/s).
    // Every card row of a host carries the same values; getPublicMiners takes
    // them once per host. Clamped like the numbers above, but NULL rather than
    // 0 when the client didn't send them (too old, or not CPU mining) or sent
    // junk, so the board can say "unknown" instead of claiming a zero.
    const ramUsed = clampOrNull(input.ramUsedMb, MAX_RAM_MB);
    const ramTotal = clampOrNull(input.ramTotalMb, MAX_RAM_MB);
    const xmrHashrate = clampOrNull(input.xmrHashrate, MAX_XMR_HASHRATE);
    const id = minerFingerprint(address, worker);
    const now = Date.now();

    await this.db.query(
      `INSERT INTO miners (id, address, worker, gpu, region, hashrate, accepted, vram_used, vram_total, version, llm_model, node_id,
                           ram_used, ram_total, xmr_hashrate, first_seen, last_seen)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $16)
       ON CONFLICT (id) DO UPDATE SET
         gpu = EXCLUDED.gpu, region = EXCLUDED.region, hashrate = EXCLUDED.hashrate,
         accepted = EXCLUDED.accepted, vram_used = EXCLUDED.vram_used,
         vram_total = EXCLUDED.vram_total, version = EXCLUDED.version,
         llm_model = EXCLUDED.llm_model, node_id = EXCLUDED.node_id,
         ram_used = EXCLUDED.ram_used, ram_total = EXCLUDED.ram_total,
         xmr_hashrate = EXCLUDED.xmr_hashrate,
         last_seen = EXCLUDED.last_seen`,
      [id, address, worker, gpu, region, hashrate, accepted, vramUsed, vramTotal, version, llmModel, nodeId,
        ramUsed, ramTotal, xmrHashrate, now]
    );
    return { success: true, id };
  }

  // The network page's view: one row per online HOST — a rig running several
  // cards on one payout address (workers "<name>/gpu0", "/gpu1", …) is combined
  // into a single row that sums the cards' hashrate/VRAM and nests them for the
  // expandable per-card breakdown. `totalOnline` counts distinct online nodes
  // (those host rows — a rig, not a payout address; one address can run several)
  // and `totalWorkers` counts individual online GPUs (so the stat cards read
  // "N nodes / M GPUs"). Prunes dead rows; offline workers excluded.
  async getPublicMiners() {
    const now = Date.now();
    await this.db.query('DELETE FROM miners WHERE last_seen < $1', [now - PRUNE_TTL]);
    // Named columns, not `SELECT *`: this is an unauthenticated endpoint that the
    // network page hits every 15 seconds per open tab, and the columns below are
    // all it serves.
    const r = await this.db.query(
      `SELECT address, worker, gpu, hashrate, accepted, vram_used, vram_total,
              version, llm_model, node_id, ram_used, ram_total, xmr_hashrate, last_seen
         FROM miners WHERE last_seen >= $1`,
      [now - OFFLINE_THRESHOLD]
    );

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
      llmModel: row.llm_model || null,
      nodeId: row.node_id || null, // set only while the host serves cluster jobs
      // The machine's, so the same on every card of a host (see groupHosts).
      // NULL is "not reported" and stays null rather than reading as 0.
      ramUsedMb: readNullable(row.ram_used, Math.round),
      ramTotalMb: readNullable(row.ram_total, Math.round),
      xmrHashrate: readNullable(row.xmr_hashrate, (n) => +n.toFixed(1)),
      lastMs: Number(row.last_seen), // always positive: the WHERE clause filters on last_seen
    })));

    const hosts = groupHosts(cards, now);
    return {
      miners: hosts,
      totalOnline: hosts.length,   // distinct online nodes (rigs), grouped by (address, worker)
      totalWorkers: cards.length,  // one entry per online GPU/worker
      totalHashrate: +cards.reduce((a, c) => a + c.hash, 0).toFixed(1),
    };
  }
}

MinerService.minerFingerprint = minerFingerprint;
MinerService.isValidAddress = isValidAddress;
MinerService.normalizeAddress = normalizeAddress;
MinerService.clampNum = clampNum;
MinerService.clampOrNull = clampOrNull;
MinerService.formatAgo = formatAgo;
MinerService.baseWorker = baseWorker;
MinerService.dropHostAggregates = dropHostAggregates;
MinerService.groupHosts = groupHosts;
MinerService.newestCard = newestCard;

module.exports = MinerService;
