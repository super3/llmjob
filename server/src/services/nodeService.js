const crypto = require('crypto');

const NODE_TTL_MS = 7 * 24 * 60 * 60 * 1000; // prune nodes not seen in a week
const OFFLINE_THRESHOLD = 15 * 60 * 1000;    // mark offline after 15 minutes

// Weight on a new speed sample. Low enough that one unlucky job (a node that
// picked up mining mid-generation) doesn't swing the number that decides what
// work it gets, high enough to follow a real hardware change within a handful of
// jobs rather than a day.
const SPEED_ALPHA = 0.3;
// A measurement older than this is treated as unknown: a node that has been idle
// this long may have swapped cards, changed models, or started co-running the
// miner, and the stale figure would gate it on evidence that no longer holds.
const SPEED_STALE_MS = 6 * 60 * 60 * 1000;
// Discard a sample claiming more than this. The token count comes from the node's
// own final-chunk metrics, and the rate now decides which jobs it is offered — so
// over-reporting tokens is a way to attract longer work than the hardware can
// actually finish. No single GPU serving this model comes near 1000 tok/s (the
// fastest card measured 45), so anything above it is a broken client or a lying
// one; either way it is not a measurement. Bogus samples are dropped rather than
// clamped, so they don't pull the average up to the ceiling either.
const MAX_SAMPLE_TPS = 1000;

// Generate a short fingerprint from a public key.
function generateNodeFingerprint(publicKey) {
  const hash = crypto.createHash('sha256').update(publicKey).digest('hex');
  return hash.substring(0, 6);
}

// Render a duration in ms as a compact uptime string, e.g. "3d 4h" or "12m".
function formatUptime(ms) {
  if (!ms || ms < 0) return '0m';
  const minutes = Math.floor(ms / 60000);
  const days = Math.floor(minutes / 1440);
  const hours = Math.floor((minutes % 1440) / 60);
  const mins = minutes % 60;
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}

const num = (v) => (v == null ? null : Number(v));

class NodeService {
  constructor(db) {
    this.db = db;
  }

  async claimNode(publicKey, name, userId) {
    const nodeId = generateNodeFingerprint(publicKey);

    const existing = await this.db.query('SELECT user_id, public_key FROM nodes WHERE node_id = $1', [nodeId]);
    if (existing.rows.length > 0) {
      const owner = existing.rows[0].user_id;
      if (owner && owner !== userId) {
        return { error: 'Node already claimed by another user' };
      }
      // The nodeId is a fingerprint of the public key, so an existing row under
      // this id with a DIFFERENT key means a fingerprint collision — a genuine
      // clash, or an attacker who ground out a key colliding with a real node's
      // id to hijack it. Refuse rather than let the upsert overwrite the
      // registered key: doing so would lock the real machine out (its signed
      // pings would stop matching) and hand the fingerprint to the caller. A
      // legitimate re-claim always presents the same key, so it's unaffected.
      const existingKey = existing.rows[0].public_key;
      if (existingKey && existingKey !== publicKey) {
        return { error: 'Node key mismatch' };
      }
    }

    const now = Date.now();
    await this.db.query(
      `INSERT INTO nodes (node_id, public_key, name, user_id, status, is_public, last_seen, claimed_at)
       VALUES ($1, $2, $3, $4, 'online', false, $5, $5)
       ON CONFLICT (node_id) DO UPDATE SET
         public_key = EXCLUDED.public_key, name = EXCLUDED.name, user_id = EXCLUDED.user_id,
         status = 'online', is_public = false, last_seen = EXCLUDED.last_seen, claimed_at = EXCLUDED.claimed_at`,
      [nodeId, publicKey, name, userId, now]
    );

    return { success: true, nodeId, message: 'Node claimed successfully' };
  }

  // Self-register a node with NO account behind it. The caller has already
  // proven it holds the secret key for `publicKey` (verifySignature), which is
  // all an unclaimed node needs: assignJobsToNode reads the row's user_id, and a
  // NULL owner can only ever be handed jobs that aren't private
  // (visibility <> 'private'), so an anonymous machine sees public work only.
  //
  // This is what lets a rig contribute without linking to an account. Previously
  // a node row existed only after a join token or a Clerk claim, so an unlinked
  // box ran the model purely for itself — ping said "claim the node first" and
  // /jobs/poll 404'd.
  //
  // Idempotent, and it NEVER touches user_id: re-registering an already-claimed
  // node just marks it online, so a claimed rig can't be silently orphaned by
  // anyone who learns its public key (they'd need its secret key to get here at
  // all, but the invariant is worth keeping explicit).
  async registerNode(publicKey, name) {
    const nodeId = generateNodeFingerprint(publicKey);
    const now = Date.now();

    const existing = await this.db.query('SELECT user_id FROM nodes WHERE node_id = $1', [nodeId]);
    if (existing.rows.length > 0) {
      await this.db.query(
        "UPDATE nodes SET status = 'online', last_seen = $2 WHERE node_id = $1",
        [nodeId, now]
      );
      return { success: true, nodeId, claimed: !!existing.rows[0].user_id };
    }

    await this.db.query(
      `INSERT INTO nodes (node_id, public_key, name, user_id, status, is_public, last_seen)
       VALUES ($1, $2, $3, NULL, 'online', false, $4)
       ON CONFLICT (node_id) DO NOTHING`,
      [nodeId, publicKey, name || `Node-${nodeId}`, now]
    );
    return { success: true, nodeId, claimed: false };
  }

  async updateNodeStatus(nodeId, publicKey, additionalData = {}) {
    const r = await this.db.query('SELECT * FROM nodes WHERE node_id = $1', [nodeId]);
    if (r.rows.length === 0) {
      return { error: 'Node not found. Please claim the node first.' };
    }
    const node = r.rows[0];
    if (node.public_key !== publicKey) {
      return { error: 'Public key mismatch' };
    }

    // Keep existing values unless the ping provides a new one. `name` is
    // stricter — only a non-null name updates, so a sparse keep-alive ping never
    // clears the stored rig name; a real name lets clients rename via Settings.
    const pick = (val, current) => (val !== undefined ? val : current);
    const capabilities = pick(additionalData.capabilities, node.capabilities);
    const name = additionalData.name != null ? additionalData.name : node.name;

    await this.db.query(
      `UPDATE nodes SET status = 'online', last_seen = $2, capabilities = $3,
         active_jobs = $4, max_concurrent_jobs = $5, device = $6, vram_total = $7,
         vram_used = $8, model = $9, quant = $10, tps = $11, name = $12
       WHERE node_id = $1`,
      [
        nodeId, Date.now(),
        capabilities == null ? null : JSON.stringify(capabilities),
        pick(additionalData.activeJobs, node.active_jobs),
        pick(additionalData.maxConcurrentJobs, node.max_concurrent_jobs),
        pick(additionalData.device, node.device),
        pick(additionalData.vramTotal, node.vram_total),
        pick(additionalData.vramUsed, node.vram_used),
        pick(additionalData.model, node.model),
        pick(additionalData.quant, node.quant),
        pick(additionalData.tps, node.tps),
        name
      ]
    );

    return { success: true, status: 'online', message: 'Node status updated' };
  }

  async getUserNodes(userId) {
    const r = await this.db.query('SELECT * FROM nodes WHERE user_id = $1 ORDER BY seq', [userId]);
    const now = Date.now();
    return r.rows.map((node) => {
      const lastSeen = num(node.last_seen);
      const claimedAt = num(node.claimed_at);
      const status = (now - lastSeen > OFFLINE_THRESHOLD) ? 'offline' : node.status;
      return {
        nodeId: node.node_id,
        name: node.name,
        status,
        isPublic: node.is_public,
        lastSeen,
        device: node.device || null,
        vramTotal: num(node.vram_total),
        vramUsed: num(node.vram_used),
        model: node.model || null,
        quant: node.quant || null,
        tps: num(node.tps),
        uptime: status === 'online' ? formatUptime(claimedAt == null ? null : now - claimedAt) : null
      };
    });
  }

  // One node's existence + liveness, for the gateway to fast-fail a request that
  // targets a node which can't possibly serve it (rather than long-polling to the
  // full timeout). `online` means the node is registered and has pinged within the
  // offline threshold. Unknown ids come back { exists: false, online: false }.
  async getNodeStatus(nodeId) {
    const r = await this.db.query('SELECT status, last_seen FROM nodes WHERE node_id = $1', [nodeId]);
    if (!r.rows.length) return { exists: false, online: false };
    const node = r.rows[0];
    const online = node.status === 'online' && (Date.now() - num(node.last_seen)) <= OFFLINE_THRESHOLD;
    return { exists: true, online };
  }

  // Fold one server-measured generation into the node's running speed.
  //
  // `tokens` is the node's own count of what it produced — that part it can be
  // trusted on, since undercounting only makes it look slower. `elapsedMs` is the
  // server's clock, which is the half that matters: a node can claim any tok/s it
  // likes on its ping, but it can't make the gateway's stopwatch run slower.
  //
  // `replace` overwrites instead of blending, for the first benchmark of a cold
  // node — that run includes model load and KV warm-up and can read many times
  // slower than steady state, so blending it in would defame a fast node for the
  // next several jobs.
  async recordSpeedSample(nodeId, tokens, elapsedMs, opts = {}) {
    const t = Number(tokens);
    const ms = Number(elapsedMs);
    if (!nodeId || !Number.isFinite(t) || t <= 0 || !Number.isFinite(ms) || ms <= 0) return null;

    const rate = t / (ms / 1000);
    if (rate > MAX_SAMPLE_TPS) return null;

    const r = await this.db.query('SELECT measured_tps, speed_samples FROM nodes WHERE node_id = $1', [nodeId]);
    if (!r.rows.length) return null;

    const prev = num(r.rows[0].measured_tps);
    const samples = Number(r.rows[0].speed_samples) || 0;
    const next = (opts.replace || prev == null) ? rate : prev * (1 - SPEED_ALPHA) + rate * SPEED_ALPHA;

    await this.db.query(
      'UPDATE nodes SET measured_tps = $2, speed_samples = $3, speed_at = $4 WHERE node_id = $1',
      [nodeId, next, samples + 1, Date.now()]
    );
    return { nodeId, tps: next, samples: samples + 1 };
  }

  // What we know about a node's speed right now. `stale` is what callers act on:
  // a measurement past SPEED_STALE_MS describes hardware that may no longer be
  // there, so it neither gates the node nor counts as "measured".
  async getSpeed(nodeId) {
    const r = await this.db.query(
      'SELECT measured_tps, speed_samples, speed_at FROM nodes WHERE node_id = $1', [nodeId]
    );
    if (!r.rows.length) return null;
    const row = r.rows[0];
    const at = num(row.speed_at);
    const tps = num(row.measured_tps);
    const stale = at == null || (Date.now() - at) > SPEED_STALE_MS;
    return { tps, samples: Number(row.speed_samples) || 0, at, stale, known: tps != null && !stale };
  }

  // Every node currently serving the fleet, with what we measured it at.
  //
  // This is the only way to enumerate the serving fleet: getPublicNodes() lists
  // nodes a user flagged public (usually none), and the miner board only shows
  // machines that are mining — so a node that serves jobs without mining appears
  // in neither, and the fleet could only be counted by probing node ids we
  // happened to have seen before.
  async listServingNodes() {
    const r = await this.db.query(
      'SELECT node_id, name, device, model, measured_tps, speed_samples, speed_at, last_seen FROM nodes WHERE last_seen >= $1 ORDER BY measured_tps DESC NULLS LAST',
      [Date.now() - OFFLINE_THRESHOLD]
    );
    const now = Date.now();
    return r.rows.map((row) => {
      const at = num(row.speed_at);
      const tps = num(row.measured_tps);
      const stale = at == null || (now - at) > SPEED_STALE_MS;
      return {
        nodeId: row.node_id,
        name: row.name || null,
        device: row.device || null,
        model: row.model || null,
        tps: tps == null ? null : Math.round(tps * 10) / 10,
        samples: Number(row.speed_samples) || 0,
        measuredAt: at,
        stale,
        lastSeen: num(row.last_seen),
      };
    });
  }

  async getPublicNodes() {
    const r = await this.db.query('SELECT * FROM nodes', []);
    const now = Date.now();
    const nodes = [];
    let totalOnline = 0;

    for (const node of r.rows) {
      const isOnline = (now - num(node.last_seen)) <= OFFLINE_THRESHOLD;
      if (isOnline && node.status === 'online') {
        totalOnline++;
      }
      if (node.is_public) {
        nodes.push({
          nodeId: node.node_id,
          name: node.name,
          status: isOnline ? node.status : 'offline',
          lastSeen: num(node.last_seen)
        });
      }
    }

    return { nodes, totalOnline };
  }

  async updateNodeVisibility(nodeId, userId, isPublic) {
    const r = await this.db.query('SELECT user_id FROM nodes WHERE node_id = $1', [nodeId]);
    if (r.rows.length === 0) {
      return { error: 'Node not found', status: 404 };
    }
    if (r.rows[0].user_id !== userId) {
      return { error: 'Unauthorized: You do not own this node', status: 403 };
    }

    await this.db.query('UPDATE nodes SET is_public = $2 WHERE node_id = $1', [nodeId, isPublic]);

    return {
      success: true,
      nodeId,
      isPublic,
      message: `Node visibility updated to ${isPublic ? 'public' : 'private'}`
    };
  }

  // Prune nodes that haven't pinged within NODE_TTL_MS and log a status summary.
  async checkNodeStatuses() {
    const now = Date.now();
    // Prune UNCLAIMED rows only. A claimed node carries state the user set and
    // cannot recreate by waiting: its name, its is_public flag, and the user_id
    // that makes it eligible for that user's `private` jobs. Deleting it because
    // the rig was off for a week silently downgraded the owner to public-only
    // routing until they noticed and re-claimed. An unclaimed row has nothing
    // worth keeping — it re-registers itself on the next ping.
    await this.db.query(
      'DELETE FROM nodes WHERE last_seen < $1 AND user_id IS NULL',
      [now - NODE_TTL_MS]
    );

    const r = await this.db.query('SELECT last_seen FROM nodes', []);
    let onlineCount = 0;
    let offlineCount = 0;
    for (const row of r.rows) {
      if (now - num(row.last_seen) <= OFFLINE_THRESHOLD) onlineCount++;
      else offlineCount++;
    }
    console.log(`Node status check: ${onlineCount} online, ${offlineCount} offline`);
  }

  async getNode(nodeId) {
    const r = await this.db.query('SELECT * FROM nodes WHERE node_id = $1', [nodeId]);
    if (r.rows.length === 0) {
      return null;
    }
    const node = r.rows[0];
    return {
      nodeId: node.node_id,
      publicKey: node.public_key,
      name: node.name,
      userId: node.user_id,
      status: node.status,
      isPublic: node.is_public,
      lastSeen: num(node.last_seen),
      claimedAt: num(node.claimed_at),
      capabilities: node.capabilities,
      activeJobs: node.active_jobs,
      maxConcurrentJobs: node.max_concurrent_jobs,
      device: node.device,
      vramTotal: num(node.vram_total),
      vramUsed: num(node.vram_used),
      model: node.model,
      quant: node.quant,
      tps: num(node.tps)
    };
  }
}

NodeService.generateNodeFingerprint = generateNodeFingerprint;
NodeService.SPEED_STALE_MS = SPEED_STALE_MS;
NodeService.SPEED_ALPHA = SPEED_ALPHA;
NodeService.MAX_SAMPLE_TPS = MAX_SAMPLE_TPS;

module.exports = NodeService;
