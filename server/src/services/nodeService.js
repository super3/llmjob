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
// And discard one that is too short to mean anything. Below this many tokens the
// wall time is mostly prompt prefill and two HTTP round trips: a 40-token reply
// from a card that really runs at 40 tok/s measures ~8, and that single reading
// was enough to gate the node out of all traffic. It is the same bias that made
// the synthetic benchmark 1024 tokens rather than 256 — here it is simply not a
// measurement, so it is dropped rather than blended in.
const MIN_SAMPLE_TOKENS = 128;

// A node id is a truncated hash of the node's public key, so its WIDTH is the
// network's collision budget.
//
// It was 6 hex characters — 24 bits, 16.7 million ids. The birthday bound puts
// two honest nodes on the same id with ~3% probability at 1,000 nodes, 52% at
// 5,000 and 95% at 10,000. And a collision is not cosmetic: the loser's row
// already exists under someone else's public key, so its signed pings are
// refused for a key mismatch, /jobs/poll 401s, and it can never be claimed to an
// account — while registerNode reported success. A rig would run the model,
// serve nothing, and say nothing about why. For a network whose whole premise is
// pooling everybody's spare GPUs, that was a ceiling in the low thousands.
//
// 16 hex characters is 64 bits, which pushes the same 50% birthday point past a
// billion nodes. It also makes grinding a key that collides with a CHOSEN node
// id cost ~2^64 hashes instead of the ~13 seconds 24 bits took on a laptop.
const NODE_ID_HEX = 16;

// The old width. Machines enrolled before the change keep their 6-character id
// in node.json and go on signing with it, so both widths must be recognised —
// see _enrolledNodeId. Nothing MINTS one of these any more.
const LEGACY_NODE_ID_HEX = 6;

function nodeIdOfWidth(publicKey, hex) {
  return crypto.createHash('sha256')
    .update(String(publicKey == null ? '' : publicKey))
    .digest('hex')
    .slice(0, hex);
}

// Generate a node id from a public key. Mirrors earn/src/shared/node.js
// fingerprint() exactly — the two must agree or a client cannot address itself.
function generateNodeFingerprint(publicKey) {
  return nodeIdOfWidth(publicKey, NODE_ID_HEX);
}

// The id the same key would have had under the old width.
function legacyNodeFingerprint(publicKey) {
  return nodeIdOfWidth(publicKey, LEGACY_NODE_ID_HEX);
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

  // The id this key is (or should be) enrolled under. New nodes get the current
  // 16-character width; the two exceptions both exist so machines that minted a
  // 6-character id under the old scheme keep working.
  //
  // `presentedId` is the id the CALLER says it is. nodeStore persists an id in
  // node.json on first run and signs every later call with it, so a machine that
  // minted one before the widening will go on presenting 6 characters forever.
  // It is honoured only when it equals this key's OWN legacy fingerprint, which
  // is a value only the holder of the key can produce — so a caller can never
  // use it to name a row it does not own.
  async _enrolledNodeId(publicKey, presentedId) {
    const legacy = legacyNodeFingerprint(publicKey);
    const r = await this.db.query('SELECT public_key FROM nodes WHERE node_id = $1', [legacy]);
    const row = r.rows[0];

    // Already ours. Re-registering must not mint a second row and orphan this
    // one, which would silently drop the rig's name, its is_public flag, and the
    // user_id that makes it eligible for its owner's private jobs.
    if (row && row.public_key === publicKey) return legacy;

    // Not enrolled yet, but the caller is signing as its legacy id — a machine
    // whose node.json predates the widening and has never registered (or whose
    // row was pruned). Deriving a wide id here would hand back an id the client
    // never uses, and its very next ping would 404.
    if (!row && presentedId === legacy) return legacy;

    // Occupied by a DIFFERENT key — the old scheme's collision, which used to
    // lock this machine out permanently — or an up-to-date client. Either way the
    // wide id, which is how a machine that used to be stranded enrolls cleanly.
    return generateNodeFingerprint(publicKey);
  }

  async claimNode(publicKey, name, userId, presentedId) {
    const nodeId = await this._enrolledNodeId(publicKey, presentedId);

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
  async registerNode(publicKey, name, presentedId) {
    const nodeId = await this._enrolledNodeId(publicKey, presentedId);
    const now = Date.now();

    const existing = await this.db.query('SELECT user_id, public_key FROM nodes WHERE node_id = $1', [nodeId]);
    if (existing.rows.length > 0) {
      // Same id, different key: a fingerprint collision. Say so.
      //
      // This used to return success, which was the worst possible answer: the
      // caller believed it had enrolled, while every signed call it made
      // afterwards — ping, /jobs/poll, /complete — was refused for a key
      // mismatch. The rig ran the model, served nothing, and had no way to learn
      // why. claimNode has always refused this case; register now matches it.
      if (existing.rows[0].public_key !== publicKey) {
        return { error: 'Node key mismatch' };
      }
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

    if (t < MIN_SAMPLE_TOKENS) return null;

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
  // The model ids the fleet is actually running right now, most-served first.
  //
  // This is what makes a model REQUESTABLE: /v1/models advertises these, and a
  // job naming one is pinned to a node that has it. A model no live node reports
  // is not offered and does not pin, which keeps the documented contract that an
  // unrecognised id is still served (by whatever node takes it) rather than
  // rejected.
  // `ownerUserId` scopes the answer to one user's own nodes. A private API key's
  // requests never leave its owner's nodes, so listing the whole fleet's models
  // told it about other people's hardware AND handed it names it could not
  // actually reach: pinning to one produced a job no eligible node could take,
  // and the caller waited out the gateway budget for a 504.
  async listNetworkModels(ownerUserId) {
    const scoped = ownerUserId != null;
    const r = await this.db.query(
      `SELECT model, COUNT(*)::int AS nodes FROM nodes
        WHERE last_seen >= $1 AND model IS NOT NULL AND model <> ''
          AND ($2::text IS NULL OR user_id = $2)
        GROUP BY model ORDER BY nodes DESC, model ASC`,
      [Date.now() - OFFLINE_THRESHOLD, scoped ? ownerUserId : null]
    );
    return r.rows.map((row) => ({ id: row.model, nodes: row.nodes }));
  }

  async listServingNodes() {
    // Deliberately narrow: node id, speed and freshness only. `name` is set by the
    // node's owner and `device` is a free-text GPU string, and this endpoint is
    // unauthenticated and covers nodes that never opted into being public — the
    // is_public flag governs getPublicNodes, not this. The network page reads
    // nodeId/tps/stale and nothing else, so publishing more would be exposure
    // without a consumer.
    const r = await this.db.query(
      'SELECT node_id, measured_tps, speed_samples, speed_at, last_seen FROM nodes WHERE last_seen >= $1 ORDER BY measured_tps DESC NULLS LAST',
      [Date.now() - OFFLINE_THRESHOLD]
    );
    const now = Date.now();
    return r.rows.map((row) => {
      const at = num(row.speed_at);
      const tps = num(row.measured_tps);
      const stale = at == null || (now - at) > SPEED_STALE_MS;
      return {
        nodeId: row.node_id,
        tps: tps == null ? null : Math.round(tps * 10) / 10,
        samples: Number(row.speed_samples) || 0,
        measuredAt: at,
        stale,
        lastSeen: num(row.last_seen),
      };
    });
  }

  // The public node board: the nodes their owners flagged public, plus a count
  // of everything online.
  //
  // Both halves are SQL. This is an UNAUTHENTICATED endpoint and it used to be
  // `SELECT * FROM nodes` — every column of every row, public keys included,
  // pulled into the process to be filtered and counted in JS. The count needs no
  // rows at all, and the listing needs four columns of the handful of rows that
  // are actually public.
  async getPublicNodes() {
    const now = Date.now();
    const [listed, counted] = await Promise.all([
      this.db.query(
        `SELECT node_id, name, status, last_seen FROM nodes
          WHERE is_public = true ORDER BY seq`,
        []
      ),
      this.db.query(
        "SELECT count(*)::int AS c FROM nodes WHERE status = 'online' AND last_seen >= $1",
        [now - OFFLINE_THRESHOLD]
      ),
    ]);

    const nodes = listed.rows.map((node) => {
      const lastSeen = num(node.last_seen);
      return {
        nodeId: node.node_id,
        name: node.name,
        status: (now - lastSeen) <= OFFLINE_THRESHOLD ? node.status : 'offline',
        lastSeen
      };
    });

    return { nodes, totalOnline: counted.rows[0].c };
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

    // Counted in SQL. This runs every 60 seconds and its only product is the log
    // line below, so reading a row per node to tally two numbers in JS was the
    // whole nodes table crossing the wire every minute to print a sentence.
    // SUM(CASE …), deliberately, and not the tidier `count(*) FILTER (WHERE …)`:
    // pg-mem accepts FILTER and then IGNORES it, counting every row into both
    // columns, so the aggregate is silently wrong under the test database while
    // looking right in production. SUM(CASE) means the same thing to both.
    // COALESCE because SUM over an empty table is NULL, not 0.
    const r = await this.db.query(
      `SELECT COALESCE(SUM(CASE WHEN last_seen >= $1 THEN 1 ELSE 0 END), 0)::int AS online,
              COALESCE(SUM(CASE WHEN last_seen IS NULL OR last_seen < $1 THEN 1 ELSE 0 END), 0)::int AS offline
         FROM nodes`,
      [now - OFFLINE_THRESHOLD]
    );
    const { online, offline } = r.rows[0];
    console.log(`Node status check: ${online} online, ${offline} offline`);
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
NodeService.legacyNodeFingerprint = legacyNodeFingerprint;
NodeService.NODE_ID_HEX = NODE_ID_HEX;
NodeService.LEGACY_NODE_ID_HEX = LEGACY_NODE_ID_HEX;
NodeService.SPEED_STALE_MS = SPEED_STALE_MS;
NodeService.SPEED_ALPHA = SPEED_ALPHA;
NodeService.MAX_SAMPLE_TPS = MAX_SAMPLE_TPS;
NodeService.MIN_SAMPLE_TOKENS = MIN_SAMPLE_TOKENS;

module.exports = NodeService;
