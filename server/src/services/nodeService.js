const crypto = require('crypto');

const NODE_TTL_MS = 7 * 24 * 60 * 60 * 1000; // prune nodes not seen in a week
const OFFLINE_THRESHOLD = 15 * 60 * 1000;    // mark offline after 15 minutes

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

    const existing = await this.db.query('SELECT user_id FROM nodes WHERE node_id = $1', [nodeId]);
    if (existing.rows.length > 0) {
      const owner = existing.rows[0].user_id;
      if (owner && owner !== userId) {
        return { error: 'Node already claimed by another user' };
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
    await this.db.query('DELETE FROM nodes WHERE last_seen < $1', [now - NODE_TTL_MS]);

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

module.exports = NodeService;
