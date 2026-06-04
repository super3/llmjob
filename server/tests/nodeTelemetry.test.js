const NodeService = require('../src/services/nodeService');
const { createTestDb } = require('./helpers/pgmem');

// Covers the dashboard telemetry surfaced through the Nodes table: extra
// fields on updateNodeStatus and the computed uptime string in getUserNodes.
describe('NodeService dashboard telemetry', () => {
  let db;
  let service;

  beforeEach(async () => {
    db = await createTestDb();
    service = new NodeService(db);
  });

  afterEach(async () => {
    if (db.end) await db.end();
  });

  // Insert a node row directly, with sensible defaults for an online node.
  const seed = (nodeId, overrides = {}) => {
    const n = {
      name: 'n', user_id: 'u', status: 'online', is_public: false,
      last_seen: Date.now(), claimed_at: Date.now(), ...overrides
    };
    return db.query(
      `INSERT INTO nodes (node_id, name, user_id, status, is_public, last_seen, claimed_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7)`,
      [nodeId, n.name, n.user_id, n.status, n.is_public, n.last_seen, n.claimed_at]
    );
  };

  it('persists telemetry on ping and exposes it in getUserNodes', async () => {
    await service.claimNode('pk-telemetry', 'rig4090', 'user1');
    const nodeId = NodeService.generateNodeFingerprint('pk-telemetry');

    await service.updateNodeStatus(nodeId, 'pk-telemetry', {
      device: 'RTX 4090', vramTotal: 24, vramUsed: 15.8,
      model: 'gemma-4-26B-A4B', quant: 'Q4_K_M', tps: 94
    });

    const nodes = await service.getUserNodes('user1');
    expect(nodes[0]).toMatchObject({
      device: 'RTX 4090', vramTotal: 24, vramUsed: 15.8,
      model: 'gemma-4-26B-A4B', quant: 'Q4_K_M', tps: 94
    });
    expect(nodes[0].uptime).toEqual(expect.any(String));
  });

  it('returns null telemetry for a freshly claimed node', async () => {
    await service.claimNode('pk-bare', 'rig-bare', 'user2');
    const nodes = await service.getUserNodes('user2');
    expect(nodes[0]).toMatchObject({
      device: null, vramTotal: null, vramUsed: null,
      model: null, quant: null, tps: null
    });
  });

  it('formats days + hours of uptime', async () => {
    const ms = (3 * 24 * 60 + 4 * 60) * 60 * 1000; // 3d 4h
    await seed('ndays', { user_id: 'u-days', claimed_at: Date.now() - ms });
    const nodes = await service.getUserNodes('u-days');
    expect(nodes[0].uptime).toMatch(/^3d 4h$/);
  });

  it('formats hours + minutes of uptime', async () => {
    const ms = (2 * 60 + 5) * 60 * 1000; // 2h 5m
    await seed('nhours', { user_id: 'u-hours', claimed_at: Date.now() - ms });
    const nodes = await service.getUserNodes('u-hours');
    expect(nodes[0].uptime).toMatch(/^2h \d+m$/);
  });

  it('formats minutes-only uptime', async () => {
    await seed('nmins', { user_id: 'u-mins', claimed_at: Date.now() - 12 * 60 * 1000 });
    const nodes = await service.getUserNodes('u-mins');
    expect(nodes[0].uptime).toMatch(/^\d+m$/);
  });

  it('falls back to 0m when claimedAt is missing', async () => {
    await seed('nnoclaim', { user_id: 'u-noclaim', claimed_at: null });
    const nodes = await service.getUserNodes('u-noclaim');
    expect(nodes[0].uptime).toBe('0m');
  });

  it('reports null uptime for an offline node', async () => {
    await seed('noff', {
      user_id: 'u-off', last_seen: Date.now() - 20 * 60 * 1000,
      claimed_at: Date.now() - 5 * 60 * 60 * 1000
    });
    const nodes = await service.getUserNodes('u-off');
    expect(nodes[0].status).toBe('offline');
    expect(nodes[0].uptime).toBeNull();
  });
});
