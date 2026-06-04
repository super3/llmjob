const NodeService = require('../src/services/nodeService');
const { createCamelClient } = require('./helpers/camelRedis');

// Covers the dashboard telemetry surfaced through the Nodes table: extra
// fields on updateNodeStatus and the computed uptime string in getUserNodes.
describe('NodeService dashboard telemetry', () => {
  let redisClient;
  let service;

  beforeEach(async () => {
    redisClient = createCamelClient();
    service = new NodeService(redisClient);
    await redisClient.flushall();
  });

  afterEach(async () => {
    await redisClient.quit();
  });

  it('persists telemetry fields on ping and exposes them in getUserNodes', async () => {
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

  // Drive each branch of the uptime formatter via crafted claimedAt values.
  const seedNode = async (userId, claimedAt, status = 'online') => {
    const nodeId = NodeService.generateNodeFingerprint(userId + claimedAt);
    await redisClient.set(`node:${nodeId}`, JSON.stringify({
      nodeId, name: 'n', userId, status, isPublic: false,
      lastSeen: Date.now(), claimedAt
    }));
    await redisClient.sAdd(`user_nodes:${userId}`, nodeId);
    return nodeId;
  };

  it('formats days + hours of uptime', async () => {
    const ms = (3 * 24 * 60 + 4 * 60) * 60 * 1000; // 3d 4h
    await seedNode('u-days', Date.now() - ms);
    const nodes = await service.getUserNodes('u-days');
    expect(nodes[0].uptime).toMatch(/^3d 4h$/);
  });

  it('formats hours + minutes of uptime', async () => {
    const ms = (2 * 60 + 5) * 60 * 1000; // 2h 5m
    await seedNode('u-hours', Date.now() - ms);
    const nodes = await service.getUserNodes('u-hours');
    expect(nodes[0].uptime).toMatch(/^2h \d+m$/);
  });

  it('formats minutes-only uptime', async () => {
    await seedNode('u-mins', Date.now() - 12 * 60 * 1000); // 12m
    const nodes = await service.getUserNodes('u-mins');
    expect(nodes[0].uptime).toMatch(/^\d+m$/);
  });

  it('falls back to 0m when claimedAt is missing', async () => {
    const nodeId = NodeService.generateNodeFingerprint('u-noclaim');
    await redisClient.set(`node:${nodeId}`, JSON.stringify({
      nodeId, name: 'n', userId: 'u-noclaim', status: 'online',
      isPublic: false, lastSeen: Date.now()
    }));
    await redisClient.sAdd('user_nodes:u-noclaim', nodeId);
    const nodes = await service.getUserNodes('u-noclaim');
    expect(nodes[0].uptime).toBe('0m');
  });

  it('reports null uptime for an offline node', async () => {
    // claimedAt long ago AND lastSeen old -> getUserNodes marks it offline.
    const nodeId = NodeService.generateNodeFingerprint('u-off');
    await redisClient.set(`node:${nodeId}`, JSON.stringify({
      nodeId, name: 'n', userId: 'u-off', status: 'online',
      isPublic: false, lastSeen: Date.now() - 20 * 60 * 1000,
      claimedAt: Date.now() - 5 * 60 * 60 * 1000
    }));
    await redisClient.sAdd('user_nodes:u-off', nodeId);
    const nodes = await service.getUserNodes('u-off');
    expect(nodes[0].status).toBe('offline');
    expect(nodes[0].uptime).toBeNull();
  });
});
