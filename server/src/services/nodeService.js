const crypto = require('crypto');
const { createRedisCompat } = require('../utils/redisCompat');

const NODE_TTL = 7 * 24 * 60 * 60; // 7 days in seconds - nodes stay in Redis for a week
const OFFLINE_THRESHOLD = 15 * 60 * 1000; // 15 minutes in milliseconds - mark offline after this
const NODE_PREFIX = 'node:';
const USER_NODES_PREFIX = 'user_nodes:';

function generateNodeFingerprint(publicKey) {
  // Generate a short fingerprint from the public key
  const hash = crypto.createHash('sha256').update(publicKey).digest('hex');
  return hash.substring(0, 6);
}

async function claimNode(redisClient, publicKey, name, userId) {
  const redis = createRedisCompat(redisClient);
  const nodeId = generateNodeFingerprint(publicKey);
  const nodeKey = `${NODE_PREFIX}${nodeId}`;
  
  // Check if node already exists
  const existingNode = await redis.get(nodeKey);
  if (existingNode) {
    const node = JSON.parse(existingNode);
    if (node.userId && node.userId !== userId) {
      return { error: 'Node already claimed by another user' };
    }
  }
  
  // Create or update node - mark as online immediately when claimed
  const nodeData = {
    nodeId,
    publicKey,
    name,
    userId,
    status: 'online',
    isPublic: false,
    lastSeen: Date.now(),
    claimedAt: Date.now()
  };
  
  // Store node data with TTL as if it just pinged
  await redis.setEx(nodeKey, NODE_TTL, JSON.stringify(nodeData));
  
  // Add to user's node list
  const userNodesKey = `${USER_NODES_PREFIX}${userId}`;
  await redis.sAdd(userNodesKey, nodeId);
  
  return {
    success: true,
    nodeId,
    message: 'Node claimed successfully'
  };
}

async function updateNodeStatus(redisClient, nodeId, publicKey, additionalData = {}) {
  const redis = createRedisCompat(redisClient);
  const nodeKey = `${NODE_PREFIX}${nodeId}`;
  
  // Get existing node
  const existingNodeData = await redis.get(nodeKey);
  if (!existingNodeData) {
    return { error: 'Node not found. Please claim the node first.' };
  }
  
  const node = JSON.parse(existingNodeData);
  
  // Verify public key matches
  if (node.publicKey !== publicKey) {
    return { error: 'Public key mismatch' };
  }
  
  // Update node status and capabilities if provided
  node.status = 'online';
  node.lastSeen = Date.now();
  
  // Add capabilities and job information if provided
  if (additionalData.capabilities) {
    node.capabilities = additionalData.capabilities;
  }
  if (additionalData.activeJobs !== undefined) {
    node.activeJobs = additionalData.activeJobs;
  }
  if (additionalData.maxConcurrentJobs !== undefined) {
    node.maxConcurrentJobs = additionalData.maxConcurrentJobs;
  }
  
  // Store with TTL
  await redis.setEx(nodeKey, NODE_TTL, JSON.stringify(node));
  
  return {
    success: true,
    status: 'online',
    message: 'Node status updated'
  };
}

// Add a new function to get a node by ID (needed by JobController)
async function getNode(nodeId) {
  const redis = createRedisCompat(this);
  const nodeKey = `${NODE_PREFIX}${nodeId}`;
  
  const nodeData = await redis.get(nodeKey);
  if (!nodeData) {
    return null;
  }
  
  return JSON.parse(nodeData);
}

async function getUserNodes(redisClient, userId) {
  const redis = createRedisCompat(redisClient);
  const userNodesKey = `${USER_NODES_PREFIX}${userId}`;
  
  // Get all node IDs for user
  const nodeIds = await redis.sMembers(userNodesKey);
  
  if (!nodeIds || nodeIds.length === 0) {
    return [];
  }
  
  // Get node data for each ID
  const nodes = [];
  for (const nodeId of nodeIds) {
    const nodeKey = `${NODE_PREFIX}${nodeId}`;
    const nodeData = await redis.get(nodeKey);
    
    if (nodeData) {
      const node = JSON.parse(nodeData);
      // Check if node should be marked as offline
      const timeSinceLastSeen = Date.now() - node.lastSeen;
      if (timeSinceLastSeen > OFFLINE_THRESHOLD) {
        node.status = 'offline';
      }
      nodes.push({
        nodeId: node.nodeId,
        name: node.name,
        status: node.status,
        isPublic: node.isPublic,
        lastSeen: node.lastSeen
      });
    }
  }
  
  return nodes;
}

async function getPublicNodes(redisClient) {
  const redis = createRedisCompat(redisClient);
  // Get all node keys
  const keys = await redis.keys(`${NODE_PREFIX}*`);
  
  if (!keys || keys.length === 0) {
    return { nodes: [], totalOnline: 0 };
  }
  
  const publicNodes = [];
  let totalOnlineCount = 0;
  
  for (const key of keys) {
    const nodeData = await redis.get(key);
    
    if (nodeData) {
      const node = JSON.parse(nodeData);
      
      // Check if node should be marked as offline
      const timeSinceLastSeen = Date.now() - node.lastSeen;
      const isOnline = timeSinceLastSeen <= OFFLINE_THRESHOLD;
      
      if (isOnline && node.status === 'online') {
        totalOnlineCount++;
      }
      
      // Only include public nodes in the detailed list
      if (node.isPublic) {
        publicNodes.push({
          nodeId: node.nodeId,
          name: node.name,
          status: isOnline ? node.status : 'offline',
          lastSeen: node.lastSeen
        });
      }
    }
  }
  
  return { nodes: publicNodes, totalOnline: totalOnlineCount };
}

async function updateNodeVisibility(redisClient, nodeId, userId, isPublic) {
  const redis = createRedisCompat(redisClient);
  const nodeKey = `${NODE_PREFIX}${nodeId}`;
  
  // Get existing node
  const existingNodeData = await redis.get(nodeKey);
  if (!existingNodeData) {
    return { error: 'Node not found', status: 404 };
  }
  
  const node = JSON.parse(existingNodeData);
  
  // Verify ownership
  if (node.userId !== userId) {
    return { error: 'Unauthorized: You do not own this node', status: 403 };
  }
  
  // Update visibility
  node.isPublic = isPublic;
  
  // Store updated node
  const ttl = await redis.ttl(nodeKey);
  if (ttl > 0) {
    await redis.setEx(nodeKey, ttl, JSON.stringify(node));
  } else {
    await redis.set(nodeKey, JSON.stringify(node));
  }
  
  return {
    success: true,
    nodeId,
    isPublic,
    message: `Node visibility updated to ${isPublic ? 'public' : 'private'}`
  };
}

async function checkNodeStatuses(redisClient) {
  const redis = createRedisCompat(redisClient);
  // This function is called periodically to clean up expired nodes
  // Redis TTL will automatically remove nodes that haven't pinged in 7 days
  // This is just for logging/monitoring purposes
  
  const keys = await redis.keys(`${NODE_PREFIX}*`);
  
  if (!keys || keys.length === 0) {
    console.log('Node status check: 0 online, 0 offline');
    return;
  }
  let onlineCount = 0;
  let offlineCount = 0;
  
  for (const key of keys) {
    const ttl = await redis.ttl(key);
    if (ttl > 0) {
      onlineCount++;
    } else {
      offlineCount++;
    }
  }
  
  console.log(`Node status check: ${onlineCount} online, ${offlineCount} offline`);
}

// Create a singleton instance with proper redis access
const nodeService = {
  claimNode,
  updateNodeStatus,
  getUserNodes,
  getPublicNodes,
  updateNodeVisibility,
  checkNodeStatuses,
  generateNodeFingerprint,
  getNode: async (nodeId, redisClient) => {
    const redis = createRedisCompat(redisClient);
    const nodeKey = `${NODE_PREFIX}${nodeId}`;
    
    const nodeData = await redis.get(nodeKey);
    if (!nodeData) {
      return null;
    }
    
    return JSON.parse(nodeData);
  }
};

module.exports = nodeService;