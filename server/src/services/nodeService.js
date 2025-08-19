const crypto = require('crypto');

const NODE_TTL = 15 * 60; // 15 minutes in seconds
const NODE_PREFIX = 'node:';
const USER_NODES_PREFIX = 'user_nodes:';

function generateNodeFingerprint(publicKey) {
  // Generate a short fingerprint from the public key
  const hash = crypto.createHash('sha256').update(publicKey).digest('hex');
  return hash.substring(0, 6);
}

async function claimNode(redis, publicKey, name, userId) {
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
  
  // Create or update node
  const nodeData = {
    nodeId,
    publicKey,
    name,
    userId,
    status: 'offline',
    isPublic: false,
    lastSeen: Date.now(),
    claimedAt: Date.now()
  };
  
  // Store node data
  await redis.set(nodeKey, JSON.stringify(nodeData));
  
  // Add to user's node list
  const userNodesKey = `${USER_NODES_PREFIX}${userId}`;
  await redis.sadd(userNodesKey, nodeId);
  
  return {
    success: true,
    nodeId,
    message: 'Node claimed successfully'
  };
}

async function updateNodeStatus(redis, nodeId, publicKey) {
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
  
  // Update node status
  node.status = 'online';
  node.lastSeen = Date.now();
  
  // Store with TTL
  await redis.setex(nodeKey, NODE_TTL, JSON.stringify(node));
  
  return {
    success: true,
    status: 'online',
    message: 'Node status updated'
  };
}

async function getUserNodes(redis, userId) {
  const userNodesKey = `${USER_NODES_PREFIX}${userId}`;
  
  // Get all node IDs for user
  const nodeIds = await redis.smembers(userNodesKey);
  
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
      if (timeSinceLastSeen > NODE_TTL * 1000) {
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

async function getPublicNodes(redis) {
  // Get all node keys
  const keys = await redis.keys(`${NODE_PREFIX}*`);
  
  if (!keys || keys.length === 0) {
    return [];
  }
  
  const publicNodes = [];
  for (const key of keys) {
    const nodeData = await redis.get(key);
    
    if (nodeData) {
      const node = JSON.parse(nodeData);
      
      // Only include public nodes
      if (node.isPublic) {
        // Check if node should be marked as offline
        const timeSinceLastSeen = Date.now() - node.lastSeen;
        if (timeSinceLastSeen > NODE_TTL * 1000) {
          node.status = 'offline';
        }
        
        publicNodes.push({
          nodeId: node.nodeId,
          name: node.name,
          status: node.status,
          lastSeen: node.lastSeen
        });
      }
    }
  }
  
  return publicNodes;
}

async function updateNodeVisibility(redis, nodeId, userId, isPublic) {
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
    await redis.setex(nodeKey, ttl, JSON.stringify(node));
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

async function checkNodeStatuses(redis) {
  // This function is called periodically to clean up expired nodes
  // Redis TTL will automatically remove nodes that haven't pinged in 15 minutes
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

module.exports = {
  claimNode,
  updateNodeStatus,
  getUserNodes,
  getPublicNodes,
  updateNodeVisibility,
  checkNodeStatuses,
  generateNodeFingerprint
};