const BaseRepository = require('./BaseRepository');
const crypto = require('crypto');

/**
 * Repository for Node-related Redis operations
 */
class NodeRepository extends BaseRepository {
  constructor(redis) {
    super(redis, 'node:');
    this.userNodesPrefix = 'user_nodes:';
    this.publicNodesKey = 'publicNodes';
    this.NODE_TTL = 7 * 24 * 60 * 60; // 7 days in seconds
    this.OFFLINE_THRESHOLD = 15 * 60 * 1000; // 15 minutes in milliseconds
  }

  generateNodeFingerprint(publicKey) {
    const hash = crypto.createHash('sha256').update(publicKey).digest('hex');
    return hash.substring(0, 6);
  }

  // Node CRUD operations
  async createNode(nodeData) {
    const { nodeId, userId } = nodeData;
    
    // Store node data
    await this.set(nodeId, nodeData, this.NODE_TTL);
    
    // Add to user's node set
    if (userId) {
      const userNodesKey = `${this.userNodesPrefix}${userId}`;
      await this.sAddDirect(userNodesKey, nodeId);
    }
    
    // Add to public nodes if marked as public
    if (nodeData.isPublic) {
      await this.sAddDirect(this.publicNodesKey, nodeId);
    }
    
    return nodeId;
  }

  async getNode(nodeId) {
    return await this.get(nodeId);
  }

  async updateNode(nodeId, updates) {
    const node = await this.getNode(nodeId);
    if (!node) {
      return null;
    }
    
    const updatedNode = { ...node, ...updates };
    await this.set(nodeId, updatedNode, this.NODE_TTL);
    
    // Handle public/private status changes
    if ('isPublic' in updates) {
      if (updates.isPublic) {
        await this.sAddDirect(this.publicNodesKey, nodeId);
      } else {
        await this.sRemDirect(this.publicNodesKey, nodeId);
      }
    }
    
    return updatedNode;
  }

  async deleteNode(nodeId) {
    const node = await this.getNode(nodeId);
    if (!node) {
      return false;
    }
    
    // Remove from user's nodes
    if (node.userId) {
      const userNodesKey = `${this.userNodesPrefix}${node.userId}`;
      await this.sRemDirect(userNodesKey, nodeId);
    }
    
    // Remove from public nodes
    await this.sRemDirect(this.publicNodesKey, nodeId);
    
    // Delete node data
    return await this.delete(nodeId);
  }

  // Node status operations
  async updateNodeStatus(nodeId, status, additionalData = {}) {
    const updates = {
      status,
      lastSeen: Date.now(),
      ...additionalData
    };
    
    return await this.updateNode(nodeId, updates);
  }

  async markNodeOnline(nodeId, additionalData = {}) {
    return await this.updateNodeStatus(nodeId, 'online', additionalData);
  }

  async markNodeOffline(nodeId) {
    return await this.updateNodeStatus(nodeId, 'offline');
  }

  async checkNodeStatus(nodeId) {
    const node = await this.getNode(nodeId);
    if (!node) {
      return null;
    }
    
    const isOnline = Date.now() - node.lastSeen < this.OFFLINE_THRESHOLD;
    
    if (!isOnline && node.status === 'online') {
      await this.markNodeOffline(nodeId);
      return 'offline';
    }
    
    return node.status;
  }

  // User nodes operations
  async getUserNodes(userId) {
    const userNodesKey = `${this.userNodesPrefix}${userId}`;
    let nodeIds = await this.sMembersDirect(userNodesKey);
    
    // Handle potential undefined from redis-mock
    if (!nodeIds || !Array.isArray(nodeIds) || nodeIds.length === 0) {
      return [];
    }
    
    const nodes = [];
    for (const nodeId of nodeIds) {
      const node = await this.getNode(nodeId);
      if (node) {
        // Check and update status based on last seen
        const isOnline = Date.now() - node.lastSeen < this.OFFLINE_THRESHOLD;
        if (!isOnline && node.status === 'online') {
          node.status = 'offline';
        }
        nodes.push(node);
      }
    }
    
    return nodes;
  }

  async countUserNodes(userId) {
    const userNodesKey = `${this.userNodesPrefix}${userId}`;
    const nodeIds = await this.sMembersDirect(userNodesKey);
    return nodeIds ? nodeIds.length : 0;
  }

  // Public nodes operations
  async getPublicNodes(limit = 100) {
    const nodeIds = await this.sMembersDirect(this.publicNodesKey);
    
    if (!nodeIds || nodeIds.length === 0) {
      return [];
    }
    
    const nodes = [];
    for (const nodeId of nodeIds.slice(0, limit)) {
      try {
        const node = await this.getNode(nodeId);
        if (node) {
          // Check TTL to see if node is still online
          const ttl = await this.ttl(nodeId);
          node.isOnline = ttl > 0;
          
          // Update status based on last seen
          if (node.isOnline) {
            const isRecentlyActive = Date.now() - node.lastSeen < this.OFFLINE_THRESHOLD;
            node.status = isRecentlyActive ? 'online' : 'offline';
          } else {
            node.status = 'offline';
          }
          
          nodes.push(node);
        }
      } catch (error) {
        console.error(`Error getting public node ${nodeId}:`, error);
      }
    }
    
    return nodes;
  }

  async addToPublicNodes(nodeId) {
    return await this.sAddDirect(this.publicNodesKey, nodeId);
  }

  async removeFromPublicNodes(nodeId) {
    return await this.sRemDirect(this.publicNodesKey, nodeId);
  }

  async isPublicNode(nodeId) {
    const members = await this.sMembersDirect(this.publicNodesKey);
    return members && members.includes(nodeId);
  }

  // Claim node operation
  async claimNode(publicKey, name, userId) {
    const nodeId = this.generateNodeFingerprint(publicKey);
    const existingNode = await this.getNode(nodeId);
    
    if (existingNode) {
      if (existingNode.userId && existingNode.userId !== userId) {
        throw new Error('Node already claimed by another user');
      }
    }
    
    const nodeData = {
      nodeId,
      publicKey,
      name,
      userId,
      status: 'online',
      lastSeen: Date.now(),
      isPublic: false,
      claimedAt: Date.now(),
      capabilities: existingNode?.capabilities || {},
      activeJobs: 0,
      maxConcurrentJobs: existingNode?.maxConcurrentJobs || 1
    };
    
    await this.createNode(nodeData);
    return nodeData;
  }

  // Node visibility operations
  async updateNodeVisibility(nodeId, userId, isPublic) {
    const node = await this.getNode(nodeId);
    
    if (!node) {
      throw new Error('Node not found');
    }
    
    if (node.userId !== userId) {
      throw new Error('Unauthorized: You do not own this node');
    }
    
    return await this.updateNode(nodeId, { isPublic });
  }

  // Cleanup operations
  async cleanupInactiveNodes(maxInactiveTime = 30 * 24 * 60 * 60 * 1000) { // 30 days
    const allNodeKeys = await this.keys('*');
    let deletedCount = 0;
    
    for (const key of allNodeKeys) {
      const nodeId = key.replace(this.keyPrefix, '');
      const node = await this.getNode(nodeId);
      
      if (node && Date.now() - node.lastSeen > maxInactiveTime) {
        await this.deleteNode(nodeId);
        deletedCount++;
      }
    }
    
    return deletedCount;
  }

  // Statistics
  async getNodeStats() {
    const allNodeKeys = await this.keys('*');
    const publicNodeIds = await this.sMembersDirect(this.publicNodesKey);
    
    let onlineCount = 0;
    let offlineCount = 0;
    
    for (const key of allNodeKeys) {
      const nodeId = key.replace(this.keyPrefix, '');
      const node = await this.getNode(nodeId);
      
      if (node) {
        const isOnline = Date.now() - node.lastSeen < this.OFFLINE_THRESHOLD;
        if (isOnline) {
          onlineCount++;
        } else {
          offlineCount++;
        }
      }
    }
    
    return {
      total: allNodeKeys.length,
      online: onlineCount,
      offline: offlineCount,
      public: publicNodeIds ? publicNodeIds.length : 0
    };
  }
}

module.exports = NodeRepository;