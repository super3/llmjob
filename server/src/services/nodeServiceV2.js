const NodeRepository = require('../repositories/NodeRepository');

/**
 * Refactored NodeService using Repository pattern
 */
class NodeServiceV2 {
  constructor(redis) {
    this.nodeRepo = new NodeRepository(redis);
  }

  /**
   * Generate node fingerprint from public key
   */
  generateNodeFingerprint(publicKey) {
    return this.nodeRepo.generateNodeFingerprint(publicKey);
  }

  /**
   * Claim a node for a user
   */
  async claimNode(publicKey, name, userId) {
    try {
      const node = await this.nodeRepo.claimNode(publicKey, name, userId);
      return {
        success: true,
        nodeId: node.nodeId,
        name: node.name,
        status: node.status
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Update node status
   */
  async updateNodeStatus(nodeId, publicKey, additionalData = {}) {
    const node = await this.nodeRepo.getNode(nodeId);
    
    if (!node) {
      return { error: 'Node not found. Please claim the node first.' };
    }

    if (node.publicKey !== publicKey) {
      return { error: 'Public key mismatch' };
    }

    // Update node status
    await this.nodeRepo.markNodeOnline(nodeId, additionalData);

    return {
      success: true,
      status: 'online',
      message: 'Node status updated'
    };
  }

  /**
   * Get a specific node
   */
  async getNode(nodeId) {
    return await this.nodeRepo.getNode(nodeId);
  }

  /**
   * Get all nodes for a user
   */
  async getUserNodes(userId) {
    return await this.nodeRepo.getUserNodes(userId);
  }

  /**
   * Get public nodes
   */
  async getPublicNodes(limit = 100) {
    const nodes = await this.nodeRepo.getPublicNodes(limit);
    return {
      success: true,
      nodes,
      count: nodes.length
    };
  }

  /**
   * Update node visibility
   */
  async updateNodeVisibility(nodeId, userId, isPublic) {
    try {
      const updatedNode = await this.nodeRepo.updateNodeVisibility(nodeId, userId, isPublic);
      
      return {
        success: true,
        nodeId,
        isPublic,
        message: `Node is now ${isPublic ? 'public' : 'private'}`
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Check and update node status based on last seen time
   */
  async checkNodeStatus(nodeId) {
    return await this.nodeRepo.checkNodeStatus(nodeId);
  }

  /**
   * Get node statistics
   */
  async getNodeStats() {
    return await this.nodeRepo.getNodeStats();
  }

  /**
   * Clean up inactive nodes
   */
  async cleanupInactiveNodes(maxInactiveTime) {
    return await this.nodeRepo.cleanupInactiveNodes(maxInactiveTime);
  }

  /**
   * Update node capabilities
   */
  async updateNodeCapabilities(nodeId, publicKey, capabilities) {
    const node = await this.nodeRepo.getNode(nodeId);
    
    if (!node) {
      return { error: 'Node not found' };
    }

    if (node.publicKey !== publicKey) {
      return { error: 'Public key mismatch' };
    }

    await this.nodeRepo.updateNode(nodeId, { capabilities });

    return {
      success: true,
      nodeId,
      capabilities
    };
  }

  /**
   * Update node job information
   */
  async updateNodeJobInfo(nodeId, activeJobs, maxConcurrentJobs) {
    const node = await this.nodeRepo.getNode(nodeId);
    
    if (!node) {
      return { error: 'Node not found' };
    }

    await this.nodeRepo.updateNode(nodeId, {
      activeJobs,
      maxConcurrentJobs,
      lastActivity: Date.now()
    });

    return {
      success: true,
      nodeId,
      activeJobs,
      maxConcurrentJobs
    };
  }

  /**
   * Get nodes by status
   */
  async getNodesByStatus(status) {
    const allNodes = await this.nodeRepo.keys('*');
    const nodes = [];

    for (const key of allNodes) {
      const nodeId = key.replace('node:', '');
      const node = await this.nodeRepo.getNode(nodeId);
      
      if (node && node.status === status) {
        nodes.push(node);
      }
    }

    return nodes;
  }

  /**
   * Count nodes for a user
   */
  async countUserNodes(userId) {
    return await this.nodeRepo.countUserNodes(userId);
  }

  /**
   * Check if a node is public
   */
  async isPublicNode(nodeId) {
    return await this.nodeRepo.isPublicNode(nodeId);
  }

  /**
   * Validate node ownership
   */
  async validateNodeOwnership(nodeId, userId) {
    const node = await this.nodeRepo.getNode(nodeId);
    
    if (!node) {
      return { valid: false, error: 'Node not found' };
    }

    if (node.userId !== userId) {
      return { valid: false, error: 'Unauthorized: You do not own this node' };
    }

    return { valid: true, node };
  }

  /**
   * Get online nodes
   */
  async getOnlineNodes() {
    return await this.getNodesByStatus('online');
  }

  /**
   * Get offline nodes
   */
  async getOfflineNodes() {
    return await this.getNodesByStatus('offline');
  }

  /**
   * Bulk update node statuses
   */
  async bulkUpdateNodeStatuses() {
    const allNodes = await this.nodeRepo.keys('*');
    let updatedCount = 0;

    for (const key of allNodes) {
      const nodeId = key.replace('node:', '');
      const status = await this.nodeRepo.checkNodeStatus(nodeId);
      if (status) {
        updatedCount++;
      }
    }

    return {
      checked: allNodes.length,
      updated: updatedCount
    };
  }
}

// Export as both class and singleton factory
module.exports = NodeServiceV2;

// Also export individual functions for backward compatibility
module.exports.createNodeService = (redis) => new NodeServiceV2(redis);
module.exports.nodeService = {
  generateNodeFingerprint: (publicKey) => {
    const repo = new NodeRepository(null);
    return repo.generateNodeFingerprint(publicKey);
  },
  claimNode: async (redisClient, publicKey, name, userId) => {
    const service = new NodeServiceV2(redisClient);
    return await service.claimNode(publicKey, name, userId);
  },
  updateNodeStatus: async (redisClient, nodeId, publicKey, additionalData) => {
    const service = new NodeServiceV2(redisClient);
    return await service.updateNodeStatus(nodeId, publicKey, additionalData);
  },
  getNode: async (nodeId, redisClient) => {
    const service = new NodeServiceV2(redisClient);
    return await service.getNode(nodeId);
  },
  getUserNodes: async (redisClient, userId) => {
    const service = new NodeServiceV2(redisClient);
    return await service.getUserNodes(userId);
  },
  getPublicNodes: async (redisClient) => {
    const service = new NodeServiceV2(redisClient);
    return await service.getPublicNodes();
  },
  updateNodeVisibility: async (redisClient, nodeId, userId, isPublic) => {
    const service = new NodeServiceV2(redisClient);
    return await service.updateNodeVisibility(nodeId, userId, isPublic);
  }
};