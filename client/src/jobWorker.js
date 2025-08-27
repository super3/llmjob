const EventEmitter = require('events');
const OllamaClient = require('./ollama');

class JobWorker extends EventEmitter {
  constructor(nodeClient, config) {
    super();
    this.nodeClient = nodeClient;
    this.config = config;
    this.ollama = new OllamaClient(config.configDir);
    this.currentJob = null;
    this.isRunning = false;
    this.pollingInterval = null;
    this.heartbeatInterval = null;
    this.capabilities = null;
    this.maxConcurrentJobs = 1;
    this.activeJobs = new Map();
  }

  async initialize() {
    // Load hardware capabilities
    this.capabilities = await this.ollama.loadCapabilities();
    
    // Ensure Ollama is running
    const isRunning = await this.ollama.checkServiceStatus();
    if (!isRunning) {
      throw new Error('Ollama service is not running. Please start it first.');
    }
    
    return this.capabilities;
  }

  // Enhanced ping with capabilities
  async pingWithCapabilities() {
    const timestamp = Date.now();
    const message = `${this.config.nodeId}:${timestamp}`;
    const signature = this.nodeClient.signMessage(message, this.config.secretKey);

    const data = {
      nodeId: this.config.nodeId,
      publicKey: this.config.publicKey,
      signature,
      timestamp,
      capabilities: this.capabilities,
      activeJobs: Array.from(this.activeJobs.keys()),
      maxConcurrentJobs: this.maxConcurrentJobs
    };

    try {
      const response = await this.nodeClient.axiosInstance.post('/api/nodes/ping', data);
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.error || error.message
      };
    }
  }

  // Poll server for jobs
  async pollForJobs() {
    const timestamp = Date.now();
    const message = `${this.config.nodeId}:${timestamp}`;
    const signature = this.nodeClient.signMessage(message, this.config.secretKey);

    const data = {
      nodeId: this.config.nodeId,
      publicKey: this.config.publicKey,
      signature,
      timestamp,
      maxJobs: this.maxConcurrentJobs - this.activeJobs.size
    };

    try {
      const response = await this.nodeClient.axiosInstance.post('/api/jobs/poll', data);
      
      if (response.data.jobs && response.data.jobs.length > 0) {
        return response.data.jobs;
      }
      
      return [];
    } catch (error) {
      this.emit('error', {
        type: 'poll_error',
        error: error.message,
        details: error.response?.data
      });
      return [];
    }
  }

  // Execute a job with Ollama
  async executeJob(job) {
    const jobId = job.id;
    
    // Mark job as active
    this.activeJobs.set(jobId, {
      job,
      startTime: Date.now(),
      status: 'running'
    });

    this.emit('job:started', { jobId, job });

    try {
      // Start heartbeat for this job
      this.startHeartbeat(jobId);

      // Execute inference with streaming
      const startTime = Date.now();
      let chunkIndex = 0;
      let totalTokens = 0;
      let currentMemory = process.memoryUsage();

      const stream = await this.ollama.generate(job.prompt, {
        model: job.model || 'llama3.2:3b',
        stream: true,
        ...job.options
      });

      // Buffer for collecting chunks
      let buffer = '';
      let lastSendTime = Date.now();
      const CHUNK_SIZE = 100; // tokens per chunk
      const CHUNK_INTERVAL = 1000; // ms between chunks

      for await (const part of stream) {
        if (part.response) {
          buffer += part.response;
          totalTokens++;

          // Check if we should send a chunk
          const shouldSend = 
            totalTokens % CHUNK_SIZE === 0 || 
            (Date.now() - lastSendTime) > CHUNK_INTERVAL;

          if (shouldSend && buffer.length > 0) {
            const elapsed = (Date.now() - startTime) / 1000;
            const tokensPerSecond = totalTokens / elapsed;
            currentMemory = process.memoryUsage();

            await this.streamChunk(jobId, {
              chunkIndex: chunkIndex++,
              content: buffer,
              metrics: {
                tokensPerSecond: Math.round(tokensPerSecond * 10) / 10,
                totalTokens,
                elapsedSeconds: Math.round(elapsed * 10) / 10,
                memoryUsageMB: Math.round(currentMemory.heapUsed / 1024 / 1024)
              }
            });

            buffer = '';
            lastSendTime = Date.now();
          }
        }

        // Check for cancellation
        if (this.activeJobs.get(jobId)?.status === 'cancelling') {
          throw new Error('Job cancelled');
        }
      }

      // Send final chunk if there's remaining content
      if (buffer.length > 0) {
        const elapsed = (Date.now() - startTime) / 1000;
        const tokensPerSecond = totalTokens / elapsed;
        currentMemory = process.memoryUsage();

        await this.streamChunk(jobId, {
          chunkIndex: chunkIndex++,
          content: buffer,
          isFinal: true,
          metrics: {
            tokensPerSecond: Math.round(tokensPerSecond * 10) / 10,
            totalTokens,
            elapsedSeconds: Math.round(elapsed * 10) / 10,
            memoryUsageMB: Math.round(currentMemory.heapUsed / 1024 / 1024)
          }
        });
      }

      // Mark job as completed
      await this.completeJob(jobId);
      
      this.emit('job:completed', { 
        jobId, 
        totalTokens,
        duration: (Date.now() - startTime) / 1000
      });

    } catch (error) {
      await this.failJob(jobId, error.message);
      this.emit('job:failed', { jobId, error: error.message });
    } finally {
      // Stop heartbeat
      this.stopHeartbeat(jobId);
      
      // Remove from active jobs
      this.activeJobs.delete(jobId);
    }
  }

  // Stream a chunk to the server
  async streamChunk(jobId, chunk) {
    const timestamp = Date.now();
    const message = `${this.config.nodeId}:${jobId}:${timestamp}`;
    const signature = this.nodeClient.signMessage(message, this.config.secretKey);

    const data = {
      nodeId: this.config.nodeId,
      publicKey: this.config.publicKey,
      jobId,
      signature,
      timestamp,
      ...chunk
    };

    const maxRetries = 3;
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        await this.nodeClient.axiosInstance.post(`/api/jobs/${jobId}/chunks`, data);
        return;
      } catch (error) {
        if (attempt === maxRetries) {
          throw new Error(`Failed to send chunk after ${maxRetries} attempts: ${error.message}`);
        }
        
        // Exponential backoff
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
      }
    }
  }

  // Send heartbeat for a job
  async sendHeartbeat(jobId) {
    const timestamp = Date.now();
    const message = `${this.config.nodeId}:${jobId}:${timestamp}`;
    const signature = this.nodeClient.signMessage(message, this.config.secretKey);

    const data = {
      nodeId: this.config.nodeId,
      publicKey: this.config.publicKey,
      jobId,
      signature,
      timestamp,
      status: 'running',
      activeJobs: Array.from(this.activeJobs.keys())
    };

    try {
      await this.nodeClient.axiosInstance.post(`/api/jobs/${jobId}/heartbeat`, data);
    } catch (error) {
      this.emit('warning', {
        type: 'heartbeat_failed',
        jobId,
        error: error.message
      });
    }
  }

  // Start heartbeat for a job
  startHeartbeat(jobId) {
    const jobInfo = this.activeJobs.get(jobId);
    if (jobInfo) {
      // Send initial heartbeat
      this.sendHeartbeat(jobId);
      
      // Set up interval (every 30 seconds)
      jobInfo.heartbeatInterval = setInterval(() => {
        this.sendHeartbeat(jobId);
      }, 30000);
    }
  }

  // Stop heartbeat for a job
  stopHeartbeat(jobId) {
    const jobInfo = this.activeJobs.get(jobId);
    if (jobInfo?.heartbeatInterval) {
      clearInterval(jobInfo.heartbeatInterval);
      jobInfo.heartbeatInterval = null;
    }
  }

  // Complete a job
  async completeJob(jobId) {
    const timestamp = Date.now();
    const message = `${this.config.nodeId}:${jobId}:${timestamp}`;
    const signature = this.nodeClient.signMessage(message, this.config.secretKey);

    const data = {
      nodeId: this.config.nodeId,
      publicKey: this.config.publicKey,
      jobId,
      signature,
      timestamp,
      status: 'completed'
    };

    try {
      await this.nodeClient.axiosInstance.post(`/api/jobs/${jobId}/complete`, data);
    } catch (error) {
      throw new Error(`Failed to mark job as complete: ${error.message}`);
    }
  }

  // Fail a job
  async failJob(jobId, reason) {
    const timestamp = Date.now();
    const message = `${this.config.nodeId}:${jobId}:${timestamp}`;
    const signature = this.nodeClient.signMessage(message, this.config.secretKey);

    const data = {
      nodeId: this.config.nodeId,
      publicKey: this.config.publicKey,
      jobId,
      signature,
      timestamp,
      status: 'failed',
      error: reason
    };

    try {
      await this.nodeClient.axiosInstance.post(`/api/jobs/${jobId}/fail`, data);
    } catch (error) {
      this.emit('error', {
        type: 'fail_report_error',
        jobId,
        error: error.message
      });
    }
  }

  // Cancel a job
  cancelJob(jobId) {
    const jobInfo = this.activeJobs.get(jobId);
    if (jobInfo) {
      jobInfo.status = 'cancelling';
      this.emit('job:cancelling', { jobId });
    }
  }

  // Start the worker
  async start(pollingInterval = 5000) {
    if (this.isRunning) {
      throw new Error('Worker is already running');
    }

    // Initialize
    await this.initialize();
    
    this.isRunning = true;
    
    // Send initial ping with capabilities
    const pingResult = await this.pingWithCapabilities();
    if (!pingResult.success) {
      this.emit('error', {
        type: 'ping_error',
        error: pingResult.error
      });
    }

    // Start polling for jobs
    this.pollingInterval = setInterval(async () => {
      if (this.activeJobs.size >= this.maxConcurrentJobs) {
        return; // Skip polling if at capacity
      }

      const jobs = await this.pollForJobs();
      
      for (const job of jobs) {
        if (this.activeJobs.size >= this.maxConcurrentJobs) {
          break; // Stop if we've reached capacity
        }
        
        // Execute job asynchronously
        this.executeJob(job).catch(error => {
          this.emit('error', {
            type: 'job_execution_error',
            jobId: job.id,
            error: error.message
          });
        });
      }
    }, pollingInterval);

    this.emit('started');
  }

  // Stop the worker
  async stop(graceful = true) {
    if (!this.isRunning) {
      return;
    }

    this.emit('stopping');
    
    // Stop polling
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }

    if (graceful && this.activeJobs.size > 0) {
      this.emit('waiting', { 
        message: 'Waiting for active jobs to complete...',
        activeJobs: Array.from(this.activeJobs.keys())
      });

      // Wait for all jobs to complete (with timeout)
      const timeout = 60000; // 1 minute timeout
      const startTime = Date.now();
      
      while (this.activeJobs.size > 0 && (Date.now() - startTime) < timeout) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      // Force cancel remaining jobs after timeout
      if (this.activeJobs.size > 0) {
        for (const [jobId] of this.activeJobs) {
          this.cancelJob(jobId);
        }
        
        // Wait a bit more for cancellations to process
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    } else if (!graceful) {
      // Force cancel all active jobs
      for (const [jobId] of this.activeJobs) {
        this.cancelJob(jobId);
      }
    }

    this.isRunning = false;
    this.emit('stopped');
  }

  // Set max concurrent jobs
  setMaxConcurrentJobs(max) {
    this.maxConcurrentJobs = Math.max(1, max);
    this.emit('config:updated', { maxConcurrentJobs: this.maxConcurrentJobs });
  }
}

module.exports = JobWorker;