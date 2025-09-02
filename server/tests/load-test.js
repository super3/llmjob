#!/usr/bin/env node

/**
 * Load Testing Script for LLMJob Queue System
 * 
 * This script simulates multiple nodes and job submissions to test
 * the system under load.
 * 
 * Usage:
 *   node server/tests/load-test.js [options]
 * 
 * Options:
 *   --nodes <n>     Number of worker nodes to simulate (default: 3)
 *   --jobs <n>      Number of jobs to submit (default: 100)
 *   --duration <s>  Test duration in seconds (default: 60)
 *   --server <url>  Server URL (default: http://localhost:3001)
 */

const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
  nodes: 3,
  jobs: 100,
  duration: 60,
  server: 'http://localhost:3001'
};

for (let i = 0; i < args.length; i += 2) {
  const arg = args[i].replace('--', '');
  const value = args[i + 1];
  if (options.hasOwnProperty(arg)) {
    options[arg] = isNaN(value) ? value : parseInt(value);
  }
}

console.log('Load Test Configuration:', options);
console.log('=' .repeat(50));

// Metrics tracking
const metrics = {
  jobsSubmitted: 0,
  jobsCompleted: 0,
  jobsFailed: 0,
  nodesRegistered: 0,
  nodesActive: 0,
  startTime: Date.now(),
  errors: []
};

// Create axios instance with defaults
const api = axios.create({
  baseURL: options.server,
  timeout: 5000,
  headers: {
    'Content-Type': 'application/json'
  }
});

/**
 * Register a simulated node
 */
async function registerNode(nodeId) {
  try {
    const response = await api.post('/api/nodes/claim', {
      nodeId,
      userId: 'load-test-user',
      name: `Load Test Node ${nodeId}`,
      signature: 'mock-signature',
      timestamp: Date.now()
    });
    
    if (response.data.success) {
      metrics.nodesRegistered++;
      console.log(`✓ Node registered: ${nodeId}`);
      return response.data;
    }
  } catch (error) {
    console.error(`✗ Failed to register node ${nodeId}:`, error.message);
    metrics.errors.push({ type: 'node_registration', nodeId, error: error.message });
  }
  return null;
}

/**
 * Simulate node polling for jobs
 */
async function pollJobs(nodeId) {
  try {
    const response = await api.post('/api/jobs/poll', {
      nodeId,
      signature: 'mock-signature',
      timestamp: Date.now(),
      maxJobs: 1
    });
    
    return response.data.jobs || [];
  } catch (error) {
    // Polling errors are expected when no jobs available
    return [];
  }
}

/**
 * Simulate job execution
 */
async function executeJob(nodeId, job) {
  const jobId = job.id;
  
  try {
    // Send heartbeat
    await api.post(`/api/jobs/${jobId}/heartbeat`, {
      nodeId,
      signature: 'mock-signature',
      timestamp: Date.now(),
      status: 'running'
    });
    
    // Simulate processing time (1-5 seconds)
    const processingTime = 1000 + Math.random() * 4000;
    await new Promise(resolve => setTimeout(resolve, processingTime));
    
    // Send mock result chunks
    const mockResponse = `The answer to "${job.prompt}" is: [simulated response]`;
    const chunks = mockResponse.match(/.{1,20}/g) || [];
    
    for (let i = 0; i < chunks.length; i++) {
      await api.post(`/api/jobs/${jobId}/chunks`, {
        nodeId,
        signature: 'mock-signature',
        timestamp: Date.now(),
        chunkIndex: i,
        content: chunks[i],
        isFinal: i === chunks.length - 1,
        metrics: {
          tokensPerSecond: 10 + Math.random() * 20
        }
      });
    }
    
    // Mark job as complete
    await api.post(`/api/jobs/${jobId}/complete`, {
      nodeId,
      signature: 'mock-signature',
      timestamp: Date.now()
    });
    
    metrics.jobsCompleted++;
    console.log(`✓ Job ${jobId} completed by ${nodeId}`);
    
  } catch (error) {
    metrics.jobsFailed++;
    console.error(`✗ Job ${jobId} failed:`, error.message);
    
    // Try to mark job as failed
    try {
      await api.post(`/api/jobs/${jobId}/fail`, {
        nodeId,
        signature: 'mock-signature',
        timestamp: Date.now(),
        error: error.message
      });
    } catch (failError) {
      // Ignore failure to mark as failed
    }
  }
}

/**
 * Simulate a worker node
 */
async function simulateNode(nodeIndex) {
  const nodeId = `load-test-node-${nodeIndex}-${uuidv4()}`;
  
  // Register node
  const registration = await registerNode(nodeId);
  if (!registration) return;
  
  metrics.nodesActive++;
  
  // Poll and execute jobs
  const endTime = Date.now() + (options.duration * 1000);
  
  while (Date.now() < endTime) {
    const jobs = await pollJobs(nodeId);
    
    if (jobs.length > 0) {
      for (const job of jobs) {
        await executeJob(nodeId, job);
      }
    } else {
      // No jobs available, wait before polling again
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  metrics.nodesActive--;
  console.log(`Node ${nodeId} finished`);
}

/**
 * Submit test jobs to the queue
 */
async function submitJobs() {
  const prompts = [
    'What is 2+2?',
    'Explain quantum computing',
    'Write a haiku about programming',
    'What is the meaning of life?',
    'How does photosynthesis work?',
    'Describe the water cycle',
    'What is machine learning?',
    'Explain blockchain technology',
    'How do computers work?',
    'What is artificial intelligence?'
  ];
  
  for (let i = 0; i < options.jobs; i++) {
    const prompt = prompts[i % prompts.length] + ` (Test job ${i + 1})`;
    
    try {
      const response = await api.post('/api/jobs', {
        prompt,
        model: 'llama3.2:3b',
        priority: Math.floor(Math.random() * 3), // 0-2 priority
        options: {
          temperature: 0.7,
          max_tokens: 100
        }
      });
      
      if (response.data.success) {
        metrics.jobsSubmitted++;
        console.log(`✓ Job submitted: ${response.data.job.id}`);
      }
    } catch (error) {
      console.error(`✗ Failed to submit job:`, error.message);
      metrics.errors.push({ type: 'job_submission', error: error.message });
    }
    
    // Stagger job submissions
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  console.log(`\nSubmitted ${metrics.jobsSubmitted}/${options.jobs} jobs`);
}

/**
 * Get and display queue statistics
 */
async function getQueueStats() {
  try {
    const response = await api.get('/api/jobs/stats');
    return response.data.stats;
  } catch (error) {
    console.error('Failed to get queue stats:', error.message);
    return null;
  }
}

/**
 * Display real-time metrics
 */
async function displayMetrics() {
  const interval = setInterval(async () => {
    const elapsed = Math.floor((Date.now() - metrics.startTime) / 1000);
    const stats = await getQueueStats();
    
    console.clear();
    console.log('Load Test Metrics');
    console.log('=' .repeat(50));
    console.log(`Elapsed Time: ${elapsed}s / ${options.duration}s`);
    console.log(`Nodes: ${metrics.nodesActive}/${metrics.nodesRegistered} active`);
    console.log(`Jobs Submitted: ${metrics.jobsSubmitted}`);
    console.log(`Jobs Completed: ${metrics.jobsCompleted}`);
    console.log(`Jobs Failed: ${metrics.jobsFailed}`);
    
    if (stats) {
      console.log('\nQueue Statistics:');
      console.log(`  Pending: ${stats.pending || 0}`);
      console.log(`  Assigned: ${stats.assigned || 0}`);
      console.log(`  Running: ${stats.running || 0}`);
      console.log(`  Completed: ${stats.completed || 0}`);
      console.log(`  Failed: ${stats.failed || 0}`);
    }
    
    if (metrics.errors.length > 0) {
      console.log(`\nErrors: ${metrics.errors.length}`);
    }
    
    // Stop when test duration is reached
    if (elapsed >= options.duration) {
      clearInterval(interval);
      await showFinalReport();
      process.exit(0);
    }
  }, 2000);
}

/**
 * Show final test report
 */
async function showFinalReport() {
  const duration = (Date.now() - metrics.startTime) / 1000;
  const throughput = metrics.jobsCompleted / duration;
  
  console.log('\n' + '=' .repeat(50));
  console.log('LOAD TEST COMPLETE');
  console.log('=' .repeat(50));
  console.log(`Duration: ${duration.toFixed(1)}s`);
  console.log(`Jobs Submitted: ${metrics.jobsSubmitted}`);
  console.log(`Jobs Completed: ${metrics.jobsCompleted}`);
  console.log(`Jobs Failed: ${metrics.jobsFailed}`);
  console.log(`Success Rate: ${((metrics.jobsCompleted / metrics.jobsSubmitted) * 100).toFixed(1)}%`);
  console.log(`Throughput: ${throughput.toFixed(2)} jobs/second`);
  console.log(`Average Time per Job: ${(duration / metrics.jobsCompleted * 1000).toFixed(0)}ms`);
  
  if (metrics.errors.length > 0) {
    console.log(`\nErrors Encountered: ${metrics.errors.length}`);
    const errorTypes = {};
    metrics.errors.forEach(e => {
      errorTypes[e.type] = (errorTypes[e.type] || 0) + 1;
    });
    Object.entries(errorTypes).forEach(([type, count]) => {
      console.log(`  ${type}: ${count}`);
    });
  }
  
  // Get final queue stats
  const finalStats = await getQueueStats();
  if (finalStats) {
    console.log('\nFinal Queue State:');
    console.log(`  Pending: ${finalStats.pending || 0}`);
    console.log(`  Assigned: ${finalStats.assigned || 0}`);
    console.log(`  Running: ${finalStats.running || 0}`);
    console.log(`  Completed: ${finalStats.completed || 0}`);
    console.log(`  Failed: ${finalStats.failed || 0}`);
  }
}

/**
 * Main test runner
 */
async function runLoadTest() {
  console.log('Starting load test...\n');
  
  // Check server availability
  try {
    await api.get('/api/health');
    console.log('✓ Server is available\n');
  } catch (error) {
    console.error('✗ Server is not available at', options.server);
    console.error('Please start the server first: npm start');
    process.exit(1);
  }
  
  // Start metrics display
  displayMetrics();
  
  // Submit jobs
  const jobSubmission = submitJobs();
  
  // Wait a bit for jobs to be queued
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Start worker nodes
  const nodePromises = [];
  for (let i = 0; i < options.nodes; i++) {
    nodePromises.push(simulateNode(i));
    // Stagger node starts
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  
  // Wait for all nodes to complete
  await Promise.all([jobSubmission, ...nodePromises]);
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n\nInterrupted - showing final report...');
  await showFinalReport();
  process.exit(0);
});

// Run the load test
runLoadTest().catch(error => {
  console.error('Load test failed:', error);
  process.exit(1);
});