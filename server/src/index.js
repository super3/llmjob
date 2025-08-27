const express = require('express');
const path = require('path');
const cors = require('cors');
const dotenv = require('dotenv');
const { createClient } = require('redis');
const { router, initJobRoutes } = require('./routes');
const { checkNodeStatuses } = require('./services/nodeService');
const JobService = require('./services/jobService');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Redis client
let redisClient;

async function connectRedis() {
  redisClient = createClient({
    url: process.env.REDIS_URL || 'redis://localhost:6379'
  });

  redisClient.on('error', (err) => console.error('Redis Client Error', err));
  
  await redisClient.connect();
  console.log('Connected to Redis');
  
  // Make redis client available to routes
  app.locals.redis = redisClient;
}

// Routes
app.use('/api', router);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Serve static files from root directory
// In production (Railway), files are at /app root
// In development, files are at project root (two levels up from server/src)
const staticPath = process.env.RAILWAY_ENVIRONMENT 
  ? '/app' 
  : path.join(__dirname, '../..');
app.use(express.static(staticPath));

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(err.status || 500).json({
    error: err.message || 'Internal server error'
  });
});

// Start server
async function startServer() {
  try {
    await connectRedis();
    
    // Initialize job routes with Redis
    initJobRoutes(redisClient);
    
    // Initialize job service for background tasks
    const jobService = new JobService(redisClient);
    
    // Check node statuses every minute
    const statusInterval = setInterval(async () => {
      await checkNodeStatuses(redisClient);
    }, 60000);
    
    // Check for timed out jobs every 30 seconds
    const timeoutInterval = setInterval(async () => {
      try {
        const timeoutJobs = await jobService.checkTimeouts();
        if (timeoutJobs.length > 0) {
          console.log(`Returned ${timeoutJobs.length} timed out jobs to queue`);
        }
      } catch (error) {
        console.error('Error checking job timeouts:', error);
      }
    }, 30000);
    
    // Clean up old jobs every hour
    const cleanupInterval = setInterval(async () => {
      try {
        const cleaned = await jobService.cleanupOldJobs();
        if (cleaned > 0) {
          console.log(`Cleaned up ${cleaned} old jobs`);
        }
      } catch (error) {
        console.error('Error cleaning up jobs:', error);
      }
    }, 3600000);
    
    const server = app.listen(PORT, '0.0.0.0', () => {
      console.log(`Server running on port ${PORT}`);
    });

    // Graceful shutdown handling
    const gracefulShutdown = async (signal) => {
      console.log(`Received ${signal}, starting graceful shutdown...`);
      
      clearInterval(statusInterval);
      clearInterval(timeoutInterval);
      clearInterval(cleanupInterval);
      
      server.close(() => {
        console.log('HTTP server closed');
        
        redisClient.quit(() => {
          console.log('Redis connection closed');
          process.exit(0);
        });
      });
      
      // Force shutdown after 10 seconds
      setTimeout(() => {
        console.error('Could not close connections in time, forcefully shutting down');
        process.exit(1);
      }, 10000);
    };

    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    process.on('SIGINT', () => gracefulShutdown('SIGINT'));
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();

module.exports = app;