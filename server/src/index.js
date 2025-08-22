const express = require('express');
const path = require('path');
const cors = require('cors');
const dotenv = require('dotenv');
const { createClient } = require('redis');
const routes = require('./routes');
const { checkNodeStatuses } = require('./services/nodeService');

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
app.use('/api', routes);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Debug endpoint to check file availability
app.get('/debug/files', (req, res) => {
  const fs = require('fs');
  const staticPath = process.env.RAILWAY_ENVIRONMENT ? '/app' : path.join(__dirname, '../..');
  
  try {
    const files = fs.readdirSync(staticPath);
    const htmlFiles = files.filter(f => f.endsWith('.html'));
    res.json({
      staticPath,
      totalFiles: files.length,
      htmlFiles,
      allFiles: files.slice(0, 50) // Limit to first 50 files
    });
  } catch (error) {
    res.json({
      error: error.message,
      staticPath
    });
  }
});

// Serve static files from root directory
// In production (Railway), files are at /app root
// In development, files are at project root (two levels up from server/src)
const staticPath = process.env.RAILWAY_ENVIRONMENT 
  ? '/app' 
  : path.join(__dirname, '../..');
console.log(`Environment: NODE_ENV=${process.env.NODE_ENV}, RAILWAY_ENVIRONMENT=${process.env.RAILWAY_ENVIRONMENT}`);
console.log(`Serving static files from: ${staticPath}`);
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
    
    // Check node statuses every minute
    const statusInterval = setInterval(async () => {
      await checkNodeStatuses(redisClient);
    }, 60000);
    
    const server = app.listen(PORT, '0.0.0.0', () => {
      console.log(`Server running on port ${PORT}`);
    });

    // Graceful shutdown handling
    const gracefulShutdown = async (signal) => {
      console.log(`Received ${signal}, starting graceful shutdown...`);
      
      clearInterval(statusInterval);
      
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