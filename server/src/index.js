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

// Serve static files from root directory
app.use(express.static(path.join(__dirname, '../..')));

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