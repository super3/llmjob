const express = require('express');
const path = require('path');
const cors = require('cors');
const dotenv = require('dotenv');
const { createPool } = require('./db');
const { corsOrigin } = require('./corsOptions');
const routes = require('./routes');
const { initJobRoutes, initOpenAiRoutes, initChatRoutes } = require('./routes');
const NodeService = require('./services/nodeService');
const JobService = require('./services/jobService');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware. CORS is restricted to our own origins (llmjob.com + the Railway
// app + previews); other websites can't call the API — including the free chat
// proxy — from a browser. Non-browser callers send no Origin and are unaffected.
app.use(cors({ origin: corsOrigin }));
app.use(express.json());

// Postgres pool
let db;

async function connectDb() {
  db = createPool();
  // Fail fast if the database is unreachable.
  await db.query('SELECT 1');
  console.log('Connected to Postgres');

  // Make the pool available to routes.
  app.locals.db = db;
}

// Routes
app.use('/api', routes);

// OpenAI-compatible gateway at the app root (POST /v1/chat/completions). Uses
// req.app.locals.db per request, so it's safe to register before the DB connects.
initOpenAiRoutes(app);

// Free public web-chat gateway (POST /api/chat/completions), proxied to
// OpenRouter. Also uses req.app.locals.db per request, so it's safe here too.
initChatRoutes(app);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Serve the built static site from dist/ (produced by `npm run build:site`,
// which the start script runs before this). GitHub Pages serves the same dist/
// output; here it lets the Railway deployment answer for the marketing pages.
// In production (Railway) the app is at /app; in dev it's the project root
// (two levels up from server/src).
const staticPath = process.env.RAILWAY_ENVIRONMENT
  ? '/app/dist'
  : path.join(__dirname, '../..', 'dist');

app.use(express.static(staticPath));

// Error handling middleware. Log the full error server-side, but only echo the
// message back for explicit client errors (4xx). For anything 500+ (or an
// unclassified throw) return a generic message so internal details — stack
// fragments, driver errors, file paths — never leak to the caller. The unused
// `next` param is required: Express only treats 4-arity functions as error
// handlers (the lint config allows unused args).
function errorHandler(err, req, res, next) {
  console.error(err.stack || err);
  const status = err.status || 500;
  const clientError = status >= 400 && status < 500;
  res.status(status).json({
    error: clientError && err.message ? err.message : 'Internal server error'
  });
}
app.use(errorHandler);

// Start server
async function startServer() {
  try {
    await connectDb();

    // Initialize job routes with the database pool
    initJobRoutes(db);

    // Initialize services for background tasks
    const jobService = new JobService(db);
    const nodeService = new NodeService(db);

    // Check node statuses every minute
    const statusInterval = setInterval(async () => {
      await nodeService.checkNodeStatuses();
    }, 60000);

    // Check for timed out jobs every 30 seconds
    const timeoutInterval = setInterval(async () => {
      try {
        const timeoutJobs = await jobService.checkTimeouts();
        if (timeoutJobs.length > 0) {
          console.log(`Returned ${timeoutJobs.length} timed out jobs to queue`);
        }
        // …and drop jobs nothing ever picked up, so they neither pile up nor get
        // run long after their caller gave up waiting.
        const expired = await jobService.expireStalePending();
        if (expired.length > 0) {
          console.log(`Expired ${expired.length} pending jobs no node picked up`);
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

      server.close(async () => {
        console.log('HTTP server closed');
        await db.end();
        console.log('Postgres connection closed');
        process.exit(0);
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

// Only boot when run directly (`node server/src/index.js`), not when required
// by tests — so the suite can exercise the pieces without opening a port or
// connecting to Postgres.
/* istanbul ignore if -- @preserve: bootstrap only runs via `node index.js` */
if (require.main === module) {
  startServer();
}

module.exports = app;
module.exports.app = app;
module.exports.startServer = startServer;
module.exports.connectDb = connectDb;
module.exports.errorHandler = errorHandler;
