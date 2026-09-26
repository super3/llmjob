const express = require('express');
const path = require('path');
const cors = require('cors');
const dotenv = require('dotenv');
const { createPool } = require('./db');
const { corsOrigin } = require('./corsOptions');
const routes = require('./routes');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware. CORS is restricted to our own origins (llmjob.com + the Railway
// app + previews); other websites can't call the API — including the waitlist
// signup — from a browser. Non-browser callers send no Origin and are unaffected.
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

// Routes. They use req.app.locals.db per request, so it's safe to register them
// before the DB connects.
app.use('/api', routes);

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

// URLs are extensionless: /network, not /network.html. Old links (bookmarks,
// posts, search results) still resolve, but they redirect to the canonical form
// instead of being served, so a page never answers on two URLs at once.
// GitHub Pages, which serves the same dist/ for llmjob.com, strips the
// extension on its own; this is the equivalent for the Railway deployment.
app.use((req, res, next) => {
  if ((req.method !== 'GET' && req.method !== 'HEAD') || !req.path.endsWith('.html')) {
    return next();
  }
  // /network.html -> /network, /index.html -> / (a directory keeps its trailing slash).
  const target = req.path.slice(0, -'.html'.length).replace(/(^|\/)index$/, '$1');
  return res.redirect(301, target + req.url.slice(req.path.length));
});

// `extensions: ['html']` is what serves dist/network.html for a request to /network.
app.use(express.static(staticPath, { extensions: ['html'] }));

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

    const server = app.listen(PORT, '0.0.0.0', () => {
      console.log(`Server running on port ${PORT}`);
    });

    // Graceful shutdown handling
    const gracefulShutdown = async (signal) => {
      console.log(`Received ${signal}, starting graceful shutdown...`);

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
