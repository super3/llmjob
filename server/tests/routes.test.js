// The dashboard-facing job routes now require auth; mock it to pass through so
// the handlers still execute end-to-end here.
jest.mock('../src/middleware/auth', () => ({
  requireAuth: (req, res, next) => { req.user = { id: 'user-routes' }; next(); },
}));

const request = require('supertest');
const express = require('express');
const router = require('../src/routes');
const { initJobRoutes } = require('../src/routes');
const { createTestDb } = require('./helpers/pgmem');

describe('Routes', () => {
  let app;
  let db;

  beforeEach(async () => {
    app = express();
    app.use(express.json());
    db = await createTestDb();
  });

  afterEach(async () => {
    if (db.end) await db.end();
  });

  describe('initJobRoutes', () => {
    it('should initialize job routes with the database', () => {
      initJobRoutes(db);
      app.use('/api', router);

      const routes = router.stack.filter((layer) => layer.route);
      const routePaths = routes.map((layer) => layer.route.path);

      expect(routePaths).toContain('/jobs/poll');
      expect(routePaths).toContain('/jobs/:jobId/heartbeat');
      expect(routePaths).toContain('/jobs/:jobId/chunks');
      expect(routePaths).toContain('/jobs/:jobId/complete');
      expect(routePaths).toContain('/jobs/:jobId/fail');
      expect(routePaths).toContain('/jobs/cleanup');
      expect(routePaths).toContain('/jobs/check-timeouts');
    });

    // Jobs are created in-process, never over HTTP. POST /api/jobs let any
    // signed-in account queue arbitrary prompts on the fleet.
    it('no longer serves the removed job routes', async () => {
      initJobRoutes(db);
      app.use('/api', router);

      const routes = router.stack.filter((layer) => layer.route);
      const routePaths = routes.map((layer) => layer.route.path);
      expect(routePaths).not.toContain('/jobs');
      expect(routePaths).not.toContain('/jobs/stats');
      expect(routePaths).not.toContain('/jobs/:jobId');
      expect((await request(app).post('/api/jobs').send({ prompt: 'x' })).status).toBe(404);
      expect((await request(app).get('/api/jobs/stats')).status).toBe(404);
    });

    // check-timeouts requeues every user's in-flight jobs, so it is admin-only.
    it('should handle check-timeouts endpoint for an admin', async () => {
      const prev = process.env.ADMIN_USER_IDS;
      process.env.ADMIN_USER_IDS = 'user-routes';
      try {
        initJobRoutes(db);
        app.use('/api', router);

        const response = await request(app)
          .post('/api/jobs/check-timeouts')
          .expect(200);

        expect(response.body).toHaveProperty('success', true);
        expect(response.body).toHaveProperty('timeoutJobs');
        expect(Array.isArray(response.body.timeoutJobs)).toBe(true);
      } finally {
        if (prev === undefined) delete process.env.ADMIN_USER_IDS;
        else process.env.ADMIN_USER_IDS = prev;
      }
    });

    it('refuses check-timeouts for a signed-in non-admin', async () => {
      const prev = process.env.ADMIN_USER_IDS;
      delete process.env.ADMIN_USER_IDS;
      try {
        initJobRoutes(db);
        app.use('/api', router);

        const response = await request(app)
          .post('/api/jobs/check-timeouts')
          .expect(403);

        expect(response.body.error).toBe('Admin access required');
      } finally {
        if (prev !== undefined) process.env.ADMIN_USER_IDS = prev;
      }
    });
  });
});
