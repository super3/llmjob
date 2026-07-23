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

      expect(routePaths).toContain('/jobs');
      expect(routePaths).toContain('/jobs/stats');
      expect(routePaths).toContain('/jobs/:jobId');
      expect(routePaths).toContain('/jobs/poll');
      expect(routePaths).toContain('/jobs/:jobId/heartbeat');
      expect(routePaths).toContain('/jobs/:jobId/chunks');
      expect(routePaths).toContain('/jobs/:jobId/complete');
      expect(routePaths).toContain('/jobs/:jobId/fail');
      expect(routePaths).toContain('/jobs/cleanup');
      expect(routePaths).toContain('/jobs/check-timeouts');
    });

    it('should handle job route requests', async () => {
      initJobRoutes(db);
      app.use('/api', router);

      const response = await request(app)
        .get('/api/jobs/stats')
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('stats');
      expect(response.body.stats).toHaveProperty('pending');
      expect(response.body.stats).toHaveProperty('assigned');
      expect(response.body.stats).toHaveProperty('running');
      expect(response.body.stats).toHaveProperty('completed');
      expect(response.body.stats).toHaveProperty('failed');
    });

    it('should handle check-timeouts endpoint', async () => {
      initJobRoutes(db);
      app.use('/api', router);

      const response = await request(app)
        .post('/api/jobs/check-timeouts')
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('timeoutJobs');
      expect(Array.isArray(response.body.timeoutJobs)).toBe(true);
    });
  });
});
