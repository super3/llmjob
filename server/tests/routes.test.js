const request = require('supertest');
const express = require('express');
const router = require('../src/routes');
const { initJobRoutes } = require('../src/routes');
const redis = require('redis-mock');

describe('Routes', () => {
  let app;
  let redisClient;

  beforeEach(() => {
    app = express();
    app.use(express.json());
    redisClient = redis.createClient();
  });

  afterEach(() => {
    if (redisClient) {
      redisClient.quit();
    }
  });

  describe('initJobRoutes', () => {
    it('should initialize job routes with redis', () => {
      // Initialize job routes
      initJobRoutes(redisClient);
      
      // Use the router in the app
      app.use('/api', router);

      // Verify routes are set up by checking they exist
      const routes = router.stack.filter(layer => layer.route);
      const routePaths = routes.map(layer => layer.route.path);
      
      // Check that job routes exist
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
      // Initialize job routes
      initJobRoutes(redisClient);
      app.use('/api', router);

      // Test stats endpoint (doesn't require auth)
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
      // Initialize job routes
      initJobRoutes(redisClient);
      app.use('/api', router);

      // Test check-timeouts endpoint
      const response = await request(app)
        .post('/api/jobs/check-timeouts')
        .expect(200);
      
      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('timeoutJobs');
      expect(Array.isArray(response.body.timeoutJobs)).toBe(true);
    });
  });
});