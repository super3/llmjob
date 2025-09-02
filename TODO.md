### Phase 5: Codebase Refactoring and Cleanup

**5A: Code Quality Improvements**
- [ ] Implement proper logging system (Winston/Pino) to replace 235+ console.log/error statements
- [ ] Extract magic numbers to configuration (timeouts: 60000ms, 30000ms, 3600000ms, job timeout: 600000ms)
- [ ] Create centralized error handling middleware for consistent error responses
- [ ] Separate business logic from controllers (move logic to service layer)
- [ ] Add input validation layer using Joi or Zod instead of manual validation

**5B: Architecture Improvements**
- [ ] Remove code duplication in Redis operations (create Repository pattern)
- [ ] Consolidate 3 separate jest.config.js files into root configuration
- [ ] Add TypeScript or comprehensive JSDoc for type safety
- [ ] Extract constants and enums (job statuses: pending, assigned, running, completed, failed)
- [ ] Implement dependency injection container (Awilix) for better testability

### Phase 6: Live Stats Dashboard

**6A: Token Tracking**
- [ ] Count tokens in JobService when jobs complete (input + output)
- [ ] Store in Redis: `stats:tokens:total` counter
- [ ] Create GET /api/stats endpoint with token count

**6B: Frontend Updates**
- [ ] Fetch stats from API on page load
- [ ] Replace hardcoded "1.2M" with real count
- [ ] Format numbers (1.2M, 3.4B) and auto-refresh every 30s

### Phase 7: API Tab Functionality

**7A: API Key Management**
- [ ] Generate unique API keys per user (lj-{userId}-{randomString})
- [ ] Store API keys in Redis with user mapping
- [ ] Add POST /api/keys/generate endpoint
- [ ] Add POST /api/keys/revoke endpoint

**7B: API Tab Features**
- [ ] Display real API key (not hardcoded "lj-abc123...")
- [ ] Add "Generate New Key" button functionality
- [ ] Show actual usage stats (requests today, total tokens)
- [ ] Add working code examples with user's real API key
- [ ] Test connection button to verify API key works