### Phase 4: LLM Integration with Ollama

Transform the node client into a wrapper for Ollama to enable distributed LLM inference.

**Core Features:**
- [ ] Ollama integration
  - [ ] Detect Ollama installation and auto-install if missing
  - [ ] Check Ollama service status (http://localhost:11434/api/version)
  - [ ] Auto-pull llama3.2:3b model if not present (MVP default)
  - [ ] Forward inference requests from job system
  - [ ] Handle streaming responses
  - [ ] Report model capabilities and requirements

- [ ] Job execution system
  - [ ] Poll server for assigned jobs and download specifications
  - [ ] Execute inference locally with Ollama
  - [ ] Stream results back to server with metrics (tokens/sec, memory)
  - [ ] Handle job cancellation gracefully

- [ ] Resource management
  - [ ] Detect and report hardware capabilities (GPU, RAM, CPU)
  - [ ] Implement resource and concurrent job limits
  - [ ] Graceful shutdown during active jobs

**Server-side changes:**
- [ ] Job queue system with Redis
- [ ] Job assignment algorithm based on node capabilities
- [ ] Result storage and status tracking (pending, assigned, running, completed, failed)
- [ ] API endpoints for job submission/results with WebSocket support

**Testing Requirements:**
- [ ] Mock Ollama API endpoints and test availability detection
- [ ] Test model listing, pulling, and job execution flow
- [ ] Test error handling and streaming responses
- [ ] Test graceful degradation when Ollama unavailable
- [ ] Integration tests with actual Ollama (optional/manual)
- [ ] Load testing for job queue system