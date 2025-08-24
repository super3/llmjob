### Phase 4: LLM Integration with Ollama

**4A: Local Client Ollama Integration (Test First)**
- [ ] Detect hardware capabilities (CPU cores, RAM, GPU type)
- [ ] Store capabilities in ~/.llmjob/capabilities.json
- [ ] Detect Ollama installation, auto-install if missing (brew/curl script)
- [ ] Check service status at http://localhost:11434/api/version
- [ ] Auto-pull llama3.2:3b model with progress reporting
- [ ] Test local inference with streaming responses
- [ ] Benchmark actual inference speed (tokens/sec)
- [ ] Add Jest tests for Ollama integration

**4B: Server Integration**
- [ ] Report capabilities during node ping
- [ ] Poll server for assigned jobs every 30 seconds
- [ ] Execute inference with Ollama API (/api/generate)
- [ ] Stream results with metrics back to server (tokens/sec, memory usage)
- [ ] Handle job cancellation and concurrent job limits
- [ ] Graceful shutdown if job is running

**4C: Server-side Changes**
- [ ] Redis job queue with priority and assignment algorithm
- [ ] Job status tracking (pending, assigned, running, completed, failed)
- [ ] Result storage with streaming support
- [ ] API endpoints for job submission and results
- [ ] WebSocket support for real-time updates

**4D: Testing Requirements**
- [ ] Mock Ollama API endpoints for unit tests
- [ ] Test installation detection and model pulling
- [ ] Test local inference execution and streaming
- [ ] Test hardware detection and benchmarking
- [ ] Test job polling, execution, and result streaming
- [ ] Test error handling and Ollama unavailability
- [ ] Integration tests with real Ollama (manual)
- [ ] Load test job queue with multiple nodes