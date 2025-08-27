### Phase 4: LLM Integration with Ollama

**4A: Local Client Ollama Integration (Test First)** ✅
- [x] Detect hardware capabilities (CPU cores, RAM, GPU type)
- [x] Store capabilities in ~/.llmjob/capabilities.json
- [x] Detect Ollama installation, auto-install if missing (brew/curl script)
- [x] Check service status at http://localhost:11434/api/version
- [x] Auto-pull llama3.2:3b model with progress reporting
- [x] Test local inference with streaming responses
- [x] Benchmark actual inference speed (tokens/sec)
- [x] Add Jest tests for Ollama integration
- [x] Refactored to use official ollama npm package for better maintainability

**4B: Server Integration** ✅
- [x] Report capabilities during node ping
  - [x] Test: Mock server responses for capability reporting
  - [x] Test: Verify capabilities are included in ping payload
- [x] Poll server for assigned jobs every 5 seconds
  - [x] Test: Mock job polling with various responses (no jobs, single job, multiple jobs)
  - [x] Test: Verify polling interval and retry logic
- [x] Execute inference with Ollama API (/api/generate)
  - [x] Test: Mock Ollama responses for job execution
  - [x] Test: Handle inference errors and timeouts
- [x] Stream results via HTTP POST chunks with metrics (tokens/sec, memory usage)
  - [x] Test: Mock chunk submission with retries
  - [x] Test: Verify metrics calculation and formatting
- [x] Handle job cancellation and concurrent job limits
  - [x] Test: Job cancellation mid-execution
  - [x] Test: Enforce concurrent job limits
- [x] Graceful shutdown if job is running
  - [x] Test: Shutdown behavior with active jobs
  - [x] Test: Job cleanup and state persistence
- [x] Created JobWorker class with comprehensive event-driven architecture
- [x] Added new CLI command `llmjob-node worker` for job processing mode

**4C: Server-side Changes**
- [ ] Redis job queue with priority and assignment algorithm
  - [ ] Test: Job queue operations (enqueue, dequeue, priority sorting)
  - [ ] Test: Assignment algorithm with multiple nodes
- [ ] Job status tracking (pending, assigned, running, completed, failed)
  - [ ] Test: State transitions and validation
  - [ ] Test: Status updates from workers
- [ ] Implement job locking with 10-minute timeout (return to queue if not completed)
  - [ ] Test: Lock acquisition and release
  - [ ] Test: Timeout behavior and job reassignment
- [ ] Worker heartbeat every 30 seconds to detect failures
  - [ ] Test: Heartbeat tracking and timeout detection
  - [ ] Test: Worker failure recovery
- [ ] Result storage with chunked streaming support
  - [ ] Test: Chunk storage and reassembly
  - [ ] Test: Handle incomplete/out-of-order chunks
- [ ] API endpoints for job submission, chunk receiving, and results
  - [ ] Test: API endpoint validation and error handling
  - [ ] Test: Rate limiting and authentication

**4D: Testing Requirements**
- [ ] Mock Ollama API endpoints for unit tests
- [ ] Test installation detection and model pulling
- [ ] Test local inference execution and streaming
- [ ] Test hardware detection and benchmarking
- [ ] Test job polling, execution, and result streaming
- [ ] Test error handling and Ollama unavailability
- [ ] Integration tests with real Ollama (manual)
- [ ] Load test job queue with multiple nodes