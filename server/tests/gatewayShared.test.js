// Direct coverage for the shared gateway helpers extracted from the two chat
// controllers (chatController + openaiController). The controllers exercise most
// of these paths through their route tests; this file pins every branch so the
// module stands on its own.
const {
  estimateTokens, errorBody, joinContent, lastUserText, nodeFailMessage,
  writeSsePreamble, pollJobResult, clampMessages, resolveMaxTokens, MAX_PROMPT_CHARS,
} = require('../src/controllers/gatewayShared');

describe('gatewayShared — pure helpers', () => {
  it('estimateTokens is ~4 chars/token and tolerates empty/nullish input', () => {
    expect(estimateTokens('abcd')).toBe(1);
    expect(estimateTokens('abcde')).toBe(2);
    expect(estimateTokens('')).toBe(0);
    expect(estimateTokens(null)).toBe(0);
    expect(estimateTokens(undefined)).toBe(0);
  });

  it('errorBody builds the OpenAI-style envelope', () => {
    expect(errorBody('nope', 'invalid_request_error')).toEqual({
      error: { message: 'nope', type: 'invalid_request_error', code: null },
    });
  });

  it('joinContent joins message text and tolerates null entries / missing content', () => {
    expect(joinContent([{ content: 'a' }, { content: 'b' }])).toBe('a\nb');
    expect(joinContent([null, { content: 'x' }])).toBe('\nx');       // null entry → ''
    expect(joinContent([{ role: 'system' }])).toBe('');              // missing content → ''
  });

  it('lastUserText returns the last user turn, else joins all content', () => {
    expect(lastUserText([{ role: 'user', content: 'a' }, { role: 'assistant', content: 'b' }])).toBe('a');
    expect(lastUserText([null, { role: 'user', content: null }, { role: 'user', content: 'z' }])).toBe('z');
    expect(lastUserText([{ role: 'assistant', content: 'x' }])).toBe('x'); // no user → join
    expect(lastUserText([null, { role: 'system' }])).toBe('\n');           // no user, null entry in the join
    expect(lastUserText([{ role: 'user', content: 42 }])).toBe('42');      // coerced to string
  });

  it('nodeFailMessage prefers the reason, falls back for empty/nullish results', () => {
    expect(nodeFailMessage({ error: 'boom' })).toBe('The node failed to run the job: boom');
    expect(nodeFailMessage({})).toBe('The node failed to run the job: unknown error');
    expect(nodeFailMessage(null)).toBe('The node failed to run the job: unknown error');
  });
});

describe('gatewayShared — writeSsePreamble', () => {
  function fakeRes(withFlush) {
    const res = { statusCode: 0, headers: {}, flushed: false, status(c) { this.statusCode = c; return this; }, setHeader(k, v) { this.headers[k] = v; } };
    if (withFlush) res.flushHeaders = function () { this.flushed = true; };
    return res;
  }

  it('sets the SSE headers including X-Accel-Buffering and flushes when available', () => {
    const res = fakeRes(true);
    writeSsePreamble(res);
    expect(res.statusCode).toBe(200);
    expect(res.headers['Content-Type']).toMatch(/text\/event-stream/);
    expect(res.headers['Cache-Control']).toBe('no-cache, no-transform');
    expect(res.headers['Connection']).toBe('keep-alive');
    expect(res.headers['X-Accel-Buffering']).toBe('no'); // the header that keeps Railway/nginx from buffering
    expect(res.flushed).toBe(true);
  });

  it('works when flushHeaders is not available', () => {
    const res = fakeRes(false);
    expect(() => writeSsePreamble(res)).not.toThrow();
    expect(res.headers['X-Accel-Buffering']).toBe('no');
  });
});

describe('gatewayShared — pollJobResult', () => {
  // A clock that advances by `step` ms on each call.
  function stepClock(step = 100) {
    let t = 0;
    return () => (t += step);
  }
  const noSleep = async () => {};
  const collect = async (gen) => { const out = []; for await (const r of gen) out.push(r); return out; };

  it('yields each poll and stops after a completed result', async () => {
    const seq = [{ status: 'running' }, { status: 'running' }, { status: 'completed', result: 'x' }];
    let i = 0;
    const jobService = { getJobResult: async () => seq[i++] };
    const out = await collect(pollJobResult({
      jobService, jobId: 'j', now: stepClock(), sleep: noSleep, pollMs: 1, timeoutMs: 10000, isAborted: () => false,
    }));
    expect(out.map((r) => r.status)).toEqual(['running', 'running', 'completed']);
  });

  it('stops after a failed result', async () => {
    const jobService = { getJobResult: async () => ({ status: 'failed', error: 'boom' }) };
    const out = await collect(pollJobResult({
      jobService, jobId: 'j', now: stepClock(), sleep: noSleep, pollMs: 1, timeoutMs: 10000, isAborted: () => false,
    }));
    expect(out).toEqual([{ status: 'failed', error: 'boom' }]);
  });

  it('yields a timeout sentinel (carrying the last result) once the deadline passes', async () => {
    const jobService = { getJobResult: async () => ({ status: 'running', assignedTo: 'n1' }) };
    // step 100ms/call, timeout 0 → the first now()-started already exceeds it.
    const out = await collect(pollJobResult({
      jobService, jobId: 'j', now: stepClock(100), sleep: noSleep, pollMs: 1, timeoutMs: 0, isAborted: () => false,
    }));
    const sentinel = out[out.length - 1];
    expect(sentinel.status).toBe('timeout');
    expect(sentinel.last).toEqual({ status: 'running', assignedTo: 'n1' }); // last poll carried through
    expect(out.some((r) => r.status === 'running')).toBe(true); // the live poll is yielded before the sentinel
  });

  it('stops immediately (yielding nothing) when already aborted', async () => {
    let called = 0;
    const jobService = { getJobResult: async () => { called++; return { status: 'running' }; } };
    const out = await collect(pollJobResult({
      jobService, jobId: 'j', now: stepClock(), sleep: noSleep, pollMs: 1, timeoutMs: 10000, isAborted: () => true,
    }));
    expect(out).toEqual([]);
    expect(called).toBe(0); // never even polled
  });

  it('stops when abort flips true between polls', async () => {
    let aborted = false;
    const jobService = { getJobResult: async () => { aborted = true; return { status: 'running' }; } };
    const out = await collect(pollJobResult({
      jobService, jobId: 'j', now: stepClock(), sleep: noSleep, pollMs: 1, timeoutMs: 10000, isAborted: () => aborted,
    }));
    // First iteration polls + yields 'running', then the next top-of-loop abort check ends it.
    expect(out).toEqual([{ status: 'running' }]);
  });
});

// The other half of the drift these helpers exist to fix: only the web-chat
// gateway ever clamped its input, so /v1 accepted an unbounded prompt and an
// unbounded max_tokens — one API key could hand a node work it would never
// finish.
describe('clampMessages', () => {
  it('keeps well-formed messages unchanged', () => {
    const msgs = [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'yo' }];
    expect(clampMessages(msgs, 100)).toEqual(msgs);
  });

  it('maps an unknown role onto user', () => {
    expect(clampMessages([{ role: 'tool', content: 'x' }], 100))
      .toEqual([{ role: 'user', content: 'x' }]);
  });

  it('drops non-objects and empty content', () => {
    const msgs = ['nope', null, { role: 'user', content: '' }, { role: 'user', content: null }, { role: 'user', content: 'keep' }];
    expect(clampMessages(msgs, 100)).toEqual([{ role: 'user', content: 'keep' }]);
  });

  it('truncates at the character budget and stops consuming', () => {
    const msgs = [{ role: 'user', content: 'abcdef' }, { role: 'user', content: 'ignored' }];
    expect(clampMessages(msgs, 4)).toEqual([{ role: 'user', content: 'abcd' }]);
  });

  it('defaults to MAX_PROMPT_CHARS when no budget is given', () => {
    const out = clampMessages([{ role: 'user', content: 'x'.repeat(MAX_PROMPT_CHARS + 500) }]);
    expect(out[0].content).toHaveLength(MAX_PROMPT_CHARS);
  });

  it('treats a missing list as empty', () => {
    expect(clampMessages(undefined, 10)).toEqual([]);
  });
});

describe('resolveMaxTokens', () => {
  it('clamps a caller value to the ceiling and floors it', () => {
    expect(resolveMaxTokens(999999, 6400)).toBe(6400);
    expect(resolveMaxTokens(10.9, 6400)).toBe(10);
  });

  // Unlike temperature, 0 is not a meaningful completion budget — it would
  // produce nothing — so it falls back rather than being preserved.
  it('falls back to the ceiling for 0, negatives and junk', () => {
    expect(resolveMaxTokens(0, 6400)).toBe(6400);
    expect(resolveMaxTokens(-5, 6400)).toBe(6400);
    expect(resolveMaxTokens('abc', 6400)).toBe(6400);
    expect(resolveMaxTokens(undefined, 6400)).toBe(6400);
    expect(resolveMaxTokens(Infinity, 6400)).toBe(6400);
  });
});
