// Direct coverage for the shared gateway helpers extracted from the two chat
// controllers (chatController + openaiController). The controllers exercise most
// of these paths through their route tests; this file pins every branch so the
// module stands on its own.
const {
  estimateTokens, errorBody, joinContent, lastUserText, nodeFailMessage,
  writeSsePreamble, pollJobResult, clampMessages, resolveMaxTokens, MAX_PROMPT_CHARS,
  contentText, MAX_IMAGES, MAX_IMAGE_CHARS,
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
    // Newest-first: the LAST message gets the budget and the earlier one is
    // dropped, not the other way round.
    const msgs = [{ role: 'user', content: 'dropped' }, { role: 'user', content: 'abcdef' }];
    expect(clampMessages(msgs, 4)).toEqual([{ role: 'user', content: 'abcd' }]);
  });

  it('keeps the newest turns and trims/drops the oldest', () => {
    const big = 'a'.repeat(30000);
    // The current question is the LAST message; it must survive even when an
    // earlier turn already exceeds the whole budget on its own. Walking
    // front-to-back dropped it and left the model answering stale context with
    // a 200 — silently, which is what made it so hard to see.
    const out = clampMessages([{ role: 'user', content: big }, { role: 'user', content: 'current question' }]);
    expect(out[out.length - 1]).toEqual({ role: 'user', content: 'current question' });
    const total = out.reduce((n, m) => n + m.content.length, 0);
    expect(total).toBeLessThanOrEqual(MAX_PROMPT_CHARS);
  });

  it('drops the oldest turn entirely when the newest already fills the budget', () => {
    const big = 'b'.repeat(MAX_PROMPT_CHARS);
    const out = clampMessages([{ role: 'user', content: 'old' }, { role: 'user', content: big }]);
    expect(out).toEqual([{ role: 'user', content: big }]);
  });

  it('returns the kept turns in chronological order', () => {
    const msgs = [
      { role: 'system', content: 'be brief' },
      { role: 'user', content: 'first' },
      { role: 'assistant', content: 'second' },
      { role: 'user', content: 'third' },
    ];
    expect(clampMessages(msgs, 1000)).toEqual(msgs);
  });

  it('spends the image budget on the newest images', () => {
    // A caller who attaches a screenshot to THIS turn must not lose it to
    // MAX_IMAGES stale ones earlier in the conversation.
    const stale = Array.from({ length: MAX_IMAGES }, (_, i) => ({
      type: 'image_url', image_url: { url: 'data:image/png;base64,old' + i },
    }));
    const out = clampMessages([
      { role: 'user', content: stale },
      { role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,NEW' } }] },
    ], 1000);
    const urls = out.flatMap((m) => (Array.isArray(m.content) ? m.content : []))
      .filter((p) => p.type === 'image_url').map((p) => p.image_url.url);
    expect(urls).toContain('data:image/png;base64,NEW');
    expect(urls).toHaveLength(MAX_IMAGES);
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

// OpenAI sends multimodal turns as an ARRAY in `content`. Before this, the
// gateway did String(m.content) on it, which yields the literal "[object
// Object]" — so a vision request was not merely unsupported, it was corrupted
// into nonsense and the node answered the nonsense.
describe('multimodal content', () => {
  const img = (url, extra = {}) => ({ type: 'image_url', image_url: { url, ...extra } });
  const txt = (text) => ({ type: 'text', text });

  test('an image survives the gateway instead of becoming "[object Object]"', () => {
    const out = clampMessages([{ role: 'user', content: [txt('what is this?'), img('data:image/png;base64,AAAA')] }], 24000);
    expect(out).toEqual([{
      role: 'user',
      content: [txt('what is this?'), { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } }],
    }]);
    expect(JSON.stringify(out)).not.toContain('[object Object]');
  });

  test('plain string content is untouched', () => {
    expect(clampMessages([{ role: 'user', content: 'hi' }], 100)).toEqual([{ role: 'user', content: 'hi' }]);
  });

  test('detail is carried through only when the caller set it', () => {
    const [m] = clampMessages([{ role: 'user', content: [img('http://x/a.png', { detail: 'high' })] }], 100);
    expect(m.content[0].image_url).toEqual({ url: 'http://x/a.png', detail: 'high' });
    const [n] = clampMessages([{ role: 'user', content: [img('http://x/a.png')] }], 100);
    expect(n.content[0].image_url).toEqual({ url: 'http://x/a.png' });
  });

  test('images are capped by count, and the cap does not eat the text', () => {
    const many = Array.from({ length: MAX_IMAGES + 4 }, (_, i) => img('data:image/png;base64,' + i));
    const [m] = clampMessages([{ role: 'user', content: [txt('describe'), ...many] }], 24000);
    expect(m.content.filter((p) => p.type === 'image_url')).toHaveLength(MAX_IMAGES);
    expect(m.content[0]).toEqual(txt('describe'));
  });

  test('an oversized image is dropped rather than truncated into a corrupt one', () => {
    // Half a data: URL decodes to nothing useful; better to drop it and answer
    // the text than to hand a node a broken image.
    const huge = 'data:image/png;base64,' + 'A'.repeat(MAX_IMAGE_CHARS);
    const [m] = clampMessages([{ role: 'user', content: [txt('hi'), img(huge)] }], 100);
    expect(m.content).toEqual([txt('hi')]);
  });

  test('image bytes are NOT charged to the text budget', () => {
    // The whole reason images get their own limit: a base64 screenshot would
    // exhaust a 24,000-char prompt budget by itself and starve the text.
    const url = 'data:image/png;base64,' + 'A'.repeat(50000);
    const [m] = clampMessages([{ role: 'user', content: [img(url), txt('still here')] }], 200);
    expect(m.content).toEqual([{ type: 'image_url', image_url: { url } }, txt('still here')]);
  });

  test('text parts still share the text budget, so mixing buys no extra room', () => {
    const [m] = clampMessages([{ role: 'user', content: [txt('abcdef'), txt('ghijkl')] }], 8);
    expect(m.content).toEqual([txt('abcdef'), txt('gh')]);
  });

  test('unknown part types are dropped, not coerced', () => {
    const out = clampMessages([{ role: 'user', content: [{ type: 'audio', data: 'x' }, txt('ok')] }], 100);
    expect(out).toEqual([{ role: 'user', content: [txt('ok')] }]);
  });

  test('a message whose parts are all unusable is dropped entirely', () => {
    expect(clampMessages([{ role: 'user', content: [{ type: 'audio' }, null, 'nope'] }], 100)).toEqual([]);
  });

  test('contentText reads text parts and never the image bytes', () => {
    expect(contentText([txt('a'), img('data:image/png;base64,ZZZZ'), txt('b')])).toBe('a\nb');
    expect(contentText('plain')).toBe('plain');
    expect(contentText(null)).toBe('');
  });

  test('lastUserText and joinContent stay text-only for multimodal turns', () => {
    const msgs = [{ role: 'user', content: [txt('describe it'), img('data:image/png;base64,ZZZZ')] }];
    expect(lastUserText(msgs)).toBe('describe it');
    expect(joinContent(msgs)).toBe('describe it');
    // Token estimation must not bill a megabyte of base64 as prompt text.
    expect(joinContent(msgs)).not.toContain('ZZZZ');
  });

  test('lastUserText falls through an image-only turn to something usable', () => {
    // No text in the user turn, so it falls back to the joined conversation
    // (the pre-existing behaviour for a turn with no usable text). What matters
    // is that it yields the prose and never the base64.
    const msgs = [{ role: 'system', content: 'be brief' }, { role: 'user', content: [img('data:image/png;base64,ZZZZ')] }];
    expect(lastUserText(msgs).trim()).toBe('be brief');
    expect(lastUserText(msgs)).not.toContain('ZZZZ');
  });
});

describe('multimodal edge cases', () => {
  const txt = (text) => ({ type: 'text', text });

  test('an empty or null text part is dropped, not emitted as ""', () => {
    expect(clampMessages([{ role: 'user', content: [txt(''), txt(null), txt('real')] }], 100))
      .toEqual([{ role: 'user', content: [txt('real')] }]);
  });

  test('a malformed image part is dropped rather than throwing', () => {
    const out = clampMessages([{
      role: 'user',
      content: [{ type: 'image_url' }, { type: 'image_url', image_url: {} },
                { type: 'image_url', image_url: { url: null } }, txt('ok')],
    }], 100);
    expect(out).toEqual([{ role: 'user', content: [txt('ok')] }]);
  });

  test('a text part sliced away by an exhausted budget is dropped', () => {
    // First part eats the budget exactly; the second slices to '' and must not
    // be pushed as an empty text part.
    const [m] = clampMessages([{ role: 'user', content: [txt('abcd'), txt('efgh')] }], 4);
    expect(m.content).toEqual([txt('abcd')]);
  });

  test('numeric text is coerced, matching the plain-string path', () => {
    const [m] = clampMessages([{ role: 'user', content: [{ type: 'text', text: 42 }] }], 100);
    expect(m.content).toEqual([txt('42')]);
  });
});
