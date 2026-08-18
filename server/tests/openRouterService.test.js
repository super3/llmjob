// The OpenRouter client shared by the two gateways that reach hosted models: the
// free web chat (chatController) and the OpenAI-compatible API gateway
// (openaiController). Both are exercised end to end in their own suites; this
// file covers the client's own contract — the configuration it resolves, the
// allow-list it enforces, and the upstream request it builds — so a change here
// can't quietly alter what either gateway sends or spends.
const OpenRouterService = require('../src/services/openRouterService');
const { deltaContent, deltaReasoning, usageMeta } = OpenRouterService;

// Two entries, though we ship one: OPENROUTER_MODELS can widen the allow-list
// without a deploy, so lookup has to hold at any length, not just at one.
const MODELS = [
  { id: 'qwen/qwen3.8-27b', label: 'Qwen3.8 27B' },
  { id: 'vendor/second-model', label: 'Second Model' },
];

describe('OpenRouterService — configuration', () => {
  const ENV_KEYS = ['OPENROUTER_API_KEY', 'OPENROUTER_BASE_URL', 'OPENROUTER_MODELS',
    'OPENROUTER_FREE_TOKEN_BUDGET', 'OPENROUTER_MAX_TOKENS', 'OPENROUTER_REFERER'];
  let saved;
  beforeEach(() => { saved = {}; ENV_KEYS.forEach((k) => { saved[k] = process.env[k]; delete process.env[k]; }); });
  afterEach(() => { ENV_KEYS.forEach((k) => { if (saved[k] === undefined) delete process.env[k]; else process.env[k] = saved[k]; }); });

  it('falls back to the built-in defaults when constructed bare', () => {
    const or = new OpenRouterService();
    expect(or.apiKey).toBeUndefined();
    expect(or.configured).toBe(false); // no key → the gateways answer 503
    expect(or.baseUrl).toBe('https://openrouter.ai/api/v1');
    expect(or.models).toBe(OpenRouterService.DEFAULT_MODELS);
    expect(or.defaultModel).toBe(OpenRouterService.DEFAULT_MODELS[0]);
    expect(or.freeBudget).toBe(1000000);
    expect(or.maxTokens).toBe(2048);
    expect(or.referer).toBe('https://llmjob.com');
    expect(or.title).toBe('LLMJob');
  });

  it('ships exactly one hosted model, the one the Chat page offers', () => {
    // The whole point of the hosted path: the same model, reachable both ways.
    expect(OpenRouterService.DEFAULT_MODELS.map((m) => m.id)).toEqual(['qwen/qwen3.8-27b']);
    // Length is the assertion that matters. Every hosted model bills against
    // one shared free budget, so a second one added here doesn't add capacity —
    // it halves how long the first one lasts. Adding one should be a decision,
    // not a drive-by, and OPENROUTER_MODELS covers the case where it isn't.
    expect(OpenRouterService.DEFAULT_MODELS).toHaveLength(1);
  });

  it('reads configuration from the environment', () => {
    process.env.OPENROUTER_API_KEY = 'env-key';
    process.env.OPENROUTER_BASE_URL = 'https://env.test/v1';
    process.env.OPENROUTER_MODELS = JSON.stringify([{ id: 'x/y', label: 'XY' }]);
    process.env.OPENROUTER_FREE_TOKEN_BUDGET = '500';
    process.env.OPENROUTER_MAX_TOKENS = '64';
    process.env.OPENROUTER_REFERER = 'https://env.example';
    const or = new OpenRouterService();
    expect(or.configured).toBe(true);
    expect(or.baseUrl).toBe('https://env.test/v1');
    expect(or.models).toEqual([{ id: 'x/y', label: 'XY' }]);
    expect(or.freeBudget).toBe(500);
    expect(or.maxTokens).toBe(64);
    expect(or.referer).toBe('https://env.example');
  });
});

describe('OpenRouterService — allow-list and ceilings', () => {
  const or = new OpenRouterService({ apiKey: 'k', models: MODELS });

  it('resolves a model by id or by friendly label', () => {
    expect(or.resolveModel('vendor/second-model')).toBe(MODELS[1]);
    expect(or.resolveModel('Qwen3.8 27B')).toBe(MODELS[0]);
  });

  it('refuses anything not on the allow-list, including no model at all', () => {
    // The allow-list is what bounds our spend, so an unknown name must never
    // reach OpenRouter — the API gateway sends those to the node network.
    expect(or.resolveModel('gpt-4')).toBeNull();
    expect(or.resolveModel(undefined)).toBeNull();
    expect(or.resolveModel(null)).toBeNull();
  });

  it('clamps max_tokens, floors it, and treats 0/invalid as unset', () => {
    const small = new OpenRouterService({ apiKey: 'k', maxTokens: 100 });
    expect(small.resolveMaxTokens(40.9)).toBe(40);
    expect(small.resolveMaxTokens(500)).toBe(100);
    expect(small.resolveMaxTokens(0)).toBe(100);
    expect(small.resolveMaxTokens('abc')).toBe(100);
    expect(small.resolveMaxTokens(undefined)).toBe(100);
  });
});

describe('OpenRouterService — upstream request', () => {
  function capture() {
    const calls = [];
    const or = new OpenRouterService({
      apiKey: 'or-k', baseUrl: 'https://or.test/v1', models: MODELS, referer: 'https://ref.test', title: 'Zed',
      fetchFn: async (url, init) => { calls.push({ url, init, body: JSON.parse(init.body) }); return { ok: true }; },
    });
    return { or, calls };
  }

  it('asks for usage accounting on a non-streaming call', async () => {
    const { or, calls } = capture();
    await or.send({ model: 'x/y', messages: [{ role: 'user', content: 'hi' }], maxTokens: 128 });
    expect(calls[0].url).toBe('https://or.test/v1/chat/completions');
    expect(calls[0].init.headers).toMatchObject({
      Authorization: 'Bearer or-k', 'HTTP-Referer': 'https://ref.test', 'X-Title': 'Zed',
    });
    expect(calls[0].body).toEqual({
      model: 'x/y', messages: [{ role: 'user', content: 'hi' }], stream: false,
      max_tokens: 128, usage: { include: true },
    });
  });

  it('asks for usage in the stream when streaming, and forwards a temperature', async () => {
    const { or, calls } = capture();
    await or.send({ model: 'x/y', messages: [], maxTokens: 8, temperature: 0, stream: true });
    expect(calls[0].body).toMatchObject({
      stream: true, stream_options: { include_usage: true },
      temperature: 0, // 0 is a real setting, not "unset"
    });
    expect(calls[0].body).not.toHaveProperty('usage');
  });
});

describe('OpenRouterService — stream and usage helpers', () => {
  it('reads a content delta, and nothing from a chunk without one', () => {
    expect(deltaContent({ choices: [{ delta: { content: 'hi' } }] })).toBe('hi');
    expect(deltaContent({ usage: { total_tokens: 3 } })).toBeUndefined();
  });

  it('reads a reasoning delta under either provider spelling', () => {
    expect(deltaReasoning({ choices: [{ delta: { reasoning: 'a' } }] })).toBe('a');
    expect(deltaReasoning({ choices: [{ delta: { reasoning_content: 'b' } }] })).toBe('b');
    // A usage-only chunk carries no delta at all — must not throw.
    expect(deltaReasoning({ usage: {} })).toBeUndefined();
  });

  it('prefers reported usage and falls back to estimates', () => {
    const base = { promptText: 'abcd', text: 'efgh', start: 0, firstTokenAt: 100, model: 'm', finish: 'stop' };
    const reported = usageMeta({ ...base, usage: { prompt_tokens: 9, completion_tokens: 3, total_tokens: 12 } }, 1100);
    expect(reported).toMatchObject({
      model: 'm', promptTokens: 9, completionTokens: 3, totalTokens: 12,
      tokensPerSecond: 3, latencyMs: 1100, ttftMs: 100,
    });
    // No usage block: ~4 chars/token, and the total is derived.
    expect(usageMeta({ ...base, usage: null }, 1100))
      .toMatchObject({ promptTokens: 1, completionTokens: 1, totalTokens: 2 });
  });

  it('reports no speed when nothing was generated, and names the requested model', () => {
    const meta = usageMeta({
      promptText: 'abcd', text: '', start: 0, firstTokenAt: 0,
      model: null, requestedLabel: 'qwen/qwen3.8-27b', usage: null, finish: 'stop',
    }, 500);
    expect(meta.model).toBe('qwen/qwen3.8-27b');
    expect(meta.tokensPerSecond).toBe(0);
    expect(meta.ttftMs).toBe(0);
  });
});
