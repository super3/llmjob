'use strict';

const {
  PROXY_MODELS, proxyChatUrl, proxyModelsUrl, parseModelsResponse, findProxyModel,
  buildProxyChatBody, parseProxyStream,
} = require('../src/shared/llmProxy');

describe('PROXY_MODELS', () => {
  test('mirrors the gateway allow-list (the two Qwen models)', () => {
    expect(PROXY_MODELS.map((m) => m.id)).toEqual(['qwen/qwen3.6-27b', 'qwen/qwen3.6-35b-a3b']);
    expect(PROXY_MODELS.every((m) => m.label)).toBe(true);
  });
});

describe('proxyChatUrl / proxyModelsUrl', () => {
  test('append the gateway paths, trimming trailing slashes', () => {
    expect(proxyChatUrl('https://llmjob.example')).toBe('https://llmjob.example/api/chat/completions');
    expect(proxyChatUrl('https://llmjob.example/')).toBe('https://llmjob.example/api/chat/completions');
    expect(proxyChatUrl('https://llmjob.example///')).toBe('https://llmjob.example/api/chat/completions');
    expect(proxyModelsUrl('https://llmjob.example/')).toBe('https://llmjob.example/api/chat/models');
  });
  test('tolerate a missing base url', () => {
    expect(proxyChatUrl(null)).toBe('/api/chat/completions');
    expect(proxyModelsUrl(undefined)).toBe('/api/chat/models');
  });
});

describe('parseModelsResponse', () => {
  test('normalizes a well-formed models list', () => {
    expect(parseModelsResponse({ models: [{ id: 'a', label: 'A' }, { id: 'b' }] }))
      .toEqual([{ id: 'a', label: 'A' }, { id: 'b', label: 'b' }]);
  });
  test('null for a malformed, empty, or entry-less response', () => {
    expect(parseModelsResponse(null)).toBeNull();
    expect(parseModelsResponse({})).toBeNull();
    expect(parseModelsResponse({ models: 'nope' })).toBeNull();
    expect(parseModelsResponse({ models: [] })).toBeNull();
    expect(parseModelsResponse({ models: [{ label: 'no id' }, null] })).toBeNull();
  });
});

describe('findProxyModel', () => {
  test('resolves by id or label, case-insensitively, from the built-in list', () => {
    expect(findProxyModel('qwen/qwen3.6-27b')).toBe(PROXY_MODELS[0]);
    expect(findProxyModel('QWEN3.6 35B A3B')).toBe(PROXY_MODELS[1]);
  });
  test('null for empty input or no match; honors a custom list; skips junk entries', () => {
    expect(findProxyModel(null)).toBeNull();
    expect(findProxyModel('gpt-4')).toBeNull();
    const custom = [null, { id: 'x/y', label: 'Zed' }]; // a falsy entry is skipped
    expect(findProxyModel('zed', custom)).toBe(custom[1]);
    expect(findProxyModel('x/y', [])).toBeNull();
    expect(findProxyModel('x/y', null)).toBeNull();
  });
});

describe('buildProxyChatBody', () => {
  test('sets the gateway model id, coerces roles/content, streams by default', () => {
    const body = buildProxyChatBody([
      { role: 'user', content: 'hi' },
      { role: 'system', content: 'be brief' },
      { role: 'assistant', content: 'ok' },
      { role: 'tool', content: 42 },   // unknown role → assistant
      null,                            // junk → assistant / empty
    ], { model: 'qwen/qwen3.6-27b' });
    expect(body.model).toBe('qwen/qwen3.6-27b');
    expect(body.stream).toBe(true);
    expect(body.messages).toEqual([
      { role: 'user', content: 'hi' },
      { role: 'system', content: 'be brief' },
      { role: 'assistant', content: 'ok' },
      { role: 'assistant', content: '42' },
      { role: 'assistant', content: '' },
    ]);
    expect(body).not.toHaveProperty('temperature');
  });

  test('non-array messages become an empty list; stream can be disabled', () => {
    const body = buildProxyChatBody('nope', { model: 'x', stream: false });
    expect(body.messages).toEqual([]);
    expect(body.stream).toBe(false);
  });

  test('defaults opts when omitted', () => {
    const body = buildProxyChatBody([{ role: 'user', content: 'hi' }]);
    expect(body).toEqual({ model: undefined, stream: true, messages: [{ role: 'user', content: 'hi' }] });
  });

  test('includes a finite temperature only', () => {
    expect(buildProxyChatBody([], { model: 'x', temperature: 0.2 }).temperature).toBe(0.2);
    expect(buildProxyChatBody([], { model: 'x', temperature: 'hot' })).not.toHaveProperty('temperature');
    expect(buildProxyChatBody([], { model: 'x' })).not.toHaveProperty('temperature');
  });
});

describe('parseProxyStream', () => {
  test('collects delta frames and keeps the trailing partial line', () => {
    const r = parseProxyStream('data: {"delta":"Hel"}\ndata: {"delta":"lo"}\ndata: {"del');
    expect(r.deltas).toEqual(['Hel', 'lo']);
    expect(r.done).toBe(false);
    expect(r.error).toBeNull();
    expect(r.rest).toBe('data: {"del');
  });

  test('a done frame and [DONE] both finish the stream', () => {
    expect(parseProxyStream('data: {"done":true,"meta":{}}\n').done).toBe(true);
    expect(parseProxyStream('data: [DONE]\n').done).toBe(true);
  });

  test('surfaces a gateway error frame', () => {
    const r = parseProxyStream('data: {"error":"quota exhausted"}\n');
    expect(r.error).toBe('quota exhausted');
    expect(r.deltas).toEqual([]);
  });

  test('skips blanks, comments, empty payloads, torn JSON and empty deltas', () => {
    const r = parseProxyStream([
      '',                        // blank
      ': keep-alive',            // comment (not a data: line)
      'data:',                   // empty payload
      'data: {bad json',         // torn frame
      'data: {"delta":""}',      // empty delta (ignored)
      'data: {"foo":"bar"}',     // no delta/error/done
      '',
    ].join('\n'));
    expect(r.deltas).toEqual([]);
    expect(r.done).toBe(false);
    expect(r.error).toBeNull();
  });

  test('null/undefined buffer parses to an empty result', () => {
    expect(parseProxyStream(null)).toEqual({ deltas: [], done: false, error: null, rest: '' });
    expect(parseProxyStream(undefined)).toEqual({ deltas: [], done: false, error: null, rest: '' });
  });
});
