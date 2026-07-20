'use strict';

const { buildChatBody, parseChatStream } = require('../src/shared/llmChat');
const { LLM } = require('../src/shared/config');

describe('buildChatBody', () => {
  test('defaults model from config, streams, coerces content, normalizes roles', () => {
    const b = buildChatBody([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'yo' },
      { role: 'system', content: 'be nice' },
      { role: 'weird', content: 42 },
    ]);
    expect(b.model).toBe(LLM.model.name);
    expect(b.stream).toBe(true);
    expect(b.temperature).toBe(0.7);
    expect(b.messages).toEqual([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'yo' },
      { role: 'system', content: 'be nice' },
      { role: 'assistant', content: '42' }, // unknown role → assistant, content stringified
    ]);
  });

  test('honors overrides and tolerates non-array / missing content', () => {
    const b = buildChatBody(undefined, { model: 'm', stream: false, temperature: 0 });
    expect(b).toEqual({ model: 'm', messages: [], stream: false, temperature: 0 });
    const b2 = buildChatBody([{ role: 'user' }]);
    expect(b2.messages[0]).toEqual({ role: 'user', content: '' });
  });
});

describe('parseChatStream', () => {
  const frame = (content) => 'data: ' + JSON.stringify({ choices: [{ delta: { content } }] }) + '\n';

  test('extracts content deltas and keeps a trailing partial line as rest', () => {
    const r = parseChatStream(frame('Hel') + frame('lo') + 'data: {"choices":[{"delta":{"content":"!"');
    expect(r.deltas).toEqual(['Hel', 'lo']);
    expect(r.done).toBe(false);
    expect(r.rest).toBe('data: {"choices":[{"delta":{"content":"!"');
  });

  test('marks done on [DONE] and ignores non-data / torn / role-only frames', () => {
    const r = parseChatStream([
      ': keep-alive comment',
      'data: {"choices":[{"delta":{"role":"assistant"}}]}', // no content
      'data: not-json',                                     // torn → ignored
      frame('hi').trim(),
      'data: [DONE]',
      '',
    ].join('\n') + '\n');
    expect(r.deltas).toEqual(['hi']);
    expect(r.done).toBe(true);
    expect(r.rest).toBe('');
  });

  test('empty / null buffer yields nothing', () => {
    expect(parseChatStream('')).toEqual({ deltas: [], done: false, rest: '' });
    expect(parseChatStream(null)).toEqual({ deltas: [], done: false, rest: '' });
  });

  test('a missing-choices object contributes no delta', () => {
    const r = parseChatStream('data: {"foo":1}\n');
    expect(r.deltas).toEqual([]);
  });
});
