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

  test('collects reasoning_content separately from content', () => {
    const r = parseChatStream([
      'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}',
      'data: {"choices":[{"delta":{"reasoning_content":"ing"}}]}',
      frame('answer').trim(),
      '',
    ].join('\n'));
    expect(r.reasoning).toEqual(['think', 'ing']);
    expect(r.deltas).toEqual(['answer']);
  });

  test('captures finish_reason, keeping the last one reported', () => {
    const cut = parseChatStream('data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n');
    expect(cut.finishReason).toBe('length');
    // A null finish_reason on every streaming frame but the last must not clobber it.
    const run = parseChatStream([
      'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}',
      'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
      '',
    ].join('\n'));
    expect(run.finishReason).toBe('stop');
  });

  test('a thinking model that spends its whole budget yields reasoning but no content', () => {
    // The silent-empty-answer case: max_tokens ran out mid-thought, so there is
    // reasoning to show and 'length' to report even though content is empty.
    const r = parseChatStream([
      'data: {"choices":[{"delta":{"reasoning_content":"still thinking"}}]}',
      'data: {"choices":[{"delta":{},"finish_reason":"length"}]}',
      'data: [DONE]',
      '',
    ].join('\n'));
    expect(r.deltas).toEqual([]);
    expect(r.reasoning).toEqual(['still thinking']);
    expect(r.finishReason).toBe('length');
    expect(r.done).toBe(true);
  });

  test('empty / null buffer yields nothing', () => {
    expect(parseChatStream('')).toEqual({ deltas: [], reasoning: [], finishReason: null, done: false, rest: '' });
    expect(parseChatStream(null)).toEqual({ deltas: [], reasoning: [], finishReason: null, done: false, rest: '' });
  });

  test('a missing-choices object contributes no delta', () => {
    const r = parseChatStream('data: {"foo":1}\n');
    expect(r.deltas).toEqual([]);
  });
});
