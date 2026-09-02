'use strict';

const { jobToChatBody } = require('../src/shared/jobs');
const { LLM } = require('../src/shared/config');

describe('jobToChatBody', () => {
  test('wraps the prompt as a user message, defaults the model, streams', () => {
    expect(jobToChatBody({ prompt: 'Hello' })).toEqual({
      model: LLM.model.name,
      messages: [{ role: 'user', content: 'Hello' }],
      stream: true,
    });
  });

  test('honors temperature + maxTokens, but ignores the job model for the node default', () => {
    // The node always serves its own loaded model; a job's requested model must not
    // reach llama-server (it would come back as metrics.model — the model that "ran").
    expect(jobToChatBody({ prompt: 'hi', model: 'gpt-4', temperature: 0.2, maxTokens: 128 })).toEqual({
      model: LLM.model.name,
      messages: [{ role: 'user', content: 'hi' }],
      stream: true,
      temperature: 0.2,
      max_tokens: 128,
    });
  });

  test('reports the model the node actually loaded, not the fleet default', () => {
    // The whole point of the tier: a 5090 running Qwen must not tell the gateway
    // it served Gemma. metrics.model is copied straight from this field, and
    // openaiController.modelName puts it in the `model` of every completion —
    // so getting it wrong here misreports the model through the public API.
    const qwen = { name: 'Qwen3.8-27B-UD-Q4_K_XL' };
    expect(jobToChatBody({ prompt: 'hi' }, qwen).model).toBe('Qwen3.8-27B-UD-Q4_K_XL');
    // A job's own `model` still loses to what is loaded — that rule is unchanged.
    expect(jobToChatBody({ prompt: 'hi', model: 'gpt-4' }, qwen).model).toBe('Qwen3.8-27B-UD-Q4_K_XL');
  });

  test('falls back to the fleet default for a caller that passes no model', () => {
    // An un-wired call site reports the old answer rather than `undefined`, which
    // would reach llama-server and the job record as a missing model name.
    expect(jobToChatBody({ prompt: 'hi' }, null).model).toBe(LLM.model.name);
    expect(jobToChatBody({ prompt: 'hi' }, {}).model).toBe(LLM.model.name);
  });

  test('drops non-finite temperature/maxTokens and coerces a missing prompt', () => {
    const b = jobToChatBody({ temperature: 'x', maxTokens: null });
    expect(b.messages[0].content).toBe('');
    expect(b).not.toHaveProperty('temperature');
    expect(b).not.toHaveProperty('max_tokens');
  });

  test('tolerates no job at all', () => {
    expect(jobToChatBody().messages[0].content).toBe('');
  });

  test('uses a full messages array (multi-turn) over the prompt when present', () => {
    const b = jobToChatBody({
      messages: [
        { role: 'system', content: 'Be terse.' },
        { role: 'user', content: 'Hi' },
        { role: 'assistant', content: 'Hey.' },
        { role: 'user', content: 'Again?' },
      ],
      prompt: 'ignored when messages is set',
    });
    expect(b.messages).toEqual([
      { role: 'system', content: 'Be terse.' },
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hey.' },
      { role: 'user', content: 'Again?' },
    ]);
  });

  test('defaults a message role to user and coerces null content; ignores an empty messages array', () => {
    const b = jobToChatBody({ messages: [null, { content: null }] });
    expect(b.messages).toEqual([
      { role: 'user', content: '' },
      { role: 'user', content: '' },
    ]);
    // Empty array falls back to the single-prompt path.
    expect(jobToChatBody({ messages: [], prompt: 'fallback' }).messages).toEqual([{ role: 'user', content: 'fallback' }]);
  });
});

describe('completion floor on a reasoning tier', () => {
  const { LLM } = require('../src/shared/config');
  const qwen = LLM.tiers.find((t) => t.minCompletionTokens);

  test('a small max_tokens is raised to the model floor', () => {
    // Measured: at 60 the model spends the whole budget thinking and returns
    // content "". The floor is what stops a caller getting an empty string.
    const b = jobToChatBody({ prompt: 'hi', maxTokens: 60 }, qwen);
    expect(b.max_tokens).toBe(qwen.minCompletionTokens);
  });

  test('a generous max_tokens is left exactly as asked', () => {
    const b = jobToChatBody({ prompt: 'hi', maxTokens: 4000 }, qwen);
    expect(b.max_tokens).toBe(4000);
  });

  test('a model without a floor is untouched — Gemma does not reason', () => {
    const b = jobToChatBody({ prompt: 'hi', maxTokens: 20 }, LLM.model);
    expect(b.max_tokens).toBe(20);
  });

  test('no max_tokens means no max_tokens, floor or not', () => {
    expect(jobToChatBody({ prompt: 'hi' }, qwen).max_tokens).toBeUndefined();
  });
});
