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
