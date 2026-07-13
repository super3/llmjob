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

  test('honors model + temperature + maxTokens when set', () => {
    expect(jobToChatBody({ prompt: 'hi', model: 'm', temperature: 0.2, maxTokens: 128 })).toEqual({
      model: 'm',
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
});
