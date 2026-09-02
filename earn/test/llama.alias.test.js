'use strict';

const { buildServerArgs } = require('../src/shared/llama');

describe('llama-server --alias', () => {
  // Without --alias, llama-server names itself after the model PATH: /v1/models
  // and the `model` field of every completion come back as an absolute local
  // filesystem path. Observed on a 5090 serving the Qwen3.8 tier before this was
  // added -- the reported id was the full /home/.../Qwen3.8-27B-UD-Q4_K_XL.gguf,
  // which leaks the node's directory layout and does not match the model name
  // the fleet reports everywhere else.
  test('passes the model name so the server does not report a filesystem path', () => {
    const args = buildServerArgs({
      modelPath: '/var/lib/llmjob/llm/Qwen3.8-27B-UD-Q4_K_XL.gguf',
      alias: 'Qwen3.8-27B-UD-Q4_K_XL',
    });
    const i = args.indexOf('--alias');
    expect(i).toBeGreaterThan(-1);
    expect(args[i + 1]).toBe('Qwen3.8-27B-UD-Q4_K_XL');
  });

  test('is omitted when no alias is supplied, rather than passing an empty flag', () => {
    expect(buildServerArgs({ modelPath: '/m.gguf' })).not.toContain('--alias');
  });
});
