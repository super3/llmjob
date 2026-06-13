// Tests for the node telemetry scripts: the ping-body helpers in install.sh
// and the usage log-shipper in scripts/llmjob-usage.sh. The shell scripts are
// exercised directly — install.sh is sourced with LLMJOB_TEST_MODE so it
// defines its helpers without side effects, and the shipper is run in
// --parse mode (stdin -> JSON records on stdout, no network).
const { execFileSync } = require('child_process');
const path = require('path');

const ROOT = path.resolve(__dirname, '../..');

// Real llama.cpp journal lines captured from the target box (one request).
const PROMPT_LINE = 'slot print_timing: id  0 | task 6080 | prompt eval time =   11352.50 ms / 32734 tokens (    0.35 ms per token,  2883.42 tokens per second)';
const EVAL_LINE = 'slot print_timing: id  0 | task 6080 |        eval time =     817.82 ms /    43 tokens (   19.02 ms per token,    52.58 tokens per second)';
const TOTAL_LINE = 'slot print_timing: id  0 | task 6080 |       total time =   12170.32 ms / 32777 tokens';
const JOURNAL_FIXTURE = [PROMPT_LINE, EVAL_LINE, TOTAL_LINE].join('\n');

// Source install.sh in test mode and run a snippet against its helpers.
function sh(snippet, env = {}) {
  return execFileSync('sh', ['-c', `. ./install.sh; ${snippet}`], {
    cwd: ROOT,
    env: { ...process.env, LLMJOB_TEST_MODE: '1', ...env },
    encoding: 'utf8'
  });
}

// Feed journal lines to the usage shipper in --parse mode; returns stdout.
function ship(input, env = {}) {
  return execFileSync('bash', ['scripts/llmjob-usage.sh', '--parse'], {
    cwd: ROOT,
    input,
    env: {
      ...process.env,
      LLMJOB_MODEL: 'qwen3.6-27b',
      LLMJOB_NODE_NAME: 'node-ab12cd',
      LLMJOB_APP: 'hermes',
      ...env
    },
    encoding: 'utf8'
  });
}

describe('install.sh quant parsing', () => {
  test.each([
    ['Qwen_Qwen3.6-27B-Q6_K.gguf', 'Q6_K'],
    ['Llama-3.1-8B-Instruct-Q4_K_M.gguf', 'Q4_K_M'],
    ['Mistral-Small-IQ4_XS.gguf', 'IQ4_XS'],
    ['gemma-2-9b-it-Q8_0.gguf', 'Q8_0'],
    ['DeepSeek-V3-BF16.gguf', 'BF16']
  ])('extracts %s -> %s', (filename, quant) => {
    expect(sh(`parse_quant '${filename}'`).trim()).toBe(quant);
  });

  test('returns empty for a filename with no quant token', () => {
    expect(sh(`parse_quant 'some-model.gguf'`).trim()).toBe('');
  });
});

describe('install.sh VRAM rounding', () => {
  test('rounds MiB to whole GB (97887 -> 96, 39125 -> 38)', () => {
    expect(sh('mib_to_gb 97887')).toBe('96');
    expect(sh('mib_to_gb 39125')).toBe('38');
  });

  test('returns empty for non-numeric input', () => {
    expect(sh(`mib_to_gb 'N/A'`)).toBe('');
  });
});

describe('install.sh tps parsing', () => {
  test('returns the generation speed, not the prefill speed', () => {
    expect(sh('parse_tps "$FIXTURE"', { FIXTURE: JOURNAL_FIXTURE })).toBe('52.6');
  });

  test('picks the most recent eval line', () => {
    const older = EVAL_LINE.replace('52.58', '40.00');
    const fixture = [older, PROMPT_LINE, EVAL_LINE, TOTAL_LINE].join('\n');
    expect(sh('parse_tps "$FIXTURE"', { FIXTURE: fixture })).toBe('52.6');
  });

  test('returns empty when no eval line exists', () => {
    expect(sh('parse_tps "$FIXTURE"', { FIXTURE: PROMPT_LINE })).toBe('');
  });
});

describe('install.sh ping body builder', () => {
  const BASE = 'build_ping_body abc123 PUBKEY== SIG== 1733700000000';

  test('full telemetry produces valid JSON with correct types', () => {
    const out = sh(
      'DEVICE="NVIDIA RTX PRO 6000 Blackwell Workstation Edition"; ' +
      'VRAM_TOTAL=96; VRAM_USED=38; MODEL="qwen3.6-27b"; QUANT="Q6_K"; TPS=52.6; ' +
      BASE
    );
    const body = JSON.parse(out);
    expect(body).toEqual({
      nodeId: 'abc123',
      publicKey: 'PUBKEY==',
      signature: 'SIG==',
      timestamp: 1733700000000,
      device: 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition',
      vramTotal: 96,
      vramUsed: 38,
      model: 'qwen3.6-27b',
      quant: 'Q6_K',
      tps: 52.6
    });
    // Numbers must be unquoted so the server stores numeric types.
    expect(typeof body.vramTotal).toBe('number');
    expect(typeof body.tps).toBe('number');
  });

  test('omits absent fields entirely (no null, no empty string)', () => {
    const out = sh(
      'DEVICE=""; VRAM_TOTAL=""; VRAM_USED=""; MODEL=""; QUANT=""; TPS=""; ' + BASE
    );
    const body = JSON.parse(out);
    expect(Object.keys(body).sort()).toEqual(
      ['nodeId', 'publicKey', 'signature', 'timestamp'].sort()
    );
    expect(out).not.toContain('null');
    expect(out).not.toContain('""');
  });

  test('drops non-numeric values from numeric fields', () => {
    const out = sh('DEVICE="GPU"; TPS="not-a-number"; VRAM_TOTAL="1.2.3"; ' + BASE);
    const body = JSON.parse(out);
    expect(body.device).toBe('GPU');
    expect(body).not.toHaveProperty('tps');
    expect(body).not.toHaveProperty('vramTotal');
  });

  test('signature input is unchanged: base fields pass through verbatim', () => {
    const body = JSON.parse(sh(
      'DEVICE="GPU"; ' +
      'build_ping_body deadbe "pk+b64==" "sig/b64==" 1733700000000'
    ));
    expect(body.nodeId).toBe('deadbe');
    expect(body.signature).toBe('sig/b64==');
    expect(body.timestamp).toBe(1733700000000);
  });

  test('JSON-escapes string telemetry values', () => {
    const body = JSON.parse(sh('DEVICE=\'GPU "Pro" \\ Edition\'; ' + BASE));
    expect(body.device).toBe('GPU "Pro" \\ Edition');
  });
});

describe('install.sh dependency bootstrap', () => {
  const fs = require('fs');
  const os = require('os');
  let stubDir;

  // A PATH containing only the named stub executables.
  function makeStubs(...names) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'llmjob-stubs-'));
    for (const name of names) {
      fs.writeFileSync(path.join(dir, name), '#!/bin/sh\nexit 0\n', { mode: 0o755 });
    }
    return dir;
  }

  afterEach(() => {
    if (stubDir) fs.rmSync(stubDir, { recursive: true, force: true });
    stubDir = undefined;
  });

  test('detect_pkg_manager finds apt-get', () => {
    stubDir = makeStubs('apt-get');
    expect(sh('PATH="$STUBS"; detect_pkg_manager', { STUBS: stubDir })).toBe('apt-get');
  });

  test('detect_pkg_manager finds brew', () => {
    stubDir = makeStubs('brew');
    expect(sh('PATH="$STUBS"; detect_pkg_manager', { STUBS: stubDir })).toBe('brew');
  });

  test('detect_pkg_manager fails when no package manager exists', () => {
    stubDir = makeStubs();
    expect(
      sh('PATH="$STUBS"; detect_pkg_manager; echo "rc=$?"', { STUBS: stubDir }).trim()
    ).toBe('rc=1');
  });

  test('ensure_dep is a no-op when the command already exists', () => {
    expect(sh('ensure_dep sh; echo "rc=$?"').trim()).toBe('rc=0');
  });

  test('ensure_dep fails gracefully when install is impossible', () => {
    stubDir = makeStubs();
    const out = sh(
      'PATH="$STUBS"; ensure_dep llmjob-no-such-cmd 2>/dev/null; echo "rc=$?"',
      { STUBS: stubDir }
    );
    expect(out).toContain("Dependency 'llmjob-no-such-cmd' not found");
    const rc = Number(out.trim().match(/rc=(\d+)$/)[1]);
    expect(rc).not.toBe(0);
  });
});

describe('llmjob-usage.sh record assembly', () => {
  test('one task produces one complete usage record', () => {
    const lines = ship(JOURNAL_FIXTURE + '\n').trim().split('\n');
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0])).toEqual({
      model: 'qwen3.6-27b',
      node: 'node-ab12cd',
      app: 'hermes',
      in: 32734,
      out: 43,
      speed: 52.6,
      finish: 'stop'
    });
  });

  test('interleaved task ids are tracked independently', () => {
    const fixture = [
      'slot print_timing: id  0 | task 100 | prompt eval time =   1000.00 ms / 500 tokens (    2.00 ms per token,   500.00 tokens per second)',
      'slot print_timing: id  1 | task 200 | prompt eval time =   2000.00 ms / 900 tokens (    2.22 ms per token,   450.00 tokens per second)',
      'slot print_timing: id  1 | task 200 |        eval time =    500.00 ms /  20 tokens (   25.00 ms per token,    40.04 tokens per second)',
      'slot print_timing: id  0 | task 100 |        eval time =    400.00 ms /  10 tokens (   40.00 ms per token,    25.06 tokens per second)',
      'slot print_timing: id  1 | task 200 |       total time =   2500.00 ms / 920 tokens',
      'slot print_timing: id  0 | task 100 |       total time =   1400.00 ms / 510 tokens'
    ].join('\n');
    const records = ship(fixture + '\n').trim().split('\n').map(JSON.parse);
    expect(records).toHaveLength(2);
    expect(records[0]).toMatchObject({ in: 900, out: 20, speed: 40.0 });
    expect(records[1]).toMatchObject({ in: 500, out: 10, speed: 25.1 });
  });

  test('a task id is never emitted twice', () => {
    const out = ship(JOURNAL_FIXTURE + '\n' + TOTAL_LINE + '\n').trim();
    expect(out.split('\n')).toHaveLength(1);
  });

  test('an incomplete task (missing eval line) emits nothing', () => {
    expect(ship(PROMPT_LINE + '\n' + TOTAL_LINE + '\n').trim()).toBe('');
  });

  test('exits non-zero with a clear error when LLMJOB_API_KEY is unset', () => {
    const env = { ...process.env, LLMJOB_NODE_NAME: 'node-ab12cd' };
    delete env.LLMJOB_API_KEY;
    expect(() =>
      execFileSync('bash', ['scripts/llmjob-usage.sh'], {
        cwd: ROOT, input: '', env, encoding: 'utf8'
      })
    ).toThrow(/LLMJOB_API_KEY is not set/);
  });
});
