'use strict';

const { parseCliArgs, buildSettings, regionChoices, USAGE, VALUE_FLAGS } = require('../src/shared/cliArgs');
const { DEFAULTS } = require('../src/shared/config');

const ADDR = 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c';
const MDL = 'mdl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c';

describe('parseCliArgs — flags', () => {
  test('non-array argv is treated as empty (address required)', () => {
    const r = parseCliArgs(undefined);
    expect(r.settings.address).toBe('');
    expect(r.errors).toContain('--address is required (your prl1p… payout address)');
  });

  test('--help short-circuits with no settings', () => {
    const r = parseCliArgs(['--help', '--address', ADDR]);
    expect(r.help).toBe(true);
    expect(r.settings).toBeNull();
    expect(r.errors).toEqual([]);
  });

  test('-h alias maps to --help', () => {
    expect(parseCliArgs(['-h']).help).toBe(true);
  });

  test('--version / -v short-circuits', () => {
    expect(parseCliArgs(['--version']).version).toBe(true);
    expect(parseCliArgs(['-v']).version).toBe(true);
    expect(parseCliArgs(['-v']).settings).toBeNull();
  });

  test('--no-report flips report to false', () => {
    const r = parseCliArgs(['--address', ADDR, '--no-report']);
    expect(r.report).toBe(false);
    expect(r.settings.report).toBe(false);
  });

  test('report defaults to true', () => {
    expect(parseCliArgs(['--address', ADDR]).settings.report).toBe(true);
  });

  test('--flag=value form', () => {
    const r = parseCliArgs(['--address=' + ADDR, '--worker=rig9']);
    expect(r.settings.address).toBe(ADDR);
    expect(r.settings.worker).toBe('rig9');
  });

  test('short alias with separate value', () => {
    const r = parseCliArgs(['-a', ADDR, '-w', 'rig5']);
    expect(r.settings.address).toBe(ADDR);
    expect(r.settings.worker).toBe('rig5');
  });

  test('missing value at end of argv', () => {
    const r = parseCliArgs(['--address']);
    expect(r.errors).toContain('missing value for --address');
  });

  test('missing value when next token is a flag', () => {
    const r = parseCliArgs(['--address', '--worker', 'rig1']);
    expect(r.errors).toContain('missing value for --address');
    expect(r.settings.worker).toBe('rig1');
  });

  test('unknown option is reported', () => {
    const r = parseCliArgs(['--address', ADDR, '--bogus']);
    expect(r.errors).toContain('unknown option: --bogus');
  });

  test('bare positional token is unknown', () => {
    const r = parseCliArgs(['whoops']);
    expect(r.errors).toContain('unknown option: whoops');
  });
});

describe('buildSettings — validation', () => {
  test('a full valid command parses cleanly', () => {
    const r = parseCliArgs([
      '-a', ADDR, '-m', MDL, '-r', 'eu1', '-w', 'rig7',
      '-d', '131072', '-g', 'RTX 4090', '--backend', 'ampere',
      '-b', '/opt/alpha-miner', '--engine-dir', '/tmp/eng',
    ]);
    expect(r.errors).toEqual([]);
    expect(r.settings).toMatchObject({
      address: ADDR,
      mdlAddress: MDL,
      region: 'eu1',
      worker: 'rig7',
      difficulty: 131072,
      gpu: 'RTX 4090',
      backend: 'ampere',
      binaryPath: '/opt/alpha-miner',
      engineDir: '/tmp/eng',
      report: true,
    });
  });

  test('empty address value triggers the required error', () => {
    const r = parseCliArgs(['--address=']);
    expect(r.errors).toContain('--address is required (your prl1p… payout address)');
  });

  test('invalid Pearl address is rejected', () => {
    const r = parseCliArgs(['--address', 'nope123']);
    expect(r.errors).toContain('invalid Pearl address: nope123');
  });

  test('invalid MDL address is rejected but Pearl still parses', () => {
    const r = parseCliArgs(['--address', ADDR, '--mdl', 'mdl1pbad']);
    expect(r.errors).toContain('invalid MDL address: mdl1pbad');
    expect(r.settings.mdlAddress).toBeNull();
  });

  test('no MDL leaves mdlAddress null', () => {
    expect(parseCliArgs(['--address', ADDR]).settings.mdlAddress).toBeNull();
  });

  test('unknown region is rejected with choices', () => {
    const r = parseCliArgs(['--address', ADDR, '--region', 'mars']);
    expect(r.errors).toContain('unknown region: mars (choices: ' + regionChoices() + ')');
  });

  test('region defaults when omitted', () => {
    expect(parseCliArgs(['--address', ADDR]).settings.region).toBe(DEFAULTS.region);
  });

  test('worker defaults when omitted', () => {
    expect(parseCliArgs(['--address', ADDR]).settings.worker).toBe(DEFAULTS.worker);
  });

  test('explicit difficulty wins', () => {
    expect(parseCliArgs(['--address', ADDR, '-d', '262144']).settings.difficulty).toBe(262144);
  });

  test('non-integer difficulty is rejected', () => {
    const r = parseCliArgs(['--address', ADDR, '--difficulty=abc']);
    expect(r.errors).toContain('invalid difficulty: abc (must be a positive integer)');
  });

  test('zero/negative difficulty is rejected', () => {
    const r = parseCliArgs(['--address', ADDR, '--difficulty=0']);
    expect(r.errors).toContain('invalid difficulty: 0 (must be a positive integer)');
  });

  test('difficulty is inferred from --gpu when not given', () => {
    // 3070 maps to 131072 in the per-card table (distinct from the default).
    const r = parseCliArgs(['--address', ADDR, '--gpu', 'RTX 3070']);
    expect(r.settings.difficulty).toBe(131072);
  });

  test('difficulty falls back to the default without gpu or flag', () => {
    expect(parseCliArgs(['--address', ADDR]).settings.difficulty).toBe(DEFAULTS.difficulty);
  });

  test('gpu is null when omitted; backend/binary/engineDir null when omitted', () => {
    const s = parseCliArgs(['--address', ADDR]).settings;
    expect(s.gpu).toBeNull();
    expect(s.backend).toBeNull();
    expect(s.binaryPath).toBeNull();
    expect(s.engineDir).toBeNull();
  });
});

describe('buildSettings — direct', () => {
  test('collects errors into the provided array', () => {
    const errors = [];
    const s = buildSettings({}, errors, true);
    expect(s.address).toBe('');
    expect(errors.length).toBeGreaterThan(0);
  });
});

describe('metadata', () => {
  test('USAGE mentions the required address flag', () => {
    expect(USAGE).toContain('--address');
  });

  test('VALUE_FLAGS includes the value-taking options', () => {
    expect(VALUE_FLAGS.has('--address')).toBe(true);
    expect(VALUE_FLAGS.has('--help')).toBe(false);
  });
});
