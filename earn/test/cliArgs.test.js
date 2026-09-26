'use strict';

const {
  parseCliArgs, buildSettings, regionChoices, USAGE, VALUE_FLAGS, RETIRED_VALUE_FLAGS, RETIRED_SWITCHES,
} = require('../src/shared/cliArgs');
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

  test('--no-update flips update to false; defaults to true', () => {
    expect(parseCliArgs(['--address', ADDR]).settings.update).toBe(true);
    const r = parseCliArgs(['--address', ADDR, '--no-update']);
    expect(r.update).toBe(false);
    expect(r.settings.update).toBe(false);
  });

  test('--help preserves the update flag in the short-circuit result', () => {
    expect(parseCliArgs(['--help', '--no-update']).update).toBe(false);
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
      '-a', ADDR, '-m', MDL, '-r', 'de', '-w', 'rig7',
      '-g', 'RTX 4090',
      '--stats-file', '/run/hive/llmjob-earn-stats.json',
    ]);
    expect(r.errors).toEqual([]);
    expect(r.settings).toMatchObject({
      address: ADDR,
      mdlAddress: MDL,
      region: 'de',
      worker: 'rig7',
      gpu: 'RTX 4090',
      statsFile: '/run/hive/llmjob-earn-stats.json',
      report: true,
      update: true,
      regionProvided: true,
      gpuProvided: true,
      workerProvided: true,
    });
  });

  test('*Provided flags are false when the knobs are omitted (auto-detect eligible)', () => {
    const s = parseCliArgs(['--address', ADDR]).settings;
    expect(s.regionProvided).toBe(false);
    expect(s.gpuProvided).toBe(false);
    expect(s.workerProvided).toBe(false);
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


  test('gpu is null when omitted', () => {
    const s = parseCliArgs(['--address', ADDR]).settings;
    expect(s.gpu).toBeNull();
    expect(s.statsFile).toBeNull();
  });
});

// The LLM options were retired with the LLM, but an old unit or flight sheet
// still passing them must keep mining after the CLI updates itself — so they
// parse, are ignored, and are reported back for one startup notice.
describe('retired LLM options', () => {
  test('are accepted with their values, ignored, and listed as retired', () => {
    const r = parseCliArgs([
      '--address', ADDR, '--mode', 'auto', '--llm-binary', '/opt/llama-server',
      '--llm-model=/m.gguf', '--llm-max-instances', '2', '--gate-port', '8000',
      '--gate-host', '0.0.0.0', '--gate-quiet', '60', '--no-serve', '--worker', 'rig3',
    ]);
    expect(r.errors).toEqual([]);
    expect(r.settings.retired).toEqual([
      '--mode', '--llm-binary', '--llm-model', '--llm-max-instances',
      '--gate-port', '--gate-host', '--gate-quiet', '--no-serve',
    ]);
    // Their values are consumed, not mistaken for the next option.
    expect(r.settings.worker).toBe('rig3');
    expect(r.settings).not.toHaveProperty('mode');
    expect(r.settings).not.toHaveProperty('gatePort');
  });

  test('a retired option with no value is still just ignored, not an error', () => {
    const r = parseCliArgs(['--address', ADDR, '--mode', '--no-report']);
    expect(r.errors).toEqual([]);
    expect(r.settings.retired).toEqual(['--mode']);
    expect(r.settings.report).toBe(false);
    expect(parseCliArgs(['--address', ADDR, '--gate-quiet']).errors).toEqual([]);
  });

  test('nothing retired means an empty list', () => {
    expect(parseCliArgs(['--address', ADDR]).settings.retired).toEqual([]);
  });

  test('an old LLM-only unit is told why it now needs an address', () => {
    expect(parseCliArgs(['--mode', 'llm']).errors).toEqual([
      '--address is required (your prl1p… payout address); --mode llm was retired and this build only mines',
    ]);
    // Any other old mode just gets the plain message.
    expect(parseCliArgs(['--mode', 'auto']).errors)
      .toEqual(['--address is required (your prl1p… payout address)']);
  });

  test('are not documented any more', () => {
    for (const flag of [...RETIRED_VALUE_FLAGS, ...RETIRED_SWITCHES]) expect(USAGE).not.toContain(flag);
    expect(USAGE).not.toContain('connect');
  });
});

describe('buildSettings — direct', () => {
  test('collects errors into the provided array; passes through report/update', () => {
    const errors = [];
    const s = buildSettings({}, errors, true, false);
    expect(s.address).toBe('');
    expect(s.report).toBe(true);
    expect(s.update).toBe(false);
    expect(s.retired).toEqual([]);
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
