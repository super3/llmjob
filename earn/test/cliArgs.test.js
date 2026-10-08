'use strict';

const fs = require('fs');
const path = require('path');
const {
  parseCliArgs, buildSettings, regionChoices, USAGE, VALUE_FLAGS, ALIASES,
} = require('../src/shared/cliArgs');
const { DEFAULTS, LEGACY_REGIONS } = require('../src/shared/config');

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

  // Old HiveOS flight sheets carry --region eu1 in Extra config. It used to be
  // an "unknown region" exit, and HiveOS restarted the miner in a loop.
  test('an old AlphaPool region maps to the nearest one and remembers what was typed', () => {
    for (const [old, now] of Object.entries(LEGACY_REGIONS)) {
      const r = parseCliArgs(['--address', ADDR, '--region', old]);
      expect(r.errors).toEqual([]);
      expect(r.settings).toMatchObject({ region: now, legacyRegion: old, regionProvided: true });
    }
    expect(parseCliArgs(['-a', ADDR, '-r', 'eu1']).settings.region).toBe('de');
    expect(Object.keys(LEGACY_REGIONS).sort()).toEqual(['eu1', 'eu2', 'hk1', 'in1', 'ru1', 'sg1', 'us1']);
  });

  test('a current region has no legacyRegion; us2 exists on both pools', () => {
    expect(parseCliArgs(['-a', ADDR, '-r', 'de']).settings.legacyRegion).toBeNull();
    expect(parseCliArgs(['-a', ADDR, '-r', 'us2']).settings).toMatchObject({ region: 'us2', legacyRegion: null });
    expect(parseCliArgs(['-a', ADDR]).settings.legacyRegion).toBeNull();
  });

  // A typo is loud: no quiet fall back to the default region.
  test('a region that is neither is still rejected, even an Object property name', () => {
    for (const bad of ['eu3', 'EU1', 'constructor', '__proto__', 'toString']) {
      expect(parseCliArgs(['-a', ADDR, '-r', bad]).errors)
        .toContain('unknown region: ' + bad + ' (choices: ' + regionChoices() + ')');
    }
  });

  test('the help says old ids still work', () => {
    expect(USAGE).toContain('An old AlphaPool id (eu1, sg1, …) maps to the nearest.');
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

describe('buildSettings — compute mode / LLM', () => {
  test('mode defaults to auto, not provided, with null LLM paths', () => {
    const s = parseCliArgs(['--address', ADDR]).settings;
    expect(s.mode).toBe('auto');
    expect(s.modeProvided).toBe(false);
    expect(s.llmBinary).toBeNull();
    expect(s.llmModel).toBeNull();
  });

  test('explicit --mode is parsed and flagged as provided', () => {
    const s = parseCliArgs(['--address', ADDR, '--mode', 'both']).settings;
    expect(s.mode).toBe('both');
    expect(s.modeProvided).toBe(true);
  });

  test('--llm-max-instances is parsed, and rejected when not a positive integer', () => {
    expect(parseCliArgs(['--address', ADDR, '--llm-max-instances', '2']).settings.llmMaxInstances).toBe(2);
    // Unset means "no operator opinion" — the planner then uses every eligible card.
    expect(parseCliArgs(['--address', ADDR]).settings.llmMaxInstances).toBeNull();
    expect(parseCliArgs(['--address', ADDR, '--llm-max-instances', '0']).errors)
      .toContain('invalid --llm-max-instances: 0 (must be a positive integer)');
    expect(parseCliArgs(['--address', ADDR, '--llm-max-instances', 'two']).errors)
      .toContain('invalid --llm-max-instances: two (must be a positive integer)');
  });

  test('unknown mode is rejected with choices', () => {
    const r = parseCliArgs(['--address', ADDR, '--mode', 'turbo']);
    expect(r.errors).toContain('unknown mode: turbo (choices: mining, llm, auto)');
  });

  test('llm-only mode does not require a payout address', () => {
    const r = parseCliArgs(['--mode', 'llm']);
    expect(r.errors).toEqual([]);
    expect(r.settings.address).toBe('');
    expect(r.settings.mode).toBe('llm');
  });

  test('both/auto still require a payout address', () => {
    expect(parseCliArgs(['--mode', 'both']).errors)
      .toContain('--address is required (your prl1p… payout address)');
    expect(parseCliArgs(['--mode', 'auto']).errors)
      .toContain('--address is required (your prl1p… payout address)');
  });

  test('an invalid address is still rejected in llm mode', () => {
    const r = parseCliArgs(['--mode', 'llm', '--address', 'nope123']);
    expect(r.errors).toContain('invalid Pearl address: nope123');
  });

  test('--llm-binary and --llm-model are captured', () => {
    const s = parseCliArgs([
      '--mode', 'both', '--address', ADDR,
      '--llm-binary', '/opt/llama-server', '--llm-model', '/models/m.gguf',
    ]).settings;
    expect(s.llmBinary).toBe('/opt/llama-server');
    expect(s.llmModel).toBe('/models/m.gguf');
  });
});

describe('buildSettings — direct', () => {
  test('collects errors into the provided array; passes through report/update', () => {
    const errors = [];
    const s = buildSettings({}, errors, true, false);
    expect(s.address).toBe('');
    expect(s.report).toBe(true);
    expect(s.update).toBe(false);
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

describe('--gate-port', () => {
  test('accepts a port', () => {
    expect(parseCliArgs(['--address', ADDR, '--gate-port', '9000']).settings.gatePort).toBe(9000);
  });
  test('accepts 0, which asks the OS to pick (used by tests)', () => {
    expect(parseCliArgs(['--address', ADDR, '--gate-port', '0']).settings.gatePort).toBe(0);
  });
  test('rejects a non-port', () => {
    const r = parseCliArgs(['--address', ADDR, '--gate-port', '99999']);
    expect(r.errors.join(' ')).toContain('invalid --gate-port');
  });
  test('is null when not given, so config decides', () => {
    expect(parseCliArgs(['--address', ADDR]).settings.gatePort).toBeNull();
  });
});

describe('--gate-host', () => {
  const errs = () => [];

  test('defaults to null, letting the gate pick loopback', () => {
    expect(buildSettings({}, errs(), true, false).gateHost).toBeNull();
  });

  test('carries an explicit bind address through', () => {
    expect(buildSettings({ '--gate-host': '0.0.0.0' }, errs(), true, false).gateHost).toBe('0.0.0.0');
  });

  test('trims the value', () => {
    expect(buildSettings({ '--gate-host': '  1.2.3.4 ' }, errs(), true, false).gateHost).toBe('1.2.3.4');
  });

  test('rejects an empty value rather than reading it as all interfaces', () => {
    // `--gate-host ""` meaning 0.0.0.0 would be the opposite of what someone
    // clearing the setting expects.
    const errors = [];
    buildSettings({ '--gate-host': '   ' }, errors, true, false);
    expect(errors.join(' ')).toContain('--gate-host');
  });

  test('is listed in the usage text', () => {
    expect(USAGE).toContain('--gate-host');
  });
});

describe('--gate-quiet', () => {
  const errs = () => [];

  test('defaults to null, letting the product default stand', () => {
    expect(buildSettings({}, errs(), true, false).gateQuietMs).toBeNull();
  });

  test('seconds in, milliseconds out', () => {
    expect(buildSettings({ '--gate-quiet': '600' }, errs(), true, false).gateQuietMs).toBe(600000);
    expect(buildSettings({ '--gate-quiet': '1.5' }, errs(), true, false).gateQuietMs).toBe(1500);
  });

  // A rig dedicated to inference wants the model resident, full stop. Releasing
  // it drops the prompt cache, so the next turn re-prefills the whole context.
  test('0 means never hand the card back', () => {
    expect(buildSettings({ '--gate-quiet': '0' }, errs(), true, false).gateQuietMs).toBe(Infinity);
  });

  test('rejects a negative or non-numeric value', () => {
    for (const bad of ['-1', 'soon', '']) {
      const errors = [];
      buildSettings({ '--gate-quiet': bad }, errors, true, false);
      expect(errors.join(' ')).toContain('--gate-quiet');
    }
  });

  test('is listed in the usage text', () => {
    expect(USAGE).toContain('--gate-quiet');
  });

  test('takes a value, so the next argv item is not read as a flag', () => {
    const { settings } = parseCliArgs(['-a', ADDR, '--gate-quiet', '300', '--no-report']);
    expect(settings.gateQuietMs).toBe(300000);
  });
});

describe('--mine-mem-clock', () => {
  const errs = () => [];

  // Null is "not given": the RTX 5090 default then applies (shared/memClock).
  test('is null unless given', () => {
    expect(buildSettings({}, errs(), true, false).mineMemClockMhz).toBeNull();
  });

  // The way to turn the RTX 5090 default off. An empty value is not 0: a
  // flight sheet with `--mine-mem-clock ""` has lost its number, not asked for
  // the driver's clock.
  test('0 means leave the driver\'s clock, and only a real 0 does', () => {
    expect(buildSettings({ '--mine-mem-clock': '0' }, errs(), true, false).mineMemClockMhz).toBe(0);
    expect(buildSettings({ '--mine-mem-clock': ' 0 ' }, errs(), true, false).mineMemClockMhz).toBe(0);
    expect(parseCliArgs(['-a', ADDR, '--mine-mem-clock=0']).settings.mineMemClockMhz).toBe(0);
    const errors = [];
    expect(buildSettings({ '--mine-mem-clock': '' }, errors, true, false).mineMemClockMhz).toBeNull();
    expect(errors).toContain('invalid --mine-mem-clock:  (must be 0, or a whole number of MHz, 100-30000)');
  });

  test('carries a clock in MHz through', () => {
    expect(buildSettings({ '--mine-mem-clock': '7001' }, errs(), true, false).mineMemClockMhz).toBe(7001);
    expect(buildSettings({ '--mine-mem-clock': ' 405 ' }, errs(), true, false).mineMemClockMhz).toBe(405);
  });

  // The range catches the unit mistakes: `7` meaning 7 GHz, `7001000` meaning
  // kHz. Either would go straight to nvidia-smi as a clock no card runs at.
  test('rejects anything that is not a plausible whole number of MHz', () => {
    for (const bad of ['', ' ', '7', '99', '30001', '7001000', '7001.5', '-7001', 'fast']) {
      const errors = [];
      const s = buildSettings({ '--mine-mem-clock': bad }, errors, true, false);
      expect(errors).toContain('invalid --mine-mem-clock: ' + bad + ' (must be 0, or a whole number of MHz, 100-30000)');
      expect(s.mineMemClockMhz).toBeNull();
    }
  });

  test('accepts the edges of the range', () => {
    expect(buildSettings({ '--mine-mem-clock': '100' }, errs(), true, false).mineMemClockMhz).toBe(100);
    expect(buildSettings({ '--mine-mem-clock': '30000' }, errs(), true, false).mineMemClockMhz).toBe(30000);
  });

  test('takes a value, in either form', () => {
    expect(parseCliArgs(['-a', ADDR, '--mine-mem-clock', '7001', '--no-report']).settings.mineMemClockMhz).toBe(7001);
    expect(parseCliArgs(['-a', ADDR, '--mine-mem-clock=7001']).settings.mineMemClockMhz).toBe(7001);
    expect(parseCliArgs(['-a', ADDR, '--mine-mem-clock']).errors).toContain('missing value for --mine-mem-clock');
  });

  // The flag's help has to carry the reason to use it, the value to use, and
  // what it needs -- nobody reads a README before a flight sheet. Now that it
  // is on by default on the RTX 5090 it also has to say so, say how to turn it
  // off, and not claim a measurement for any card but the one measured.
  test('is documented with the number, the default, the off switch and the requirement', () => {
    expect(VALUE_FLAGS.has('--mine-mem-clock')).toBe(true);
    expect(USAGE).toContain('--mine-mem-clock <MHz>');
    expect(USAGE).toContain('7001');
    expect(USAGE).toContain('NOPASSWD');
    expect(USAGE).toMatch(/Default: 7001 on the RTX 5090/);
    expect(USAGE).toMatch(/0 leaves/);
    expect(USAGE).toMatch(/Only the\s+5090 has been measured/);
  });
});

// Which cards mine. HiveOS's h-config.sh writes this from the cards turned off
// there, so a bad value has to be loud, not read as "every card".
describe('--gpu-index', () => {
  test('not given: every card (null)', () => {
    expect(parseCliArgs(['-a', ADDR]).settings.gpuIndices).toBeNull();
  });

  test('a card or a comma list, in either form, sorted and deduplicated', () => {
    expect(parseCliArgs(['-a', ADDR, '--gpu-index', '1']).settings.gpuIndices).toEqual([1]);
    expect(parseCliArgs(['-a', ADDR, '--gpu-index=2,0']).settings.gpuIndices).toEqual([0, 2]);
    expect(parseCliArgs(['-a', ADDR, '--gpu-index', '0,2,0']).settings.gpuIndices).toEqual([0, 2]);
    expect(parseCliArgs(['-a', ADDR, '--gpu-index', ' 0 , 2 ']).settings.gpuIndices).toEqual([0, 2]);
  });

  test('none: an empty list', () => {
    const r = parseCliArgs(['-a', ADDR, '--gpu-index=none']);
    expect(r.errors).toEqual([]);
    expect(r.settings.gpuIndices).toEqual([]);
  });

  test('empty, or not card numbers: an error that says what it takes', () => {
    expect(parseCliArgs(['-a', ADDR, '--gpu-index=']).errors)
      .toContain('invalid --gpu-index:  (must not be empty)');
    expect(parseCliArgs(['-a', ADDR, '--gpu-index', 'GPU-1a2b']).errors).toContain(
      'invalid --gpu-index: GPU-1a2b (give GPU numbers from nvidia-smi separated by commas, such as 0,2, or none)');
    expect(parseCliArgs(['-a', ADDR, '--gpu-index', '0,']).errors[0]).toMatch(/^invalid --gpu-index: 0, /);
    expect(parseCliArgs(['-a', ADDR, '--gpu-index']).errors).toContain('missing value for --gpu-index');
  });

  // The user's Extra config comes after h-config.sh's flags, and a repeated
  // flag keeps its last value.
  test('given twice, the last one wins', () => {
    expect(parseCliArgs(['-a', ADDR, '--gpu-index=0,2', '--gpu-index', '1']).settings.gpuIndices).toEqual([1]);
  });

  test('is in the help, with the list form, none, and the variable', () => {
    expect(VALUE_FLAGS.has('--gpu-index')).toBe(true);
    expect(USAGE).toContain('--gpu-index <list>');
    expect(USAGE).toMatch(/such as\s+0,2, or "none" for no GPU/);
    expect(USAGE).toContain('PEARL_GPU_INDEX');
  });
});

// earn/README.md has a copy of the usage. It listed flags that had been
// removed (--difficulty, --binary) for several releases, so check it against
// what the parser takes and what --help lists.
describe('the README\'s usage block', () => {
  // Option rows only: "  -a, --address <…>" or "      --mode <…>".
  const flagsIn = (text) => {
    const flags = [];
    for (const line of text.split('\n')) {
      const m = line.match(/^ {2}(?:(-[a-z]), )? *(--[a-z][a-z0-9-]*)/);
      if (m) flags.push({ short: m[1] || null, long: m[2] });
    }
    return flags;
  };
  const readme = fs.readFileSync(path.join(__dirname, '..', 'README.md'), 'utf8');
  const block = readme.match(/```\nUsage: llmjob-earn-cli [\s\S]*?```/);
  const documented = block ? flagsIn(block[0]) : [];

  test('is there, and lists options', () => {
    expect(block).not.toBeNull();
    expect(documented.length).toBeGreaterThan(10);
  });

  test('names only flags the CLI takes, with the right short forms', () => {
    for (const { short, long } of documented) {
      expect(parseCliArgs([long]).errors).not.toContain('unknown option: ' + long);
      if (short) expect(ALIASES[short]).toBe(long);
    }
  });

  test('lists every flag --help lists', () => {
    const names = documented.map((f) => f.long);
    for (const { long } of flagsIn(USAGE)) expect(names).toContain(long);
  });
});
