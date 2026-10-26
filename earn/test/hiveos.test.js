'use strict';

const H = require('../src/shared/hiveos');

describe('parseKeyValues', () => {
  test('reads KEY=VALUE lines and drops comments, blanks and quotes', () => {
    const text = '# a manifest\nCUSTOM_NAME=llmjob-earn\r\n\nCUSTOM_VERSION="0.7.0"\nLABEL=\'x\'\nnot a pair\n=nokey\n';
    expect(H.parseKeyValues(text)).toEqual({ CUSTOM_NAME: 'llmjob-earn', CUSTOM_VERSION: '0.7.0', LABEL: 'x' });
  });

  test('nothing in, nothing out', () => {
    expect(H.parseKeyValues(null)).toEqual({});
    expect(H.parseKeyValues(undefined)).toEqual({});
  });
});

describe('trial file', () => {
  test('round-trips, in the KEY=VALUE form h-run.sh reads with sed', () => {
    const text = H.formatTrial({ from: '0.6.9', to: '0.7.0', atMs: 1700000000999 });
    expect(text).toBe('UPDATE_FROM=0.6.9\nUPDATE_TO=0.7.0\nUPDATE_AT=1700000000\n');
    expect(H.parseTrial(text)).toEqual({ from: '0.6.9', to: '0.7.0' });
  });

  test('without a version to try, there is no trial', () => {
    expect(H.parseTrial('')).toBeNull();
    expect(H.parseTrial('UPDATE_FROM=0.6.9\n')).toBeNull();
  });

  test('a trial with no from still has its to', () => {
    expect(H.parseTrial('UPDATE_TO=0.7.0\n')).toEqual({ from: null, to: '0.7.0' });
  });
});

describe('isSkipped', () => {
  const skip = 'SKIP_VERSION=0.7.0\nSKIP_UNTIL=2000\n';
  test('holds back the version it names until the time it names', () => {
    expect(H.isSkipped(skip, '0.7.0', 1999999)).toBe(true);
    expect(H.isSkipped(skip, '0.7.0', 2000000)).toBe(false);
  });

  test('never holds back another version', () => {
    expect(H.isSkipped(skip, '0.7.1', 0)).toBe(false);
  });

  test('a missing or unreadable file holds back nothing', () => {
    expect(H.isSkipped('', '0.7.0', 0)).toBe(false);
    expect(H.isSkipped('SKIP_VERSION=0.7.0\nSKIP_UNTIL=soon\n', '0.7.0', 0)).toBe(false);
  });
});

test('packageAssetName is the versioned tarball', () => {
  expect(H.packageAssetName('0.7.0')).toBe('llmjob-earn-0.7.0.tar.gz');
});

describe('trialVerdict', () => {
  test('a hashrate passes at once', () => {
    expect(H.trialVerdict({ elapsedMs: 0, hashing: true, mines: true })).toBe('pass');
  });

  test('a run that should mine waits, then fails at ten minutes', () => {
    expect(H.trialVerdict({ elapsedMs: H.TRIAL_FAIL_MS - 1, hashing: false, mines: true })).toBe('wait');
    expect(H.trialVerdict({ elapsedMs: H.TRIAL_FAIL_MS, hashing: false, mines: true })).toBe('fail');
    expect(H.TRIAL_FAIL_MS).toBe(10 * 60 * 1000);
  });

  test('a run that mines nothing on purpose passes once it has run a while', () => {
    expect(H.trialVerdict({ elapsedMs: H.TRIAL_IDLE_PASS_MS - 1, hashing: false, mines: false })).toBe('wait');
    expect(H.trialVerdict({ elapsedMs: H.TRIAL_IDLE_PASS_MS, hashing: false, mines: false })).toBe('pass');
  });
});

describe('dashboard posts', () => {
  test('the same text goes out at most once an hour', () => {
    expect(H.shouldPost(undefined, 5)).toBe(true);
    expect(H.shouldPost(null, 5)).toBe(true);
    expect(H.shouldPost(1000, 1000 + H.MESSAGE_REPEAT_MS - 1)).toBe(false);
    expect(H.shouldPost(1000, 1000 + H.MESSAGE_REPEAT_MS)).toBe(true);
  });

  test('the record keeps recent posts and drops ones old enough to go out again', () => {
    const now = 10 * H.MESSAGE_REPEAT_MS;
    const posted = { old: now - H.MESSAGE_REPEAT_MS, recent: now - 1 };
    expect(H.recordPost(posted, 'new', now)).toEqual({ recent: now - 1, new: now });
    expect(H.recordPost(null, 'new', now)).toEqual({ new: now });
  });
});

describe('startupMessage', () => {
  test('a bad address points at the wallet template', () => {
    expect(H.startupMessage(['invalid Pearl address: krxXYZ', 'unknown option: --x']))
      .toBe('LLMJob Earn can\'t start: krxXYZ is not a Pearl address. Put your prl1p… address in the flight sheet\'s wallet template.');
    expect(H.startupMessage(['--address is required (your prl1p… payout address)']))
      .toBe('LLMJob Earn can\'t start: there is no wallet address. Put your prl1p… address in the flight sheet\'s wallet template.');
  });

  test('anything else points at Extra config', () => {
    expect(H.startupMessage(['unknown option: --difficulty', 'unknown region: mars (choices: us)']))
      .toBe('LLMJob Earn can\'t start: unknown option: --difficulty; unknown region: mars (choices: us). Check Extra config arguments in the flight sheet.');
  });
});

describe('coreLoadMessage', () => {
  test('a glibc or libstdc++ the system lacks means the image is too old', () => {
    const line = 'Pearl core: /hive/miners/custom/llmjob-earn/pearl_core.node is there but would not load: '
      + "/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found (required by pearl_core.node)";
    expect(H.coreLoadMessage(line)).toBe('LLMJob Earn can\'t mine: this HiveOS image is too old (the mining core '
      + 'needs GLIBC_2.34). Update the rig to the latest HiveOS image.');
    expect(H.coreLoadMessage("x: version 'GLIBCXX_3.4.29' not found")).toContain('needs GLIBCXX_3.4.29');
  });

  test('anything else is passed on, cut to a length the dashboard shows', () => {
    expect(H.coreLoadMessage('pearl_core_hip.node is there but would not load: libamdhip64.so.7: cannot open'))
      .toBe('LLMJob Earn can\'t mine: pearl_core_hip.node is there but would not load: libamdhip64.so.7: cannot open');
    const long = H.coreLoadMessage('x'.repeat(400));
    expect(long).toBe('LLMJob Earn can\'t mine: ' + 'x'.repeat(297) + '…');
    expect(H.coreLoadMessage(null)).toBe('LLMJob Earn can\'t mine: ');
  });
});

describe('Blackwell cards on an old driver', () => {
  const lspci = [
    '00:02.0 0300: 8086:3e92',            // an iGPU
    '01:00.0 0300: 10de:2684 (rev a1)',   // RTX 4090
    '01:00.1 0403: 10de:22ba (rev a1)',   // its audio
    '02:00.0 0300: 10de:2b85 (rev a1)',   // RTX 5090
    '03:00.0 0302: 10de:2901 (rev a1)',   // B200, a 3D controller
  ].join('\n');
  const proc550 = 'NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.127.05  Tue Oct 8 03:22:07 UTC 2024\n';
  const proc575 = 'NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  575.64.03  Release Build\n';

  test('finds the cards from 0x2900 up, display and 3D controllers only', () => {
    expect(H.blackwellSlots(lspci)).toEqual(['02:00.0', '03:00.0']);
    expect(H.blackwellSlots(null)).toEqual([]);
  });

  test('reads the driver major from either kernel module', () => {
    expect(H.driverMajor(proc550)).toBe(550);
    expect(H.driverMajor(proc575)).toBe(575);
    expect(H.driverMajor('')).toBeNull();
    expect(H.driverMajor(null)).toBeNull();
  });

  test('says which cards need which driver', () => {
    expect(H.driverMessage(lspci, proc550)).toBe('LLMJob Earn: 2 RTX 50 or other Blackwell cards need NVIDIA driver '
      + '570 or newer, and this rig has 550. Run nvidia-driver-update in Hive Shell.');
    expect(H.driverMessage('02:00.0 0300: 10de:2b85', '')).toBe('LLMJob Earn: the RTX 50 or other Blackwell card at '
      + '02:00.0 needs NVIDIA driver 570 or newer, and this rig has none loaded. Run nvidia-driver-update in Hive Shell.');
  });

  test('says nothing when the driver is new enough or there is no such card', () => {
    expect(H.driverMessage(lspci, proc575)).toBeNull();
    expect(H.driverMessage('01:00.0 0300: 10de:2684', proc550)).toBeNull();
  });
});
