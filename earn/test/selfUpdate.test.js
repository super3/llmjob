'use strict';

const {
  LATEST_RELEASE_API, assetNameFor, normalizeVersion, compareVersions,
  isNewer, parseRelease, planUpdate,
} = require('../src/shared/selfUpdate');

describe('assetNameFor', () => {
  test('linux gets the CLI binary name', () => {
    expect(assetNameFor('linux')).toBe('llmjob-earn-cli-linux');
  });
  test('other platforms have no binary', () => {
    expect(assetNameFor('win32')).toBeNull();
    expect(assetNameFor('darwin')).toBeNull();
  });
});

describe('normalizeVersion', () => {
  test('strips a leading v and whitespace', () => {
    expect(normalizeVersion(' v0.1.11 ')).toBe('0.1.11');
    expect(normalizeVersion('V2.0.0')).toBe('2.0.0');
    expect(normalizeVersion('1.2.3')).toBe('1.2.3');
  });
  test('handles null/undefined', () => {
    expect(normalizeVersion(null)).toBe('');
    expect(normalizeVersion(undefined)).toBe('');
  });
});

describe('compareVersions', () => {
  test('greater / lesser / equal', () => {
    expect(compareVersions('0.1.12', '0.1.11')).toBe(1);
    expect(compareVersions('0.1.11', '0.1.12')).toBe(-1);
    expect(compareVersions('1.0.0', '1.0.0')).toBe(0);
  });
  test('different lengths and non-numeric parts count as zero', () => {
    expect(compareVersions('1.2', '1.2.0')).toBe(0);
    expect(compareVersions('1.2.1', '1.2')).toBe(1);
    expect(compareVersions('1.x', '1.0')).toBe(0);
  });
  test('major beats minor/patch', () => {
    expect(compareVersions('2.0.0', '1.9.9')).toBe(1);
  });
});

describe('isNewer', () => {
  test('true only when latest exceeds current', () => {
    expect(isNewer('0.1.11', '0.1.12')).toBe(true);
    expect(isNewer('0.1.11', '0.1.11')).toBe(false);
    expect(isNewer('0.1.12', '0.1.11')).toBe(false);
  });
});

describe('parseRelease', () => {
  test('extracts version and asset map', () => {
    const r = parseRelease({
      tag_name: 'v0.1.12',
      assets: [
        { name: 'llmjob-earn-cli-linux', browser_download_url: 'https://x/cli' },
        { name: 'LLMJob-Earn-0.1.12.AppImage', browser_download_url: 'https://x/app' },
      ],
    });
    expect(r.version).toBe('0.1.12');
    expect(r.assets['llmjob-earn-cli-linux']).toBe('https://x/cli');
  });
  test('falls back to name when tag_name is absent, tolerates missing/blank assets', () => {
    const r = parseRelease({ name: '0.2.0', assets: [null, { name: 'x' }, { foo: 1 }] });
    expect(r.version).toBe('0.2.0');
    expect(r.assets.x).toBeNull();
  });
  test('handles a null payload and non-array assets', () => {
    expect(parseRelease(null)).toEqual({ version: '', assets: {} });
    expect(parseRelease({ tag_name: 'v1.0.0', assets: 'nope' }).assets).toEqual({});
  });
});

describe('planUpdate', () => {
  const linuxRelease = {
    version: '0.1.12',
    assets: { 'llmjob-earn-cli-linux': 'https://x/cli' },
  };

  test('update available on a newer linux release', () => {
    const p = planUpdate({ currentVersion: '0.1.11', release: linuxRelease, platform: 'linux' });
    expect(p).toMatchObject({
      updateAvailable: true,
      reason: 'update-available',
      currentVersion: '0.1.11',
      latestVersion: '0.1.12',
      assetName: 'llmjob-earn-cli-linux',
      downloadUrl: 'https://x/cli',
    });
  });

  test('unsupported platform', () => {
    const p = planUpdate({ currentVersion: '0.1.11', release: linuxRelease, platform: 'win32' });
    expect(p.updateAvailable).toBe(false);
    expect(p.reason).toBe('unsupported-platform');
  });

  test('no release info', () => {
    const p = planUpdate({ currentVersion: '0.1.11', release: { version: '', assets: {} }, platform: 'linux' });
    expect(p.reason).toBe('no-release');
  });

  test('up to date', () => {
    const p = planUpdate({ currentVersion: '0.1.12', release: linuxRelease, platform: 'linux' });
    expect(p.reason).toBe('up-to-date');
  });

  test('newer release but the asset is missing', () => {
    const p = planUpdate({ currentVersion: '0.1.11', release: { version: '0.1.12', assets: {} }, platform: 'linux' });
    expect(p.reason).toBe('asset-missing');
    expect(p.updateAvailable).toBe(false);
  });

  test('defaults: no opts and no release object', () => {
    const p = planUpdate();
    expect(p.updateAvailable).toBe(false);
    // no platform -> unsupported
    expect(p.reason).toBe('unsupported-platform');
  });
});

describe('constants', () => {
  test('latest-release API points at the repo', () => {
    expect(LATEST_RELEASE_API).toContain('super3/llmjob');
    expect(LATEST_RELEASE_API).toContain('releases/latest');
  });
});
