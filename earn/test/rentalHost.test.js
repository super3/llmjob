'use strict';

// The real side of rented containers, with child_process, the server and the
// GPU probes faked.

jest.mock('child_process', () => ({ execFile: jest.fn(), spawn: jest.fn() }));
jest.mock('../src/main/io', () => ({ postJson: jest.fn() }));
jest.mock('../src/main/probe', () => ({
  detectGpusVram: jest.fn(), detectAmdGpus: jest.fn(), detectAmdBusIds: jest.fn(),
}));
jest.mock('../src/main/nodeStore', () => ({
  nodePath: jest.fn(() => '/store/node.json'),
  getOrCreateNode: jest.fn(),
}));
jest.mock('../src/main/nodeClient', () => ({ serverOf: jest.fn(() => 'https://srv') }));

const { EventEmitter } = require('events');
const path = require('path');
const fs = require('fs');
const { execFile, spawn } = require('child_process');
const { postJson } = require('../src/main/io');
const probe = require('../src/main/probe');
const nodeStore = require('../src/main/nodeStore');
const nodeProto = require('../src/shared/node');
const host = require('../src/main/rentalHost');
const { RentalRunner } = require('../src/main/rentalRunner');

// execFile answers: a map from "bin first-arg" to [err, stdout, stderr].
function answer(map) {
  execFile.mockImplementation((bin, args, opts, cb) => {
    const key = bin + ' ' + args[0];
    const a = map[key] || [Object.assign(new Error('not found'), { code: 'ENOENT' }), '', ''];
    cb(a[0], a[1], a[2]);
  });
}

const fail = (code, stderr = '') => [Object.assign(new Error('exit ' + code), { code }), '', stderr];

describe('rentalHost', () => {
  const env = { ...process.env };
  beforeEach(() => {
    jest.clearAllMocks();
    delete process.env.LLMJOB_DOCKER;
    delete process.env.LLMJOB_COSIGN;
    delete process.env.LLMJOB_RENT_REQUIRE_SIGNATURE;
  });
  afterAll(() => { process.env = env; });

  test('exec resolves the code, never rejects', async () => {
    answer({ 'x ok': [null, 'out', 'err'], 'x bad': fail(3, 'boom') });
    expect(await host.exec('x', ['ok'], { A: '1' })).toEqual({ code: 0, stdout: 'out', stderr: 'err' });
    expect(execFile.mock.calls[0][2].env.A).toBe('1');
    expect(await host.exec('x', ['bad'], {})).toEqual({ code: 3, stdout: '', stderr: 'boom' });
    // A missing binary is 127, with the error's words.
    expect(await host.exec('x', ['none'], {})).toEqual({ code: 127, stdout: '', stderr: 'not found' });
    execFile.mockImplementationOnce((b, a, o, cb) => cb(null, undefined, undefined));
    expect(await host.exec('x', ['y'], {})).toEqual({ code: 0, stdout: '', stderr: '' });
  });

  test('docker: available, run, cleanup and wait, through LLMJOB_DOCKER', async () => {
    process.env.LLMJOB_DOCKER = '/fake/docker';
    answer({
      '/fake/docker version': [null, '27.0', ''],
      '/fake/docker ps': [null, 'aaa\nbbb\n', ''],
      '/fake/docker rm': [null, '', ''],
      '/fake/docker wait': [null, '3\n', ''],
      '/fake/docker pull': [null, 'pulled', ''],
    });
    expect(await host.docker.available()).toBe(true);
    expect((await host.docker.run(['pull', 'img'], { K: 'v' })).stdout).toBe('pulled');
    expect(await host.docker.cleanup()).toBe(2);
    expect(execFile.mock.calls.find((c) => c[1][0] === 'rm')[1]).toEqual(['rm', '-f', 'aaa', 'bbb']);
    expect(await host.docker.wait('n')).toBe(3);

    answer({ '/fake/docker ps': [null, '', ''], '/fake/docker wait': fail(1, 'no such container\nmore') });
    expect(await host.docker.cleanup()).toBe(0);
    await expect(host.docker.wait('n')).rejects.toThrow('no such container');
    expect(await host.docker.available()).toBe(false);
    answer({ '/fake/docker ps': fail(1), '/fake/docker wait': [null, 'x', ''] });
    expect(await host.docker.cleanup()).toBe(0);
    await expect(host.docker.wait('n')).rejects.toThrow('no exit code');
  });

  test('docker.logs follows both streams line by line', () => {
    const child = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.kill = jest.fn();
    spawn.mockReturnValue(child);
    const lines = [];
    const f = host.docker.logs('n', (l) => lines.push(l));
    expect(spawn).toHaveBeenCalledWith('docker', ['logs', '-f', 'n'], expect.any(Object));
    child.stdout.emit('data', Buffer.from('a\nb'));
    child.stderr.emit('data', 'e1\n');
    child.stdout.emit('data', 'c\n');
    child.emit('error', new Error('gone'));
    expect(lines).toEqual(['a', 'e1', 'bc']);
    f.stop();
    expect(child.kill).toHaveBeenCalled();
  });

  test('cosign: available and verify with our identity', async () => {
    process.env.LLMJOB_COSIGN = '/fake/cosign';
    answer({ '/fake/cosign version': [null, 'v2', ''], '/fake/cosign verify': [null, 'ok', ''] });
    expect(await host.cosign.available()).toBe(true);
    expect((await host.cosign.verify('img')).code).toBe(0);
    const args = execFile.mock.calls.pop()[1];
    delete process.env.LLMJOB_COSIGN;
    answer({});
    expect(await host.cosign.available()).toBe(false);
    expect(execFile.mock.calls.pop()[0]).toBe('cosign');
    expect(args).toEqual(['verify', '--certificate-identity-regexp', host.COSIGN_IDENTITY,
      '--certificate-oidc-issuer', host.COSIGN_ISSUER, 'img']);
  });

  test('rentalCards: NVIDIA first, else AMD', async () => {
    probe.detectGpusVram.mockResolvedValueOnce([{ index: 0, name: 'RTX 4090', usedMb: 1, totalMb: 24564 }]);
    expect(await host.rentalCards()).toEqual([{ index: 0, gpu: 'RTX 4090', vramMb: 24564, vendor: 'nvidia' }]);
    probe.detectGpusVram.mockResolvedValueOnce([]);
    probe.detectAmdGpus.mockReturnValueOnce([{ index: 0, name: 'RX 6800', usedMb: 0, totalMb: 16368 }]);
    expect(await host.rentalCards()).toEqual([{ index: 0, gpu: 'RX 6800', vramMb: 16368, vendor: 'amd' }]);
  });

  test('driverVersion', async () => {
    answer({ 'nvidia-smi --query-gpu=driver_version': [null, '580.65\n580.65\n', ''] });
    expect(await host.driverVersion()).toBe('580.65');
    answer({ 'nvidia-smi --query-gpu=driver_version': [null, '\n', ''] });
    expect(await host.driverVersion()).toBeNull();
    answer({});
    expect(await host.driverVersion()).toBeNull();
  });

  test('parseNvidiaTemps', () => {
    expect(host.parseNvidiaTemps('0, 65, 40, 99\n1, [N/A], 0, 80\nbad\n')).toEqual({
      0: { core: 65, hotspot: null, fan: 40, util: 99 },
      1: { core: null, hotspot: null, fan: 0, util: 80 },
    });
  });

  test('amdReading reads edge, junction, fan and load from sysfs', () => {
    const files = {
      '/sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon4/temp1_input': '61000\n',
      '/sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon4/temp2_input': '78000',
      '/sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon4/fan1_input': '1500',
      '/sys/bus/pci/devices/0000:03:00.0/gpu_busy_percent': '97',
    };
    const fsx = {
      readFileSync: (p) => { if (p in files) return files[p]; throw new Error('ENOENT'); },
      readdirSync: (p) => { if (p.endsWith('0000:03:00.0/hwmon')) return ['name', 'hwmon4']; throw new Error('ENOENT'); },
    };
    expect(host.amdReading('0000:03:00.0', fsx)).toEqual({ core: 61, hotspot: 78, fan: 1500, util: 97 });
    expect(host.amdReading('0000:09:00.0', fsx)).toEqual({ core: null, hotspot: null, fan: null, util: null });
    const noHwmon = { readFileSync: fsx.readFileSync, readdirSync: () => ['name'] };
    expect(host.amdReading('0000:03:00.0', noHwmon)).toEqual({ core: null, hotspot: null, fan: null, util: 97 });
    // A hwmon folder missing its readings.
    const empty = { readFileSync: () => { throw new Error('ENOENT'); }, readdirSync: () => ['hwmon0'] };
    expect(host.amdReading('0000:03:00.0', empty)).toEqual({ core: null, hotspot: null, fan: null, util: null });
    // The default fs on a machine with no such card.
    expect(host.amdReading('none').core).toBeNull();
  });

  test('rentalTemps: nvidia-smi, else the AMD cards', async () => {
    answer({ 'nvidia-smi --query-gpu=index,temperature.gpu,fan.speed,utilization.gpu': [null, '0, 70, 50, 99', ''] });
    expect(await host.rentalTemps()).toEqual({ 0: { core: 70, hotspot: null, fan: 50, util: 99 } });
    answer({});
    probe.detectAmdBusIds.mockReturnValueOnce({ 0: '0000:03:00.0' });
    const fsx = { readFileSync: () => { throw new Error('x'); }, readdirSync: () => { throw new Error('x'); } };
    expect(await host.rentalTemps(fsx)).toEqual({ 0: { core: null, hotspot: null, fan: null, util: null } });
  });

  test('requestTakeBack writes the file beside node.json', () => {
    const fsx = { writeFileSync: jest.fn() };
    const p = host.requestTakeBack(fsx);
    expect(p).toBe(path.join('/store', 'rental-takeback'));
    expect(fsx.writeFileSync).toHaveBeenCalledWith(p, expect.any(String));
    const write = jest.spyOn(fs, 'writeFileSync').mockImplementationOnce(() => {});
    host.requestTakeBack();
    expect(write).toHaveBeenCalledWith(p, expect.any(String));
    write.mockRestore();
  });

  test('rentalApi signs every call as this machine', async () => {
    const kp = nodeProto.generateKeypair();
    nodeStore.getOrCreateNode.mockReturnValue({ nodeId: 'abcdef0123456789', publicKey: kp.publicKey, secretKey: kp.secretKey });
    postJson.mockResolvedValue({ status: 200, data: { job: null } });
    const api = host.rentalApi();
    expect(await api.poll({ cards: [] })).toEqual({ job: null });
    const [url, body] = postJson.mock.calls[0];
    expect(url).toBe('https://srv/api/rigs/poll');
    expect(body).toMatchObject({ nodeId: 'abcdef0123456789', publicKey: kp.publicKey, cards: [] });
    expect(body.signature).toEqual(expect.any(String));
    expect(JSON.stringify(body)).not.toContain(kp.secretKey);
    await api.heartbeat('cr/1', { state: 'running' });
    expect(postJson.mock.calls[1][0]).toBe('https://srv/api/rigs/runs/cr%2F1/heartbeat');
    await api.end('cr_1', { reason: 'exited' });
    expect(postJson.mock.calls[2][0]).toBe('https://srv/api/rigs/runs/cr_1/end');
    postJson.mockResolvedValueOnce({ status: 401, data: { error: 'Node id does not match its key' } });
    await expect(api.poll({})).rejects.toThrow('HTTP 401: Node id does not match its key');
    postJson.mockResolvedValueOnce({ status: 502, data: null });
    await expect(api.poll({})).rejects.toThrow(/^HTTP 502$/);
  });

  test('createRental wires a runner to all of the above', async () => {
    process.env.LLMJOB_RENT_REQUIRE_SIGNATURE = '1';
    const opts = { log: jest.fn(), pauseCard: jest.fn(), resumeCard: jest.fn(), llmBusy: () => false, worker: 'w' };
    const r = host.createRental(opts);
    expect(r).toBeInstanceOf(RentalRunner);
    expect(r.d).toMatchObject({ worker: 'w', platform: process.platform, requireSignature: true });
    expect(r.d.caps.cpus).toBeGreaterThanOrEqual(1);
    const exists = jest.spyOn(fs, 'existsSync').mockReturnValueOnce(true);
    expect(r.d.takeBackRequested()).toBe(true);
    exists.mockRestore();
    const unlink = jest.spyOn(fs, 'unlinkSync').mockImplementationOnce(() => {}).mockImplementationOnce(() => { throw new Error('gone'); });
    r.d.clearTakeBack();
    r.d.clearTakeBack();
    expect(unlink).toHaveBeenCalledTimes(2);
    unlink.mockRestore();
    answer({});
    probe.detectAmdBusIds.mockReturnValueOnce({});
    expect(await r.d.temps()).toEqual({});
    expect(host.createRental(Object.assign({ platform: 'win32' }, opts)).d.platform).toBe('win32');
    expect(host.takeBackPath()).toBe(path.join('/store', 'rental-takeback'));
  });
});
